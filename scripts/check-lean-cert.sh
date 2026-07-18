#!/usr/bin/env bash
# Lean-certificate drift guard — the cross-repo analog of
# scripts/check-release-consistency.sh, for the `pounce certify` emitter.
#
# The emitter (`pounce certify`) and the external pounce-lean codegen
# (`codegen/gen_lean.py` → `lake build`) talk only through the
# `pounce.lean-cert/v1` schema. If either side drifts from that contract a
# certificate silently stops verifying. This guard pins both directions against
# committed golden fixtures.
#
# Two layers, by cost:
#
#   1. POUNCE-side (ALWAYS, fast, no Lean toolchain): regenerate the golden
#      `cert.json` from the committed `.nl`/`.sol` and diff byte-for-byte. The
#      emitter is deterministic (exact rational arithmetic + content-addressed
#      hashes of fixed bytes), so any change in emitted bytes is real drift.
#      This is the part wired into POUNCE CI — it keeps the multi-GB Mathlib
#      build off POUNCE's critical path, exactly as the architecture intends.
#
#   2. Cross-repo (OPT-IN, set POUNCE_LEAN_DIR=/path/to/pounce-lean): run that
#      repo's codegen on the golden cert and diff the golden `expected.lean`;
#      and if LAKE_BUILD=1, `lake build` the generated module so the whole
#      emit → codegen → kernel-check loop is exercised. The lake build proper
#      lives in pounce-lean's own CI (its check_fixtures.py); this is for local
#      end-to-end validation.
#
# Usage:
#   scripts/check-lean-cert.sh                          # layer 1 only
#   POUNCE_LEAN_DIR=../pounce-lean scripts/check-lean-cert.sh
#   POUNCE_LEAN_DIR=../pounce-lean LAKE_BUILD=1 scripts/check-lean-cert.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

FIX="crates/pounce-cli/tests/fixtures"

# Each fixture: "<basename> <Lean module>". The basename names the committed
# <basename>.{nl,cert.json,expected.lean} triple. The `.sol` is NOT committed —
# `$FIX/.gitignore` excludes `*.sol` because they are solver byproducts — so we
# solve each `.nl` below to produce it. That also makes this a true end-to-end
# check: f64 solve, then exact certification of the refined point.
FIXTURES=(
  "certify_qp    PounceLean.CertifyQP"     # free variables, one general constraint
  "certify_box   PounceLean.CertifyBox"    # box variable bounds (folded to rows)
  "certify_range PounceLean.CertifyRange"  # two-sided range constraint (split to rows)
  "certify_eq    PounceLean.CertifyEq"     # equality constraint (free-sign μ = -1)
  "certify_lp    PounceLean.CertifyLP"     # LP: Q = 0, vertex optimum at 4/3 (not an f64)
)

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# `pounce certify` stamps the live crate version into `binding.solver`
# (certify.rs: format!("pounce {}", env!("CARGO_PKG_VERSION"))), and the codegen
# copies it into the generated .lean header. That is correct for a real
# certificate -- it is provenance -- but it means a freshly emitted cert differs
# from a committed golden on every release version bump, failing this guard for
# a reason that has nothing to do with drift.
#
# So normalize just that one token before diffing. Deliberately narrow: only
# `pounce <semver>` is rewritten, so every other byte is still compared exactly.
# This is safe because `solver` is pure metadata -- it is not part of the
# problem re-derivation `cert-verify` performs, and no theorem mentions it.
nv() { sed -E 's/pounce [0-9]+\.[0-9]+\.[0-9]+/pounce <version>/g' "$1" >"$2"; }

# Diff two files with the solver version normalized. Args: label, expected, actual
diff_nv() {
  nv "$2" "$tmp/.exp.nv"
  nv "$3" "$tmp/.act.nv"
  diff -u --label "$2" --label "$3" "$tmp/.exp.nv" "$tmp/.act.nv"
}

# --- layer 1: emitter reproduces the golden certificate ---------------------
echo "== certificate regeneration (pounce certify) =="
if [[ -n "${POUNCE_BIN:-}" ]]; then
  PNC=("$POUNCE_BIN")
else
  echo "  building pounce (cargo) ..."
  cargo build -q -p pounce-cli --bin pounce
  PNC=("target/debug/pounce")
fi

for entry in "${FIXTURES[@]}"; do
  read -r base module <<<"$entry"
  golden_cert="$FIX/$base.cert.json"
  # Produce the (gitignored) .sol by actually solving; see FIXTURES comment.
  if ! "${PNC[@]}" "$FIX/$base.nl" >/dev/null 2>&1; then
    echo "FAIL — $base: solve of $FIX/$base.nl failed" >&2
    exit 1
  fi
  if [[ ! -f "$FIX/$base.sol" ]]; then
    echo "FAIL — $base: solve did not write $FIX/$base.sol" >&2
    exit 1
  fi
  "${PNC[@]}" certify "$FIX/$base.nl" "$FIX/$base.sol" -o "$tmp/$base.cert.json"
  if ! diff_nv "$base" "$golden_cert" "$tmp/$base.cert.json"; then
    echo "FAIL — emitted certificate drifted from $golden_cert" >&2
    echo "       (intentional? regenerate: pounce certify $FIX/$base.nl $FIX/$base.sol -o $golden_cert)" >&2
    exit 1
  fi
  echo "  OK — $base: emitted cert matches golden"
  # Consumer-side binding check: the golden cert must verify against its own .nl
  # (re-derived problem == cert.problem, hash matches).
  if ! "${PNC[@]}" cert-verify "$FIX/$base.nl" "$golden_cert" >/dev/null; then
    echo "FAIL — $base: cert-verify rejected the golden cert against its own .nl" >&2
    exit 1
  fi
  echo "  OK — $base: cert-verify binds cert ↔ .nl"
done

# --- layer 2 (opt-in): codegen + optional lake build ------------------------
if [[ -z "${POUNCE_LEAN_DIR:-}" ]]; then
  echo "== codegen / lake build: SKIPPED (set POUNCE_LEAN_DIR to enable) =="
  echo "check-lean-cert: OK (layer 1, ${#FIXTURES[@]} fixtures)"
  exit 0
fi

GEN="$POUNCE_LEAN_DIR/codegen/gen_lean.py"
if [[ ! -f "$GEN" ]]; then
  echo "FAIL — POUNCE_LEAN_DIR=$POUNCE_LEAN_DIR has no codegen/gen_lean.py" >&2
  exit 1
fi

# Files we may write into the pounce-lean checkout for LAKE_BUILD; cleaned up.
declare -a placed=()
cleanup() { rm -rf "$tmp"; for f in "${placed[@]}"; do rm -f "$f"; done; }
trap cleanup EXIT

for entry in "${FIXTURES[@]}"; do
  read -r base module <<<"$entry"
  golden_lean="$FIX/$base.expected.lean"

  echo "== $base: codegen reproduces the golden .lean =="
  python3 "$GEN" "$FIX/$base.cert.json" -m "$module" -o "$tmp/$base.lean"
  if ! diff_nv "$base" "$golden_lean" "$tmp/$base.lean"; then
    echo "FAIL — codegen drifted from $golden_lean" >&2
    echo "       (regenerate: python3 $GEN $FIX/$base.cert.json -m $module -o $golden_lean)" >&2
    exit 1
  fi
  echo "  OK — $base: codegen output matches golden"

  if [[ "${LAKE_BUILD:-0}" == "1" ]]; then
    echo "== $base: lake build + axiom audit ($module) =="
    dest="$POUNCE_LEAN_DIR/${module//.//}.lean"   # PounceLean.CertifyQP -> PounceLean/CertifyQP.lean
    if [[ -e "$dest" ]]; then
      # pounce-lean now commits these modules as its own regressions (they are
      # byte-identical to our goldens — same cert, same codegen, same -m). Do
      # not overwrite a tracked file; verify it instead. A mismatch here is a
      # genuine cross-repo drift and the whole point of this guard.
      if ! diff_nv "$base" "$golden_lean" "$dest"; then
        echo "FAIL — $base: pounce-lean's committed $module drifted from our golden" >&2
        exit 1
      fi
      echo "  OK — $base: pounce-lean's committed module matches our golden"
    else
      mkdir -p "$(dirname "$dest")"
      cp "$golden_lean" "$dest"
      placed+=("$dest")
    fi
    # Audit the trust base of the verdict: print the axioms `global_min` rests
    # on. `lake build` exits 0 even on a `sorry` (it only warns), so the exit
    # code alone is NOT sufficient — the axiom set is the real gate.
    #
    # The audit lives in a throwaway module rather than being appended to
    # `$dest`, so a committed module is never mutated. PounceLean/Generated/
    # gitignores *.lean precisely for this.
    audit_mod="PounceLean.Generated.Audit_$base"
    audit_dest="$POUNCE_LEAN_DIR/${audit_mod//.//}.lean"
    mkdir -p "$(dirname "$audit_dest")"
    printf 'import %s\n\n#print axioms %s.global_min\n' "$module" "$module" > "$audit_dest"
    placed+=("$audit_dest")

    # Build the audit module, not `$module` — the `#print axioms` line lives
    # there, and building it pulls in `$module` as a dependency anyway.
    out="$( cd "$POUNCE_LEAN_DIR" && lake build "$audit_mod" 2>&1 )" || {
      printf '%s\n' "$out" | grep -iE "error" | head -5 >&2
      echo "FAIL — $base: lake build failed" >&2
      exit 1
    }
    # The `#print axioms` info line for global_min.
    axline="$(printf '%s\n' "$out" | grep "global_min' depends on axioms" || true)"
    if [[ -z "$axline" ]]; then
      echo "FAIL — $base: no axiom report for $module.global_min (did the theorem build?)" >&2
      exit 1
    fi
    if printf '%s' "$axline" | grep -q "sorryAx"; then
      echo "FAIL — $base: proof depends on 'sorryAx' (a sorry slipped through a green build)" >&2
      exit 1
    fi
    # Anything beyond Lean's three standard axioms is a forbidden trust escalation.
    extras="$(printf '%s' "$axline" \
      | sed 's/.*\[//; s/\].*//; s/,/ /g' | tr ' ' '\n' \
      | grep -vE '^(propext|Classical\.choice|Quot\.sound)?$' || true)"
    if [[ -n "$extras" ]]; then
      echo "FAIL — $base: proof rests on non-standard axioms: $(echo "$extras" | tr '\n' ' ')" >&2
      exit 1
    fi
    echo "  OK — $base: kernel-checks; axioms = {propext, Classical.choice, Quot.sound}, no sorry"
  fi
done

echo "check-lean-cert: OK"
