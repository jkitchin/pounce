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

# Each fixture: "<basename> <Lean module> <theorem> [certify flags]". The
# basename names the committed <basename>.{nl,cert.json,expected.lean} triple.
# The `.sol` is usually NOT committed — `$FIX/.gitignore` excludes `*.sol`
# because they are solver byproducts — so we solve each `.nl` below to produce
# it. That also makes this a true end-to-end check: f64 solve, then exact
# certification of the refined point. A fixture whose `.sol` *is* committed
# (see certify_feasible) is used as committed instead of re-solved.
# Third field: the theorem the axiom audit targets. It is NOT always
# `global_min` — an `infeasible` certificate proves `infeasible` instead, and
# hardcoding the optimality name silently skipped that fixture's real check.
# Fourth field (optional): extra flags for `pounce certify`.
FIXTURES=(
  "certify_qp    PounceLean.CertifyQP    global_min"  # free vars, one general constraint
  "certify_box   PounceLean.CertifyBox   global_min"  # box variable bounds (folded to rows)
  "certify_range PounceLean.CertifyRange global_min"  # two-sided range (split to rows)
  "certify_eq    PounceLean.CertifyEq    global_min"  # equality (free-sign μ = -1)
  "certify_lp    PounceLean.CertifyLP    global_min"  # LP: Q = 0, optimum 4/3 (not an f64)
  "certify_infeasible PounceLean.CertifyInfeasible infeasible"  # no solution; Farkas ray
  "certify_unbounded  PounceLean.CertifyUnbounded  unbounded"   # unbounded below; recession
  # The same verdict with a nonzero Hessian, and the difference is the whole
  # point: for an LP the recession conditions are all inequalities, which
  # survive f64→ℚ, so the solver's diverging iterate IS the direction. A
  # nonzero Q adds the equality `Q d = 0`, which a float never satisfies, so
  # `d` here is the exact projection of that iterate onto ker Q — visibly
  # ![0, 1] rather than a 16-digit dyadic.
  "certify_unbounded_qp PounceLean.CertifyUnboundedQP unbounded"  # curved but flat along d
  # The two SOS fixtures differ in *verdict*, and the difference is the point:
  # x⁴−2x²+2 attains its bound at the rational x = 1, so it certifies a global
  # minimum; x⁴−3x²+2 minimizes at ±√(3/2), which no rational point reaches, so
  # the honest verdict stays a bound. Losing either one would hide a regression
  # in exactly one direction.
  "certify_sos        PounceLean.CertifySOS        global_min"  # nonconvex quartic; bound attained
  "certify_sos_bound  PounceLean.CertifySOSBound   global_lb"   # irrational minimizer; bound only
  # The one verdict that is not about optimality: an indefinite Hessian, so the
  # global-min path refuses (Ldl(Indefinite)) and there is nothing to certify
  # about the solve EXCEPT that its answer is a real point of the real feasible
  # set. Its `.sol` is committed rather than re-solved, because unlike every
  # other fixture the certificate is about the *float* the solver returned — a
  # regenerated `.sol` would make this golden track solver noise instead of
  # emitter behaviour.
  "certify_feasible   PounceLean.CertifyFeasible   feasible_point_exists --feasible"
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
  read -r base module thm flags <<<"$entry"
  golden_cert="$FIX/$base.cert.json"
  if git ls-files --error-unmatch "$FIX/$base.sol" >/dev/null 2>&1; then
    # A committed .sol is the fixture; re-solving would overwrite it.
    echo "  .. $base: using the committed .sol (not re-solving)"
  else
    # Produce the (gitignored) .sol by actually solving; see FIXTURES comment.
    # The exit code is deliberately ignored: an INFEASIBLE solve exits 1 while
    # still writing a perfectly good .sol, and that .sol is exactly what the
    # `infeasible` fixture certifies. What matters is that the file appears.
    "${PNC[@]}" "$FIX/$base.nl" >/dev/null 2>&1 || true
    if [[ ! -f "$FIX/$base.sol" ]]; then
      echo "FAIL — $base: solve did not write $FIX/$base.sol" >&2
      exit 1
    fi
  fi
  # shellcheck disable=SC2086  # $flags is a deliberate word-split of the 4th field
  "${PNC[@]}" certify $flags "$FIX/$base.nl" "$FIX/$base.sol" -o "$tmp/$base.cert.json"
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
  read -r base module thm _flags <<<"$entry"
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
    printf 'import %s\n\n#print axioms %s.%s\n' "$module" "$module" "$thm" > "$audit_dest"
    placed+=("$audit_dest")

    # Build the audit module, not `$module` — the `#print axioms` line lives
    # there, and building it pulls in `$module` as a dependency anyway.
    # Generated proofs run with an unlimited heartbeat budget (see gen_lean.py:
    # no constant is right for every problem size). Wall clock is bounded
    # HERE instead, which is what actually keeps CI from hanging. afiro-sized
    # instances take ~8 minutes, so the default is generous.
    out="$( cd "$POUNCE_LEAN_DIR" && timeout "${LEAN_BUILD_TIMEOUT:-1800}" \
        lake build "$audit_mod" 2>&1 )" || {
      printf '%s\n' "$out" | grep -iE "error" | head -5 >&2
      echo "FAIL — $base: lake build failed" >&2
      exit 1
    }
    # The `#print axioms` info line for global_min.
    axline="$(printf '%s\n' "$out" | grep "$thm' depends on axioms" || true)"
    if [[ -z "$axline" ]]; then
      echo "FAIL — $base: no axiom report for $module.$thm (did the theorem build?)" >&2
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


# --- layer 3 (needs LAKE_BUILD): forged witnesses must NOT verify -----------
#
# The whole trust model rests on one claim: a wrong witness makes the proof
# fail to typecheck, so a certificate can never prove something false. Until
# this layer existed that claim was asserted in every document and tested
# nowhere. A guard that only checks *good* certificates cannot tell a sound
# pipeline from one that proves anything it is handed.
#
# So: each fixture below carries a deliberately corrupted witness. The codegen
# must still ACCEPT it — codegen does not judge witnesses, that is the point —
# and `lake build` must then REJECT it. A forged certificate that builds is a
# soundness failure and fails this script.
#
# The axiom audit in layer 2 catches a `sorry`; it cannot catch "this witness
# is wrong and we proved it anyway". That is what this layer is for.
if [[ "${LAKE_BUILD:-0}" == "1" ]]; then
  # "<basename> <module> <what was corrupted>"
  NEGATIVE_FIXTURES=(
    "certify_qp_forged_dual       PounceLean.Generated.ForgedDual  KKT dual (breaks stationarity)"
    "certify_qp_forged_psd        PounceLean.Generated.ForgedPsd   objective Hessian (Q indefinite)"
    "certify_infeasible_forged    PounceLean.Generated.ForgedFarkas Farkas ray (breaks Aᵀy = 0)"
    # The SOS verdict rests on TWO obligations, so it needs two forgeries: one
    # each must be able to reject on its own.
    "certify_sos_forged_bound     PounceLean.Generated.ForgedSosBound bound γ (identity no longer closes)"
    "certify_sos_forged_psd       PounceLean.Generated.ForgedSosPsd   Gram (satisfies the identity but is indefinite)"
    # Attainment is a third, independent obligation: this cert's bound, Gram and
    # identity are all genuine — only the exhibited minimizer is wrong. Every
    # other check passes, so `p xstar = γ` is the sole thing standing between a
    # true bound and a false claim about where it is reached.
    "certify_sos_forged_candidate PounceLean.Generated.ForgedSosCand  minimizer (bound holds, but not at that point)"
    # The `feasible` verdict's existence claim rests on two properties of the
    # witness — that it is exactly feasible, and that it is within ε of the
    # reported point — and neither implies the other, so each gets a forgery
    # that leaves the other intact. (The third obligation, ε-feasibility of the
    # candidate, shares ε with the closeness check and so cannot be corrupted
    # in isolation; it is covered by both of these.)
    "certify_feasible_forged_witness PounceLean.Generated.ForgedFeasWitness witness (within ε of x*, but not feasible — x* itself)"
    "certify_feasible_forged_far     PounceLean.Generated.ForgedFeasFar     witness (exactly feasible, but far outside ε)"
    # `Q d = 0` is the obligation that only exists once Q is nonzero, so it
    # needs its own forgery. This direction still satisfies every condition the
    # LP slice ever checked — A d = 2 ≥ 0, c·d = −1 < 0 — and fails only the
    # new one. Without it, a regression that dropped `hQd` would still pass.
    "certify_unbounded_qp_forged_dir PounceLean.Generated.ForgedRecessionDir direction (feasible and descending, but Q d ≠ 0)"
  )
  echo "== forged witnesses must be rejected by the kernel =="
  for entry in "${NEGATIVE_FIXTURES[@]}"; do
    read -r base module what <<<"$entry"
    dest="$POUNCE_LEAN_DIR/${module//.//}.lean"
    mkdir -p "$(dirname "$dest")"
    if ! python3 "$GEN" "$FIX/$base.cert.json" -m "$module" -o "$dest" 2>/dev/null; then
      echo "FAIL — $base: codegen refused it, so the kernel is never exercised." >&2
      echo "       Codegen must accept forged witnesses; only Lean may reject them." >&2
      rm -f "$dest"
      exit 1
    fi
    placed+=("$dest")
    if ( cd "$POUNCE_LEAN_DIR" && timeout "${LEAN_BUILD_TIMEOUT:-1800}" \
         lake build "$module" >/dev/null 2>&1 ); then
      echo "FAIL — $base: a certificate with a corrupted $what BUILT." >&2
      echo "       This is a soundness failure: the proof does not depend on the" >&2
      echo "       witness being correct." >&2
      exit 1
    fi
    echo "  OK — $base: rejected by the kernel (corrupted $what)"
  done
fi
echo "check-lean-cert: OK"
