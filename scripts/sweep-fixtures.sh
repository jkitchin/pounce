#!/usr/bin/env bash
# Trajectory sweep over the CLI fixture corpus.
#
# Runs every fixture in crates/pounce-cli/tests/fixtures and records status,
# objective AND iteration count, one line per model, sorted and diffable.
#
# WHY THIS EXISTS (pounce gh#592, and gh#544 before it). The CLI test suite
# asserts *status* and *objective*. It does not assert trajectory length, and
# a change to the step computation can leave both of those untouched while
# taking four times as many iterations to get there. That is exactly what
# gh#544 did to pooling_rt2stp -- 206 -> 812 iterations -- and nothing in the
# suite could see it; it surfaced three days before the 0.10.0 release as a
# wall-clock timeout, was misattributed, and the cap was raised. The defect it
# was a symptom of shipped, and came back as gh#592.
#
# So: any change that reroutes WHICH correction the solver reaches for, or
# reorders/rescales the steps it takes, needs this sweep. "It cannot produce a
# wrong answer" is not the relevant safety property -- trajectory changes are
# invisible to the answer.
#
# TWO LEGS, BOTH RUN BY DEFAULT (pounce #677). Every fixture is swept twice:
# once on the default exact-Hessian path, once with
# `hessian_approximation=limited-memory`. The L-BFGS leg exists because this
# corpus ran exact-Hessian only, and that blind spot shipped a wrong default
# for the L-BFGS initial Hessian scalar: `limited_memory_initialization` was
# registered and never read, so every limited-memory solve used `scalar2`
# where Ipopt uses `scalar1`. The unit tests could not see it -- they check
# that `initial_hessian_scalar` computes each formula correctly, which is
# true either way, and nothing asserted which formula was SELECTED. The
# corpus could not see it either, because it never ran the path at all.
#
# The L-BFGS leg is not exotic coverage. Both the Python frontend and the
# CasADi plugin switch to `limited-memory` on their own whenever no exact
# Lagrangian Hessian is available, so it is what an embedder gets by
# default -- reached without any user ever typing the option.
#
# Usage:
#   scripts/sweep-fixtures.sh <pounce-binary> <outfile> [extra solver opts...]
#
#   git stash && cargo build --release && cp target/release/pounce /tmp/p-base
#   git stash pop && cargo build --release
#   scripts/sweep-fixtures.sh /tmp/p-base            /tmp/base.txt
#   scripts/sweep-fixtures.sh target/release/pounce  /tmp/new.txt
#   diff /tmp/base.txt /tmp/new.txt
#
# Output lines are prefixed with the leg (`exact` / `lbfgs`) so one file
# covers both and stays sorted and diffable. Extra solver opts, if given,
# apply to both legs.
#
# An empty diff is the expected result for a change that is not meant to move
# the corpus. Every line that does move should be explainable before merge --
# not after a user reports it.
set -uo pipefail

BIN="${1:?usage: sweep-fixtures.sh <pounce-binary> <outfile> [opts...]}"
OUT="${2:?usage: sweep-fixtures.sh <pounce-binary> <outfile> [opts...]}"
shift 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FX="$ROOT/crates/pounce-cli/tests/fixtures"
[ -d "$FX" ] || { echo "no fixture dir at $FX" >&2; exit 2; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
: > "$OUT"

# Sweep every fixture under one leg. $1 is the leg name; the rest are
# solver options that define it.
sweep_leg() {
  leg="$1"; shift
  for f in "$FX"/*.nl; do
    n=$(basename "$f" .nl)
    # A fixture that hangs must not hang the sweep; it shows up as NO_JSON.
    timeout 300 "$BIN" "$f" "$W/$n.sol" --json-output "$W/$n.json" "$@" \
      >/dev/null 2>&1
    if [ -f "$W/$n.json" ]; then
      python3 - "$W/$n.json" "$n" "$leg" >> "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
s = d.get("solution", {})
st = d.get("statistics", {})
obj = s.get("objective")
print("%-6s %-40s %-32s it=%-6s obj=%s" % (
    sys.argv[3],
    sys.argv[2],
    s.get("status"),
    st.get("iteration_count", "?"),
    "%.10g" % obj if obj is not None else "none",
))
PY
      # Each leg re-uses the same scratch names; a stale json from the
      # previous leg would otherwise be reported as this leg's result.
      rm -f "$W/$n.json"
    else
      printf '%-6s %-40s NO_JSON\n' "$leg" "$n" >> "$OUT"
    fi
  done
}

sweep_leg exact "$@"
sweep_leg lbfgs hessian_approximation=limited-memory "$@"

sort -o "$OUT" "$OUT"
echo "swept $(wc -l < "$OUT" | tr -d ' ') fixture-legs (exact + lbfgs) -> $OUT"
