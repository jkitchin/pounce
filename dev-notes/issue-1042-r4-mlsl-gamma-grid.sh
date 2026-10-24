#!/usr/bin/env bash
# gh#1042 R4 / R4b: MLSL 10-seed gamma grid.
#
# Runs `--minima mlsl` at the sweep-minima.sh budget (--n-minima 6
# --max-solves 30 --patience 8) over seeds 0-9 for the five fixtures that
# find a minimum in that sweep, at each gamma given, and writes one TSV row
# per run. The embedded models are taken verbatim from scripts/sweep-minima.sh
# at run time, so the grid and the sweep can never drift apart.
#
# Columns: fixture gamma seed status solves found exit_optimal
# exit_infeasible exit_other. The exit_* columns count the per-solve `EXIT:`
# banners on stdout, which include the base-problem dual-recovery re-solve
# `recover_duals` runs once per *accepted* minimum when m > 0 (the two
# constrained fixtures). So for circle_parabola_t0p5 / haverly:
#   launches  = exit_optimal + exit_infeasible + exit_other - found
#   re-finds  = exit_optimal - 2*found   (Optimal launches not accepted)
# (identity checked on all 80 rows of each constrained fixture in the R4
# grid; wavy_1e12 prints extra banners and is not used for this.)
#
# Usage: dev-notes/issue-1042-r4-mlsl-gamma-grid.sh <pounce> <out.tsv> [gamma...]
# Default gammas: 0.4 0.45 0.5 0.55 0.6 0.75 1.0 2.0.
set -uo pipefail
BIN="$(realpath "${1:?usage: $0 <pounce> <out.tsv> [gamma...]}")"
OUT="${2:?usage: $0 <pounce> <out.tsv> [gamma...]}"
shift 2
GAMMAS=("$@")
[ ${#GAMMAS[@]} -eq 0 ] && GAMMAS=(0.4 0.45 0.5 0.55 0.6 0.75 1.0 2.0)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FX="$ROOT/crates/pounce-cli/tests/fixtures"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
# The model block of sweep-minima.sh: from `H='...` to the wavy_1e12 printf.
eval "$(sed -n "/^H='/,/wavy_1e12.nl\"\$/p" "$ROOT/scripts/sweep-minima.sh")"
cp "$FX/circle_parabola_t0p5.nl" "$FX/haverly_multistart_t3_brf0.nl" "$W/"
printf 'fixture\tgamma\tseed\tstatus\tsolves\tfound\texit_optimal\texit_infeasible\texit_other\n' > "$OUT"
for g in "${GAMMAS[@]}"; do
  for n in camel circle_parabola_t0p5 haverly_multistart_t3_brf0 wavy wavy_1e12; do
    for s in 0 1 2 3 4 5 6 7 8 9; do
      rm -f "$W/r.json"
      timeout 300 "$BIN" "$W/$n.nl" --minima mlsl --n-minima 6 --max-solves 30 \
        --patience 8 --seed "$s" --gamma "$g" --json-output "$W/r.json" \
        --sol-output "$W/r.sol" > "$W/r.out" 2>&1
      eo=$(grep -c '^EXIT: Optimal Solution Found' "$W/r.out")
      ei=$(grep -c '^EXIT: Converged to a point of local infeasibility' "$W/r.out")
      ea=$(grep -c '^EXIT:' "$W/r.out")
      python3 - "$W/r.json" "$n" "$g" "$s" "$eo" "$ei" "$((ea - eo - ei))" >> "$OUT" <<'PY'
import json, sys
p, *rest = sys.argv[1:]
n, g, s, eo, ei, ex = rest
try:
    m = json.load(open(p))["minima"]
    st, so, fo = m["status"], m["n_solves"], m["n_minima"]
except Exception:
    st, so, fo = "NO_JSON", "-", "-"
print("\t".join(map(str, [n, g, s, st, so, fo, eo, ei, ex])))
PY
    done
  done
done
