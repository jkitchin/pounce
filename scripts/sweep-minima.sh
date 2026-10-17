#!/usr/bin/env bash
# Trajectory sweep for the `--minima` global search.
#
# Runs every fixture below under each of the six `--minima` methods and
# records what the search returned -- stop status, solves spent, minima found,
# kkt_tol rejections, every minimum's objective, and the best point's x and
# base-problem duals -- one line per (fixture, method), sorted and diffable.
#
# WHY THIS EXISTS (gh#966). scripts/sweep-fixtures.sh never passes
# `--minima`, so it is uniform in exactly the dimension a change to the
# search acts on: an empty sweep-fixtures diff says nothing about it. The
# search is a loop of local solves whose starts depend on every earlier
# accept/reject decision, so a change that moves one decision moves the rest
# of the run -- the found set, the order, the solve count -- while each
# individual answer stays a correct local minimum. This sweep is the witness
# for that path, the way sweep-fixtures.sh is for the single solve.
#
# Use it for any change to crates/pounce-minima or crates/pounce-cli/src/
# minima: the gh#966 extraction (a move must diff EMPTY), a change to
# acceptance, sampling, the penalty wrappers or the hook that captures a
# solve's answer. A line that moves must be explainable before merge.
#
# The fixtures are the embedded models below (each has several minima and a
# box, so every method has something to find and the sampler has a box to
# draw from) plus a few corpus models, chosen for the branch each reaches:
#   circle_parabola_t0p5, haverly_multistart_t3_brf0 -- constrained, so the
#     per-minimum dual recovery (`lambda=` column) is exercised;
#   saddle_axis_first, nonconvex_two_escapes, nonconvex_qcqp -- every solve
#     converges to a saddle, so every line reads found=0: that is the
#     saddle-rejection (PSD) branch, not a broken fixture;
#   wavy_accept_anything -- the acceptable-level test opened so wide that each
#     solve returns its own start as a success, so every candidate reaches the
#     kkt_tol veto (`kkt_rej=` column);
#   wavy_1e12 -- the same model at gradient scale 1e12, where the relative
#     bound must veto nothing.
# Seed 0 throughout: the search is deterministic for a fixed binary and seed.
#
# Usage:
#   scripts/sweep-minima.sh <pounce-binary> <outfile> [extra solver opts...]
#
#   cp target/release/pounce /tmp/p-base     # before the change
#   cargo build --release -p pounce-cli      # after
#   scripts/sweep-minima.sh /tmp/p-base           /tmp/base.txt
#   scripts/sweep-minima.sh target/release/pounce /tmp/new.txt
#   diff /tmp/base.txt /tmp/new.txt
set -uo pipefail

BIN="${1:?usage: sweep-minima.sh <pounce-binary> <outfile> [opts...]}"
OUT="${2:?usage: sweep-minima.sh <pounce-binary> <outfile> [opts...]}"
shift 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FX="$ROOT/crates/pounce-cli/tests/fixtures"
[ -d "$FX" ] || { echo "no fixture dir at $FX" >&2; exit 2; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
: > "$OUT"

H='g3 1 1 0
 2 0 1 0 0
 0 1
 0 0
 0 2 0
 0 0 0 1
 0 0 0 0 0
 0 2
 0 0
 0 0 0 0 0
O0 0'
G='k1
0
G0 2
0 0
1 0'

# Six-hump camel on [-3,3]x[-2,2]: six minima, two saddles.
printf '%s\n' "$H" o54 6 o2 n4 o5 v0 n2 o2 n-2.1 o5 v0 n4 o3 o5 v0 n6 n3 \
  o2 v0 v1 o2 n-4 o5 v1 n2 o2 n4 o5 v1 n4 x2 '0 0.5' '1 0.5' b '0 -3 3' \
  '0 -2 2' "$G" > "$W/camel.nl"
# sum(sin(3x) + 0.1x^2) on [-5,5]^2: a grid of minima.
printf '%s\n' "$H" o54 4 o2 n1 o41 o2 n3 v0 o2 n0.1 o5 v0 n2 \
  o2 n1 o41 o2 n3 v1 o2 n0.1 o5 v1 n2 x2 '0 0.3' '1 -0.2' b '0 -5 5' \
  '0 -5 5' "$G" > "$W/wavy.nl"
# The same at gradient scale 1e12 (the gh#983 relative-bound case).
printf '%s\n' "$H" o54 4 o2 n1e12 o41 o2 n3 v0 o2 n1e11 o5 v0 n2 \
  o2 n1e12 o41 o2 n3 v1 o2 n1e11 o5 v1 n2 x2 '0 0.3' '1 -0.2' b '0 -5 5' \
  '0 -5 5' "$G" > "$W/wavy_1e12.nl"

cp "$W/wavy.nl" "$W/wavy_accept_anything.nl"
ACCEPT_ANYTHING="acceptable_tol=100 acceptable_compl_inf_tol=100 \
acceptable_obj_change_tol=1e20 acceptable_iter=1"

FIXTURES=("$W/camel.nl" "$W/wavy.nl" "$W/wavy_1e12.nl" "$W/wavy_accept_anything.nl")
for n in nonconvex_two_escapes saddle_axis_first circle_parabola_t0p5 \
         haverly_multistart_t3_brf0 nonconvex_qcqp; do
  [ -f "$FX/$n.nl" ] && FIXTURES+=("$FX/$n.nl")
done

for f in "${FIXTURES[@]}"; do
  n=$(basename "$f" .nl)
  fopts=()
  [ "$n" = wavy_accept_anything ] && read -r -a fopts <<< "$ACCEPT_ANYTHING"
  for method in multistart mlsl basinhopping flooding deflation tunneling; do
    j="$W/$n.$method.json"
    # A run that hangs must not hang the sweep; it shows up as NO_JSON.
    timeout 300 "$BIN" "$f" --minima "$method" --n-minima 6 --max-solves 30 \
      --patience 8 --seed 0 --json-output "$j" --no-sol "${fopts[@]}" "$@" \
      > "$W/$n.$method.out" 2>&1
    python3 - "$j" "$n" "$method" >> "$OUT" <<'PY'
import json, sys
path, name, method = sys.argv[1:]
try:
    d = json.load(open(path))
except Exception:
    print(f"{name:28s} {method:12s} NO_JSON")
    sys.exit(0)
m = d.get("minima", {})
s = d.get("solution", {})
g = lambda v: "%.10g" % v
vals = ",".join(g(v) for v in m.get("values", []))
x = ",".join(g(v) for v in s.get("x", []) or [])
lam = ",".join(g(v) for v in s.get("lambda", []) or [])
print(f"{name:28s} {method:12s} {m.get('status', '?'):18s} "
      f"solves={m.get('n_solves', '?')} found={m.get('n_minima', '?')} "
      f"kkt_rej={m.get('n_kkt_rejected', '-')} values=[{vals}] "
      f"x=[{x}] lambda=[{lam}]")
PY
  done
done
sort -o "$OUT" "$OUT"
