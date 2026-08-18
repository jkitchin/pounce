#!/usr/bin/env bash
# Empirical complexity probe: does per-iteration cost grow linearly in n?
#
# WHY THIS EXISTS (pounce #677). The limited-memory path advertises
# `O(n·m)` storage and `O(n·m²)` work per iteration, with `m =
# limited_memory_max_history` (6 by default). Nothing measured that. The
# fixture corpus is all small models -- the largest is a few thousand
# variables -- so an accidental `O(n²)` in that path would cost nothing
# visible there and would only surface as "the solver is unusably slow"
# on somebody's 60,000-variable collocation model. That is exactly the
# size of problem the L-BFGS path exists to serve, and exactly the size
# nothing in this repo runs.
#
# The measurement is per-iteration wall time against n, on the SAME
# problem family at geometrically increasing sizes. Two refinements make
# the number mean what it says:
#
#   1. Total time is the wrong statistic -- it folds in the iteration
#      count, which moves with n for its own reasons and would swamp the
#      signal.
#   2. Setup (reading the .nl, building the structures) is itself O(n)
#      and can be most of a short run. Left in, it *biases the answer
#      toward "linear"* and would mask a quadratic in the iteration loop
#      -- the exact failure this probe exists to catch.
#
# So each size is solved TWICE, at a low and a high `max_iter`, and the
# per-iteration cost is the difference:
#
#     per_iter = (wall_hi - wall_lo) / (iters_hi - iters_lo)
#
# Setup is identical in both runs and cancels. The reported slope is a
# log-log least-squares fit of that difference against n:
#
#     slope ~= 1.0   linear      -- expected
#     slope ~= 1.5   n^1.5       -- worth explaining (sparse factor fill)
#     slope ~= 2.0   quadratic   -- a defect
#
# Sparse-factorization cost is superlinear for genuine structural reasons
# (fill-in), so a slope above 1 is not automatically a bug -- compare the
# `exact` and `lbfgs` legs. It is the *difference* between them that
# isolates the quasi-Newton code from the linear algebra underneath it.
#
# Usage:
#   scripts/scaling-probe.sh <pounce-binary> <nl-dir-glob> [extra opts...]
#
#   # generate a size family first (needs pyomo):
#   for T in 1000 2000 4000 8000 16000; do
#     python3 benchmarks/large_scale/generate_nl.py optcontrol \
#       --optcontrol-t $T --out-dir /tmp/scale/t$T
#   done
#   scripts/scaling-probe.sh target/release/pounce '/tmp/scale/t*'
#
# Sizes are read from the .nl header, so any family of one problem at
# increasing sizes works; the directories need not be named for n.
set -uo pipefail

BIN="${1:?usage: scaling-probe.sh <pounce-binary> <nl-dir-glob> [opts...]}"
GLOB="${2:?usage: scaling-probe.sh <pounce-binary> <nl-dir-glob> [opts...]}"
shift 2

ITER_LO="${SCALING_PROBE_ITER_LO:-5}"
ITER_HI="${SCALING_PROBE_ITER_HI:-25}"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

printf '%-10s %-14s %-14s %-14s %s\n' n iters_lo/hi wall_lo/hi_s per_iter_ms nl
rows="$W/rows.txt"
: > "$rows"

# One (n, wall, iters) measurement. Echoes "<iters> <seconds>".
run_once() {
  local f="$1" cap="$2"; shift 2
  local start end it
  start=$(python3 -c 'import time;print(time.time())')
  timeout 900 "$BIN" "$f" "$W/o.sol" --json-output "$W/o.json" \
    "max_iter=$cap" print_level=0 "$@" >/dev/null 2>&1
  end=$(python3 -c 'import time;print(time.time())')
  if [ -f "$W/o.json" ]; then
    it=$(python3 -c "import json;print(json.load(open('$W/o.json'))['statistics']['iteration_count'])" 2>/dev/null || echo 0)
  else
    it=0
  fi
  rm -f "$W/o.json"
  echo "$it $(python3 -c "print($end - $start)")"
}

for d in $GLOB; do
  f=$(ls "$d"/*.nl 2>/dev/null | head -1)
  [ -n "$f" ] || continue
  # n from the .nl header: line 2 of the header block is
  # "<nvars> <ncons> <nobjs> ...".
  n=$(sed -n '2p' "$f" | awk '{print $1}')

  read -r it_lo t_lo <<< "$(run_once "$f" "$ITER_LO" "$@")"
  read -r it_hi t_hi <<< "$(run_once "$f" "$ITER_HI" "$@")"

  python3 - "$n" "$it_lo" "$t_lo" "$it_hi" "$t_hi" "$(basename "$d")" "$rows" <<'PY'
import sys
n, it_lo, t_lo, it_hi, t_hi, tag, rows = sys.argv[1:]
it_lo, it_hi = int(it_lo), int(it_hi)
t_lo, t_hi = float(t_lo), float(t_hi)
d_it = it_hi - it_lo
if d_it <= 0:
    # The solve converged before the low cap, so both runs did the same
    # work and the difference is noise. Reporting it as a data point
    # would be inventing a measurement.
    print("%-10s %-14s %-14s %-14s %s" % (
        n, "%d/%d" % (it_lo, it_hi), "-", "CONVERGED<lo", tag))
else:
    per = (t_hi - t_lo) / d_it * 1000.0
    print("%-10s %-14s %-14s %-14.3f %s" % (
        n, "%d/%d" % (it_lo, it_hi), "%.2f/%.2f" % (t_lo, t_hi), per, tag))
    if per > 0:
        open(rows, "a").write("%s %.9f\n" % (n, per))
PY
done

python3 - "$rows" <<'PY'
import sys, math
pts = []
for line in open(sys.argv[1]):
    n, per = line.split()
    if float(per) > 0:
        pts.append((float(n), float(per)))
# Directory globs sort lexically ("t16000" before "t2000"), so the
# pairwise ratios below are meaningless unless we sort by n first.
pts.sort()
if len(pts) < 3:
    print("\nnot enough points for a fit (need 3+)")
    raise SystemExit(0)
xs = [math.log(p[0]) for p in pts]
ys = [math.log(p[1]) for p in pts]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
den = sum((x - mx) ** 2 for x in xs)
slope = num / den
# R^2 so a ragged fit is not read as a clean exponent.
pred = [my + slope * (x - mx) for x in xs]
ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
ss_tot = sum((y - my) ** 2 for y in ys)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
print("\nlog-log slope of per-iteration time vs n: %.2f  (R^2 = %.3f, %d points)"
      % (slope, r2, len(pts)))
verdict = ("LINEAR — as advertised" if slope < 1.25 else
           "SUPERLINEAR — explain it" if slope < 1.75 else
           "QUADRATIC OR WORSE — treat as a defect")
print("verdict: %s" % verdict)
# Pairwise ratios: a single bad size shows here but not in the fit.
print("\nper-iteration cost ratio between consecutive sizes"
      " (2.0 = linear when n doubles):")
for (n0, p0), (n1, p1) in zip(pts, pts[1:]):
    print("  n %-8g -> %-8g   x%.2f  (n x%.2f)" % (n0, n1, p1 / p0, n1 / n0))
PY
