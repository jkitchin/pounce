#!/usr/bin/env bash
# EXPERIMENT BRANCH ONLY (exp/feral-onstale). A/B of feral#228's
# ReanalysisPolicy::OnStale against Never, crossed with PR #1046's
# forget-the-init-analysis, end to end on real models.
#
#   scripts/onstale-ab.sh <pounce-binary> <outdir> <model.nl>...
#
# Every model runs under four legs, serially (timings are compared):
#   main/never     POUNCE_EXP_FORGET_INIT=0  POUNCE_FERAL_REANALYSIS=never
#   main/on_stale  POUNCE_EXP_FORGET_INIT=0  POUNCE_FERAL_REANALYSIS=on_stale
#   1046/never     POUNCE_EXP_FORGET_INIT=1  POUNCE_FERAL_REANALYSIS=never
#   1046/on_stale  POUNCE_EXP_FORGET_INIT=1  POUNCE_FERAL_REANALYSIS=on_stale
# Override with LEGS="main/never main/on_stale" etc. Per-run timeout is
# TIMEOUT seconds (default 3600). Extra args after the models are not
# supported; set solver options through an options file if needed.
#
# Output: <outdir>/results.tsv (one row per run) and <outdir>/<model>.<leg>.{json,log}.
# Re-running skips any run whose JSON already exists, so an interrupted
# sweep resumes.
set -uo pipefail

BIN="${1:?usage: onstale-ab.sh <pounce-binary> <outdir> <model.nl>...}"
OUT="${2:?usage: onstale-ab.sh <pounce-binary> <outdir> <model.nl>...}"
shift 2
[ $# -gt 0 ] || { echo "no models given" >&2; exit 2; }
LEGS="${LEGS:-main/never main/on_stale 1046/never 1046/on_stale}"
TIMEOUT="${TIMEOUT:-3600}"
mkdir -p "$OUT"
TSV="$OUT/results.tsv"
[ -f "$TSV" ] || printf 'model\tleg\tstatus\titers\tsecond_opinion\tmu_fallback\tq_esc\twall_s\tfactor_s\tfactor_flops\tdelayed_cols\tlast_nnz_l\tlast_preprocess\tobjective\n' > "$TSV"

for f in "$@"; do
  n=$(basename "$f" .nl)
  for leg in $LEGS; do
    case "$leg" in
      main/*) forget=0 ;;
      1046/*) forget=1 ;;
      *) echo "bad leg $leg" >&2; exit 2 ;;
    esac
    pol="${leg#*/}"
    tag="${leg//\//_}"
    js="$OUT/$n.$tag.json"
    if [ -f "$js" ]; then echo "skip $n $leg (have JSON)"; continue; fi
    echo "run  $n $leg"
    t0=$(date +%s.%N)
    POUNCE_EXP_FORGET_INIT=$forget POUNCE_FERAL_REANALYSIS=$pol \
      timeout "$TIMEOUT" "$BIN" "$f" "$OUT/$n.$tag.sol" --json-output "$js.tmp" \
      > "$OUT/$n.$tag.log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    wall=$(python3 -c "print('%.2f' % ($t1 - $t0))")
    if [ -f "$js.tmp" ]; then mv "$js.tmp" "$js"; fi
    python3 - "$js" "$n" "$leg" "$wall" "$rc" >> "$TSV" <<'PY'
import json, sys
path, name, leg, wall, rc = sys.argv[1:]
try:
    d = json.load(open(path))
except Exception:
    print("\t".join([name, leg, "NO_JSON(rc=%s)" % rc] + ["-"] * 4 + [wall] + ["-"] * 6))
    sys.exit(0)
s, st, ls = d.get("solution", {}), d.get("statistics", {}), d.get("linear_solver", {})
so = d.get("second_opinion")
if not so:
    second = "-"
elif so.get("promoted_by"):
    second = "%s@%s/%d,tot=%d" % (so["promoted_by"], so.get("base_status", "?"),
        so.get("base_iteration_count", -1), so.get("total_iteration_count", 0))
else:
    second = "kept(%d),tot=%d" % (len(so.get("tried", [])), so.get("total_iteration_count", 0))
mf = st.get("mu_strategy_fallback")
if not mf:
    mu = "-"
elif mf.get("promoted"):
    mu = "%s@%s/%d,tot=%d" % (mf.get("retry_mu_strategy", "?"), mf.get("base_status", "?"),
        mf.get("base_iteration_count", -1),
        mf.get("base_iteration_count", 0) + mf.get("retry_iteration_count", 0))
else:
    mu = "kept(%s:%s/%d),tot=%d" % (mf.get("retry_mu_strategy", "?"), mf.get("retry_status", "?"),
        mf.get("retry_iteration_count", -1),
        mf.get("base_iteration_count", 0) + mf.get("retry_iteration_count", 0))
obj = s.get("objective")
print("\t".join(str(x) for x in [
    name, leg, s.get("status"), st.get("iteration_count", "?"), second, mu,
    st.get("quality_escalations", 0), wall,
    "%.2f" % ls["total_factor_secs"] if "total_factor_secs" in ls else "-",
    "%.3e" % ls["total_factor_flops"] if "total_factor_flops" in ls else "-",
    ls.get("total_delayed_cols", "-"), ls.get("last_nnz_l", "-"),
    ls.get("last_ordering_preprocess", "-"),
    "%.12g" % obj if obj is not None else "none",
]))
PY
  done
done

echo
echo "== $TSV"
column -t -s $'\t' "$TSV"
