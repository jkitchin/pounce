#!/bin/bash
# Run the full matrix, 4 jobs at a time, 900 s cap per job.
# Needs jax[cpu] plus the built POUNCE extension (make python-ext) or, with
# CALIPSO_SOLVER=ipopt, cyipopt. PYTHON overrides the interpreter, OUT the
# results file; IPOPT_LINEAR_SOLVER / IPOPT_HSLLIB pick Ipopt's backend.
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
OUT="${OUT:-results.jsonl}"
export PYTHON
SEEDS="0 1 2 3 4"
ROUTES="defaults_eq direct ncp_eq ncp_eq_l1 scholtes_then_ncp scholtes_then_ineq"
if [ "${CALIPSO_SOLVER:-pounce}" = ipopt ]; then
  ROUTES="${ROUTES/ncp_eq_l1 /}"   # l1_exact_penalty_barrier is POUNCE-only
fi
{
for s in $SEEDS; do
  for r in $ROUTES; do
    echo "bunnyhop $r $s"
    echo "drifting_norm $r $s"
    echo "drifting_squared $r $s"
  done
  echo "drifting_norm jordan_eq $s"
  echo "drifting_squared jordan_eq $s"
done
} | xargs -P 4 -L 1 sh -c 'timeout 900 $PYTHON -I -c "import sys; sys.path.insert(0, \".\"); sys.argv=[\"run_one\", \"$0\", \"$1\", \"$2\"]; import run_one; run_one.main()" || echo "{\"model\": \"$0\", \"route\": \"$1\", \"seed\": $2, \"error\": \"timeout_or_crash\"}"' > "$OUT" 2> "${OUT%.jsonl}.errors.log"
echo done
