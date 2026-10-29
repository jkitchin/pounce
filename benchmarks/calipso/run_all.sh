#!/bin/bash
# Run the full matrix, 4 jobs at a time, 900 s cap per job.
# Needs the built extension (make python-ext) and jax[cpu]; PYTHON overrides the interpreter.
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
export PYTHON
SEEDS="0 1 2 3 4"
{
for s in $SEEDS; do
  for r in ipopt_defaults_eq direct ncp_eq ncp_eq_l1 scholtes_then_ncp scholtes_then_ineq; do
    echo "bunnyhop $r $s"
    echo "drifting_norm $r $s"
    echo "drifting_squared $r $s"
  done
  echo "drifting_norm jordan_eq $s"
  echo "drifting_squared jordan_eq $s"
done
} | xargs -P 4 -L 1 sh -c 'timeout 900 $PYTHON -I -c "import sys; sys.path.insert(0, \".\"); sys.argv=[\"run_one\", \"$0\", \"$1\", \"$2\"]; import run_one; run_one.main()" || echo "{\"model\": \"$0\", \"route\": \"$1\", \"seed\": $2, \"error\": \"timeout_or_crash\"}"' > results.jsonl 2> errors.log
echo done
