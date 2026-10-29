#!/bin/bash
# The whole CALIPSO corpus, every baseline that is available, one command:
#
#   bash run_corpus.sh
#
# Baselines, each skipped when its interpreter is not set:
#   POUNCE   PYTHON        (needs the built extension, make python-ext, + jax[cpu])
#   Ipopt    IPOPT_PYTHON  (needs cyipopt + jax[cpu]; IPOPT_LINEAR_SOLVER picks
#                           mumps / spral / ma57, the last with IPOPT_HSLLIB)
#   CALIPSO  JULIA + CALIPSO_JL (a CALIPSO.jl checkout) + JULIA_PROJECT (an env
#                           with CALIPSO, RoboDojo 0.1.6 and JSON; Julia 1.8)
#
# MODELS restricts the run (comma-separated names from models.py), JOBS sets
# the parallelism (default 4), and every job is capped at 900 s. Results land
# in results_<solver>.jsonl; the CALIPSO answers are scored by the port in
# results_calipso_eval.jsonl; the summary is printed and written to
# summary.md.
set -u
cd "$(dirname "$0")"
PYTHON="${PYTHON:-}"
IPOPT_PYTHON="${IPOPT_PYTHON:-}"
JULIA="${JULIA:-}"
MODELS="${MODELS:-}"
JOBS="${JOBS:-4}"
LS="${IPOPT_LINEAR_SOLVER:-mumps}"
files=()

run_matrix() {  # $1 = interpreter, $2 = solver name for models.jobs, $3 = output
    "$1" -I models.py "$2" "$MODELS" |
    xargs -P "$JOBS" -L 1 sh -c 'timeout 900 "$PY" -I -c "import sys; sys.path.insert(0, \".\"); sys.argv=[\"run_one\", \"$0\", \"$1\", \"$2\"]; import run_one; run_one.main()" || echo "{\"model\": \"$0\", \"route\": \"$1\", \"seed\": $2, \"error\": \"timeout_or_crash\"}"' > "$3" 2> "${3%.jsonl}.errors.log"
    rm -f sol_*.npy
}

if [ -n "$PYTHON" ]; then
    echo "pounce ..."
    export PY="$PYTHON" CALIPSO_SOLVER=pounce
    run_matrix "$PYTHON" pounce results_pounce.jsonl
    files+=(results_pounce.jsonl)
fi

if [ -n "$IPOPT_PYTHON" ]; then
    echo "ipopt ($LS) ..."
    export PY="$IPOPT_PYTHON" CALIPSO_SOLVER=ipopt IPOPT_LINEAR_SOLVER="$LS" \
        OMP_CANCELLATION=TRUE OMP_PROC_BIND=TRUE OMP_NUM_THREADS=1
    run_matrix "$IPOPT_PYTHON" ipopt "results_ipopt_$LS.jsonl"
    files+=("results_ipopt_$LS.jsonl")
fi

if [ -n "$JULIA" ]; then
    echo "calipso ..."
    : > results_calipso.jsonl
    "${PYTHON:-$IPOPT_PYTHON}" -I -c "
import sys; sys.path.insert(0, '.')
import models
names = sys.argv[1].split(',') if sys.argv[1] else models.ALL
seen = []
for n in names:
    ex = models.get(n)[2]
    if ex not in seen:
        seen.append(ex)
        print(ex, '0' if n in models.DETERMINISTIC else '0,1,2,3,4')
" "$MODELS" | while read -r ex seeds; do
        timeout 7200 "$JULIA" --project="$JULIA_PROJECT" calipso_baseline.jl "$CALIPSO_JL" "$ex" "$seeds" results_calipso.jsonl \
            > "calipso_$(basename "$ex" .jl).log" 2>&1 || echo "calipso $ex failed (see calipso_$(basename "$ex" .jl).log)"
    done
    "${PYTHON:-$IPOPT_PYTHON}" -I evaluate_calipso.py results_calipso.jsonl > results_calipso_eval.jsonl
    files+=(results_calipso_eval.jsonl)
fi

"${PYTHON:-${IPOPT_PYTHON:-python3}}" -I summarize.py "${files[@]}" | tee summary.md
