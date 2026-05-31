#!/usr/bin/env bash
# Single- or dual-solver .nl benchmark driver.
#
# Solves every *.nl under a suite directory (the AMPL solver protocol)
# with pounce, ipopt-ma57, or both, and writes a results JSON array in
# the standard schema:
#
#   [{"solver":"pounce|ipopt", "name":..., "n":..., "m":...,
#     "status":..., "objective":..., "iterations":..., "solve_time":...}, ...]
#
# Usage:
#   run_nl_bench.sh <suite_name> <nl_dir> <results_json> \
#                   <pounce_bin> <ipopt_bin> <time_limit_seconds> [mode]
#
# mode (default "both"):
#   pounce  — run only pounce         (ipopt_bin may be "-"/unused)
#   ipopt   — run only ipopt-ma57     (pounce_bin may be "-"/unused)
#   both    — run both (legacy behaviour)
#
# The pounce-vs-ipopt split lets the expensive ipopt reference be run
# once and saved (mode=ipopt → benchmarks/<suite>/ipopt_ma57.json) while
# each release reruns only pounce (mode=pounce → <suite>/pounce.json).
# Output feeds benchmark_report.py, which merges the two per suite.

set -u

SUITE="$1"
NL_DIR="$2"
RESULT="$3"
POUNCE_BIN="$4"
IPOPT_BIN="$5"
TIMELIMIT="${6:-300}"
MODE="${7:-both}"

# Ipopt's compiled default linear solver is ma27, even in an HSL/MA57
# build — so we ask for ma57 explicitly, otherwise the "ipopt-ma57"
# reference would silently run ma27. Override via the env var.
IPOPT_LINEAR_SOLVER="${IPOPT_LINEAR_SOLVER:-ma57}"

case "$MODE" in
  pounce) RUN_POUNCE=1; RUN_IPOPT=0 ;;
  ipopt)  RUN_POUNCE=0; RUN_IPOPT=1 ;;
  both)   RUN_POUNCE=1; RUN_IPOPT=1 ;;
  *) echo "run_nl_bench.sh: invalid mode '$MODE' (want pounce|ipopt|both)" >&2; exit 2 ;;
esac

LOGDIR="$(dirname "$RESULT")/logs/${SUITE}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

# Locate binaries — only the ones the selected mode needs.
check_bin() {
  local b="$1"
  if [ ! -x "$b" ] && ! command -v "$b" >/dev/null 2>&1; then
    echo "run_nl_bench.sh: binary not found or not executable: $b" >&2
    exit 2
  fi
}
[ "$RUN_POUNCE" = 1 ] && check_bin "$POUNCE_BIN"
[ "$RUN_IPOPT" = 1 ] && check_bin "$IPOPT_BIN"

shopt -s nullglob
nl_files=("$NL_DIR"/*.nl)
total=${#nl_files[@]}
if [ "$total" -eq 0 ]; then
  echo "run_nl_bench.sh: no .nl files under $NL_DIR" >&2
  exit 2
fi

# Helpers ------------------------------------------------------------

# Parse n, m from line 2 of an AMPL .nl file: "nvar ncon ... ".
parse_nm() {
  local nl="$1"
  awk 'NR==2 {gsub(/[\t#].*/,""); print $1, $2; exit}' "$nl"
}

# Map ipopt's free-form termination message → cutest-style status label.
ipopt_status_from_log() {
  local log="$1"
  if grep -q "Optimal Solution Found" "$log"; then echo "Solve_Succeeded"; return; fi
  if grep -q "Solved To Acceptable Level" "$log"; then echo "Solved_To_Acceptable_Level"; return; fi
  if grep -q "Maximum Number of Iterations Exceeded" "$log"; then echo "Maximum_Iterations_Exceeded"; return; fi
  if grep -q "Maximum CPU time exceeded" "$log"; then echo "Maximum_CpuTime_Exceeded"; return; fi
  if grep -q "Converged to a point of local infeasibility" "$log"; then echo "Infeasible_Problem_Detected"; return; fi
  if grep -q "Restoration Failed" "$log"; then echo "Restoration_Failed"; return; fi
  if grep -q "Search Direction is becoming Too Small" "$log"; then echo "Search_Direction_Becomes_Too_Small"; return; fi
  if grep -q "Diverging Iterates" "$log"; then echo "Diverging_Iterates"; return; fi
  if grep -q "Invalid number" "$log"; then echo "Invalid_Number_Detected"; return; fi
  echo "Unknown_Error"
}

# pounce CLI prints `Status: Solve_Succeeded` (or similar Status: <X>)
# at the end. Fall back to log scraping if we can't find it.
pounce_status_from_log() {
  local log="$1"
  local s
  s=$(grep -oE '^[Ss]tatus:[[:space:]]+\w+' "$log" | tail -1 | awk '{print $2}')
  if [ -n "$s" ]; then echo "$s"; return; fi
  # Pounce mirrors Ipopt's stdout for the common cases
  ipopt_status_from_log "$log"
}

# Extract iter count and objective from solver stdout (both use Ipopt's
# "Number of Iterations....: N" and "Objective...........: V" lines).
extract_iters() {
  # Prefer the end-of-run summary line.
  local n
  n=$(grep -oE 'Number of Iterations[. :]+[0-9]+' "$1" | tail -1 | grep -oE '[0-9]+$')
  if [ -n "$n" ]; then echo "$n"; return; fi
  # Fallback for timed-out / killed runs that never printed the summary:
  # the leading integer of the last iteration-table row (handles the
  # optional "r" restoration-phase marker) is the iteration count reached.
  grep -oE '^[[:space:]]*[0-9]+r?[[:space:]]' "$1" | tail -1 | grep -oE '[0-9]+' | head -1
}
extract_obj() {
  # Prefer the "Objective..." line; fall back to "Final objective value: V".
  local v
  v=$(grep -oE 'Objective[. :]+[-+0-9.eE]+' "$1" | tail -1 | grep -oE '[-+0-9.eE]+$')
  if [ -n "$v" ]; then echo "$v"; return; fi
  grep -oE 'Final objective[. :]+[-+0-9.eE]+' "$1" | tail -1 | grep -oE '[-+0-9.eE]+$'
}

# Run one solver on one .nl. $1=label, $2=binary, $3=nl path, $4=ampl_protocol
# Emits one JSON object on stdout (no trailing comma).
run_one() {
  local label="$1" bin="$2" nl="$3" ampl_protocol="$4"
  local problem nm n m start end elapsed log rc
  problem=$(basename "$nl" .nl)
  nm=$(parse_nm "$nl"); n=${nm%% *}; m=${nm##* }
  log="${LOGDIR}/${problem}.${label}.log"

  start=$(python3 -c 'import time; print(time.time())')
  if [ "$ampl_protocol" = "yes" ]; then
    timeout "$TIMELIMIT" "$bin" "$nl" -AMPL \
      linear_solver="$IPOPT_LINEAR_SOLVER" max_cpu_time="$TIMELIMIT" > "$log" 2>&1
    rc=$?
  else
    timeout "$TIMELIMIT" "$bin" "$nl" > "$log" 2>&1
    rc=$?
  fi
  end=$(python3 -c 'import time; print(time.time())')
  elapsed=$(python3 -c "print(f'{$end - $start:.6f}')")

  local status
  if [ "$rc" = "124" ]; then
    status="Maximum_CpuTime_Exceeded"
  elif [ "$rc" -ne 0 ]; then
    # Try log-scrape first; many real status outcomes still produce
    # non-zero rc (Infeasible_Problem_Detected, etc.).
    if [ "$ampl_protocol" = "yes" ]; then
      status=$(ipopt_status_from_log "$log")
    else
      status=$(pounce_status_from_log "$log")
    fi
    if [ -z "$status" ] || [ "$status" = "Unknown_Error" ]; then
      status="Solver_Error"
    fi
  else
    if [ "$ampl_protocol" = "yes" ]; then
      status=$(ipopt_status_from_log "$log")
    else
      status=$(pounce_status_from_log "$log")
    fi
  fi

  local obj iter
  obj=$(extract_obj "$log"); obj=${obj:-null}
  iter=$(extract_iters "$log"); iter=${iter:-0}

  # JSON solver label: "ipopt" for the AMPL-protocol invocation (so the
  # report's load_domain_results() finds it under the canonical key).
  local solver_label
  case "$label" in
    pounce) solver_label="pounce" ;;
    ipopt*) solver_label="ipopt" ;;
    *) solver_label="$label" ;;
  esac

  printf '  {"solver":"%s","name":"%s","n":%s,"m":%s,"status":"%s","objective":%s,"iterations":%s,"solve_time":%s}' \
    "$solver_label" "$problem" "$n" "$m" "$status" "$obj" "$iter" "$elapsed"
}

# Main loop ----------------------------------------------------------

# Emit one solver's record, prefixing a comma+newline for every record
# after the first so the array stays valid regardless of which solvers
# run.
first=1
emit() {  # $1=label $2=bin $3=nl $4=ampl_protocol
  if [ $first -eq 0 ]; then printf ",\n" >> "$RESULT"; fi
  first=0
  run_one "$@" >> "$RESULT"
}

echo "[" > "$RESULT"
i=0
for nl in "${nl_files[@]}"; do
  i=$((i+1))
  problem=$(basename "$nl" .nl)
  printf "[%2d/%d] %-30s " "$i" "$total" "$problem"

  if [ "$RUN_POUNCE" = 1 ]; then
    printf "pounce..."
    emit pounce "$POUNCE_BIN" "$nl" no
  fi
  if [ "$RUN_IPOPT" = 1 ]; then
    printf " ipopt..."
    emit ipopt-ma57 "$IPOPT_BIN" "$nl" yes
  fi

  printf " done\n"
done

echo "" >> "$RESULT"
echo "]" >> "$RESULT"
echo "wrote $RESULT"
