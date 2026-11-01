# MPCC `.nl` suite (`benchmarks/mpcc_nl/`)

AMPL `.nl` models with complementarity constraints (`r`-segment kind 5, what
AMPL `complements` and Pyomo `mpec` write). The CLI reads them as declared
complementarity pairs and solves them through the complementarity driver
(`pounce_restoration::mpcc_driver`). This suite checks the **reported
answer** of that path against each model's known optimum.

It is the counterpart of [`../mpcc/`](../mpcc/README.md), the gh#794 Gate 0
harness, which builds its models in Python, compares lowering routes, and
reads no `.nl`. This suite exercises the path a user actually takes: a
`.nl` file in, the CLI's answer out.

## Prerequisites

- A pounce binary with declared-complementarity support (PR #1103, on `main`
  since `e7873df`). An
  earlier binary refuses every file at load (`complementarity (kind 5)
  bounds are not supported`), and every row fails.
- The data root: `$POUNCE_BENCH_DATA/mpcc_nl/` holds the `.nl` files
  (`nl/`), `answers.json`, the Pyomo generator `generate.py`, and a README
  describing each model.

## Running it

```sh
make -C benchmarks mpcc-nl-run      # solve (incremental) and check
make -C benchmarks mpcc-nl-rerun    # force a re-solve, then check
make -C benchmarks mpcc-nl-check    # re-check an existing pounce.json
```

The run goes through the shared driver (`scripts/run_nl_bench.sh`) into
`mpcc_nl/pounce.json` (gitignored), and `scripts/check_nl_answers.py` then
holds each row to `answers.json`. The target fails if any row fails. The
whole suite takes about 20 s, 14 s of it on `chain_60000`.

## What a pass means

| `expected_status` | passes when |
|---|---|
| `SolveSucceeded` | the run converged (`Solve_Succeeded` or `Solved_To_Acceptable_Level`) and `\|f − f*\| ≤ 1e-6·max(1, \|f*\|)` |
| `InfeasibleProblemDetected` | the run reported `Infeasible_Problem_Detected` (a relaxed solve's success fails) |

Any other `expected_status` fails as a malformed `answers.json`. There was a
`refused` status while the CLI declined two-sided pairs; PR #1107 made it
solve them (`mixed_pair`), and nothing emits a refusal any more.

`f*` is a **global** optimum on every model that has one, so a run that
settles on a worse S-stationary branch fails. That is the case
`eq_split` (global 1, local 2) and the `chain_N` family (global
`N·813/1004`, continuation-only `N`) are there to catch.

## What it does not check

- **The stationarity class.** The driver certifies its answer (S / W /
  none / not determined), but the JSON solve report does not carry the
  certificate yet, so this suite can't hold a run to it. The console
  narration prints it.
- **Iteration counts.** The driver reads `statistics.iteration_count`,
  which counts only the lowered solve whose answer was reported, not every
  solve the complementarity driver ran (it can read 0 when the answer came
  from a warm-started finish). The total is on the console line `pounce:
  complementarity: answered by ... (<k> lowered solves, <N> iterations in
  all ...)` in `mpcc_nl/logs/`.
- **An Ipopt comparison.** The suite is not in `REF_SUITES`; the reference
  is `answers.json`.

## Reference result

`main` at `26991b7` (PRs #1103–#1109 merged), CLI defaults: **16/16
pass**. `mixed_pair` holds a two-sided pair, which the CLI refused until PR
#1107. Every model with an answer certifies S except `ralph1` and
`mixed_pair`, whose minimizers are only weakly stationary (W). The
per-model table (status, answering stage, certificate, wall time) is in the data root's `mpcc_nl/README.md`.

`answers.json` assumes the default `complementarity_strategy=both`. Under
`exact` alone `eq_split` lands on its local optimum, and under
`continuation` alone every `chain_N` does; both fail here by design. That
is the suite measuring what running both stages buys.
