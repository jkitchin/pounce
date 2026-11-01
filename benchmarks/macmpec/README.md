# MacMPEC (`benchmarks/macmpec/`)

Sven Leyffer's MacMPEC, the standard MPCC test collection: 178 AMPL `.nl`
models with complementarity rows, solved through the CLI's
declared-complementarity path (PR #1103, on `main` since `e7873df`) and
classified against the
collection's published best-known objectives.

The data, provenance, checksums, the full answers file and the reference
results, including the comparison with published B-stationarity rates and
the PR defects this suite found, live in the data root:
`$POUNCE_BENCH_DATA/macmpec/README.md`.

## Running it

```sh
make -C benchmarks macmpec-run      # solve (incremental) and classify
make -C benchmarks macmpec-rerun    # force a re-solve
make -C benchmarks macmpec-check    # re-classify an existing pounce.json
```

The shared driver (`scripts/run_nl_bench.sh`) writes `macmpec/pounce.json`
(gitignored), and `scripts/check_nl_answers.py` classifies each row against
`answers.json`. At the default 300 s limit every model finishes except
`qpec-200-2`, which needs about 430 s (`BENCH_TIMELIMIT=900`).

## What the outcome means

The references are best-known values found by local solvers, so a converged
run is classified rather than failed, in the model's own objective sense
(13 models maximize):

- `at`: within `1e-4·max(1, |f*|)` of the best known value.
- `worse`: another local solution.
- `BETTER`: past the published value. Check the point's feasibility before
  believing it.
- `unsolved`: the run did not converge.

The target fails only on a contradiction of the reference: converging on a
model published as infeasible, or reporting infeasible where a feasible
point is known.

## Reference result

`main` at `26991b7` (PRs #1107–#1109 merged), with `BENCH_TIMELIMIT=900` for
`qpec-200-2`: of 170 models with a best-known value, 154 `at`, 6 `BETTER`
(each verified feasible by direct evaluation), 10 `worse`, and none
`unsolved`. All 4 published-infeasible models are reported infeasible. No
FAILs. `gnash10m` (no reference) solves at −230.823, the value its one-sided
twin `gnash10` reaches. At the default 300 s limit `qpec-200-2` alone is
`unsolved` (`Maximum_CpuTime_Exceeded`), which is not a FAIL.

`main` at `e7873df` (PR #1103) had 144 `at`, 6 `BETTER`, 10 `worse`, and 10
`unsolved`, all of them two-sided refusals: `bilevel1m` and
`gnash11m`–`gnash19m`. Two changes since:

- **PR #1107** solves two-sided pairs. `gnash11m`–`gnash19m` reach the
  published optimum. `bilevel1m` reaches 5 against a published −55, the
  local solution its one-sided twin `ex9.2.3` also reaches, and joins
  `worse`.
- **PR #1109** checks B-stationarity of non-S answers and escapes along a
  descending branch. `siouxfls` moves from `worse` (207.365) to `at`
  (208.258, maximize).

None of these models' `answers.json` entries expected `refused`; they carry
the published best-known values (gh#1110).

**Not yet re-measured with this harness since PRs #1112 and #1121.** Both
change outcomes, measured on the 167 public MacMPEC `.nl` stubs (outside
`make macmpec-check`, so the counts are not directly comparable with the
table above):

- **PR #1112** budgets the exact stage and the finish: `pack-rig1p-32` moves
  from `worse` to `at`.
- **PR #1121** adds a relaxed path to the exact stage: `dempe` (28.25 against
  31.25) and `ex9.2.5` (5 against 9) move from `at` to `BETTER`, both checked
  feasible by direct evaluation, and `pack-comp2p-8` from `worse` to `at`.

Re-run `make macmpec-run` to refresh the reference result above.
