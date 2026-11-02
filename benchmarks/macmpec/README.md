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

`main` at `5882e01` (PR #1121 merged; measured at `72ad1d7`, the same
tree), with `BENCH_TIMELIMIT=900` for every model: of 170 models with a best-known
value, 154 `at`, 7 `BETTER`, 9 `worse`, and none `unsolved`. All 4
published-infeasible models are reported infeasible. No FAILs. `gnash10m`
(no reference) solves at −230.823, the value its one-sided twin `gnash10`
reaches.

Against the previous reference (`main` at `26991b7`, PRs #1107–#1109; 154
`at`, 6 `BETTER`, 10 `worse`), seven models move:

| model | before | now | f* | from |
|---|---|---|---|---|
| `pack-rig1p-32` | worse 0.8813 | at 0.85091 | 0.8509 | #1112 (per its PR) |
| `pack-comp1p-32` | at | worse 0.67714 | 0.652979 | before #1121: `main` at `45c18dd` gives the same answer (between `26991b7` and `45c18dd`; not bisected) |
| `dempe` | worse 31.25 | at 28.25 | 28.25 | #1121 |
| `ex9.2.5` | worse 9 | BETTER 5 | 6 | #1121 (checked feasible by direct evaluation) |
| `pack-comp2p-8` | worse 0.67360 | at 0.67312 | 0.67312 | #1121 |
| `incid-set2c-8` | at 0.0054719 | worse 0.0056302 | 0.005471 | #1121 |
| `pack-comp2p-16` | at | worse 0.72864 | 0.727134 | already worse on `main` at `45c18dd` (0.72769; not bisected); #1121 moves it further |

**Not yet re-measured with this harness since PR #1126**, which changes the
driver's exact-product schedule. On the 167 public `.nl` stubs (outside this
harness, 300 s limit) it moves `pack-comp2p-32` from `worse` back to `at`
and cuts total time by about 30%; there, `incid-set2c-8` and
`pack-comp2p-16` are `at` both before and after it. Re-run `make
macmpec-run` to refresh the table above.

The three new `worse` lines converge to feasible points, so they are other
local solutions, not wrong answers. #1121's own measurement used the 167
public `.nl` stubs outside this harness, where `incid-set2c-8` and
`pack-comp2p-16` did not move, so that run is not directly comparable with
this table.

`main` at `26991b7` had 154 `at`, 6 `BETTER`, 10 `worse`. Before that,
`main` at `e7873df` (PR #1103) had 144 `at`, 6 `BETTER`, 10 `worse`, and 10
`unsolved`, all of them two-sided refusals: `bilevel1m` and
`gnash11m`–`gnash19m`. Two changes between those:

- **PR #1107** solves two-sided pairs. `gnash11m`–`gnash19m` reach the
  published optimum. `bilevel1m` reaches 5 against a published −55, the
  local solution its one-sided twin `ex9.2.3` also reaches, and joins
  `worse`.
- **PR #1109** checks B-stationarity of non-S answers and escapes along a
  descending branch. `siouxfls` moves from `worse` (207.365) to `at`
  (208.258, maximize).

None of these models' `answers.json` entries expected `refused`; they carry
the published best-known values (gh#1110).
