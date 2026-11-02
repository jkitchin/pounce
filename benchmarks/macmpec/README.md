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
`answers.json`. At the default 300 s limit two models do not finish: `siouxfls`
(676 s) and `qpec-200-2` (335 s). Use `BENCH_TIMELIMIT=900`.

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

PR #1126 merged plus gh#1128 (branch `fix/issues-1127-1129`, measured
2026-11-01 with `BENCH_TIMELIMIT=900` for every model, 14-core Apple silicon,
with the baseline run alongside on the same host, so times are indicative), against the corrected `answers.json` (gh#1129: `bilevel1m`
−55 → 0 and `ex9.2.3` −55 → 5, global optima by LP branch enumeration; see
the data README). Of 170 models with a best-known value: **158 `at`, 7
`BETTER`, 5 `worse`**, none `unsolved`. All 4 published-infeasible models
are reported infeasible. No FAILs. `gnash10m` (no reference) solves at
−230.823, the value its one-sided twin `gnash10` reaches. Total solve time
2,634 s. Two models take over 300 s: `siouxfls` (676 s) and `qpec-200-2`
(335 s).

| outcome | models |
|---|---|
| BETTER (7) | `bem-milanc30-s` 0.09443 vs 83.2825; `ex9.2.5` 5 vs 6; `water-FL` 3285.22 vs 3411.92; `qpec-100-4` −4.0648 vs −3.98212; `qpec-200-2` −24.1170 vs −24.0299; `qpec-200-4` −6.2110 vs −6.19323; `tap-09` 109.1310 vs 109.143 (the same seven as before) |
| worse (5) | `bilevel1m` 5 vs 0; `qpec-100-3` −5.48167 vs −5.48287; `qpec-200-1` −1.8511 vs −1.93483; `TrafficSignalCycle` 86.40 vs 83.55; `water-net` 953.23 vs 929.17 |

Against the previous reference (`main` at `5882e01`, PR #1121; 154 `at`, 7
`BETTER`, 9 `worse`):

| model | before | now | f* | from |
|---|---|---|---|---|
| `ex9.2.3` | worse 5 | at 5 | 5 (was −55) | gh#1129: the reference was wrong |
| `bilevel1m` | worse 5 vs −55 | worse 5 vs 0 | 0 (was −55) | gh#1129: still another local solution, against the corrected value |
| `incid-set2c-8` | worse 0.0056302 | at 0.0054713 | 0.005471 | #1126 |
| `pack-comp2p-16` | worse 0.72864 | at 0.72717 | 0.727134 | #1126 |
| `pack-comp1p-32` | worse 0.67714 | at 0.65304 | 0.652979 | between `5882e01` and #1126 (not bisected) |
| `qpec-100-3` | worse −5.4488 | worse −5.48167 | −5.48287 | between `5882e01` and #1126 (not bisected) |

**`incid-set2c-8` and `pack-comp2p-16` were real, not stale, and not the stub
files** (the question in gh#1129). Run alone on this harness's stubs at CLI
defaults, both `5882e01` and `ffa39d64` (main just before #1126 merged)
reproduce the old `worse` values exactly: 0.0056302256 and 0.7286357
(Solved_To_Acceptable_Level). #1126's head gives `at` on both. The two stubs
are byte-identical to the ones in `MacMPEC-nl.tar.gz` from the Internet
Archive, so #1126's public-stub measurement (`at` before and after it) did
not come from different files. It must have differed in how the runs were
set up.

**gh#1128 on this suite.** gh#1128 makes the drift and agreement tests
independent of the objective's scale. The same harness run on #1126's head
classifies all 178 models identically. Only one line moves:
`incid-set2c-32`, 2 → 1 iterations at the same objective (6,241 → 6,240
iterations in total; time 2,635 s → 2,634 s).

Earlier references: `main` at `26991b7` (PRs #1107–#1109) had 154 `at`, 6
`BETTER`, 10 `worse`. `main` at `e7873df` (PR #1103) had 144 `at`, 6
`BETTER`, 10 `worse`, and 10 `unsolved`, all of them two-sided refusals:
`bilevel1m` and `gnash11m`–`gnash19m`. PR #1107 solved two-sided pairs
(`gnash11m`–`gnash19m` reach the published optimum, `bilevel1m` reaches 5).
PR #1109 added the B-stationarity check and branch escape (`siouxfls`
207.365 → 208.258, maximize).

None of these models' `answers.json` entries expected `refused`; they carry
the published best-known values (gh#1110), except the two that gh#1129
corrected.
