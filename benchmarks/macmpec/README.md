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
- `unsolved`: the run did not converge (or the CLI refused a pair it cannot
  lower).

The target fails only on a contradiction of the reference: converging on a
model published as infeasible, or reporting infeasible where a feasible
point is known.

## Reference result

With mixed (two-sided) pairs solved (PR #1107): of 170 models with a
best-known value, 153 `at`, 6 `BETTER` (each verified feasible by direct
evaluation), 11 `worse`, and none `unsolved`. All 4 published-infeasible
models are reported infeasible. No FAILs.

Before it, `main` at `e7873df` (PR #1103) had 144 `at`, 6 `BETTER`, 10
`worse`, and 10 `unsolved`, all of them two-sided refusals: `bilevel1m` and
`gnash11m`–`gnash19m`. Nine of those now reach the published optimum.
`bilevel1m` reaches 5 against a published −55, the local solution its
one-sided twin `ex9.2.3` also reaches, and is the eleventh `worse`.
