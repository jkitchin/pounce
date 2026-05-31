# LP suite — Mittelmann `lpopt` (curated, pounce-tractable subset)

Linear programs drawn from Hans Mittelmann's **`lpopt`** benchmark
(<https://plato.asu.edu/ftp/lpopt.html>), the de-facto modern LP
benchmark. This suite is a deliberately **curated subset**, not the full
set: `lpopt` lists 65 instances (49 disclosed + 16 undisclosed/unavailable)
and several run to *millions* of rows (up to ~30M), which produce
multi-GB `.nl` and are far beyond what POUNCE's interior-point method can
handle. We take the smaller disclosed instances under a size cap.

Like every other suite, this is `.nl`-driven through
`benchmarks/scripts/run_nl_bench.sh`, so each problem is recorded in the
standard schema — `{solver, name, n, m, status, objective, iterations,
solve_time}` — in `lpopt/pounce.json`, and merged into the composite
`BENCHMARK_REPORT.md`. That is the per-problem **iterations / solve time /
status / objective** tracking.

## How the `.nl` are produced

`build_subset.py` fetches each instance as bzip2'd standard MPS from the
plato lptestset mirror (<https://plato.asu.edu/ftp/lptestset/>),
decompresses it, screens its dimensions against a size cap, and converts
it to `.nl` via `mps_to_nl.py` (parse with HiGHS `highspy`, rebuild as a
Pyomo model, write `.nl` with Pyomo's ASL writer — the same `.nl`
pipeline as `large_scale/generate_nl.py`). The plato `lptestset` files are
plain `.mps.bz2`, so no netlib `emps` expansion is needed; sourcing other
`lpopt` instances from the netlib / Mészáros mirrors would.

Convention (matches MPS and HiGHS):

    minimize   c' x + offset
    subject to row_lower <= A x <= row_upper,  col_lower <= x <= col_upper

Size cap (in `build_subset.py`): `MAX_VARS=200k`, `MAX_CONS=200k`,
`MAX_NNZ=2M`. Instances above the cap are reported as
`deferred(too large)` and not converted.

## Current subset

Converted (under the cap):

| instance | n (vars) | m (cons) | nnz |
|---|---:|---:|---:|
| `qap15` | 22,275 | 6,330 | 94,950 |
| `supportcase10` | 14,770 | 165,684 | 555,082 |
| `irish-electricity` | 61,728 | 104,259 | 523,257 |
| `ex10` | 17,680 | 69,608 | 1,162,000 |

Deferred as too large (raise the cap in `build_subset.py` to include):
`datt256` (262k vars), `graph40-40` (361k cons), `s100` (364k vars),
`savsched1` (329k vars), `woodlands09` (382k vars). The 18–70M-compressed
instances (`square41`, `scpm1`, `s82`, `set-cover-model`) and the
multi-million-row LPs are intentionally omitted.

Note: `lpopt` is a *hard*-LP benchmark, so even small-dimension instances
can be slow or time out for an IPM (e.g. `qap15`, a QAP relaxation, is
degenerate and does not converge quickly). A timeout is a legitimate,
recorded benchmark outcome.

## Running

```sh
# (re)generate / extend the .nl subset (downloads MPS, converts under cap)
make -C benchmarks lpopt-generate
# add/raise coverage: edit CANDIDATES / the caps in build_subset.py, or
python3 benchmarks/lpopt/build_subset.py qap15 ex10   # specific instances

# run POUNCE on the suite -> benchmarks/lpopt/pounce.json (iters/time/status/obj)
# these are hard -- use a long per-problem limit
make -C benchmarks lpopt-run BENCH_TIMELIMIT=1800

# refresh the ipopt-ma57 reference for this suite (rare)
make -C benchmarks ipopt-ref-lpopt
```

## Comparing against a dedicated QP/LP solver

The shared driver runs any AMPL-protocol solver; the `ipopt-ma57`
reference column is produced exactly that way. To compare POUNCE against
another solver (e.g. a dedicated QP/LP solver or HiGHS), run that solver's
AMPL executable over `lpopt/nl/*.nl` through `run_nl_bench.sh` and merge its
JSON the same way the Ipopt reference is merged.
