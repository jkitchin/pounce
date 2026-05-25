# Benchmarks

Comparison harnesses that exercise POUNCE against upstream Ipopt across
several NLP test suites. Each suite lives in its own subdirectory with
its own README explaining the problems, prerequisites, and how to run
it. All suites feed a single composite report at
[`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md), with run metadata (POUNCE
version, git SHA, Ipopt version, linear solver) emitted in the report
header.

The benchmark *inputs* (large `.nl` exports, compiled SIF problem
libraries) and the per-run *outputs* (logs, JSON results, generated
reports) are regenerated locally and not tracked in the repository.
The per-suite README files, the source harnesses, and
`benchmarks/scripts/` are tracked.

## Suites

| Suite | Problem class | Size | Notes |
|-------|---------------|------|-------|
| [`cho/`](cho/README.md)             | Parameter estimation, kinetic ODE | 1 problem, medium dense | CHO cell kinetics from Pyomo `parmest`; stress test for restoration |
| [`cutest/`](cutest/README.md)       | Canonical NLP collection | 727 / 1542 problems | Gould–Orban–Toint MASTSIF; requires `prepare.sh` to compile SIF libraries |
| [`electrolyte/`](electrolyte/README.md) | Gibbs free-energy minimization | 13 problems | Aqueous electrolyte equilibrium; ill-conditioned by construction |
| [`gas/`](gas/README.md)             | Gas-pipeline network NLP | 4 problems, up to 21k vars | Finite-volume Euler discretization, GasLib networks |
| [`grid/`](grid/README.md)           | AC optimal power flow | MATPOWER cases | Polar-form ACOPF; canonical nonconvex grid NLP |
| [`large_scale/`](large_scale/README.md) | Synthetic large sparse NLPs | up to 100k vars | Bratu, OptControl, PoissonControl, SparseQP — stresses sparse linsol |
| [`mittelmann/`](mittelmann/README.md) | Mittelmann ampl-nlp | 47 problems, up to 261k vars | Standard public NLP benchmark |
| [`water/`](water/README.md)         | Water-network design | 6 problems | MINLPLib instances, signomial nonlinearities |

The GAMS nlpbench harness ([`gams/nlpbench/`](../gams/nlpbench/)) is
also aggregated into the composite report, but it lives outside this
directory because it uses the GAMS solver-link protocol rather than the
.nl driver.

## Running everything

All targets are driven through this directory's `Makefile`. The
top-level `Makefile` provides three convenience shims (`make benchmark`,
`make benchmark-report`, `make benchmark-<suite>`) that delegate here.

All `*-run` targets are **incremental** — they skip work if the suite's
`results.json` is newer than its inputs (the `.nl` files, problem list,
or the `pounce` binary). To force a rerun, use the corresponding
`*-rerun` target, which wipes the suite's `results.json` first.

```sh
# Incremental sweep — only suites with stale/missing results.json rerun,
# then the composite report regenerates
make -C benchmarks benchmark

# Force everything: wipes all results then full rebuild
make -C benchmarks benchmark-rerun

# Just regenerate the composite report from existing JSONs (no suite reruns)
make -C benchmarks benchmark-report

# One suite at a time (incremental)
make -C benchmarks cutest-run
make -C benchmarks water-run
make -C benchmarks gas-run
make -C benchmarks electrolyte-run
make -C benchmarks grid-run
make -C benchmarks cho-run
make -C benchmarks large-scale
make -C benchmarks mittelmann-run
make -C benchmarks gams-bench

# Force a rerun of one suite (wipes its results.json first)
make -C benchmarks water-rerun
make -C benchmarks cutest-rerun
# …etc for every suite

# Or from the repo root via the shim
make benchmark
make benchmark-water
```

`make -C benchmarks help` lists every target.

## How the comparison runs

Two paths for invoking Ipopt:

1. **Rust FFI** (cutest, large_scale). The harness binaries link
   `libipopt` via `pkg-config`. `benchmarks/Makefile` exports
   `PKG_CONFIG_PATH` to point at `ref/Ipopt/install-ma57/`, which is
   built by `make -C benchmarks build-ipopt-ma57`. This guarantees
   POUNCE and Ipopt see identical problem data and are linked against
   the same linear solver family (MA57 on the Ipopt side; FERAL on the
   POUNCE side by default).
2. **AMPL solver protocol** (cho, electrolyte, gas, grid, water,
   mittelmann). `benchmarks/scripts/run_nl_bench.sh` invokes
   `pounce <file.nl>` and `ipopt <file.nl> -AMPL`, parses stdout for
   iteration count and objective, and writes a single
   `<suite>/results.json` in the standard schema consumed by
   `benchmark_report.py`.

`make -C benchmarks check-ipopt-ma57-link` verifies the FFI route
resolves to the MA57 install (run this after a fresh checkout).

## Adding a new suite

1. Create `benchmarks/<suite>/` with a `README.md` describing the
   problem class, sources, prerequisites, and how to run.
2. If the suite is .nl-driven, just call `benchmarks/scripts/run_nl_bench.sh`
   from a new `<suite>-run` target in `benchmarks/Makefile`. The
   composite loader (`load_domain_results`) picks up
   `<suite>/results.json` automatically once you add the suite name
   to the loop in `benchmark_report.py:main()`.
3. The `.gitignore` whitelists every `benchmarks/*/README.md`, the
   `cutest/` source tree, and `benchmarks/scripts/`. Everything else
   under `benchmarks/` is ignored by default — add explicit
   `!benchmarks/<suite>/<file>` lines for any other tracked source.
