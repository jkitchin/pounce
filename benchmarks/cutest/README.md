# CUTEst Suite

CUTEst (Constrained and Unconstrained Testing Environment, revisited) is
the canonical large-scale NLP test collection from Gould, N.I.M., Orban,
D., and Toint, P.L., "CUTEst: a Constrained and Unconstrained Testing
Environment with safe threads for mathematical optimization", *Computational
Optimization and Applications* 60:545-557 (2015). The MASTSIF archive
provides roughly 1500 problems spanning a few variables to 100K+, covering
every constraint class and a wide range of numerical pathologies.

The curated ripopt subset in `problem_list.txt` is 727 problems chosen to
exercise every solver path (bounds, equality, inequality, NE, LS, and
mixed), with `problem_list_full.txt` covering all 1542 SIF problems.

## Contents

- `run_cutest.rs` — the `cutest_suite` binary (ripopt + optional Ipopt
  comparison, runs each problem in a subprocess with timeout)
- `cutest_ffi.rs`, `cutest_problem.rs` — thin Rust wrappers around the
  CUTEst Fortran interface
- `collect_kkt.rs` — `collect_kkt` binary that dumps per-iteration KKT
  matrices for offline analysis
- `prepare.sh` — compiles SIF problems from `~/.local/cutest/mastsif/` into
  shared libraries under `problems/`
- `compare.py` — per-problem comparison report from `results.json`
- `problem_list.txt` — curated 727-problem subset
- `problem_list_full.txt` — full 1542-problem list
- `problems/` — compiled `lib<NAME>.dylib`/`.so` and `<NAME>_OUTSDIF.d`
  files (gitignored)
- `results.json` — latest benchmark results
- `CUTEST_COMPARATIVE_REPORT.org` — long-form analysis

## Prerequisites

1. CUTEst toolchain installed to `~/.local/cutest/` (run
   `bash install_cutest.sh` from the repo root, or `make cutest-install`)
2. MASTSIF archive at `~/.local/cutest/mastsif/`
3. gfortran (for compiling SIF problems)
4. Source the environment before running: `source ~/.local/cutest/env.sh`

## How to run

From the repo root:

```bash
make cutest-prepare         # compile SIF -> shared libs (slow, one-time)
make cutest-run             # run the 727 problems (~10 min)
make cutest-report          # generate comparison report
# or all three in sequence:
make cutest
```

For the full 1542-problem sweep:

```bash
make cutest-full
```

The underlying cargo commands (if you want to bypass the Makefile):

```bash
RESULTS_FILE=benchmarks/cutest/results.json \
    cargo run --release --bin cutest_suite --features cutest,ipopt-native
```

Individual problems can be passed directly:

```bash
cargo run --release --bin cutest_suite --features cutest,ipopt-native -- \
    ROSENBR HS71 ACOPP14
```

## Output

- `results.json` — ripopt and Ipopt per-problem results
- `benchmark_stderr.txt` — solver chatter
- `problems/` — compiled SIF artefacts (gitignored)

The CUTEst suite feeds the composite `benchmarks/BENCHMARK_REPORT.md`.
