# pounce-feral

POUNCE's default sparse symmetric LDLᵀ backend, built on the pure-Rust
[FERAL](https://github.com/jkitchin/feral) library. Implements
[`SparseSymLinearSolverInterface`](../pounce-linsol) so the upstream
`TSymLinearSolver` wrapper requires no changes versus the MA57 path.

This crate is what makes the default POUNCE build pure Rust — no
Fortran, no HSL, no system BLAS required.

## Lifecycle

The wrapper mirrors `pounce-hsl::Ma57SolverInterface`:

1. `matrix_format()` returns `EMatrixFormat::TripletFormat` (1-based,
   lower-triangle COO).
2. `initialize_structure` caches 0-based row/col arrays and allocates
   the values buffer; the upper-triangle entries that POUNCE's KKT
   assembly emits are canonicalized to the lower triangle here, because
   FERAL's `CscMatrix::from_triplets` silently drops upper-triangle
   entries during LDLᵀ.
3. `multi_solve` rebuilds the CSC matrix from the cached pattern +
   caller-filled values, then dispatches to `feral::Solver::factor` /
   `solve_many`. FERAL's pattern-fingerprint cache reuses the symbolic
   factorization across iterates with identical structure — the IPM
   common case.
4. `increase_quality` delegates to `feral::Solver::increase_quality`
   and uses MA57's `pivtol_changed` / `CallAgain` protocol so the
   upper-layer reload-and-retry semantics line up.

## When to use FERAL vs MA57

FERAL is the default. MA57 is generally faster for very large sparse
KKT systems (≳ 50k variables) and remains the option of choice when
`libcoinhsl` is available; enable it with `--features ma57` on
`pounce-cli`. See `benchmarks/BENCHMARK_REPORT.md` for a head-to-head
comparison across the `.nl` benchmark suites.

## License

EPL-2.0.
