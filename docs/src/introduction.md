# Introduction

POUNCE is a general interior-point method, implemented in pure Rust — one
numerical backbone that now spans nonlinear, conic/quadratic, and polynomial
global optimization rather than a single problem class. Its
nonlinear-programming core began as a faithful port of the
[Ipopt](https://github.com/coin-or/Ipopt) filter line-search method —
the algorithm, console output, and option semantics follow upstream Ipopt
closely enough that anyone used to reading `ipopt` logs can drop in
`pounce` without relearning where the numbers live — and it has since grown
into a *family* of solvers sharing that backbone:

- **Nonlinear programming** — the filter line-search interior-point method
  (the Ipopt port) plus an active-set SQP path, for general smooth problems

  ```text
  min  f(x)
  s.t. g_L <= g(x) <= g_U
       x_L <=   x  <= x_U
  ```

  where `f` and `g` are twice-continuously-differentiable.
- **Conic & quadratic** — LP, convex QP, second-order (SOCP),
  positive-semidefinite (SDP), and the non-symmetric exponential and power
  cones, each solved to the global optimum.
- **Global optimization** — certified global optima for nonconvex
  **polynomial** problems via SOS / Lasserre relaxations. (A general-purpose
  spatial branch-and-bound solver, `pounce-global`, is in development on the
  `feature/global` branch and not part of this release.)

See [Choosing a Solver](choosing-a-solver.md) for which solver fits which
problem.

## Pure Rust by default

The default build is pure Rust — no Fortran, no commercial solver, no system BLAS
required. The bundled FERAL backend provides a sparse symmetric LDLᵀ
factorization. The HSL MA57 backend is available behind the optional
`ma57` feature for users who have a license for `libcoinhsl` and have it installed (see
[Installation](installation.md)).

## Status

Production-ready for the core IPM workflow. The algorithm-side core,
NLP interface, line search, filter, barrier update (monotone +
Mehrotra adaptive), KKT solve, restoration phase, AMPL `.nl` reader,
the C ABI (`pounce-cinterface`), the Python wrapper (`pounce-solver`),
and the CLI all solve a wide range of NLPs from the standard test
suites (Hock-Schittkowski, CUTEst, Mittelmann ampl-nlp, CHO parameter
estimation, gas/water network design). Sensitivity analysis (sIPOPT
port), reduced-Hessian computation, the auxiliary-equality + FBBT
presolve, and the [active-set SQP path](active-set-sqp.md) are all wired
in and available behind option keys. Existing PyIpopt / cyipopt / JuMP / AMPL clients
link against `libpounce_cinterface` in place of `libipopt`
unchanged.

The conic and global solvers are wired end-to-end alongside the NLP
core: the convex interior-point solver (`pounce-convex`) handles
LP / QP, SOCP, exponential / power cones, and small SDPs — with a Conic
Benchmark Format (`.cbf`) reader cross-checked against the CBLIB tier —
and adds SOS / Lasserre polynomial global optimization (`sos_minimize`).
These are reachable from the CLI, the Python package, and the JSON solve
report. A deterministic spatial branch-and-bound solver for general
factorable nonconvex problems (`pounce-global`) is in development on the
`feature/global` branch and not part of this release.

## License

EPL-2.0, the same license as upstream Ipopt.

## Where to go next

- [Installation](installation.md) — build and install POUNCE.
- [Quick Start](quick-start.md) — solve your first problem.
- [Running Solves](cli.md) — the command-line driver in depth.
- [Acknowledgments](acknowledgments.md) — the papers behind the
  algorithm.
