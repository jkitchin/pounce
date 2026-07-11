# POUNCE

[![CI](https://github.com/jkitchin/pounce/actions/workflows/ci.yml/badge.svg)](https://github.com/jkitchin/pounce/actions/workflows/ci.yml)
[![Docs](https://github.com/jkitchin/pounce/actions/workflows/docs.yml/badge.svg)](https://jkitchin.github.io/pounce/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20387011.svg)](https://doi.org/10.5281/zenodo.20387011)

[![PyPI: pounce-solver](https://img.shields.io/pypi/v/pounce-solver.svg?label=pypi%3A%20pounce-solver)](https://pypi.org/project/pounce-solver/)
[![Downloads: pounce-solver](https://static.pepy.tech/badge/pounce-solver)](https://pepy.tech/project/pounce-solver)
[![PyPI: pyomo-pounce](https://img.shields.io/pypi/v/pyomo-pounce.svg?label=pypi%3A%20pyomo-pounce)](https://pypi.org/project/pyomo-pounce/)
[![Downloads: pyomo-pounce](https://static.pepy.tech/badge/pyomo-pounce)](https://pepy.tech/project/pyomo-pounce)

![POUNCE](logos/pounce_A_pounce.png)

POUNCE is a pure-Rust interior-point optimization solver. Its
nonlinear-programming core began as a faithful port of
[Ipopt](https://github.com/coin-or/Ipopt) — the same filter line-search
algorithm, console output, and option semantics, so anyone used to reading
`ipopt` logs can drop in `pounce` without relearning where the numbers
live — and it has since grown into a *family* of solvers sharing one
numerical backbone:

- **Nonlinear programming** — the filter line-search interior-point method
  (the Ipopt port), plus an active-set SQP path, for general smooth problems
  `min f(x)  s.t.  g_L ≤ g(x) ≤ g_U,  x_L ≤ x ≤ x_U`.
- **Conic & quadratic** — dedicated interior-point solvers for LP, convex QP,
  second-order (SOCP), positive-semidefinite (SDP), and the non-symmetric
  exponential and power cones — each solved to the global optimum, with
  infeasibility certificates, warm starts, and post-optimal sensitivity.
- **Global optimization** — certified global optima for nonconvex
  **polynomial** problems via SOS / Lasserre relaxations. (A general-purpose
  spatial branch-and-bound solver, `pounce-global`, is in development on the
  `feature/global` branch and not part of this release.)

Convex and conic problems are solved to global optimality; nonconvex problems
are solved locally by default, or — for polynomials — to a certified global
optimum via the SOS path. See **[Choosing a Solver](https://jkitchin.github.io/pounce/choosing-a-solver.html)**
for the full map of which solver fits which problem.

The default build is pure Rust — no Fortran, no HSL, no system BLAS required.
The [FERAL](crates/pounce-feral) backend provides a sparse symmetric LDLᵀ
factorization that is also in pure Rust. The HSL MA57 backend is available
behind the optional `ma57` feature for users who have `libcoinhsl` installed.

License: EPL-2.0 (same as upstream Ipopt).

## Status

Production-ready for the core IPM workflow. The algorithm-side core,
NLP interface, line search, filter, barrier update (monotone + Mehrotra
adaptive), KKT solve, restoration phase, AMPL `.nl` reader, the C ABI
(`pounce-cinterface`), the Python wrapper (`pounce-solver`), and the
CLI all solve a wide range of NLPs from the standard test suites
(Hock-Schittkowski, CUTEst, Mittelmann ampl-nlp, CHO parameter
estimation, gas/water network design). Sensitivity analysis (sIPOPT
port) and reduced-Hessian computation are wired end-to-end; the
`pounce-presolve` pass (auxiliary-equality elimination + FBBT +
bound-tightening) and the active-set SQP path (`pounce-qp`-backed)
are available behind option keys.

Beyond the NLP core, the solver family is wired end-to-end and validated
against external suites:

- **Convex & conic** (`pounce-convex`) — LP / convex-QP, SOCP, the
  exponential and power cones (geometric programming, entropy, logistic,
  `p`-norms), and small dense SDPs, with a Conic Benchmark Format (`.cbf`)
  reader cross-checked against the CBLIB tier. The CLI's `auto` routing
  classifies an `.nl` and sends LP / convex-QP problems here automatically.
- **Global (polynomials)** — SOS / Lasserre polynomial optimization
  (`sos_minimize` / `pounce.sos_minimize`): a single SDP certifies the global
  minimum and recovers the global minimizers. A general-purpose spatial
  branch-and-bound solver (`pounce-global`, with McCormick relaxations,
  OBBT/FBBT bound tightening, and a certified optimality gap) is in development
  on the `feature/global` branch and not part of this release.

The shipped solvers — NLP, conic, and SOS — are reachable from the CLI, the
Python package, and the JSON solve report.

See `benchmarks/` for the comparison harness against upstream Ipopt.

## Documentation

The full user guide lives in [`docs/`](docs/src) as an
[mdbook](https://rust-lang.github.io/mdBook/) — installation, the CLI,
solver options, the JSON solve report, sensitivity analysis, and the
Pyomo / Python integrations. Browse the Markdown sources directly on
GitHub, or render the book locally:

```sh
make book       # builds docs/book/ (requires `cargo install mdbook`)
```

## Workspace layout

| Crate                                             | Purpose                                                                                                                       |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| [`pounce-common`](crates/pounce-common)           | Types, exceptions, journalist, options, tagged objects, cached results (Ipopt `src/Common`).                                  |
| [`pounce-linalg`](crates/pounce-linalg)           | BLAS-1, dense/compound vectors and matrices, triplet storage, CSC conversion (Ipopt `src/LinAlg`).                            |
| [`pounce-linsol`](crates/pounce-linsol)           | Symmetric linear-solver trait layer — no FFI; backends plug in below.                                                         |
| [`pounce-feral`](crates/pounce-feral)             | Pure-Rust sparse symmetric LDLᵀ backend. Default.                                                                             |
| [`pounce-hsl`](crates/pounce-hsl)                 | MA57 backend via `libcoinhsl` (optional, behind `ma57` feature).                                                              |
| [`pounce-nlp`](crates/pounce-nlp)                 | TNLP trait, TNLPAdapter, `IpoptApplication` entry point (Ipopt `src/Interfaces`).                                             |
| [`pounce-algorithm`](crates/pounce-algorithm)     | IteratesVector, IpoptData, calculated quantities, KKT, line search, mu update, conv check, main loop (Ipopt `src/Algorithm`). |
| [`pounce-restoration`](crates/pounce-restoration) | Restoration phase (Ipopt `Algorithm/Resto*`).                                                                                 |
| [`pounce-presolve`](crates/pounce-presolve)       | NLP preprocessing — auxiliary-equality elimination, FBBT, bound tightening, redundant-row removal.                            |
| [`pounce-l1penalty`](crates/pounce-l1penalty)     | Thierry-Biegler ℓ₁-exact penalty-barrier wrapper for degenerate / MPCC problems.                                              |
| [`pounce-sensitivity`](crates/pounce-sensitivity) | Post-optimal sensitivity + reduced-Hessian (port of upstream sIPOPT).                                                         |
| [`pounce-qp`](crates/pounce-qp)                   | Sparse parametric active-set QP subproblem solver — drives the SQP path and the sensitivity corrector.                        |
| [`pounce-convex`](crates/pounce-convex)           | Convex/conic interior-point solver — LP, QP, SOCP, exponential/power cones, small SDP, and SOS polynomial optimization.       |
| [`pounce-solve-report`](crates/pounce-solve-report) | `pounce.solve-report/v1` JSON writer (shared by `pounce-cli --json-output` and `IpoptWriteSolveReport`).                     |
| [`pounce-observability`](crates/pounce-observability) | `tracing` subscriber install + per-iteration collector layer that feeds the iteration stream into the solve report.       |
| [`pounce-cinterface`](crates/pounce-cinterface)   | C ABI shim — `CreateIpoptProblem` / `IpoptSolve` / `FreeIpoptProblem` / `IpoptWriteSolveReport`.                              |
| [`pounce-py`](crates/pounce-py)                   | PyO3 bindings — the cyipopt-compatible `pounce` Python package (the `pounce-solver` wheel).                                  |
| [`pounce-cli`](crates/pounce-cli)                 | The `pounce` command-line driver.                                                                                             |
| [`pounce-studio-core`](crates/pounce-studio-core) | Solve-report / iter-dump parsers and diagnostic analysis (foundation for the `pounce-studio` GUI / MCP server).               |
| [`pounce-studio-pyo3`](crates/pounce-studio-pyo3) | PyO3 `_native` extension exposing `pounce-studio-core` to the `pounce-studio-mcp` Python MCP server.                          |

## Build

Prerequisites: a stable Rust toolchain. Nothing else for the default
build.

```sh
make            # release build of the workspace
make test       # run all tests
make clippy     # lint
make doc        # rustdoc
```

To build with the HSL MA57 backend (requires `libcoinhsl` discoverable
by the linker):

```sh
cargo build -p pounce-cli --release --features ma57
```

## Install

```sh
make install                # installs to $HOME/.local
sudo make install PREFIX=/usr/local   # or system-wide
```

This drops the `pounce` binary into `$PREFIX/bin` and the
`libpounce_cinterface` shared library into `$PREFIX/lib`. Make sure
`$HOME/.local/bin` is on your `PATH`.

## Usage

Solve an AMPL `.nl` file:

```sh
pounce problem.nl
pounce problem.nl print_level=8 max_iter=500 tol=1e-10
pounce problem.nl linear_solver=ma57       # with --features ma57
```

Trailing `KEY=VALUE` pairs follow the same syntax and semantics as the
upstream Ipopt CLI; they override values loaded from `--options-file`.

List available built-in test problems:

```sh
pounce --list-problems
pounce --problem rosenbrock
```

Full help:

```sh
pounce --help
```

### Solution output (`.sol`)

Following the AMPL solver convention, solving a positional `.nl` file
writes a sibling `<stub>.sol` next to it — `pounce problem.nl`
produces `problem.sol`. The file carries the primal `x` and dual
`lambda` blocks plus an `objno` line with the AMPL `solve_result_num`,
so AMPL (or any `.sol` reader) can pull the solution back:

```sh
pounce problem.nl                       # writes problem.sol
pounce problem.nl --sol-output out.sol  # write to an explicit path
pounce problem.nl --no-sol              # skip the .sol write
```

A `.sol` is written even when the solve fails, so the
`solve_result_num` is always recoverable. Built-in problems
(`--problem …`) have no `.nl` stub, so they only produce a `.sol`
when `--sol-output` is given explicitly.

### AMPL imported (external) functions

`.nl` files that import functions from a shared library (declared in
`F` segments, called via `f<id>` expression tokens) are supported.
Set `AMPLFUNC` to a newline-separated list of library paths — the
same convention upstream Ipopt uses — and pounce loads each library
through the standard AMPL `funcadd_ASL` ABI:

```sh
AMPLFUNC=$HOME/.idaes/bin/general_helmholtz_external.dylib \
  pounce helmholtz.nl
```

Multiple libraries: `AMPLFUNC=$(printf '%s\n%s\n' /path/lib1 /path/lib2) pounce …`.
Without `AMPLFUNC` set, problems that need external functions fail
with a clear error naming the offending function.

### Pyomo

Because pounce speaks the AMPL NL/SOL protocol, it drops into
[Pyomo](https://www.pyomo.org/) through the AMPL Solver Library
interface — exactly how Pyomo drives Ipopt. The
[`pyomo-pounce`](pyomo-pounce) package registers `pounce` as a Pyomo
`SolverFactory` solver:

```python
import pyomo_pounce  # registers 'pounce'
from pyomo.environ import *

solver = SolverFactory('pounce')
solver.solve(model)
```

To invoke pounce directly as an AMPL solver, pass `-AMPL`
(`pounce problem.nl -AMPL`); in that mode the termination is conveyed
through the `.sol` file rather than the process exit code.

### Machine-readable output (JSON)

Pass `--json-output PATH` to write a structured solve report alongside
the regular console output:

```sh
pounce problem.nl --json-output result.json
pounce problem.nl --json-output result.json --json-detail full
```

The report is FAIR-aligned (Wilkinson et al. 2016, DOI
[10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18)) — every
field documented in [`docs/src/schema/solve-report-v1.md`](docs/src/schema/solve-report-v1.md).
`--json-detail summary` (default) emits status, primal `x`, dual
`lambda`, and aggregate statistics; `--json-detail full` adds the
per-iteration trajectory (`iter`, `objective`, `inf_pr`, `inf_du`, `mu`,
`||d||`, alphas, line-search trials) for debugging.

The schema is versioned (`pounce.solve-report/v1`) so downstream
tooling can pin against a major version. Consumers should tolerate
unknown fields — additive changes don't bump the version.

### Logging & diagnostics

POUNCE emits diagnostics through the [`tracing`](https://docs.rs/tracing)
ecosystem. Logs go to **stderr**; program output (the iteration table,
summary, `--dump`) stays on **stdout**, so the two never collide when
redirected. The per-iteration table is colored when stdout is a terminal.

| Variable | Effect |
|---|---|
| `RUST_LOG` | Verbosity / per-target filtering (default `info`); e.g. `RUST_LOG=pounce::linsol=debug`. |
| `POUNCE_LOG_FORMAT` | `text` (default) or `json` — structured JSON sink on stderr for Studio / CI. |
| `NO_COLOR` / `CLICOLOR_FORCE` | Disable / force the colored iteration table. |

See [`docs/src/options.md`](docs/src/options.md) and
[`docs/src/troubleshooting.md`](docs/src/troubleshooting.md) for details.

### Interactive solver debugger (`--debug`)

POUNCE ships an interactive debugger for the interior-point loop — a *pdb
for the IPM*. Pause the solve at well-defined checkpoints, inspect and
**mutate** the live state (iterate, multipliers, the barrier parameter μ),
set breakpoints by iteration / numeric condition / solver event, step
through an iteration's internal phases, rewind, and re-solve with new
options. It has **zero effect on the solve when not attached**.

```sh
pounce problem.nl --debug             # human REPL (history, Tab-complete)
pounce problem.nl --debug-on-error    # run freely; drop in only on failure
pounce problem.nl --debug-json        # newline-delimited JSON for agents/tools
```

`--debug-json` speaks a self-describing protocol: the first line is a
`hello` handshake advertising every command, event, checkpoint, metric,
and capability, so an **LLM agent, script, or visual debugger** can drive
the solver with no out-of-band docs. Full guide:
[`docs/src/debugger.md`](docs/src/debugger.md). Post-mortem analysis of a
finished solve is also available through the **pounce-studio MCP server**
([`studio/mcp`](studio/mcp)) and the JSON solve report.

### Sensitivity analysis (sIPOPT-compatible)

The `pounce-sensitivity` crate is a Rust port of upstream Ipopt's
`contrib/sIPOPT/` (Pirnay, López-Negrete & Biegler 2012, [DOI
10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2)).
Four entry points cover the common workflows:

* **AMPL CLI** — the main `pounce` driver auto-detects the sIPOPT
  suffixes (`sens_state_1`, `sens_state_value_1`, `sens_init_constr`)
  in an input `.nl`, runs a post-optimal sensitivity step after the
  solve, and writes the perturbed primal back as `sens_sol_state_1` —
  no separate binary or flag needed:

  ```sh
  pounce problem.nl                   # writes problem.sol
  pounce problem.nl out.sol --json-output result.json --json-detail full
  ```

  `pounce_sens` is retained as a thin backward-compatibility alias:
  `pounce_sens in.nl out.sol` is identical to `pounce in.nl out.sol`,
  so existing AMPL / solver scripts keep working unchanged.

* **Rust library** — `SensSolve` is a builder that wraps the
  `on_converged` callback plumbing into a single call:

  ```rust
  use pounce_sensitivity::SensSolve;
  let result = SensSolve::new(vec![2, 3])
      .with_deltas(vec![0.05, 0.0])
      .with_reduced_hessian()
      .run(&mut app, tnlp);
  // result.dx, result.reduced_hessian, result.status
  ```

* **Python (`pounce.Problem`)** — `solve_with_sens` exposes the same
  capability from the cyipopt-compatible Python wrapper. See
  [`python/notebooks/04_sensitivity.ipynb`](python/notebooks/04_sensitivity.ipynb)
  for a walkthrough.

* **Pyomo (`pyomo_pounce`)** — declare the parameters that matter when
  building the model, solve normally, then query derivatives; no
  suffixes and no upfront perturbation values:

  ```python
  from pyomo_pounce import declare_sens_param, gradient, estimate
  declare_sens_param(m.p)
  SolverFactory("pounce").solve(m)     # keeps the KKT factorization
  gradient(m.x, wrt=m.p)               # dx*/dp; constraints give dlambda/dp
  estimate(m, [(m.p, 2.5)])            # perturbed-solution estimate
  ```

  See [`docs/src/sensitivity.md`](docs/src/sensitivity.md) and
  [`python/notebooks/25_pyomo_sensitivity.ipynb`](python/notebooks/25_pyomo_sensitivity.ipynb)
  (an optimal-control example where the first-move gradients are the
  NMPC feedback gains).

All three are verified against upstream sIPOPT 3.14.19's
`parametric_cpp` golden output to 1e-8. Reduced-Hessian eigendecomposition
is available via `--rh-eigendecomp` (AMPL CLI),
`SensSolve::with_reduced_hessian_eigen` (Rust), and
`solve_with_sens(rh_eigendecomp=True)` (Python). Bound projection of the
perturbed step is available via `--sens-boundcheck [--sens-bound-eps EPS]`
(AMPL CLI), `SensSolve::with_boundcheck(eps)` (Rust), and
`solve_with_sens(sens_boundcheck=True, sens_bound_eps=…)` (Python). The
bound projection is a single-pass clamp; upstream's iterative Schur
refinement (re-factorize on each violation) is intentionally not ported.

### Sessions: factor-once / solve-many

For workflows that issue several follow-up operations against the
converged KKT factor — sensitivity sweeps, reduced Hessians over many
pinned-row sets, raw KKT back-solves — the **session APIs** hold the
factor alive between calls:

* **Python** — `pounce.Solver(problem)` with `.solve(...)`,
  `.parametric_step(...)`, `.reduced_hessian(...)`, `.kkt_solve(...)`.
* **C** — `IpoptCreateSolver(&prob)` / `IpoptSolverSolve` /
  `IpoptSolverParametricStep` / `IpoptSolverReducedHessian` /
  `IpoptSolverKktSolve` / `IpoptFreeSolver`. The classic
  `IpoptSolve` API is unchanged and unaffected.
* **Rust** — `pounce_sensitivity::Solver`; or
  `pounce_linsol::Factorization` for the underlying factor-only
  primitive (no IPM in the loop).

See [`docs/src/sessions.md`](docs/src/sessions.md) for the full
walkthrough.

## Benchmarks

`benchmarks/` contains comparison harnesses against upstream Ipopt
across several suites (CUTEst, Mittelmann ampl-nlp, CHO, electrolyte,
grid, gas, water, large-scale synthetic NLPs, GAMS nlpbench). All
suites feed a single composite report at `benchmarks/BENCHMARK_REPORT.md`
with provenance metadata (versions, git SHA, linear solvers).

```sh
make benchmark              # full sweep: every suite + composite report
make benchmark-report       # regenerate the composite report only
make benchmark-water        # one suite at a time (water, gas, cutest, …)
```

The Ipopt comparison side runs against a locally-built Ipopt-MA57
(`ref/Ipopt/install-ma57/`). Build it once with
`make -C benchmarks build-ipopt-ma57`. See `benchmarks/README.md` for
the full list and per-suite details.

## Acknowledgments

POUNCE's nonlinear-programming core is a Rust port of
[Ipopt](https://github.com/coin-or/Ipopt), the interior-point nonlinear
programming solver by Andreas Wächter, Lorenz T. Biegler, and the COIN-OR
community. Its algorithm, console output, and option semantics are modeled
directly on that codebase, which is released under the EPL-2.0.

It is a sibling of [ripopt](https://github.com/jkitchin/ripopt), an
earlier memory-safe interior-point NLP optimizer in Rust by the same
author ([doi:10.5281/zenodo.19542664](https://doi.org/10.5281/zenodo.19542664)).

I want to thank Carl Laird and Victor Alves for encouraging this particular
development path. In ripopt I codeveloped the ipm and linear algebra solvers,
which led to a plateau in progress because of the difficulty in debugging which
side the problems were on. They encouraged me to instead use a good, known
linear algebra library and just focus on the ipm development. In parallel, I
also did the same for the linear algebra library, and separated out feral to
focus only on that. This package is the union of these two efforts, and it is
much more robust than ripopt is. 

### Key references

- Wächter, A., Biegler, L.T. "On the implementation of an
  interior-point filter line-search algorithm for large-scale
  nonlinear programming." *Mathematical Programming* 106(1), 25–57
  (2006). [doi:10.1007/s10107-004-0559-y](https://doi.org/10.1007/s10107-004-0559-y)
  — the algorithm POUNCE implements.
- Wächter, A., Biegler, L.T. "Line search filter methods for nonlinear
  programming: Motivation and global convergence." *SIAM Journal on
  Optimization* 16(1), 1–31 (2005).
  [doi:10.1137/S1052623403426556](https://doi.org/10.1137/S1052623403426556)
- Wächter, A., Biegler, L.T. "Line search filter methods for nonlinear
  programming: Local convergence." *SIAM Journal on Optimization*
  16(1), 32–48 (2005).
  [doi:10.1137/S1052623403426544](https://doi.org/10.1137/S1052623403426544)
- Fletcher, R., Leyffer, S. "Nonlinear programming without a penalty
  function." *Mathematical Programming* 91(2), 239–269 (2002).
  [doi:10.1007/s101070100244](https://doi.org/10.1007/s101070100244)
  — the filter concept underlying the line search.
- Pirnay, H., López-Negrete, R., Biegler, L.T. "Optimal sensitivity
  based on IPOPT." *Mathematical Programming Computation* 4(4),
  307–331 (2012).
  [doi:10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2)
  — the sIPOPT method behind `pounce-sensitivity`.
- Duff, I.S. "MA57—a code for the solution of sparse symmetric
  definite and indefinite systems." *ACM Transactions on Mathematical
  Software* 30(2), 118–144 (2004).
  [doi:10.1145/992200.992202](https://doi.org/10.1145/992200.992202)
  — the optional `ma57` linear-solver backend.
