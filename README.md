# POUNCE

![POUNCE](logos/pounce_A_pounce.png)

POUNCE is a pure-Rust port of the [Ipopt](https://github.com/coin-or/Ipopt)
interior-point nonlinear programming solver. It solves problems of the
form

```
min  f(x)
s.t. g_L <= g(x) <= g_U
     x_L <=   x  <= x_U
```

where `f` and `g` are twice-continuously-differentiable. The algorithm,
console output, and option semantics follow upstream Ipopt closely enough
that anyone used to reading `ipopt` logs can drop in `pounce` without
relearning where the numbers live.

The default build is pure Rust — no Fortran, no HSL, no system BLAS
required. The bundled [FERAL](crates/pounce-feral) backend provides a
sparse symmetric LDLᵀ factorization. The HSL MA57 backend is available
behind the optional `ma57` feature for users who have `libcoinhsl`
installed.

License: EPL-2.0 (same as upstream Ipopt).

## Status

Work in progress. The algorithm-side core, NLP interface, line search,
filter, barrier update, KKT solve, restoration phase, AMPL `.nl` reader,
and CLI are in place and solve a wide range of NLPs from the standard
test suites (Hock-Schittkowski, CUTEst, Mittelmann ampl-nlp, CHO
parameter estimation, gas/water network design). The C ABI shim
(`pounce-cinterface`) is scaffolded so existing PyIpopt / cyipopt / JuMP
/ AMPL clients can link against it; full coverage lands incrementally.

See `benchmarks/` for the comparison harness against upstream Ipopt.

## Workspace layout

| Crate                                          | Purpose |
|------------------------------------------------|---------|
| [`pounce-common`](crates/pounce-common)        | Types, exceptions, journalist, options, tagged objects, cached results (Ipopt `src/Common`). |
| [`pounce-linalg`](crates/pounce-linalg)        | BLAS-1, dense/compound vectors and matrices, triplet storage, CSC conversion (Ipopt `src/LinAlg`). |
| [`pounce-linsol`](crates/pounce-linsol)        | Symmetric linear-solver trait layer — no FFI; backends plug in below. |
| [`pounce-feral`](crates/pounce-feral)          | Pure-Rust sparse symmetric LDLᵀ backend. Default. |
| [`pounce-hsl`](crates/pounce-hsl)              | MA57 backend via `libcoinhsl` (optional, behind `ma57` feature). |
| [`pounce-nlp`](crates/pounce-nlp)              | TNLP trait, TNLPAdapter, `IpoptApplication` entry point (Ipopt `src/Interfaces`). |
| [`pounce-algorithm`](crates/pounce-algorithm)  | IteratesVector, IpoptData, calculated quantities, KKT, line search, mu update, conv check, main loop (Ipopt `src/Algorithm`). |
| [`pounce-restoration`](crates/pounce-restoration) | Restoration phase (Ipopt `Algorithm/Resto*`). |
| [`pounce-cinterface`](crates/pounce-cinterface) | C ABI shim — `IpoptCreate` / `IpoptSolve` / `IpoptFreeProblem`. |
| [`pounce-cli`](crates/pounce-cli)              | The `pounce` command-line driver. |

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
pounce --problem hs071
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
field documented in [`docs/schema/solve-report-v1.md`](docs/schema/solve-report-v1.md).
`--json-detail summary` (default) emits status, primal `x`, dual
`lambda`, and aggregate statistics; `--json-detail full` adds the
per-iteration trajectory (`iter`, `objective`, `inf_pr`, `inf_du`, `mu`,
`||d||`, alphas, line-search trials) for debugging.

The schema is versioned (`pounce.solve-report/v1`) so downstream
tooling can pin against a major version. Consumers should tolerate
unknown fields — additive changes don't bump the version.

### Sensitivity analysis (sIPOPT-compatible)

The `pounce-sensitivity` crate is a Rust port of upstream Ipopt's
`contrib/sIPOPT/` (Pirnay, López-Negrete & Biegler 2012, [DOI
10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2)).
Three entry points cover the common workflows:

* **AMPL CLI** — `pounce_sens` consumes `.nl` files annotated with
  sIPOPT suffixes (`sens_state_1`, `sens_state_value_1`,
  `sens_init_constr`) and writes the perturbed primal back as
  `sens_sol_state_1`:

  ```sh
  pounce_sens problem.nl              # writes problem.sol
  pounce_sens problem.nl out.sol --json-output result.json --json-detail full
  ```

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

All three are verified against upstream sIPOPT 3.14.19's
`parametric_cpp` golden output to 1e-8. Reduced-Hessian eigendecomposition
is available via `--rh-eigendecomp` (AMPL CLI),
`SensSolve::with_reduced_hessian_eigen` (Rust), and
`solve_with_sens(rh_eigendecomp=True)` (Python). Bound projection of the
perturbed step is available via `--sens-boundcheck [--sens-bound-eps EPS]`
(AMPL CLI), `SensSolve::with_boundcheck(eps)` (Rust), and
`solve_with_sens(sens_boundcheck=True, sens_bound_eps=…)` (Python); the
implementation is a single-pass clamp rather than upstream's iterative
Schur refinement — track progress in
[pounce#7](https://github.com/jkitchin/pounce/issues/7).

## Benchmarks

`benchmarks/` contains comparison harnesses against upstream Ipopt
across several suites (Hock-Schittkowski, CUTEst, Mittelmann ampl-nlp,
CHO, gas pipelines, water networks, large-scale synthetic NLPs).
Common targets:

```sh
make bench-cho          # CHO parameter-estimation
make bench-gas          # GasLib pipelines
make bench-water        # Water network design
make bench-mittelmann   # Mittelmann ampl-nlp
make bench-cutest       # CUTEst (requires one-time `make bench-cutest-prepare`)
```

See `benchmarks/README.md` for the full list and per-suite details.
