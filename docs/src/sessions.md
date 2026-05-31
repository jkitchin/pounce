# Sessions: Factor-Once / Solve-Many

POUNCE's IPM converges to a KKT linear system that, once factored,
answers a number of useful follow-up questions cheaply: parametric
steps, reduced Hessians, custom back-solves. The **session** APIs let
you hold that factor alive between operations, rather than
rebuilding it on every call. The same machinery serves two workloads:

* **Sensitivity / many-RHS.** After one solve, issue many cheap
  operations against the converged factor — parametric steps for
  several parameter perturbations, reduced Hessians over several
  pinned-row sets, raw KKT back-solves.
* **Factor-only.** For non-IPM uses (shift-invert eigensolves, custom
  Newton iterations) the underlying [`Factorization`] handle in
  `pounce-linsol` exposes factor / refactor / back-solve directly,
  without the IPM in the loop.

## Which layer do I want?

| You want…                                                            | Use                                       |
|----------------------------------------------------------------------|-------------------------------------------|
| One solve plus a few sensitivity queries, from Python                | `pounce.Solver` (Python)                  |
| The same, from C                                                     | `IpoptSolver` (C ABI)                     |
| The same, from Rust                                                  | `pounce_sensitivity::Solver`              |
| Just a sparse symmetric factor — no IPM involved                     | `pounce_linsol::Factorization`            |
| A one-shot sensitivity computation with a fluent builder             | `pounce_sensitivity::SensSolve` (Rust) or `Problem.solve_with_sens` (Python) |

The session API does **not** rebuild the IPM. Each `solve()` call runs
the full barrier method from scratch. What it reuses is the **factor
that exists at convergence**: KKT back-solves and sensitivity
operations skip the symbolic factor, AMD ordering, and numeric
factorization.

## Python

```python
import pounce

problem = pounce.Problem(...)
solver = pounce.Solver(problem)

x, info = solver.solve(x0=x0)
assert solver.converged

# Parametric step ∂x*/∂p · Δp, with p pinned by g(x) row indices.
dx = solver.parametric_step([2, 3], [-0.5, 0.0])

# Reduced Hessian B K⁻¹ Bᵀ over the same pinned-row set.
hr = solver.reduced_hessian([2, 3])

# Raw KKT back-solve, useful for custom workflows.
dim = solver.kkt_dim
rhs = np.zeros(dim)
lhs = solver.kkt_solve(rhs)
```

The KKT compound vector is laid out as
`x || s || y_c || y_d || z_l || z_u || v_l || v_u`. `pin` indices in
`parametric_step` / `reduced_hessian` are 0-based row indices into
`g(x)`; they are shifted internally to the y_c block.

`pounce.Problem.solve()` and `Problem.solve_with_sens()` still work
unchanged — each internally builds a fresh session — but new code that
issues more than one sensitivity query per solve should prefer
`pounce.Solver` to skip rebuilding the application.

## C

```c
IpoptProblem prob = CreateIpoptProblem(...);
AddIpoptStrOption(prob, "linear_solver", "feral");

/* Consumes prob — the IpoptSolver is now the sole owner.
   prob is NULLed; calling FreeIpoptProblem(prob) on the now-null
   pointer is harmless. */
IpoptSolver sol = IpoptCreateSolver(&prob);

double x[n], obj;
IpoptSolverSolve(sol, x, NULL, &obj, NULL, NULL, NULL, user_data);

Index dim = IpoptSolverGetKktDim(sol);     /* compound KKT dim     */
double rhs[dim], lhs[dim];                  /* memset rhs as needed */
IpoptSolverKktSolve(sol, rhs, lhs);

Index pins[2] = {2, 3};
double deltas[2] = {-0.5, 0.0};
double dx[n];
IpoptSolverParametricStep(sol, 2, pins, deltas, dx);

double hr[2 * 2];                           /* column-major dense   */
IpoptSolverReducedHessian(sol, 2, pins, 1.0, hr);

IpoptFreeSolver(sol);
```

The classic `IpoptSolve` API is unchanged and unaffected; the session
handle lives alongside it.

## Rust

```rust
use pounce_sensitivity::Solver;

let mut solver = Solver::new(app, tnlp);
solver.solve();
assert!(solver.converged().is_some());

let dx = solver.parametric_step(&[2, 3], &[-0.5, 0.0])?;
let hr = solver.compute_reduced_hessian(&[2, 3], 1.0)?;

let mut lhs = vec![0.0; solver.kkt_dim().unwrap()];
solver.kkt_solve(&rhs, &mut lhs)?;
```

For purely linear-algebra uses with no IPM in the loop:

```rust
use pounce_linsol::Factorization;

let mut fact = Factorization::new(dim, ia, ja, &values, backend)?;
fact.solve(&mut rhs, 1)?;          // back-substitute in place
fact.refactor(&new_values)?;       // pattern preserved; numeric reuse
fact.solve_one(&mut another_rhs)?;
```

## What's preserved across operations

* **Symbolic factor / AMD ordering.** Owned by the linear-solver
  backend; reused on every back-solve and on `refactor()`.
* **Numeric factor.** Reused on every back-solve until you refactor.
* **The converged primal-dual state** (`x*`, multipliers, `g(x*)`,
  iteration stats).

## What's not preserved across `solve()` calls

The session is currently a **factor-and-query** value: one solve,
many follow-up operations. A separate `resolve()` that re-runs the
IPM while reusing the symbolic factor + AMD ordering across top-level
solves (for MPC / B&B / warm-start workloads) is planned but not yet
implemented. Each `solve()` call today runs a fresh IPM.

## Verification

All session entry points are tested for numerical equivalence with the
corresponding one-shot APIs:

* `pounce.Solver.solve` ≡ `Problem.solve` (1e-12).
* `pounce.Solver.parametric_step` ≡ `Problem.solve_with_sens(deltas=…)['dx']`
  (1e-10).
* `pounce.Solver.reduced_hessian` ≡
  `Problem.solve_with_sens(compute_reduced_hessian=True)['reduced_hessian']`
  (1e-10).
* `pounce_sensitivity::Solver::parametric_step` ≡
  `SensSolve::with_deltas` (1e-10).

See `python/tests/test_solver_session.py` and
`crates/pounce-sensitivity/tests/solver_session.rs` for the full test
matrix.
