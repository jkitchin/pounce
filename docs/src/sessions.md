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
| The same, from Rust                                                  | `pounce_rs::sensitivity::Solver`          |
| Just a sparse symmetric factor — no IPM involved                     | `pounce_rs::linsol::Factorization`        |
| A one-shot sensitivity computation with a fluent builder             | `pounce_rs::sensitivity::SensSolve` (Rust) or `Problem.solve_with_sens` (Python) |
| Re-solving an NLP family with presolve *and* warm starts (MPC, oximo) | `pounce_rs::session::TnlpPresolveSession` (Rust) |
| Re-solving a convex QP family with presolve *and* warm starts         | `pounce_convex::ConvexPresolveSession` (Rust; `pounce_rs::convex` with `convex`) |

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
# NOTE: over pin rows this is −H_R, not H_R — negate to read curvature.
hr = solver.reduced_hessian([2, 3])

# Raw KKT back-solve, useful for custom workflows.
dim = solver.kkt_dim
rhs = np.zeros(dim)
lhs = solver.kkt_solve(rhs)

# Which bounds and rows actually hold the solution, in user index order.
rep = solver.classify_activity()          # needs bound_relax_factor=0
rep["var_status"], rep["row_status"]

# Constraint-row gradient in user space, natural units.
a = solver.row_normal(j)

# Exact Lagrangian Hessian times a user-space vector, natural units.
hv = solver.hessian_vec(v)
```

The KKT compound vector is laid out as
`x || s || y_c || y_d || z_l || z_u || v_l || v_u`. `pin` indices in
`parametric_step` / `reduced_hessian` are 0-based row indices into
`g(x)`; they are mapped internally to the matching y_c rows (through
the equality/inequality split, so inequalities may precede the pins).
That mapping is also why `reduced_hessian` /
`compute_reduced_hessian` report **`−H_R`** rather than `H_R`: over
`y_c` rows `B K⁻¹ Bᵀ` is the multiplier sensitivity `∂λ/∂p`, which is
the same minus that makes `−inv(hr)` the parameter covariance. An
all-negative spectrum here is the convention, not indefiniteness, and
the ascending eigenvalues run stiffest-first — see [The reduced
Hessian comes back
negated](sensitivity.md#the-reduced-hessian-comes-back-negated).

All back-solves are in **natural (unscaled) units** — any NLP scaling
the IPM applied internally is undone, so results are independent of
`nlp_scaling_method`
([#128](https://github.com/jkitchin/pounce/issues/128)). The
solver-space values remain available via
`reduced_hessian(pins, scaled=True)` / `kkt_solve(rhs, scaled=True)`,
and the factors via the `Solver.nlp_scaling` dict — see
[Sensitivity Analysis](sensitivity.md#units-and-nlp-scaling).

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
IpoptSolverReducedHessian(sol, 2, pins, 1.0, hr);   /* writes -H_R */

IpoptFreeSolver(sol);
```

The classic `IpoptSolve` API is unchanged and unaffected; the session
handle lives alongside it.

## Rust

Both session APIs come through the `pounce-rs` facade. `Solver` needs the
`sensitivity` feature; the bare `Factorization` below needs `convex` or `qp`,
whichever you are already using — either one enables `pounce_rs::linsol`.

```rust
use pounce_rs::sensitivity::Solver;

let mut solver = Solver::new(app, tnlp);
solver.solve();
assert!(solver.converged().is_some());

let dx = solver.parametric_step(&[2, 3], &[-0.5, 0.0])?;
let hr = solver.compute_reduced_hessian(&[2, 3], 1.0)?;  // −H_R, see above

let mut lhs = vec![0.0; solver.kkt_dim().unwrap()];
solver.kkt_solve(&rhs, &mut lhs)?;
```

For purely linear-algebra uses with no IPM in the loop:

```rust
use pounce_rs::linsol::{Factorization, backend};

let mut fact = Factorization::new(dim, ia, ja, values, backend())?;
fact.solve(&mut rhs, 1)?;          // back-substitute in place
fact.refactor(&new_values)?;       // pattern preserved; numeric reuse
fact.solve_one(&mut another_rhs)?;
```

## The analysis layer above these calls

`Solver` is the primitive; `pounce.sensitivity` is the analysis built on
it — the parametric step in every mode, what the step did about the
bounds, the active-set events along a path, parameter covariance and the
information matrix — over a session that bundles a solved NL with its
held factor. It has no modelling-layer dependency; `pyomo_pounce` is one
of its callers. See
[Sensitivity Analysis](sensitivity.md#the-analysis-layer-pouncesensitivity).

## What's preserved across operations

* **Symbolic factor / AMD ordering.** Owned by the linear-solver
  backend; reused on every back-solve and on `refactor()`.
* **Numeric factor.** Reused on every back-solve until you refactor.
* **The converged primal-dual state** (`x*`, multipliers, `g(x*)`,
  iteration stats).

## Re-solving families: warm starts through presolve

The sessions above hold a *factor* across queries. A different persistence
need is re-solving a *family* of nearby problems — MPC steps, parametric
sweeps, oximo's persistent IPM — seeding each solve from the last solution.
That seed lives in original space, but `presolve=yes` solves a reduced
problem (tightened bounds, dropped rows, eliminated variables), so embedders
previously had to pick presolve **or** warm starts. The presolve sessions map
every original-space warm point into the reduced space the solver sees. They
reuse a retained transformation only when its fingerprint matches; otherwise
they run presolve again and map the same warm point through the fresh
transformation:

```rust
use std::cell::RefCell;
use std::rc::Rc;
use pounce_rs::prelude::*;
use pounce_rs::session::TnlpPresolveSession;

// `inner` is your TNLP, mutated in place between solves.
let mut session = TnlpPresolveSession::new(inner)?;
session.set_option_str("presolve", "yes")?;

let first = session.solve_cold()?;
assert!(first.success);

// Re-solve the unchanged model warm through the retained transform.
let second = session.solve_warm_last()?;
assert!(second.success);
assert!(second.presolve_reused);
```

Cache + validate: before each solve the session fingerprints what the
transformation was computed from (dims, Jacobian structure + linear-row
values, constraint and variable linearity tags, bounds, presolve options).
A match reuses the wrapper; anything else rebuilds it and maps the warm point
through the fresh transform. Either way the solve is warm *and* presolved.
For an MPC loop that changes at least one hashed RHS or bound every step, this
means the warm-start projection engages but the transformation reuse rate is
0% after the initial build.

The two sessions deliberately have different objective policies:

| Change between solves | TNLP session | Convex-QP session |
|---|---|---|
| No fingerprinted data changes | Reuse | Reuse |
| Objective only | Reuse when auxiliary Phase 0 is off | Rebuild |
| RHS, bounds, or linear constraint coefficients | Rebuild | Rebuild |
| Any other numeric QP matrix value | Not applicable | Rebuild |

The TNLP wrappers keep calling the live problem for objective evaluations, so
outside auxiliary Phase 0 the transform is constraint-derived and a pure cost
change is safe to reuse. Convex `Presolve`, by contrast, owns a reduced numeric
`QpProblem` and retains the original numbers needed by postsolve, so its
fingerprint includes `c`, `P`, and every other numeric field. Reusing convex
presolve across changing numerical data would require a separate plan-refresh
API; the current session does not provide one. Dropped-row dual
mass is reported on `SessionSolution::warm_report`, folded through the
linear-eq elimination as well as presolve (redundant rows carry 0
at the optimum; aux-eliminated and elimination-consumed rows are re-derived
from KKT stationarity at postsolve), and `SessionSolution::warm_point()`
threads `final_mu` into the next `mu_init`.

Two classes of TNLP changes are invisible to the fingerprint and need
`invalidate()`: FBBT expression-tape swaps, and objective or nonlinear data
changes that can affect `presolve_auxiliary=yes` decisions.

The convex counterpart keeps a retained `Presolve` over the IPM instead of
an application: `ConvexPresolveSession::solve` fingerprints every numeric
field of the `QpProblem`, skips the recompute only on an exact match, projects
the `QpWarmStart` through
`Presolve::project_warm`, and postsolves back (threading `obj_offset()`
into `obj_constant` as the CLI does).

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
* `pounce_rs::sensitivity::Solver::parametric_step` ≡
  `SensSolve::with_deltas` (1e-10).

See `python/tests/test_solver_session.py` and
`crates/pounce-sensitivity/tests/solver_session.rs` for the full test
matrix.
