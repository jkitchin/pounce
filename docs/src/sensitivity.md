# Sensitivity Analysis

POUNCE includes a parametric sensitivity capability compatible with
upstream Ipopt's `contrib/sIPOPT/` (Pirnay, López-Negrete & Biegler
2012, DOI
[10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2)).
It computes the first-order change in the optimal primal solution with
respect to a problem parameter, reusing the KKT factorization from the
converged solve. Four entry points cover the common workflows.

## AMPL CLI

The main `pounce` driver auto-detects the sIPOPT suffixes
(`sens_state_1`, `sens_state_value_1`, `sens_init_constr`) in an input
`.nl`, runs a post-optimal sensitivity step after the solve, and
writes the perturbed primal back as a `sens_sol_state_1` suffix — no
separate binary or flag needed:

```sh
pounce problem.nl                   # writes problem.sol
pounce problem.nl out.sol --json-output result.json --json-detail full
```

`pounce_sens` is retained as a thin backward-compatibility alias:
`pounce_sens in.nl out.sol` is identical to `pounce in.nl out.sol`, so
existing AMPL / solver scripts keep working unchanged.

Related flags:

- `--sens-boundcheck` / `--sens-bound-eps EPS` — clamp the perturbed
  primal `x* + Δx` onto the declared `[x_l, x_u]` box.
- `--compute-red-hessian` / `--rh-eigendecomp` — compute the reduced
  Hessian (and its eigendecomposition) over the variables tagged by
  the `red_hessian` integer var-suffix.

## Rust library

`SensSolve` is a builder that wraps the `on_converged` callback
plumbing into a single call:

```rust
use pounce_sensitivity::SensSolve;

let result = SensSolve::new(vec![2, 3])
    .with_deltas(vec![0.05, 0.0])
    .with_reduced_hessian()
    .run(&mut app, tnlp);
// result.dx, result.reduced_hessian, result.status
```

`with_reduced_hessian_eigen()` adds the eigendecomposition;
`with_boundcheck(eps)` enables the bound projection.

## Python

`solve_with_sens` exposes the same capability from the
cyipopt-compatible Python wrapper:

```python
# pin_constraint_indices is required; pass deltas=..., compute_reduced_hessian=True,
# or both. Returns (x, info) — sensitivity outputs live in the info dict.
x, info = prob.solve_with_sens(x0, pin_constraint_indices=[2, 3],
                               deltas=[0.05, 0.0], sens_boundcheck=True)
# info["dx"], info["reduced_hessian"], info["reduced_hessian_eigenvalues"], ...
```

`compute_reduced_hessian=True` returns the reduced Hessian in
`info["reduced_hessian"]`; `rh_eigendecomp=True` adds its
eigendecomposition; `sens_bound_eps=…` tunes the bound projection. See
[`python/notebooks/04_sensitivity.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/04_sensitivity.ipynb)
for a walkthrough.

## Pyomo

`pyomo_pounce` wraps the same machinery in a declare-then-query
interface: flag the parameters that matter while building the model
(no perturbed values required), solve normally, then ask for
derivatives. Parameters are declared with `declare_sens_param`
(mutable `Param` or fixed `Var`, scalar or indexed); when declarations
are present, `SolverFactory("pounce").solve(m)` runs in-process and
keeps the converged KKT factorization, so every query afterwards is a
single backsolve.

```python
import pyomo.environ as pyo
import pyomo_pounce
from pyomo_pounce import declare_sens_param, gradient, estimate

m.p = pyo.Param(initialize=2.0, mutable=True)
declare_sens_param(m.p)                 # a flag, not a perturbation

pyo.SolverFactory("pounce").solve(m)    # ordinary solve

gradient(m.x, wrt=m.p)                  # dx*/dp (float)
gradient(m.con, wrt=m.p)                # d(multiplier of con)/dp
G = gradient(m.z, wrt=m.r)              # containers -> Gradient object
G[m.z[1], m.r[2]]; G.to_dataframe()     # element access / full Jacobian
estimate(m, [(m.p, 2.5)])               # first-order solution estimate at
                                        # new values, clamped to bounds
```

`gradient` returns exact first-order derivatives (unit-perturbation
backsolves, no finite differencing); `estimate` combines the stored
derivative columns for arbitrary perturbed values after the fact, and
warns when the linear step leaves the variable bounds (a single-pass
projection analogous to the CLI's `--sens-boundcheck`). Multiplier
sensitivities are available for equality constraints. Models without
declarations solve through the ordinary AMPL/CLI path, unchanged. See
[`python/notebooks/25_pyomo_sensitivity.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/25_pyomo_sensitivity.ipynb)
for a worked optimal-control example (initial conditions as
parameters; the first-move gradient IS the NMPC feedback gain).

## Parameter covariance and identifiability

For a parameter-estimation model whose objective is a **plain sum of
squared residuals**, the factorization from ONE ordinary solve yields
the asymptotic covariance of the estimated parameters. Declare the
fitted variables (they stay free) and the residual container while
building the model, solve, and ask:

```python
from pyomo_pounce import covariance, declare_estimated, declare_residual

m.A = pyo.Var(); m.k = pyo.Var()        # the fitted parameters, free
declare_estimated(m.A)
declare_estimated(m.k)

m.r = pyo.Var(m.I)                      # residuals, one per data point
m.res = pyo.Constraint(m.I, rule=...)   # r[i] == y[i] - model(A, k, t[i])
declare_residual(m.r)

m.obj = pyo.Objective(expr=sum(m.r[i]**2 for i in m.I))
pyo.SolverFactory("pounce").solve(m)    # one solve

cov = covariance(m)                     # no further information needed

cov[m.A, m.k]               # covariance entry (either order)
cov.std_err[m.k]            # standard error of one parameter
cov.correlation[m.A, m.k]   # correlation matrix entry
cov.matrix                  # dense numpy array, ordered like cov.params
w, V = cov.eigen()          # eigendecomposition, for identifiability
```

The recipe: the parameter block of the inverse KKT matrix, one
backsolve per parameter against the held factor, equals the inverse
reduced Hessian of the eliminated problem, and for a sum-of-squares
objective `cov = 2 sigma^2 (K^-1)_pp`. The factor 2 belongs to the
unscaled sum of squares (a Gaussian negative log-likelihood objective,
`SSR / (2 sigma^2)`, would drop it). The scaling is pinned by test
against the analytical linear-regression covariance
`sigma^2 inv(X^T X)` (`pyomo-pounce/tests/test_covariance.py`).

The noise variance comes from, in order of precedence: `sigma_sq=`
(known measurement variance); the declared residuals (estimated as
`SSR / (n - n_params)`, with both numbers derived from the container);
or the `n_data=` fallback for models without explicit residuals. The
solve warns if the declared residuals do not reproduce the objective
value (weights or regularization terms would silently corrupt the
estimate).

**Groups.** `declare_residual(m.r_conc, group="conc")` partitions
residuals into noise groups by arbitrary user strings: containers
sharing a group (or all ungrouped containers) pool into one estimated
variance; distinct groups get their own (`cov.sigma_sq` becomes a
dict), and the covariance switches to the heteroscedastic sandwich
form, whose per-group pieces come from the same backsolves. When
groups genuinely differ, weighting the objective itself (dividing each
group's residuals by its sigma) is the statistically efficient fix;
the sandwich is the truthful report on the unweighted fit.

`cov.eigen()` returns ascending eigenvalues and matching eigenvectors.
An eigenvalue much larger than the rest flags a poorly identified
problem: its eigenvector is the parameter combination the data cannot
pin down, and the corresponding `cov.correlation` entries approach
+/-1. `covariance` warns when the held factor carries
inertia-correction perturbations (typically an exactly unidentifiable
parameterization), when an estimated parameter sits on a bound at the
optimum (the asymptotics are invalid there), and when the covariance
diagonal comes out negative (not a least-squares minimum).

**Relation to `pyomo.contrib.parmest`.** parmest is an estimation
workflow: multi-experiment data management, bootstrap resampling, and
likelihood-ratio confidence regions, at the price of restructuring the
problem into its experiment framework, with covariance computed by
finite differences or an ipopt re-solve. `covariance()` is a
post-solve primitive: the model as written, one declaration per
component, the asymptotic covariance and identifiability diagnostics
from the factorization the solve already produced. Use parmest for
multi-experiment campaigns and non-asymptotic intervals; use this to
interrogate the fit you already have.

See
[`python/notebooks/26_parameter_covariance.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/26_parameter_covariance.ipynb)
for a worked example with a Monte Carlo validated confidence ellipse
and an identifiability diagnosis.

## Units and NLP scaling

All sensitivity outputs are in **natural (unscaled) units**. The IPM
holds its converged KKT factor in an internally scaled space whenever
NLP scaling is active (the default `nlp_scaling_method =
"gradient-based"` fires when an objective gradient or constraint row
exceeds `nlp_scaling_max_gradient = 100` at the starting point);
pounce undoes that scaling in every held-factor back-solve, so `dx`,
`kkt_solve`, and the reduced Hessian are independent of how the
problem was scaled internally
([#128](https://github.com/jkitchin/pounce/issues/128)).

In particular, for a parameter-estimation NLP with the parameters
pinned by equality constraints, `-inv(info["reduced_hessian"])` is
directly the parameter covariance — no per-problem scale factor, no
need to set `nlp_scaling_method = "none"`. (Sign convention: over pin
*constraint* rows, `B K⁻¹ Bᵀ` equals the multiplier sensitivity
`∂λ/∂p = −∂²f*/∂p²`, hence the minus in the covariance recipe.)

For callers that calibrated against the pre-#128 behavior, the
solver-space value and the factors that relate the two are exposed:

- Python: `info["reduced_hessian_scaled"]`,
  `info["obj_scaling_factor"]`, `info["pin_g_scaling"]`;
  `Solver.reduced_hessian(pins, scaled=True)`,
  `Solver.kkt_solve(rhs, scaled=True)`, and the `Solver.nlp_scaling`
  dict (`{"obj": df, "c_scale": …, "d_scale": …}`).
- Rust: `SensResult::{reduced_hessian_scaled, obj_scaling_factor,
  pin_g_scaling}`, `Solver::{compute_reduced_hessian_scaled,
  kkt_solve_scaled, nlp_scaling, pin_g_scaling}`, and
  `PdSensBacksolver::solve_scaled_space`.

The relation is `H_scaled[i,j] = df / (dc_i·dc_j) · H[i,j]`, where
`df` is the objective scaling factor and `dc_i` the pin rows'
constraint scaling factors.

One caveat: the IPM's inertia-correction perturbations (`δ_x`, `δ_s`,
`δ_c`, `δ_d`) are added to the factor in *scaled* space, so on a
problem whose final factorization needed regularization (e.g.
linearly dependent pin rows) the unscaling maps a slightly different
perturbed system per scaling method. The perturbations are reported —
`info["kkt_perturbations"]` / `Solver.kkt_perturbations` (Python),
`SensResult::kkt_perturbations` / `Solver::kkt_perturbations` (Rust)
— so a covariance workflow can assert they are all zero before
trusting `-inv(reduced_hessian)`; on well-posed estimation problems
the final factor is unregularized and the invariance is exact.

## Verification

All three entry points are verified against upstream sIPOPT 3.14.19's
`parametric_cpp` golden output to within roughly 6e-9 per component.
The bound projection is a single-pass clamp; upstream's iterative
Schur refinement (re-factorize on each violation) is intentionally not
ported.
