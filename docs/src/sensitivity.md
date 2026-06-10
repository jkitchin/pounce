# Sensitivity Analysis

POUNCE includes a parametric sensitivity capability compatible with
upstream Ipopt's `contrib/sIPOPT/` (Pirnay, López-Negrete & Biegler
2012, DOI
[10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2)).
It computes the first-order change in the optimal primal solution with
respect to a problem parameter, reusing the KKT factorization from the
converged solve. Three entry points cover the common workflows.

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

One caveat: the IPM's inertia-correction perturbations (`delta_w`,
`delta_c`) are added to the factor in *scaled* space, so on a problem
whose final factorization needed regularization (e.g. linearly
dependent pin rows) the unscaling maps a slightly different perturbed
system per scaling method. On well-posed estimation problems the
final factor is unregularized and the invariance is exact.

## Verification

All three entry points are verified against upstream sIPOPT 3.14.19's
`parametric_cpp` golden output to within roughly 6e-9 per component.
The bound projection is a single-pass clamp; upstream's iterative
Schur refinement (re-factorize on each violation) is intentionally not
ported.
