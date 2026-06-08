# POUNCE notebooks

Runnable, progressive Jupyter notebooks for the Python API. They are numbered
to suggest a reading order, but each is self-contained. Every notebook here is
executed against the current release before it is committed, so the saved
outputs match what you will see.

Run one with:

```sh
pip install pounce-solver          # the solver + Python bindings
jupyter lab python/notebooks/01_getting_started.ipynb
```

## Foundations

| # | Notebook | What it shows |
|---|---|---|
| 01 | [`01_getting_started.ipynb`](01_getting_started.ipynb) | First solve — define a problem, call `minimize`, read the result. |
| 02 | [`02_jax_autodiff.ipynb`](02_jax_autodiff.ipynb) | Let JAX supply exact gradients/Hessians. |
| 05 | [`05_pyomo.ipynb`](05_pyomo.ipynb) | Drive POUNCE from a Pyomo model. |
| 07 | [`07_scaling.ipynb`](07_scaling.ipynb) | NLP scaling and why it matters for convergence. |
| 08 | [`08_fbbt.ipynb`](08_fbbt.ipynb) | Feasibility-based bound tightening. |

## Differentiating through the solver

| # | Notebook | What it shows |
|---|---|---|
| 03 | [`03_implicit_differentiation.ipynb`](03_implicit_differentiation.ipynb) | Implicit differentiation of the optimum w.r.t. parameters. |
| 04 | [`04_sensitivity.ipynb`](04_sensitivity.ipynb) | Parametric sensitivity (sIPOPT-style). |
| 09 | [`09_differentiable_layer.ipynb`](09_differentiable_layer.ipynb) | A differentiable constrained-projection layer. |
| 13 | [`13_post_solve_jacobian.ipynb`](13_post_solve_jacobian.ipynb) | Post-solve Jacobian/sensitivities from the held KKT factor. |
| 17 | [`17_differentiable_convex.ipynb`](17_differentiable_convex.ipynb) | Differentiable convex optimization with `pounce.jax`. |

## Performance & sparsity

| # | Notebook | What it shows |
|---|---|---|
| 06 | [`06_sqp_parametric_continuation.ipynb`](06_sqp_parametric_continuation.ipynb) | Active-set SQP and parametric continuation. |
| 10 | [`10_dense_to_sparse.ipynb`](10_dense_to_sparse.ipynb) | Dense → sparse: choosing `factor_reuse` as the problem grows. |
| 11 | [`11_batched_warm_start.ipynb`](11_batched_warm_start.ipynb) | Warm-starting a batched differentiable solve. |
| 12 | [`12_kkt_solve_many_perf.ipynb`](12_kkt_solve_many_perf.ipynb) | Batched `kkt_solve_many` performance. |
| 14 | [`14_path_following.ipynb`](14_path_following.ipynb) | Predictor–corrector path following & inverse mapping. |

## Convex & conic (`pounce.qp`)

| # | Notebook | What it shows |
|---|---|---|
| 15 | [`15_convex_qp.ipynb`](15_convex_qp.ipynb) | Convex QP & LP with `pounce.qp`. |
| 16 | [`16_socp.ipynb`](16_socp.ipynb) | Second-order cone programs with `pounce.qp.solve_socp`. |

## Global optimization

| # | Notebook | What it shows |
|---|---|---|
| 18 | [`18_sos_global_optimization.ipynb`](18_sos_global_optimization.ipynb) | Certified polynomial global optimization with `pounce.sos_minimize` (sum-of-squares / moment relaxation). |

> **Spatial branch-and-bound** (`pounce-global`, for general factorable
> `exp`/`log`/trig problems) ships in the engine but is **not yet wired to a
> Python entry point** in this release, so it has no notebook here. SOS (18)
> covers the polynomial case. See [`docs/src/global-optimization.md`](../../docs/src/global-optimization.md).

## Finding many minima

| # | Notebook | What it shows |
|---|---|---|
| 19 | [`19_find_minima_repulsion.ipynb`](19_find_minima_repulsion.ipynb) | `find_minima` — the repulsion/deflation family. |
| 20 | [`20_find_minima_restart.ipynb`](20_find_minima_restart.ipynb) | `find_minima` — the multistart/restart family. |
| 21 | [`21_find_minima_hopping.ipynb`](21_find_minima_hopping.ipynb) | `find_minima` — the basin-hopping family. |

## Curve fitting

| # | Notebook | What it shows |
|---|---|---|
| 22 | [`22_curve_fit.ipynb`](22_curve_fit.ipynb) | `pounce.curve_fit` — SciPy-style nonlinear least squares with exact Jacobians, covariance, and confidence intervals. |
| 23 | [`23_curve_fit_minima.ipynb`](23_curve_fit_minima.ipynb) | `pounce.curve_fit_minima` — find *every* parameter set that explains the data, each a full `CurveFitResult`. |
