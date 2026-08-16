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
| 07 | [`07_scaling.ipynb`](07_scaling.ipynb) | NLP scaling and why it matters for convergence. |
| 08 | [`08_fbbt.ipynb`](08_fbbt.ipynb) | Feasibility-based bound tightening. |
| 28 | [`28_pyomo_initialization_repair.ipynb`](28_pyomo_initialization_repair.ipynb) | The two ways an initialization goes bad, on a 569-equation distillation train: a structurally singular specification (diagnosed by `block_analyze`, repaired by `block_repair_plan`, applied automatically by `initialize`) and a garbage starting point (rebuilt by the block-triangular solve, even from all zeros). |

## Modeling-language front ends

| # | Notebook | What it shows |
|---|---|---|
| 05 | [`05_pyomo.ipynb`](05_pyomo.ipynb) | Drive POUNCE from a Pyomo model. |
| 35 | [`35_casadi.ipynb`](35_casadi.ipynb) | POUNCE as a CasADi `nlpsol` plugin: MX models with parameters, `Opti`, warm-started MPC, differentiating through the solve (and a bilevel problem built on it), L-BFGS restricted to the nonlinear variables, and a side-by-side check against CasADi's bundled Ipopt. |

## Differentiating through the solver

| # | Notebook | What it shows |
|---|---|---|
| 03 | [`03_implicit_differentiation.ipynb`](03_implicit_differentiation.ipynb) | Implicit differentiation of the optimum w.r.t. parameters. |
| 04 | [`04_sensitivity.ipynb`](04_sensitivity.ipynb) | Parametric sensitivity (sIPOPT-style). |
| 09 | [`09_differentiable_layer.ipynb`](09_differentiable_layer.ipynb) | A differentiable constrained-projection layer. |
| 13 | [`13_post_solve_jacobian.ipynb`](13_post_solve_jacobian.ipynb) | Post-solve Jacobian/sensitivities from the held KKT factor. |
| 17 | [`17_differentiable_convex.ipynb`](17_differentiable_convex.ipynb) | Differentiable convex optimization with `pounce.jax`. |
| 25 | [`25_pyomo_sensitivity.ipynb`](25_pyomo_sensitivity.ipynb) | Declared-parameter sensitivity from Pyomo: an optimal-control example whose first-move gradients are the NMPC feedback gains (needs `pyomo-cvp`). |
| 26 | [`26_parameter_covariance.ipynb`](26_parameter_covariance.ipynb) | Parameter covariance and identifiability from the reduced Hessian: standard errors, a Monte Carlo validated confidence ellipse, and a sloppy-direction diagnosis. |
| 31 | [`31_information_identifiability.ipynb`](31_information_identifiability.ipynb) | The information matrix as the un-inverted view of the covariance: eigen() naming a poorly identified combination, the duality check, and zero variance versus finite information at a bound. |
| 32 | [`32_wrt_blocks_and_retain_kkt.ipynb`](32_wrt_blocks_and_retain_kkt.ipynb) | One solve, many questions: wrt= sub-block marginals, confidence and prediction bands on undeclared prediction variables, conditioned_on, retain_kkt() with nothing declared, and release_kkt() to give the factor's memory back. |

## Performance & sparsity

| # | Notebook | What it shows |
|---|---|---|
| 06 | [`06_sqp_parametric_continuation.ipynb`](06_sqp_parametric_continuation.ipynb) | Active-set SQP and parametric continuation. |
| 10 | [`10_dense_to_sparse.ipynb`](10_dense_to_sparse.ipynb) | Dense → sparse: choosing `factor_reuse` as the problem grows. |
| 11 | [`11_batched_warm_start.ipynb`](11_batched_warm_start.ipynb) | Warm-starting a batched differentiable solve. |
| 12 | [`12_kkt_solve_many_perf.ipynb`](12_kkt_solve_many_perf.ipynb) | Batched `kkt_solve_many` performance. |
| 14 | [`14_path_following.ipynb`](14_path_following.ipynb) | Predictor–corrector path following & inverse mapping. |
| 33 | [`33_asdex_sparsity.ipynb`](33_asdex_sparsity.ipynb) | Automatic sparsity detection and coloring with [`asdex`](https://github.com/adrhill/asdex): derive `jac_pattern`/`hess_pattern` from the jaxpr instead of by hand, feed them to `from_jax(sparse=True)` or a raw `pounce.Problem`, and see why a graph-derived pattern beats a probed one on branchy (`where`/`clip`) models. |

## Convex & conic (`pounce.qp`)

| # | Notebook | What it shows |
|---|---|---|
| 15 | [`15_convex_qp.ipynb`](15_convex_qp.ipynb) | Convex QP & LP with `pounce.qp`. |
| 30 | [`30_active_set_qp.ipynb`](30_active_set_qp.ipynb) | The opt-in active-set engine (`method="active-set"` / `solver_selection="qp-active-set"`): reaching it from either surface, where it is ~2x faster than the IPM, and — measured — where it is much worse (71/138 vs 137/138 on Maros-Mészáros) plus the limitations to know before opting in. |
| 16 | [`16_socp.ipynb`](16_socp.ipynb) | Second-order cone programs with `pounce.qp.solve_socp`. |

## Global optimization

| # | Notebook | What it shows |
|---|---|---|
| 18 | [`18_sos_global_optimization.ipynb`](18_sos_global_optimization.ipynb) | Certified polynomial global optimization with `pounce.sos_minimize` (sum-of-squares / moment relaxation). |

> SOS (18) covers the **polynomial** case, and it is the only certified-global
> path POUNCE has — there is no spatial branch-and-bound solver for general
> factorable `exp`/`log`/trig problems. For those, use the multistart notebooks
> (19–21) below and accept an uncertified answer. See
> [`docs/src/global-optimization.md`](../../docs/src/global-optimization.md).

## Finding many minima

| # | Notebook | What it shows |
|---|---|---|
| 19 | [`19_find_minima_repulsion.ipynb`](19_find_minima_repulsion.ipynb) | `find_minima` — the repulsion/deflation family. |
| 20 | [`20_find_minima_restart.ipynb`](20_find_minima_restart.ipynb) | `find_minima` — the multistart/restart family. |
| 21 | [`21_find_minima_hopping.ipynb`](21_find_minima_hopping.ipynb) | `find_minima` — the basin-hopping family. |

## Glass box / black box

| # | Notebook | What it shows |
|---|---|---|
| 29 | [`29_trust_region_filter.ipynb`](29_trust_region_filter.ipynb) | `pounce.trf_minimize` — optimize a model that is part algebra, part opaque simulation. Why fitting a surrogate and optimizing it converges to a local *maximum*, how the zero/first-order corrections fix it, why an affine basis is provably useless (only curvature survives), and how fitting a basis once and freezing it cuts truth-model calls from 10 to 4. |

## Curve fitting

| # | Notebook | What it shows |
|---|---|---|
| 22 | [`22_curve_fit.ipynb`](22_curve_fit.ipynb) | `pounce.curve_fit` — SciPy-style nonlinear least squares with exact Jacobians, covariance, and confidence intervals. |
| 23 | [`23_curve_fit_minima.ipynb`](23_curve_fit_minima.ipynb) | `pounce.curve_fit_minima` — find *every* parameter set that explains the data, each a full `CurveFitResult`. |

## Boundary value problems

| # | Notebook | What it shows |
|---|---|---|
| 24 | [`24_boundary_value_problems.ipynb`](24_boundary_value_problems.ipynb) | `pounce.solve_bvp` — SciPy-compatible BVPs (fast FERAL Newton + adaptive refinement), differentiable JAX/Torch solves, and constrained / optimal-control BVPs unique to pounce. |

## ODE / DAE initial value problems

| # | Notebook | What it shows |
|---|---|---|
| 27 | [`27_dae_manifold_projection.ipynb`](27_dae_manifold_projection.ipynb) | `pounce.ode.solve_ivp` for index-1 DAEs via a singular mass matrix — keeping the reported solution on the constraint manifold with `consistent=` (project an inconsistent initial condition) and `project_output=` (Newton-polish requested output points), and why linear conservation laws need neither. |
