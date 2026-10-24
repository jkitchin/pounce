# pounce-sens-core

The engine-agnostic core of POUNCE's sensitivity layer: the `SensBacksolver`
contract and everything that can be built on it without knowing which solver
produced the KKT system.

## What is here

Everything in the table below is written against one trait, `SensBacksolver`,
whose entire required surface is `dim()` and `solve(rhs, lhs)`. Any engine
that can back-solve against its converged factor gets these modules:

| module | what it provides |
|---|---|
| `backsolver` | the `SensBacksolver` contract, `BoundRow`, and the synthetic `DenseLuBacksolver` that validates the math without an IPM |
| `boundcheck` | fix-relax refinement, path following through active-set breakpoints, the directional derivative at a kink |
| `sens_app` | the sIPOPT `SensApplication` driver, reduced-Hessian entry point, option registration |
| `p_calculator`, `schur_data`, `schur_driver`, `step_calc`, `reduced_hessian` | the `P = K⁻¹A` / Schur-complement stack |

Two consumers exist in tree. [`pounce-sensitivity`](../pounce-sensitivity)
implements the trait over the NLP filter-IPM's KKT factor, and
[`pounce-convex`](../pounce-convex) implements it over the convex active-set
KKT. Neither depends on the other, and this crate depends on neither. It
depends on `pounce-common` and `pounce-linalg`, on `pounce-qp` and
`pounce-feral` for the reduced QP the directional derivative solves, and on
`faer` for the dense algebra of the statistics.

## Barrier iterates

The corrector and the activity classification need more than a back-solve.
They read the interior-point iterate itself, so they are written against a
second trait, `BarrierIterate`, which an engine implements beside
`SensBacksolver`:

| module | what it provides |
|---|---|
| `barrier` | the `BarrierIterate` contract: the block layout of the compound iterate `(x, s, y_c, y_d, z_l, z_u, v_l, v_u)`, the converged point and its barrier level, the barrier KKT residual at a point, the Newton operator at a predicted point, and the converged quantities the classification reads (`ActivityData`) |
| `corrector` | the corrector, which re-solves the barrier KKT conditions from a parametric step's predicted point |
| `activity` | the activity classification of every bound and inequality row, and the reduced reports over a chosen set of variables or rows |
| `converged` | `ConvergedSolve`, every calculation over a converged factor: the parametric steps in each mode and degeneracy treatment, the corrector, the classification, the reduced Hessian, the KKT solves and the maps between the model and the engine's blocks |
| `stats` | the covariance and the information matrix of a fit, the correlation, and the bounds a block is conditioned on |
| `step` | the solution estimate at perturbed parameters, the report of what its step does at the bounds, the active-set changes along a path step, the multiplier row of a model row, and the total derivative of the objective |
| `model` | `ModelFunctions`, the model's bounds, constraints, Jacobian and objective gradient, which a step report evaluates at the moved point |
| `dense` | the dense algebra of `stats`: the inverse, the solve, the rank and null space at NumPy's tolerance, and the symmetric eigendecomposition with each eigenvector's largest component positive |
| `index` | index newtypes for the model's variables (full-x) and the factor's (var-x), used where a conversion between the two feeds a direct array index |

`pounce-sensitivity` implements `BarrierIterate` over the filter-IPM's
`IpoptData`, and its `Solver` is a wrapper over `ConvergedSolve`. An engine
outside pounce implements the trait from its own final iterate, its own model
callbacks and its own factorization of the KKT matrix, and gets the same
calculations through `ConvergedSolve::new`. The classification assumes the
solve ran with `bound_relax_factor = 0`, which pounce's `Solver` checks and an
outside engine must ensure itself.

The statistics and the steps are the numerics behind `pounce.sensitivity`'s
`covariance`, `information`, `solution`, `solution_report` and
`active_set_changes`. They return each warning condition as a finding and
each failure as a typed error, and the Python front ends write the messages.

## Algorithmic reference

> Pirnay, H., López-Negrete, R., and Biegler, L.T. (2012).
> *Optimal sensitivity based on IPOPT.*
> Mathematical Programming Computation, **4**(4), 307–331.
> DOI: [10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2).

## Upstream source

The port follows the upstream sIPOPT contrib at
[`ref/Ipopt/contrib/sIPOPT/src/`][upstream-src] in this repo (EPL-2.0,
© Hans Pirnay 2009–2011 per the headers). The files mapped here moved out of
`pounce-sensitivity` when this crate was split off; the mapping is unchanged:

| pounce-sens-core                             | upstream                                                                                    |
|----------------------------------------------|---------------------------------------------------------------------------------------------|
| `src/schur_data.rs` — `SchurData` trait      | [`SensSchurData.hpp`](../../ref/Ipopt/contrib/sIPOPT/src/SensSchurData.hpp) (lines 17–177)   |
| `src/schur_data.rs` — `IndexSchurData`       | [`SensIndexSchurData.{hpp,cpp}`](../../ref/Ipopt/contrib/sIPOPT/src/SensIndexSchurData.hpp)  |
| `src/p_calculator.rs` — `PCalculator` trait  | [`SensPCalculator.hpp`](../../ref/Ipopt/contrib/sIPOPT/src/SensPCalculator.hpp) (lines 17–133) |
| `src/backsolver.rs` — `SensBacksolver` trait | [`SensBacksolver.hpp`](../../ref/Ipopt/contrib/sIPOPT/src/SensBacksolver.hpp)                |

Every public item ported from sIPOPT documents the upstream symbol it
mirrors, with line numbers when they're stable. `barrier`, `converged`,
`stats`, `step`, `model` and `dense` are not ports: they hold pounce's own
calculations, moved here from `pounce-sensitivity` and the Python front end,
and say so inline where one follows sIPOPT.

## License

EPL-2.0, matching upstream Ipopt and the sIPOPT contrib.

[upstream-src]: ../../ref/Ipopt/contrib/sIPOPT/src/
