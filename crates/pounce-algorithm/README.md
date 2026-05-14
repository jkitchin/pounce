# pounce-algorithm

The interior-point core of POUNCE. Port of Ipopt's `src/Algorithm/`.
Wires together the iterate, lazy-cache, KKT, line-search, μ-update,
convergence-check, and Hessian-approximation strategies into the main
`IpoptAlgorithm::optimize()` loop.

Internal crate. The user-facing entry point is [`IpoptApplication`],
re-exported as `pounce_algorithm::IpoptApplication`. Callers should
implement [`TNLP`](../pounce-nlp) and call
`IpoptApplication::optimize_tnlp(...)`.

## Subsystems

| Module          | What it does                                               | Upstream                          |
|-----------------|------------------------------------------------------------|-----------------------------------|
| `iterates_vector` | Compound iterate (x, s, λ, ν, z_L, z_U)                  | `IpIteratesVector.{hpp,cpp}`      |
| `ipopt_data`    | Mutable state (current iterate, perturbations, μ, ...)     | `IpIpoptData.{hpp,cpp}`           |
| `ipopt_cq`      | Lazy-cached derived quantities (norms, residuals, ...)     | `IpIpoptCalculatedQuantities.*`   |
| `kkt`           | Augmented system, perturbation handler, PD full-space solver, search-direction calculator | `Ip{AugSystemSolver,PdFullSpaceSolver,SearchDirCalculator}.*` |
| `line_search`   | Filter + backtracking, second-order correction, watchdog   | `IpFilterLineSearch.{hpp,cpp}`    |
| `mu`            | Barrier-parameter update (monotone)                        | `IpMonotoneMuUpdate.{hpp,cpp}`    |
| `conv_check`    | Optimality / acceptable-level / iteration-cap termination  | `IpOptErrorConvCheck.{hpp,cpp}`   |
| `eq_mult`       | Equality-multiplier initial estimate                       | `IpEqMultCalculator.{hpp,cpp}`    |
| `init`          | Starting-point projection and bound-multiplier seeding     | `IpDefaultIterateInitializer.*`   |
| `hess`          | Exact / quasi-Newton Hessian dispatch                      | `IpHessianUpdater.{hpp,cpp}`      |
| `scaling`       | NLP-side rescaling chain                                   | `IpNLPScalingObject.{hpp,cpp}`    |
| `ipopt_alg`     | The main optimize() loop                                   | `IpIpoptAlg.{hpp,cpp}`            |
| `alg_builder`   | Strategy wire-up (`BuildBasicAlgorithm`)                   | `IpAlgBuilder.{hpp,cpp}`          |
| `application`   | `IpoptApplication` entry point                             | `IpoptApplication.{hpp,cpp}`      |
| `timing_stats`  | Wall-clock accumulators                                    | `IpTimingStatistics.{hpp,cpp}`    |
| `output`        | Per-iteration table                                        | `IpOrigIterationOutput.{hpp,cpp}` |

## Choosing strategies

`AlgorithmBuilder` carries enum knobs:

- `LinearSolverChoice` — `Feral` (default) or `Ma57` (requires `ma57`).
- `MuStrategyChoice` — `Monotone`. (Adaptive lands in Phase 10.)
- `HessianApproxChoice` — `Exact` or `LBfgs` (exact landed first; L-BFGS
  is wired up but not yet validated for all problem classes).
- `LineSearchChoice` — `Filter`.
- `NlpScalingChoice` — `None`, `Gradient`, `User`.

Most users won't touch the builder directly — `IpoptApplication::new()`
plus the option machinery covers the standard knobs (`linear_solver`,
`hessian_approximation`, `mu_strategy`, etc.) and matches upstream
option names verbatim.

## Restoration

When the line search cannot accept any step, control switches into the
[restoration phase](../pounce-restoration) (port of Ipopt's
`Algorithm/Resto*`).

## License

EPL-2.0.

[`IpoptApplication`]: src/application.rs
