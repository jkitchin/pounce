# Julia wrapper validation notes

The first wrapper was developed against native POUNCE commit
`b4e5ffb641990738fd69cb53c02ae5966fd68edc` (0.15.0).
The Rust solver is unchanged by the wrapper contribution.
Ipopt.jl sources and adapted tests are pinned in `Pounce.jl/NOTICE.md`.

## An inherited test exposed a native limited-memory limitation

The Ipopt.jl test `test_manually_evaluated_primal_status` constructs:

- feasibility objective (zero);
- one variable, `0 <= x <= 1`;
- an affine interval `0.1 <= x <= 1.1`;
- a vector nonlinear oracle imposing `0.25 <= x <= 0.75`;
- a scalar nonlinear constraint `-1 <= log(x) <= 1`.

The oracle supplies no Hessian, so the adapter selects limited-memory.
The original test supplies no start and the wrapper starts at zero, projected
onto the variable bounds. On the native commit above, this returns
`Restoration_Failed`. Its result-classification assertions assume a feasible
solve without checking the status, and therefore fail.

The kill-switch control on that exact model disabled
`infeas_stationarity_tol`, `presolve`, `acceptable_iter`,
`acceptable_progress_kappa`, `obj_scale_certificate_threshold`,
`primal_noise_floor_kappa`, `dual_divergence_retry`, and NLP scaling.
It returned the same failure and point.

The same functions passed directly to the C API, bypassing MOI, reproduce:

| Hessian | Starting x | Native status |
|---|---:|---|
| limited-memory | 0 | Restoration_Failed |
| limited-memory | 0.5 | Solve_Succeeded |
| exact | 0 | Solve_Succeeded |
| exact | 0.5 | Solve_Succeeded |

Reproduce with:

```sh
julia --project=Pounce.jl dev-notes/julia-feasibility-repro.jl
```

This establishes a native limited-memory limitation, not its internal cause.
No solver fix is included here. The result-classification test now starts
at the known strictly feasible point `x=0.5` and asserts successful termination
before manipulating the stored results. Other tests exercise limited-memory
solves. This should not be read as a general robustness or OPF benchmark.

## Test adaptations

Ipopt-specific SPRAL/HSL library tests and its extension alias test are omitted.
The vector-oracle timing test uses an infinity sentinel to verify timers reset
between solves, instead of sleeping and assuming Ipopt's evaluation counts.
Callback exceptions are rethrown only after the C API returns, including
interrupts, so the interrupt test expects an exception and no usable result.

Native `check_derivatives_for_naninf=yes` is refused by POUNCE (gh #483);
the wrapper instead checks callback outputs for finiteness.

Binary distribution, Windows CI, and BMOPFTools integration benchmarks remain
follow-up work. The first contribution does not claim those are validated.
