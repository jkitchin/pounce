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

## Wrapper design study and model updates

The [Ipopt.jl and MadNLP study](julia-wrapper-design-study.md) identified stale
nonlinear bounds after a solve–modify–solve sequence. The fix, regression tests,
and comparative reproducer are committed at
`8b142ab0adb8752d9925bd35e781a6484f3e792d`.

On that commit, using the unchanged native library from
`b4e5ffb641990738fd69cb53c02ae5966fd68edc`, the full package suite passed
3,453 assertions on each of Julia 1.10.12 and 1.12.6 (macOS arm64). This includes
82 new assertions for nonlinear bound updates, fresh-model comparisons,
cached/direct JuMP models, parameter updates, and consistent result invalidation.
The runs used MathOptInterface 1.54.0, JuMP 1.32.1, and Ipopt.jl 1.16.0.

The committed reproducer also ran against registered MadNLP 0.11.0 and
Ipopt.jl 1.16.0. Updated Pounce and its fresh-model control both returned
`1.3862943686119529` for the new bound. Both reference packages retained their
old result after the bound update; their fresh-model controls returned the
expected new optimum. Package identities and the pre-fix values are recorded
in the study. No upstream issue or PR was published as part of this work.

## Statistics and callback refactor

Implementation commit: `cc42dec83da1f1a1e95cdf29b50f3b905b4bc2c4`.
Native library: unchanged POUNCE 0.15.0 built from
`b4e5ffb641990738fd69cb53c02ae5966fd68edc`.

On macOS arm64, the full package suite passed 3,586 assertions on each of
Julia 1.10.12 and 1.12.6, with MathOptInterface 1.54.0, JuMP 1.32.1, and
Ipopt.jl 1.16.0. The 133 additional assertions cover native statistics,
callback installation/replacement/removal, snapshot freshness and task scope,
restoration restrictions, exceptions/recovery, recursive solve rejection, and
direct-model parameter introspection. The callback tests exposed missing
parameter constraint getters; these are now implemented too.

The documentation consistency check passed (54 reachable pages, no dead links),
as did whitespace checks. No Rust source changed, so the native test suite was
not rerun for this Julia-only refactor. Hosted CI remains unrun/unpublished.

The allocation probe used Julia 1.12.6 and the same native library/MOI version
before and after the change:

| Variables queried | Baseline `85cb9d43` | Refactor `cc42dec8` |
|---|---:|---:|
| 128 | 155,648 bytes | 0 bytes |
| 512 | 2,179,072 bytes | 0 bytes |

These are Julia allocations for a warmed scalar-query sweep inside one regular
callback. Warmup populates the new cache, so the first snapshot copy is excluded.
They are not total callback/solver allocations or a wall-time speedup measurement.
Both probes record wrapper and native source revisions, Julia/MOI/native versions,
library path, and working-tree state. The baseline wrapper code was unchanged;
only the new roadmap document was untracked.

Reproduce from the repository root in an environment containing Pounce and
MathOptInterface, setting the native revision to the commit actually used to
build the selected library:

```sh
POUNCE_NATIVE_REVISION=b4e5ffb641990738fd69cb53c02ae5966fd68edc \
  julia --project=/path/to/environment --startup-file=no dev-notes/julia-callback-allocation-probe.jl
```

The same script can be copied to a baseline checkout for comparison. The probe's
diagonal quadratic model is self-contained and uses no external benchmark data.

## Explicit model/cache/result lifecycle

At wrapper commit `b8a9f4e3`, the full suite passed 3,642 assertions on each of
Julia 1.10.12 and 1.12.6 on macOS arm64, using the unchanged native library from
`b4e5ffb641990738fd69cb53c02ae5966fd68edc`. The 56 new lifecycle assertions check
derivative-cache reuse across bound/parameter updates, rebuilds after structural
edits, result retention for settings/starts, option resets, failed-attempt
invalidation/recovery, and rejection of callback mutations before data changes.
Existing tests continue to compare updated nonlinear models with fresh models.

Public MOI/JuMP entry points and low-level C bindings are retained. Tests that
inspect implementation details now use the explicit native/data containers.
These fields are not a supported user API; model/result queries use MOI.
