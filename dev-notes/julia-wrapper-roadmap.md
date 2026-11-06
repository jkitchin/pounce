# Pounce.jl refactoring roadmap

This roadmap follows the [Ipopt.jl/MadNLP study](julia-wrapper-design-study.md).
The starting point is `85cb9d43`, with the nonlinear bound-update fix and
3,453 passing assertions on each of Julia 1.10.12 and 1.12.6.

Keep `Model(Pounce.Optimizer)`, the supported MOI model types, and the low-level
C interface usable throughout. Make each stage independently reviewable. The
native Rust solver and its ABI remain unchanged in the first stage. Publishing
the branch or opening a PR requires the user's permission.

## 1. Native statistics and callback lifetime — implementing

Expose `GetIpoptIterCount` and `GetIpoptSolveTime`. Obtain final MOI iteration
counts from native statistics so ordinary solves need no intermediate callback.
Retain `MOI.SolveTimeSec` as elapsed wrapper time; introduce
`Pounce.NativeSolveTimeSec` for the native solver's reported wall time. Neither
timer includes model construction before `optimize!`; compilation can affect
both, so benchmarks must distinguish first and subsequent calls.

Allow callback installation, replacement, and removal between solves. Cache a
single primal snapshot lazily during each regular callback invocation. Keep its
buffer reusable, invalidate the contents at entry and exit (including errors),
and reject queries outside that callback's task or during restoration. Move
callback methods into their own source file as a small first separation.

Acceptance: C binding lifetime/statistics tests; no registered callback on the
default solve path; callback transitions on an existing optimizer; fresh values
across iterations and solves; cleanup after stop/errors; native and wrapper
statistics after successful, interrupted, and failed attempts; full package
suite on Julia 1.10 and 1.12. A version-stamped allocation probe must show that
reading n scalar callback values no longer copies n entire primal vectors.

## 2. Explicit model, cache, and result state — next

Replace implicit coupling between `inner === nothing`, `needs_new_inner`, and
`inner.solved` with separate validity for the model/evaluator, native problem,
and results. Centralize invalidation in named helpers and document a mutation
table: structural changes, bound changes, parameter values, starts, options,
callback changes, and failed solves.

Separate MOI model storage/mutation, evaluator assembly, solve orchestration,
and results into focused files. Preserve current behavior with scenario tests
before moving code. Refresh nonlinear bounds without rebuilding AD only when
the state contract and a fresh-model comparison prove it safe. Preserve explicit
starts; do not silently warm-start from a previous solution.

Acceptance: all supported edits have consistent result availability; failed
setup/solves cannot expose stale results or statistics; nonstructural updates
preserve usable derivative caches; repeated solves match fresh-model controls.

## 3. Optional MOI and predictable startup — after state separation

Make the low-level package independent of MOI through a Julia package extension.
Keep `Pounce.Optimizer` discoverable after loading JuMP/MOI, and document the
extension's concrete optimizer type for code that requires type stability.
Retain runtime library selection and same-process coexistence with Ipopt.

Measure `using Pounce`, model construction, first solve, and subsequent solves
before choosing a small PrecompileTools workload. Keep native paths/handles out
of serialized caches and handle a missing native build clearly.

Acceptance: low-level-only installation/load; both Pounce/MOI load orders;
JuMP entry-point compatibility; subprocess library-override checks; measured
startup effect on a pinned environment. Extension conversion and precompile
workloads may be separate commits.

## 4. Application validation and benchmark protocol — before OPF claims

Add a small, reproducible BMOPFTools case pinned to its source version, with
application-level feasibility checks and documented solver/backend options.
Exercise parameter sweeps, explicit primal/dual starts, and failure recovery.
Compare Pounce with independently loaded Ipopt; add MadNLP only where the same
model and derivatives can be compared fairly.

Report model/AD construction, first-call compilation effects, total solve
elapsed time, native time, iterations, and residuals separately. Record wrapper,
native solver, model, dependency, and linear-backend versions. Follow the
repository's benchmark source/licence/citation rules when adding suite data.

Acceptance: reproducible scripts and residual checks, no performance claim
based only on solver status or a different timer boundary. This is not required
to describe the initial package as a source-checkout JuMP adapter.

## 5. Binary distribution and release — after the adapter is validated

Build a Pounce JLL/artifact distribution for supported platforms, retain an
explicit custom-library override, and test clean-machine installation. Add
Windows coverage once its binary path is supported. Establish native-version
compatibility checks and release/registration documentation.

Acceptance: installing the Julia package needs no Rust toolchain on supported
platforms; artifact and custom-library paths both pass the package suite;
hosted CI is green before release. Registry submission and publication remain
separately authorized actions.

## Separate future feature: persistent solver sessions

Keeping a C handle is not a promise of symbolic factorization reuse. If measured
parameter-sweep workloads justify it, expose POUNCE's native session interface
with explicit permitted updates, ownership, starts, invalidation, and statistics
semantics. An NLPModels interface can be considered independently for users of
that ecosystem. Neither belongs in the initial MOI refactor by default.
