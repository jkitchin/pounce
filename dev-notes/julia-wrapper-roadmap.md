# Pounce.jl refactoring roadmap

This roadmap follows the [Ipopt.jl/MadNLP study](julia-wrapper-design-study.md).
The starting point is `85cb9d43`, with the nonlinear bound-update fix and
3,453 passing assertions on each of Julia 1.10.12 and 1.12.6.

Keep `Model(Pounce.Optimizer)`, the supported MOI model types, and the low-level
C interface usable throughout. Make each stage independently reviewable. The
native Rust solver and its ABI remain unchanged in the first stage. Publishing
the branch or opening a PR requires the user's permission.

## 1. Native statistics and callback lifetime — completed

Implemented in `cc42dec83da1f1a1e95cdf29b50f3b905b4bc2c4`. Both supported local
Julia test runs pass; version-stamped results and the allocation comparison are
in the [validation notes](julia-wrapper-validation.md#statistics-and-callback-refactor).

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

## 2. Explicit model, cache, and result state — completed

Implemented in `b8a9f4e3` and verified by 3,642 assertions on each of Julia
1.10.12 and 1.12.6. Model data, evaluator cache, native cache, and completed
results now have separate containers/revisions. The MOI implementation is split
into focused state, model, evaluator, solve, results, and callback files.

Replace implicit coupling between `inner === nothing`, `needs_new_inner`, and
`inner.solved` with separate validity for the model/evaluator, native problem,
and results. Centralize invalidation in named helpers and document a mutation
table: structural changes, bound changes, parameter values, starts, options,
callback changes, and failed solves.

Group existing fields into model data, evaluator cache, native problem, and
solve result/statistics containers. Track the model revision each cache/result
belongs to. Keep native ownership in the low-level problem; do not duplicate
large result buffers unless their lifetime requires it. The intended rules are:

| Change | Evaluator/derivatives | Native problem | Results |
|---|---|---|---|
| Variables, constraints, expressions, AD backend | Rebuild | Rebuild | Invalidate |
| Bounds only | Refresh bounds; reuse derivative structure | Rebuild until a supported update path exists | Invalidate |
| Parameter values | Update values in place | Reuse compatible handle | Invalidate |
| Primal/dual starts | Reuse | Apply on next solve | Keep last completed result until next attempt |
| Solver options | Reuse unless derivative requirements change | Apply or rebuild when options are removed | Keep last result until next attempt |
| Callback registration | Reuse | Install/remove before next solve | Keep last result until next attempt |
| New solve attempt | Reuse valid caches | Reuse compatible handle | Clear before setup; publish only completed results |

Test these rules at the optimizer level and through JuMP's caching model;
JuMP may independently guard result access after model edits. Evaluate option
removal explicitly so omitted settings cannot survive on a reused handle.

Separate MOI model storage/mutation, evaluator assembly, solve orchestration,
and results into focused files. Preserve current behavior with scenario tests
before moving code. Refresh nonlinear bounds without rebuilding AD only when
the state contract and a fresh-model comparison prove it safe. Preserve explicit
starts; do not silently warm-start from a previous solution.

Acceptance: all supported edits have consistent result availability; failed
setup/solves cannot expose stale results or statistics; nonstructural updates
preserve usable derivative caches; repeated solves match fresh-model controls.

## 3. Optional MOI and predictable startup — completed

Implemented in `b1e02486`, with derivative correction `9247e5a3` and subprocess
test fix `7527a8fe`. Both Julia 1.10.12 and 1.12.6 pass 3,653 assertions.
The measured import/build/first-solve median fell from 6.34 s to 3.28 s on the
HS071 probe; subsequent solves stayed near 1.85 ms. See the versioned
[validation evidence](julia-wrapper-validation.md#optional-moi-and-startup).

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

## 4. Application validation and benchmark protocol — completed

The [pinned BMOPFTools harness](../benchmarks/julia_bmopf/README.md) at `06f3d312`
passes ten solve scenarios per solver, repeated in three measured fresh processes
for Pounce and Ipopt. Checks include independent SI branch-current reconstruction
and terminal KCL, parameter sweeps with fresh controls, explicit primal/dual
starts, and iteration-limit recovery. The full dependency graph and measurements
are retained. This small teaching feeder does not support utility-scale claims.

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

## 5. Binary distribution and release — in progress

The pinned native source passes all 95 C-interface tests in a local Linux arm64
container; wrapper `7527a8fe` also passes all 3,653 Julia assertions there.
At `1d985b52`, native-version/symbol checks pass the complete 3,662-assertion
suite on both macOS Julia versions. The local-only BinaryBuilder recipe and
binary-backed release staging tool are under validation. JLL builds, artifact
installation and remaining platform gates are not yet complete. The working
source-checkout installation still requires a native build. No binary has been
published and no registry submission or PR has been opened.

A first staged JLL candidate loads from a fresh Linux artifact depot with no
Rust toolchain and no custom-library override, while leaving MOI unloaded.
Its audit identified the Linux unwind runtime. Adding that dependency at
`350f204c` alone did not resolve it: the default compiler selected an unavailable
runtime ABI tag. The recipe now selects GCC 10 and makes audit failures fatal;
the resulting platform tags resolve the runtime artifacts on both Linux targets.
The full Linux arm64 rebuild at `b892f46a` passes the fatal audit and resolves
the runtime dependency. Full artifact-backed tests and the remaining target
builds are still required before this stage is complete.
The corrected macOS arm64 artifact at `73f45e96` also passes the fatal audit;
its unused system libiconv dependency is gone. Linux x86_64 additionally needs
the GNU-linker selection at `28297b0b`, verified in a matching linker control;
the complete rebuild and Intel macOS artifact are still pending. The inspected
arm64 build baselines are glibc 2.19 on Linux and macOS 11.0 on Apple platforms.

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
