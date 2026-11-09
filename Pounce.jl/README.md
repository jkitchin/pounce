# Pounce.jl

Julia and [JuMP](https://jump.dev/) interface to POUNCE's C API.
Use `Model(Pounce.Optimizer)` for continuous nonlinear optimization.

This is a source-checkout package: build the native library first. It is not
yet registered in General and does not yet ship a JLL binary dependency.

## Install

Requires Julia 1.10 or later and the Rust toolchain used by POUNCE.

From the POUNCE repository:

```sh
cargo build --release --locked -p pounce-cinterface
```

In your Julia project:

```julia
using Pkg
Pkg.develop(path="/absolute/path/to/pounce/Pounce.jl")
Pkg.add("JuMP")
using JuMP, Pounce
```

By default the package loads the library from the repository's
`target/release` directory. For a custom Cargo target directory or an installed
library, set `ENV["POUNCE_LIBRARY_PATH"]` **before** `using Pounce` to the
directory containing the library or to its full filename. The setting is read
at runtime, including when using a precompiled package. Restart Julia to change
libraries; do not rebuild a loaded library in a running Julia process.

`Pounce.library_path()` identifies the loaded file;
`Pounce.GetIpoptVersion()` reports the POUNCE version compiled into it.
The C entry-point names retain the Ipopt spelling used in `pounce.h`.
Pounce.jl 0.1 accepts native POUNCE 0.15.x and 0.16.x and verifies all required C symbols
before exposing the library. A new native minor version requires a wrapper ABI
review; an arbitrary Ipopt library is not compatible with POUNCE's callback ABI.

POUNCE and Ipopt.jl can be loaded in the same process. Pounce does not replace
Ipopt_jll or modify Ipopt's library preferences.

The low-level C API can be installed and loaded without MathOptInterface.
Loading JuMP or MathOptInterface activates `PounceMathOptInterfaceExt` and makes
`Pounce.Optimizer`, `Pounce.CallbackFunction`, and `Pounce.NativeSolveTimeSec`
available. Either load order works. Existing `using JuMP, Pounce` code keeps the
same API. For code that needs the concrete optimizer type as a constant, use
`Base.get_extension(Pounce, :PounceMathOptInterfaceExt).Optimizer` after loading
both packages, or pass `Pounce.Optimizer` through a function barrier.

The extension precompiles a small nonlinear solve to reduce first-solve latency.
Library selection still happens at runtime; precompilation stores no native
problem or library handle in a global cache.

## JuMP example

```julia
using JuMP, Pounce

model = Model(Pounce.Optimizer)
set_silent(model)
@variable(model, 1 <= x[1:4] <= 5)
set_start_value.(x, [1.0, 5.0, 5.0, 1.0])
@objective(model, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
@constraint(model, prod(x) >= 25)
@constraint(model, sum(x.^2) == 40)
optimize!(model)

@show termination_status(model)
@show objective_value(model)
@show value.(x)
```

The same example is in `examples/hs071.jl`. Run it from a Julia environment
containing Pounce and JuMP.

## Supported interface

- Continuous variables and lower, upper, fixed, and interval bounds.
- Affine, quadratic, and scalar nonlinear objectives and constraints.
- Modern JuMP nonlinear expressions and the legacy `@NL...` / MOI `NLPBlock`
  interface (do not mix the two nonlinear APIs in one model).
- Sparse derivatives through MOI's evaluator; limited-memory Hessians when
  an evaluator does not provide exact Hessians.
- MOI parameters, user-defined nonlinear operators, and vector nonlinear oracles.
- Primal and constraint/bound dual results, termination status, solve time,
  and iteration count.
- Primal/dual starts, repeated solves, and an intermediate callback.

Discrete variables and global optimality certificates are not supported.
Use MOI bridges (as JuMP does by default) for reformulations into the supported
scalar constraint types. This first interface covers the C API's NLP solve
path; it does not expose POUNCE's full CLI routing or sensitivity-session API.

## Options and starts

With `hessian_approximation="limited-memory"`, the wrapper automatically tells
POUNCE which variables can have nonzero Hessian rows or columns. It uses the
complete structural Hessian pattern from quadratic terms, the nonlinear
evaluator, and vector nonlinear oracles (mapped into model variable order).
This keeps purely linear variables out of the L-BFGS approximation; it does
not infer linearity from derivative values at the starting point.

If any evaluator or oracle lacks Hessian structure, or the loaded native
library lacks the optional hint symbols, automatic detection is skipped.
An explicit `num_linear_variables` option also disables automatic detection
and preserves the native prefix convention. Set it to `0` to use the original
full approximation space. Exact and finite-difference solves receive no
automatic hints. Structural edits recompute the subset; parameter and bound
updates retain the conservative structural pattern.

Low-level users can call `Pounce.supports_nonlinear_variable_hints()`,
`Pounce.IpoptSetNonlinearVariables(prob, [2, 5])`, and
`Pounce.IpoptClearNonlinearVariables(prob)`. The list uses distinct one-based
indices. An empty list declares a structurally zero Hessian; clearing a list
restores the native default. Direct declarations override `num_linear_variables`
and may also guide native finite-difference sparsity. Only omit variables whose
Hessian rows/columns are identically zero, and change hints between solves.

```julia
set_optimizer_attribute(model, "tol", 1e-8)
set_optimizer_attribute(model, "max_iter", 500)
set_time_limit_sec(model, 30.0)
```

Native options use their registered types: `Float64`, integer, or string.
Unknown names or values rejected by the C option setters raise an error when
optimization sets them. Other native validation can return `INVALID_OPTION`.
`set_silent` takes precedence over `print_level`.

For a warm start, set primal values with `set_start_value` and constraint
multipliers with `set_dual_start_value`, then enable
`set_optimizer_attribute(model, "warm_start_init_point", "yes")`.
Use JuMP/MOI dual conventions; the wrapper converts signs for POUNCE, including
maximization. Repeated solves do not automatically reuse the previous solution.
They also do not promise reuse of native solver factorizations.

Model edits invalidate results immediately. Bound changes refresh the bound
snapshot while retaining the initialized derivative evaluator; parameter updates
reuse compatible evaluator and native-handle caches. Structural edits rebuild
them. Changes to starts, solver options, or callbacks retain the last completed
optimizer result until another solve starts (JuMP may apply its own access
guards). Every new attempt clears result availability before setup. Options are
applied to a fresh handle when changed, including removal of a time limit, so
old native option values cannot leak into the next solve. Mutating a model from
inside its callback is rejected before the model data is changed.

Optimizer fields are implementation details. Use MOI attributes to query models
and results; use the qualified low-level C bindings for explicitly owned native
problems.

After `optimize!`, two timers are available:

```julia
MOI.get(model, MOI.SolveTimeSec())          # entire optimizer call, including setup
MOI.get(model, Pounce.NativeSolveTimeSec()) # native solver's reported wall time
```

The wrapper timer uses a monotonic clock. Native time includes Julia callbacks
executed inside the solve, but excludes wrapper setup. Model construction before
`optimize!` is outside both timers. Measure first-call compilation separately
from subsequent solves. After a thrown error, the optimizer records elapsed
wrapper time, resets iterations to zero, and leaves native time unavailable
(`NaN`); JuMP's normal result-access guards still apply.

## Native diagnostics and solve reports

```julia
MOI = JuMP.MOI
MOI.set(model, Pounce.RecordIterationHistory(), true) # opt-in, before solving
optimize!(model)
residuals = MOI.get(model, Pounce.NativeResiduals())
restoration = MOI.get(model, Pounce.RestorationStatistics())
linear = MOI.get(model, Pounce.LinearSolverStatistics())
fd = MOI.get(model, Pounce.FiniteDifferenceHessianStatistics())
write("solve.json", MOI.get(model, Pounce.SolveReport("full")))
```

These attributes work with direct and cached JuMP models. Reports use the native
`pounce.solve-report/v1` JSON schema and can be consumed by pounce-studio tools.
Their solution arrays use native solver column/row order, including any bridge
variables; the report does not provide a JuMP name/index map.
`SolveReport()` defaults to summary detail, which omits iteration history.
The wrapper returns a JSON string and does not load a JSON parser; report
serialization uses a temporary file that is removed before returning.

`NativeResiduals()` returns `primal_infeasibility`, `dual_infeasibility`, and
`complementarity` in the solver's internally scaled convention. The last value
measures NLP bound/multiplier complementarity, not declared MPCC pairs. Continue
to recompute application residuals from returned points for physical validation.
The native getters return zero when a completed native call has no statistics;
zero alone is not a feasibility certificate.

The statistics attributes return owned named tuples. `LinearSolverStatistics()`
returns `nothing` if no backend summary exists; optional native fields retain
`NaN`/-1 sentinels. The returned `fd.pattern_used` is -1 when finite differences did not run, 0 for declared
sparsity, or 1 for Jacobian-derived sparsity. Counts and flags describe the
pattern actually used. Linear statistics cover counts, fill, pivots and inertia;
the C struct does not expose separate factorization timings.

Diagnostics reject queries before a solve, during callbacks, after a thrown
solve, and after model changes invalidate the result. Completed unsuccessful
solves can still supply diagnostics. At the MOI optimizer level, changing options
or history capture keeps the previous result until the next solve starts. JuMP
may additionally mark a cached model dirty when a setting changes, requiring
`optimize!` before querying through the JuMP model again. History is off by default;
toggling it takes effect on the next solve. Turning it off rebuilds the native
handle because the C API only supports enabling it. Like other optimizer
options, the history setting survives `MOI.empty!`.

Core-only users can call `GetIpoptPrimalInf`, `GetIpoptDualInf`,
`GetIpoptComplInf`, `GetPounceRestorationStats`, `GetPounceLinearSolverStats`,
`GetPounceFdHessianStats`, `IpoptEnableIterHistory`, and
`IpoptWriteSolveReport(prob, path; detail="full")`, all qualified with `Pounce.`.
The new diagnostic getters require a completed solve and reject freed handles.
Report writing accepts UTF-8 paths, rejects embedded NULs, raises an error on a
failed write, and overwrites an existing file. Low-level handles must be
recreated to disable history capture.

## Callbacks and low-level use

```julia
MOI = JuMP.MOI
MOI.set(model, Pounce.CallbackFunction(), (args...) -> args[2] < 100)
# Between solves, remove the callback:
MOI.set(model, Pounce.CallbackFunction(), nothing)
```

The callback receives the C intermediate-callback arguments; return `false`
to stop. `args[1]` distinguishes regular iterations (0) from restoration (1).
Restoration metrics describe the feasibility subproblem; current-iterate
inspection is unavailable during restoration.

Within a regular callback, `MOI.CallbackVariablePrimal` reads a single lazily
copied primal snapshot, shared by all variable queries in that invocation.
The snapshot cannot be queried outside that callback's task or after it returns.
Each new invocation refreshes it. Callback replacement/removal takes effect on
the next solve. Ordinary solves install no intermediate callback; final iteration
counts come directly from the native solver's statistics.

The qualified low-level API includes `Pounce.CreateIpoptProblem`,
`Pounce.IpoptSolve`, `Pounce.FreeIpoptProblem`, the option setters, and iterate
inspectors. See the constructor's docstring and `test/cinterface.jl`.
`Pounce.GetIpoptIterCount` and `Pounce.GetIpoptSolveTime` expose native statistics
between solves; they return zero before the first solve and reject use during a
solve or after freeing the problem. `Pounce.SetIntermediateCallback(prob, nothing)`
removes a low-level callback between solves.
Indices are one-based and Hessian structure is lower triangular. Callback
buffers are borrowed: do not retain them or resize solve buffers.
Use separate problem objects for separate solves; recursive solves on the
same object and freeing it during a callback are rejected. Concurrent access to
one problem is unsupported. The wrapper pins the solving task to its current
thread for the duration of the native call and restores its prior scheduling
policy afterward. Callbacks may yield, but a different task cannot start another
POUNCE solve on that occupied thread; independent solves on other threads remain
available. This protects the native thread-local iterate and callback state.
Idle problems may be freed or finalized on another thread; explicit
`FreeIpoptProblem` gives deterministic cleanup.

If the low-level constructor receives `eval_h = nothing`, select
`hessian_approximation=limited-memory` or `finite-difference` before solving.
Exact mode requires a Hessian callback, including an explicit empty callback for
a structurally zero Hessian. String options and output filenames reject embedded
NUL characters instead of silently truncating them.

A callback `DomainError` or nonfinite numerical output rejects the evaluation,
allowing the solver to backtrack. Other exceptions, including interrupts, are
caught at the C boundary and rethrown after the solver returns. Exceptions in
the intermediate callback always propagate this way. No Julia exception
unwinds through Rust.

## Tests and development

```sh
cargo build --release --locked -p pounce-cinterface
julia --project=Pounce.jl --startup-file=no -e 'using Pkg; Pkg.test()'
```

Tests cover direct C callbacks, forced garbage collection, callback failures,
JuMP's modern/legacy APIs, warm starts, dual signs, an independent Ipopt solve,
and MOI conformance. Ipopt and JuMP are test dependencies, not runtime
dependencies. CI is configured to build and test the library on Linux and macOS;
Windows packaging is not part of this first release.
Local validation covers macOS arm64 on Julia 1.10.12/1.12.6 and Linux arm64 on
Julia 1.12.6. The minimum MOI 1.48.0 / JuMP 1.29.0 combination also passes on
Julia 1.10.12. Exact revisions and test counts are in the validation notes below.

For BMOPFTools, use
`solve_opf(net; optimizer=Pounce.Optimizer)`. Validate the returned point with
`profile_solution(net, result)`. The separate
[application harness](../benchmarks/julia_bmopf/README.md) pins a small teaching
feeder and dependency graph, exercises parameter/start/recovery scenarios, and
adds independent SI current/KCL checks. It does not establish utility-scale
performance. Local JLL release preparation is documented in
[packaging/README.md](packaging/README.md); those artifacts remain unpublished.

## Attribution

Native solver differences found during wrapper development, including a
limited-memory feasibility example, are recorded in the repository's
[validation notes](../dev-notes/julia-wrapper-validation.md).

The MOI adapter, utility code, and portions of the C bindings/tests are adapted
from Ipopt.jl under its MIT license. See [NOTICE.md](NOTICE.md) for the pinned
upstream source and changes, and [LICENSE.md](LICENSE.md) for the license.
POUNCE's Rust library retains its own EPL-2.0 license.
