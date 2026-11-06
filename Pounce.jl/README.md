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
Pounce.jl 0.1 accepts native POUNCE 0.15.x and verifies all required C symbols
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
same object and freeing it during a callback are rejected.

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
dependencies. CI builds the library and runs the suite on Linux and macOS;
Windows packaging is not part of this first release.

For BMOPFTools, the intended integration is
`solve_opf(net; optimizer=Pounce.Optimizer)`. Validate the returned point with
`profile_solution(net, result)`. The package test suite does not yet run
BMOPFTools or claim distribution-network benchmark results.

## Attribution

Native solver differences found during wrapper development, including a
limited-memory feasibility example, are recorded in the repository's
[validation notes](../dev-notes/julia-wrapper-validation.md).

The MOI adapter, utility code, and portions of the C bindings/tests are adapted
from Ipopt.jl under its MIT license. See [NOTICE.md](NOTICE.md) for the pinned
upstream source and changes, and [LICENSE.md](LICENSE.md) for the license.
POUNCE's Rust library retains its own EPL-2.0 license.
