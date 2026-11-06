# Julia and JuMP

`Pounce.jl` provides a Julia interface to POUNCE's C API and a
MathOptInterface optimizer for JuMP. The package currently installs from a
source checkout and requires a locally built native library.

## Installation

From the POUNCE repository:

```sh
cargo build --release --locked -p pounce-cinterface
```

Then, in your Julia environment (Julia 1.10 or later):

```julia
using Pkg
Pkg.develop(path="/absolute/path/to/pounce/Pounce.jl")
Pkg.add("JuMP")
using JuMP, Pounce
```

The default library is in the repository's `target/release` directory.
For another build, set `ENV["POUNCE_LIBRARY_PATH"]` to the library file or its
directory before loading Pounce. `Pounce.library_path()` reports the loaded
file, and `Pounce.GetIpoptVersion()` reports its compiled POUNCE version.
There is no registered Julia package or bundled JLL binary yet.

## Solve a model

```julia
model = Model(Pounce.Optimizer)
set_silent(model)
@variable(model, x >= 0, start = 1)
@variable(model, y >= 0, start = 1)
@constraint(model, x + y >= 2)
@objective(model, Min, x^2 + y^2)
optimize!(model)

@show termination_status(model)
@show objective_value(model)
@show value(x), value(y)
```

The wrapper supports continuous affine, quadratic, and nonlinear models,
sparse derivatives, primal/dual results and starts, modern nonlinear
expressions, and legacy `@NL...` models. It does not support discrete
variables or provide global optimality certificates. This interface covers
the C API's NLP path, not the full CLI's solver routing.

Pass native options through `set_optimizer_attribute`; for example,
`set_optimizer_attribute(model, "tol", 1e-8)`.
`set_time_limit_sec` and `set_silent` are also supported.
Enable `warm_start_init_point="yes"` after supplying primal/dual starts.
JuMP/MOI dual signs are converted by the wrapper, including maximization.

## Comparing with Ipopt

Pounce and Ipopt.jl can coexist in one Julia process. Construct otherwise
identical models with `Pounce.Optimizer` and `Ipopt.Optimizer`, preserving
starts, formulation, scaling, and tolerances. Recompute residuals from the
returned points before comparing timing; solver status alone is not a
validation of application-level feasibility. Record the POUNCE commit,
library path, package versions, options, and linear solver with measurements.

BMOPFTools accepts `solve_opf(net; optimizer=Pounce.Optimizer)`; its
`profile_solution(net, result)` can check the returned network solution.
The wrapper's own tests do not yet constitute a BMOPFTools benchmark.

## Testing and callback details

```sh
julia --project=Pounce.jl --startup-file=no -e 'using Pkg; Pkg.test()'
```

See the repository's
[Pounce.jl README](https://github.com/jkitchin/pounce/tree/main/Pounce.jl)
for the full callback contract, library configuration, supported interfaces,
and test instructions. In particular, callbacks must not retain borrowed
buffers, exceptions are rethrown only after returning from the C API, and
restoration callback metrics refer to the feasibility subproblem.
