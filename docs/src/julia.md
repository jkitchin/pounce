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
Pounce.jl 0.1 accepts native POUNCE 0.15.x, 0.16.x and 0.17.x and checks the required C symbols
at import. Other native minor versions require a wrapper ABI review.
The existing NLP bindings and diagnostics work across these versions. Native
0.17's complementarity API is optional and detected at runtime (see below).
Convex/conic entry points are not yet exposed; see the [C API](c-api.md).
The existing linear-solver statistics struct remains unchanged, without the
`struct_size` field used by the new native output structs.
There is no registered Julia package or bundled JLL binary yet.

JuMP/MathOptInterface support is an optional package extension. Load JuMP or
MathOptInterface along with Pounce before constructing `Pounce.Optimizer`;
either load order works. Low-level C API users need not install MOI.

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

## L-BFGS nonlinear-variable hints

For `hessian_approximation="limited-memory"`, the wrapper automatically declares
the variables appearing in either coordinate of the complete structural
Hessian pattern. It combines quadratic terms, nonlinear evaluators, and vector
nonlinear oracles with their indices mapped to model columns. Purely linear
variables then remain outside the L-BFGS approximation. Numerical zeros at a
particular point do not establish linearity.

If any evaluator/oracle lacks Hessian structure, or the native library lacks
the optional entry points, the wrapper keeps the previous behavior. Setting
`num_linear_variables` explicitly overrides automatic detection; `0` requests
the original full approximation space. Exact and finite-difference solves do
not receive automatic hints. Structural changes rebuild the declaration;
parameter and bound updates retain the structural pattern.

For low-level problems, `Pounce.supports_nonlinear_variable_hints()` checks native
support. `Pounce.IpoptSetNonlinearVariables(prob, [2, 5])` declares distinct,
one-based columns; an empty vector declares a structurally zero Hessian.
`Pounce.IpoptClearNonlinearVariables(prob)` restores native defaults. Direct
declarations take precedence over `num_linear_variables` and may also guide
native finite-difference sparsity. Only omit identically zero Hessian
rows/columns, and set or clear declarations between solves.

## Comparing with Ipopt

Pounce and Ipopt.jl can coexist in one Julia process. Construct otherwise
identical models with `Pounce.Optimizer` and `Ipopt.Optimizer`, preserving
starts, formulation, scaling, and tolerances. Recompute residuals from the
returned points before comparing timing; solver status alone is not a
validation of application-level feasibility. Record the POUNCE commit,
library path, package versions, options, and linear solver with measurements.

`MOI.get(model, MOI.SolveTimeSec())` measures the entire optimizer call,
including wrapper setup. `MOI.get(model, Pounce.NativeSolveTimeSec())` reports
native solver wall time, including Julia callbacks invoked during that solve.
Model construction is outside both timers; distinguish first-call compilation
from subsequent solves. Final iteration counts do not require a user callback.

BMOPFTools accepts `solve_opf(net; optimizer=Pounce.Optimizer)`; its
`profile_solution(net, result)` can check the returned network solution.
The repository's `benchmarks/julia_bmopf/` harness pins a small teaching feeder
and its Julia dependency graph. It compares parameter updates, explicit starts,
and recovery with Ipopt, including independent current/KCL reconstruction.
Its README defines the timing boundaries, residual checks, and reproduction
commands; the results do not establish performance on large OPF networks.

## Native diagnostics

Use `MOI.get(model, attr)` with `Pounce.NativeResiduals()`,
`Pounce.RestorationStatistics()`, `Pounce.LinearSolverStatistics()`, and
`Pounce.FiniteDifferenceHessianStatistics()` to inspect a completed solve.
The residuals use the native scaled convention; they do not replace independent
physical residual checks. The complementarity statistic concerns NLP bounds and
multipliers, not declared MPCC pairs. A missing linear-backend summary returns
`nothing`; other optional fields retain their documented native sentinels.

```julia
MOI = JuMP.MOI
MOI.set(model, Pounce.RecordIterationHistory(), true)
optimize!(model)
write("solve.json", MOI.get(model, Pounce.SolveReport("full")))
```

The JSON follows the [solve-report schema](schema/solve-report-v1.md).
`SolveReport()` defaults to summary detail. Full reports include the trajectory
only if capture was enabled before solving. History defaults to off; changing
it affects the next solve. Disabling it rebuilds the native handle at that point.
Diagnostics obey result invalidation and are unavailable during callbacks.
See the wrapper README for low-level entry points and complete field semantics.

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

## Native complementarity

With a native library providing the finalized 0.17 C API,
`Pounce.supports_complementarity()` is true and the optimizer accepts
`MOI.Complements` in vector-of-variables, vector affine, vector quadratic and
vector nonlinear form. The second half must represent individual variables.
For example:

```julia
model = Model(Pounce.Optimizer)
@variable(model, x >= 0, start = 0.5)
@constraint(model, pair, complements(2x - 1, x))
optimize!(model)
value(x) # approximately 0.5
MOI.get(model, Pounce.ComplementarityStatistics())
```

Each function `F` gets a dedicated free native row. Its paired variable's bounds
determine the meaning: lower-only means `F ≥ 0`, upper-only `F ≤ 0`, a boxed
variable has mixed complementarity, a free variable requires `F = 0`, and a
fixed variable imposes no condition on `F`. Existing scalar constraints remain
independent. A variable may be paired only once, including fixed/free cases;
repeat pairings are rejected before modifying the model. Parameters can occur
in `F`, but cannot be paired variables. The wrapper does not silently introduce
auxiliary variables to transform unsupported declarations.

Bound edits invalidate results immediately and rebuild the native handle from
the original declarations, including changes between these five classes.
Parameter updates reuse the evaluators and handle. Ordinary scalar constraints,
legacy NLP blocks, and vector nonlinear oracles may coexist with these rows.
Complementarity functions use the selected automatic-differentiation backend;
unknown Hessian information retains the existing limited-memory fallback.

If a declared complementarity variable has a lower bound greater than its upper
bound when `optimize!` is called, the wrapper returns `INVALID_MODEL` with no
solution. `RawStatusString` identifies the variable and bounds. Correct the
bounds and call `optimize!` again to recover; a temporary crossing repaired
before the call does not invalidate the final model. Low-level native API
validation failures still throw `NativeAPIError`.

### Results and certificates

`ConstraintPrimal` returns `[F(x); x]`. `PrimalStatus` independently checks the
original bounds/rows and the mixed-complementarity natural map
`x - clamp(x - F, lower, upper)`, even when the native status says success.
It evaluates the equivalent `clamp(F, x - upper, x - lower)` to avoid losing
a nonzero residual when `x` is large relative to `F`.
All these primal residuals must satisfy `tol` for `FEASIBLE_POINT`, or
`acceptable_tol` for `NEARLY_FEASIBLE_POINT`; otherwise the result is
`INFEASIBLE_POINT`. This classification can be stricter than native acceptance:
the current driver's branch-polish trigger is `10 * tol`, not the wrapper's
feasibility threshold. A residual between `tol` and `10 * tol` is nearly
feasible only if it and the other primal residuals satisfy `acceptable_tol`.
Applications should also check their physical residuals in appropriate units.

The native termination status describes the formulation that returned the
point; it is not an MPCC stationarity certificate. Read the certificate and its
residuals separately:

- `ComplementarityStatistics()` returns an owned named tuple, including the
  certificate class, tightness, natural residual, reported formulation,
  B-stationarity verdict, polishing information and total driver iterations.
- `ComplementaritySolves()` returns per-formulation statistics in driver order.
- `ComplementarityText(which)` returns the summary or a polishing/B-check reason;
  `which` is a `Pounce.POUNCE_CC_TEXT_*` constant.
- `SolveReport()` includes the native `complementarity` block automatically.

Stats/text return `nothing` and the solve list is empty when no driver ran
(including models whose declarations all reduce to fixed/free NLP cases).
Before a completed solve or after a model edit, getters reject the query.
Native absent-value sentinels (`-1`, `NaN`, empty strings) are preserved, and
integer categories use the corresponding `Pounce.POUNCE_CC_*` constants.
`BarrierIterations()` remains the reported solve's count; use
`stats.total_iterations` for work across the driver.

The returned multipliers are stationarity multipliers and may be nonunique.
For `[F; x]`, `ConstraintDual` reports the function-row multipliers followed by
zeros. The variable-side contributions remain on the explicit bound constraints,
so they are not counted twice. A paired lower/upper bound returns its signed
native multiplier in the wrapper's usual objective-sense convention; it is not
clamped to the ordinary NLP sign. Interval bounds report the combined value.
Fixed/free declarations retain ordinary NLP bound handling. Because these are
not convex dual certificates, `DualStatus` is `UNKNOWN_RESULT_STATUS` for models
with complementarity declarations. Inspect MPCC certificates explicitly.
Primal and bound-dual starts remain explicit; complementarity `ConstraintDualStart`
is not supported. Live primal inspection is unavailable during the native MPCC
driver, although intermediate callbacks can observe its scalar progress.

The low-level optional bindings mirror `pounce.h`, with one-based row/variable
indices and one-based `GetPounceComplementaritySolve(prob, k)` indexing. They
initialize `struct_size` before each output call. Negative extension codes throw
`Pounce.NativeAPIError` with the operation, native code, message and optional
one-based batch position. Nonnegative declaration outcomes are successful;
existing Bool-returning APIs retain their own convention. Clearing native pairs
does **not** undo row edits made by `AddPounceComplementsConstraint`; recreate the
handle to undo that lowering. Older libraries load normally and report
complementarity as unsupported; no product-constraint fallback is applied.

Application validation is reproducible with
[`benchmarks/julia_mpcc_opf`](https://github.com/jkitchin/pounce/tree/main/benchmarks/julia_mpcc_opf),
which uses the original MPCCModels volt-var/volt-watt cases and independently
checks `min(G,H)` and the controller laws.
