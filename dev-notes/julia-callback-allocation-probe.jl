# Allocation probe for repeated scalar queries within one callback.
# Self-contained diagonal quadratic fixture; no external benchmark data.
# This does not measure total solve allocations or wall-time speedups.
import Pounce
const MOI = Pounce.MOI
println("Wrapper commit: ", readchomp(`git rev-parse HEAD`))
println("Working tree: ", readchomp(`git status --short`))
println(
    "Native build commit: ",
    get(
        ENV,
        "POUNCE_NATIVE_REVISION",
        "unknown; set POUNCE_NATIVE_REVISION to the library build commit",
    ),
)
println("Julia ", VERSION, "; MOI ", pkgversion(MOI), "; POUNCE ", Pounce.GetIpoptVersion())
println("Library: ", Pounce.library_path())
function read_primals(model, xs)
    value = 0.0
    for x in xs
        value += MOI.get(model, MOI.CallbackVariablePrimal(model), x)
    end
    return value
end
function probe(n)
    model = Pounce.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.RawOptimizerAttribute("presolve"), "no")
    xs = MOI.add_variables(model, n)
    f = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(2.0, x, x) for x in xs],
        [MOI.ScalarAffineTerm(-2.0, x) for x in xs],
        Float64(n),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.set(
        model,
        Pounce.CallbackFunction(),
        function (args...)
            read_primals(model, xs) # Compile and warm this query path first.
            bytes = @allocated read_primals(model, xs)
            println(
                "n=",
                n,
                "; warmed scalar-query sweep Julia allocations=",
                bytes,
                " bytes",
            )
            return false
        end,
    )
    MOI.optimize!(model)
    @assert MOI.get(model, MOI.TerminationStatus()) == MOI.INTERRUPTED
end
for n in (128, 512)
    probe(n)
end
