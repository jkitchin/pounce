# Optional arguments: Ipopt MadNLP (installed in the selected environment).
# See julia-wrapper-design-study.md for the original versions and measurements.
import Pounce
import MathOptInterface as MOI

repo = normpath(joinpath(@__DIR__, ".."))
println("Wrapper checkout: ", readchomp(`git -C $repo rev-parse HEAD`))
println("Working tree changes:\n", readchomp(`git -C $repo status --short`))
println("Julia ", VERSION, "; MOI ", pkgversion(MOI))
println("POUNCE ", Pounce.GetIpoptVersion(), " at ", Pounce.library_path())
println("Record the native library's build commit separately if it was rebuilt.")

function make_model(Optimizer, upper)
    model = Optimizer()
    MOI.set(model, MOI.Silent(), true)
    x, _ = MOI.add_constrained_variable(model, MOI.Interval(0.0, 3.0))
    c = MOI.add_constraint(
        model,
        MOI.ScalarNonlinearFunction(:exp, Any[x]),
        MOI.LessThan(upper),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    return model, x, c
end

function probe(Optimizer)
    model, x, c = make_model(Optimizer, 2.0)
    MOI.optimize!(model)
    first = MOI.get(model, MOI.VariablePrimal(), x)
    MOI.set(model, MOI.ConstraintSet(), c, MOI.LessThan(4.0))
    MOI.optimize!(model)
    second = MOI.get(model, MOI.VariablePrimal(), x)
    fresh, y, _ = make_model(Optimizer, 4.0)
    MOI.optimize!(fresh)
    @show Optimizer first second MOI.get(model, MOI.TerminationStatus())
    @show MOI.get(fresh, MOI.VariablePrimal(), y) log(4.0)
end

probe(Pounce.Optimizer)
if "Ipopt" in ARGS
    @eval import Ipopt
    println("Ipopt.jl ", pkgversion(Ipopt), "; native Ipopt ", Ipopt.GetIpoptVersion())
    probe(Ipopt.Optimizer)
end
if "MadNLP" in ARGS
    @eval import MadNLP
    println("MadNLP ", pkgversion(MadNLP))
    probe(MadNLP.Optimizer)
end
