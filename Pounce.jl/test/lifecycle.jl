module TestLifecycle
using Test
import Pounce
import JuMP
const MOI = JuMP.MOI

function fixture()
    model = Pounce.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    x, b = MOI.add_constrained_variable(model, MOI.Interval(0.0, 3.0))
    p, pc = MOI.add_constrained_variable(model, MOI.Parameter(1.0))
    c = MOI.add_constraint(
        model,
        MOI.ScalarNonlinearFunction(:exp, Any[x]),
        MOI.LessThan(2.0),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    f = MOI.ScalarNonlinearFunction(:*, Any[x, p])
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
    return model, x, b, pc, c
end

function no_result(model)
    @test MOI.get(model, MOI.ResultCount()) == 0
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
    @test isnan(MOI.get(model, MOI.SolveTimeSec()))
    @test isnan(MOI.get(model, Pounce.NativeSolveTimeSec()))
end

@testset "Nonstructural edits preserve derivative caches" begin
    model, x, b, pc, c = fixture()
    evaluator = MOI.get(model, MOI.NLPBlock()).evaluator
    handle = model.native.problem
    MOI.set(model, MOI.ConstraintSet(), c, MOI.LessThan(4.0))
    no_result(model)
    @test handle.ipopt_problem == C_NULL
    MOI.optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ log(4) atol = 1e-6
    @test MOI.get(model, MOI.NLPBlock()).evaluator === evaluator
    MOI.set(model, MOI.ConstraintSet(), b, MOI.Interval(0.0, 1.0))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 1.0 atol = 1e-6
    @test MOI.get(model, MOI.NLPBlock()).evaluator === evaluator

    handle = model.native.problem
    MOI.set(model, MOI.ConstraintSet(), pc, MOI.Parameter(2.0))
    no_result(model)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 2.0 atol = 1e-6
    @test MOI.get(model, MOI.NLPBlock()).evaluator === evaluator
    @test model.native.problem === handle
    # A structural change must rebuild the derivative graph.
    y = MOI.add_variable(model)
    MOI.add_constraint(model, y, MOI.EqualTo(0.0))
    no_result(model)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
    @test MOI.get(model, MOI.NLPBlock()).evaluator !== evaluator
end

@testset "Settings retain results until the next attempt" begin
    model, x, _, _, _ = fixture()
    objective = MOI.get(model, MOI.ObjectiveValue())
    evaluator = MOI.get(model, MOI.NLPBlock()).evaluator
    handle = model.native.problem
    for (attr, value) in (
        (MOI.Silent(), false),
        (MOI.TimeLimitSec(), 1e-12),
        (MOI.TimeLimitSec(), nothing),
        (MOI.RawOptimizerAttribute("max_iter"), 0),
        (Pounce.CallbackFunction(), (args...) -> true),
    )
        MOI.set(model, attr, value)
        @test MOI.get(model, MOI.ResultCount()) == 1
        @test MOI.get(model, MOI.ObjectiveValue()) == objective
    end
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.VariablePrimalStart(), x, 0.5)
    @test MOI.get(model, MOI.ObjectiveValue()) == objective
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.ITERATION_LIMIT
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 0.5 atol = 1e-6
    @test handle.ipopt_problem == C_NULL
    @test MOI.get(model, MOI.NLPBlock()).evaluator === evaluator
    MOI.set(model, MOI.RawOptimizerAttribute("max_iter"), 100)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ objective atol = 1e-6

    MOI.set(model, Pounce.CallbackFunction(), (args...) -> error("lifecycle callback"))
    @test_throws ErrorException("lifecycle callback") MOI.optimize!(model)
    @test MOI.get(model, MOI.ResultCount()) == 0
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
    @test isfinite(MOI.get(model, MOI.SolveTimeSec()))
    @test isnan(MOI.get(model, Pounce.NativeSolveTimeSec()))
    MOI.set(model, Pounce.CallbackFunction(), nothing)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
end

@testset "Callback mutations fail before changing model data" begin
    model, x, _, pc, c = fixture()
    MOI.set(
        model,
        Pounce.CallbackFunction(),
        function (args...)
            @test_throws ErrorException MOI.set(
                model,
                MOI.ConstraintSet(),
                c,
                MOI.LessThan(10.0),
            )
            @test_throws ErrorException MOI.set(
                model,
                MOI.ConstraintSet(),
                pc,
                MOI.Parameter(3.0),
            )
            @test_throws ErrorException MOI.add_variable(model)
            @test_throws ErrorException MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            @test_throws ErrorException MOI.set(
                model,
                MOI.RawOptimizerAttribute("max_iter"),
                0,
            )
            @test MOI.get(model, MOI.ConstraintSet(), c) == MOI.LessThan(2.0)
            @test MOI.get(model, MOI.ConstraintSet(), pc) == MOI.Parameter(1.0)
            return false
        end,
    )
    MOI.optimize!(model)
    MOI.set(model, Pounce.CallbackFunction(), nothing)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.VariablePrimal(), x) ≈ log(2) atol = 1e-6
end

@testset "Objective replacement updates objective curvature" begin
    model = Pounce.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    x, _ = MOI.add_constrained_variable(model, MOI.Interval(1.0, 3.0))
    q = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(2.0, x, x)],
        MOI.ScalarAffineTerm{Float64}[],
        0.0,
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(q)}(), q)
    MOI.optimize!(model)
    f = MOI.ScalarNonlinearFunction(:^, Any[x, 3])
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
    H = zeros(length(MOI.hessian_lagrangian_structure(model)))
    MOI.eval_hessian_lagrangian(model, H, [2.0], 1.0, Float64[])
    @test sum(H) == 12.0 # d²(x³)/dx² at x=2; no contribution from old x².
    MOI.set(model, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
    MOI.optimize!(model)
    MOI.eval_hessian_lagrangian(model, H, [2.0], 1.0, Float64[])
    @test sum(H) == 0.0
    MOI.add_constraint(model, q, MOI.LessThan(9.0))
    MOI.optimize!(model)
    H = zeros(length(MOI.hessian_lagrangian_structure(model)))
    MOI.eval_hessian_lagrangian(model, H, [2.0], 1.0, [4.0])
    @test sum(H) == 8.0 # Feasibility mode still includes constraint curvature.
    MOI.set(model, MOI.ObjectiveFunction{typeof(q)}(), q)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    H = zeros(length(MOI.hessian_lagrangian_structure(model)))
    MOI.eval_hessian_lagrangian(model, H, [2.0], 1.0, [0.0])
    @test sum(H) == 2.0
end
end
