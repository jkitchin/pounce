module TestModelUpdates

using Test
import JuMP
import Pounce
const MOI = JuMP.MOI

function nonlinear_bound_model(optimizer, set, sense)
    model = optimizer()
    MOI.set(model, MOI.Silent(), true)
    x, _ = MOI.add_constrained_variable(model, MOI.Interval(0.0, 3.0))
    c = MOI.add_constraint(model, MOI.ScalarNonlinearFunction(:exp, Any[x]), set)
    MOI.set(model, MOI.ObjectiveSense(), sense)
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
    return model, x, c
end

@testset "Nonlinear bound updates use current bounds" begin
    for (first_set, next_set, sense, first_x, next_x) in (
        (MOI.LessThan(2.0), MOI.LessThan(4.0), MOI.MAX_SENSE, log(2), log(4)),
        (MOI.GreaterThan(2.0), MOI.GreaterThan(4.0), MOI.MIN_SENSE, log(2), log(4)),
        (MOI.EqualTo(2.0), MOI.EqualTo(4.0), MOI.MIN_SENSE, log(2), log(4)),
        (MOI.Interval(1.5, 2.0), MOI.Interval(3.0, 4.0), MOI.MAX_SENSE, log(2), log(4)),
    )
        model, x, c = nonlinear_bound_model(Pounce.Optimizer, first_set, sense)
        MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ first_x atol = 1e-6
        MOI.set(model, MOI.ConstraintSet(), c, next_set)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        @test MOI.get(model, MOI.RawStatusString()) == "Optimize not called"
        @test MOI.get(model, MOI.ResultCount()) == 0
        MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ next_x atol = 1e-6
        @test MOI.get(model, MOI.ConstraintPrimal(), c) ≈ exp(next_x) atol = 1e-6
        fresh, fresh_x, fresh_c = nonlinear_bound_model(Pounce.Optimizer, next_set, sense)
        MOI.optimize!(fresh)
        @test MOI.get(fresh, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈
              MOI.get(fresh, MOI.VariablePrimal(), fresh_x) atol = 1e-6
        @test MOI.get(model, MOI.ConstraintDual(), c) ≈
              MOI.get(fresh, MOI.ConstraintDual(), fresh_c) atol = 1e-6
    end
end

@testset "JuMP cached and direct nonlinear updates" begin
    for model in (JuMP.Model(Pounce.Optimizer), JuMP.direct_model(Pounce.Optimizer()))
        JuMP.set_silent(model)
        JuMP.@variable(model, 0 <= x <= 3)
        c = JuMP.@constraint(model, exp(x) <= 2)
        JuMP.@objective(model, Max, x)
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == MOI.LOCALLY_SOLVED
        @test JuMP.value(x) ≈ log(2) atol = 1e-6
        # JuMP may represent this as exp(x) - 2 <= 0. Relax the stored
        # upper bound by two, preserving the expression's constant term.
        set = MOI.get(model, MOI.ConstraintSet(), c)
        MOI.set(model, MOI.ConstraintSet(), c, MOI.LessThan(set.upper + 2.0))
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == MOI.LOCALLY_SOLVED
        @test JuMP.value(x) ≈ log(4) atol = 1e-6
    end
end

@testset "Bound edits invalidate results consistently" begin
    for kind in (:variable, :affine, :quadratic)
        model = Pounce.Optimizer()
        MOI.set(model, MOI.Silent(), true)
        x = MOI.add_variable(model)
        MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
        f = if kind == :variable
            x
        elseif kind == :affine
            MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
        else
            MOI.ScalarQuadraticFunction(
                [MOI.ScalarQuadraticTerm(2.0, x, x)],
                MOI.ScalarAffineTerm{Float64}[],
                0.0,
            )
        end
        c = MOI.add_constraint(model, f, MOI.LessThan(4.0))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), x)
        MOI.set(model, MOI.VariablePrimalStart(), x, 1.0)
        MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
        MOI.set(model, MOI.ConstraintSet(), c, MOI.LessThan(9.0))
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        @test MOI.get(model, MOI.RawStatusString()) == "Optimize not called"
        @test MOI.get(model, MOI.ResultCount()) == 0
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
        @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
        MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ (kind == :quadratic ? 3.0 : 9.0) atol =
            1e-6
    end
end

@testset "Parameter edits invalidate results and update the next solve" begin
    model = JuMP.direct_model(Pounce.Optimizer())
    JuMP.set_silent(model)
    JuMP.@variable(model, 0 <= x <= 3)
    JuMP.@variable(model, p in JuMP.Parameter(1.0))
    JuMP.@objective(model, Min, (x - p)^2)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.LOCALLY_SOLVED
    @test JuMP.value(x) ≈ 1.0 atol = 1e-6
    JuMP.set_parameter_value(p, 2.0)
    @test JuMP.termination_status(model) == MOI.OPTIMIZE_NOT_CALLED
    @test JuMP.result_count(model) == 0
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.LOCALLY_SOLVED
    @test JuMP.value(x) ≈ 2.0 atol = 1e-6
end

end # module
