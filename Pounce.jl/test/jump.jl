using JuMP
import Ipopt
const MOI = JuMP.MOI

function hs071(optimizer; legacy = false)
    model = Model(optimizer)
    set_silent(model)
    @variable(model, 1 <= x[1:4] <= 5)
    set_start_value.(x, [1.0, 5.0, 5.0, 1.0])
    if legacy
        @NLobjective(model, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
        @NLconstraint(model, prod(x[i] for i = 1:4) >= 25)
        @NLconstraint(model, sum(x[i]^2 for i = 1:4) == 40)
    else
        @objective(model, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
        @constraint(model, prod(x) >= 25)
        @constraint(model, sum(x .^ 2) == 40)
    end
    optimize!(model)
    return model, x
end

@testset "JuMP and independent Ipopt library" begin
    @test solver_name(Model(Pounce.Optimizer)) == "POUNCE"
    @test Pounce.GetIpoptVersion() != Ipopt.GetIpoptVersion()
    oracle, xi = hs071(Ipopt.Optimizer)
    for legacy in (false, true)
        model, x = hs071(Pounce.Optimizer; legacy)
        @test termination_status(model) == MOI.LOCALLY_SOLVED
        @test objective_value(model) ≈ objective_value(oracle) atol = 1e-6
        @test value.(x) ≈ value.(xi) atol = 1e-5
        @test prod(value.(x)) >= 25 - 1e-6
        @test sum(abs2, value.(x)) ≈ 40 atol = 1e-6
        @test primal_status(model) == MOI.FEASIBLE_POINT
        @test dual_status(model) == MOI.FEASIBLE_POINT
    end
    # Solve Ipopt again after POUNCE to catch symbol/library contamination.
    optimize!(oracle)
    @test termination_status(oracle) == MOI.LOCALLY_SOLVED
end

@testset "Mixed constraints, dual signs, and repeated solves" begin
    for optimizer in (Pounce.Optimizer, Ipopt.Optimizer), maximize in (false, true)
        model = Model(optimizer)
        set_silent(model)
        @variable(model, 0 <= x <= 3, start = 1)
        @variable(model, 0 <= y <= 3, start = 1)
        c = @constraint(model, x + y >= 2)
        @constraint(model, x^2 + y^2 <= 8)
        if maximize
            @objective(model, Max, -(x^2 + y^2))
        else
            @objective(model, Min, x^2 + y^2)
        end
        optimize!(model)
        @test termination_status(model) == MOI.LOCALLY_SOLVED
        @test value.([x, y]) ≈ [1, 1] atol = 1e-5
        @test dual(c) ≈ 2 atol = 1e-4
        starts, dual_start = value.([x, y]), dual(c)
        set_start_value.([x, y], starts)
        set_dual_start_value(c, dual_start)
        set_optimizer_attribute(model, "warm_start_init_point", "yes")
        optimize!(model)
        @test objective_value(model) ≈ (maximize ? -2 : 2) atol = 1e-5
        set_normalized_rhs(c, 3.0)
        optimize!(model)
        @test value.([x, y]) ≈ [1.5, 1.5] atol = 1e-5
    end
end

@testset "Callback exception and recovery" begin
    model = Model(Pounce.Optimizer)
    set_silent(model)
    @variable(model, x >= 1, start = 2)
    @objective(model, Min, x^2)
    MOI.set(model, Pounce.CallbackFunction(), (args...) -> error("user callback"))
    @test_throws ErrorException("user callback") optimize!(model)
    @test result_count(model) == 0
    MOI.set(model, Pounce.CallbackFunction(), (args...) -> true)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test value(x) ≈ 1 atol = 1e-6
end

@testset "Option resets, result invalidation, and native cleanup" begin
    model = Model(Pounce.Optimizer)
    set_silent(model)
    @variable(model, 1 <= x[1:4] <= 5)
    set_start_value.(x, [1.0, 5.0, 5.0, 1.0])
    @objective(model, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
    @constraint(model, prod(x) >= 25)
    @constraint(model, sum(x .^ 2) == 40)
    set_time_limit_sec(model, 1e-12)
    optimize!(model)
    @test termination_status(model) == MOI.TIME_LIMIT
    unset_time_limit_sec(model)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 17.014017145179164 atol = 1e-6

    optimizer = unsafe_backend(model)
    problem = optimizer.native.problem
    MOI.empty!(optimizer)
    @test problem.ipopt_problem == C_NULL
    @test MOI.get(optimizer, MOI.ResultCount()) == 0
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED

    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.INVALID_MODEL
    MOI.add_variable(optimizer)
    MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED

    MOI.set(optimizer, MOI.RawOptimizerAttribute("not_an_option"), 1)
    @test_throws ErrorException MOI.optimize!(optimizer)
    @test MOI.get(optimizer, MOI.ResultCount()) == 0
end
