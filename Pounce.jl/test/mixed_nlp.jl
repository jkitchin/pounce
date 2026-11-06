module TestMixedNLP

using Test, JuMP
import Pounce
const MOI = JuMP.MOI

function mixed_model(sense)
    model = Model(Pounce.Optimizer)
    set_silent(model)
    @variable(model, x >= 0, start = 3.1)
    @constraint(model, 2 * x >= 1)
    if sense == MOI.MIN_SENSE
        @objective(model, Min, (x - 2)^2)
    else
        @objective(model, Max, -(x - 2)^2)
    end
    legacy = [@NLconstraint(model, x^2 <= 16), @NLconstraint(model, (x - 3)^2 <= 1)]
    set = MOI.VectorNonlinearOracle(;
        dimension = 1,
        l = [3.0, -Inf],
        u = [Inf, 10.0],
        eval_f = (ret, x) -> (ret .= (x[1], 2 * x[1]); nothing),
        jacobian_structure = [(1, 1), (2, 1)],
        eval_jacobian = (ret, x) -> (ret .= (1.0, 2.0); nothing),
        hessian_lagrangian_structure = Tuple{Int,Int}[],
        eval_hessian_lagrangian = (ret, x, mu) -> nothing,
    )
    first_oracle = @constraint(model, [x] in set)
    set = MOI.VectorNonlinearOracle(;
        dimension = 1,
        l = [-Inf],
        u = [10.0],
        eval_f = (ret, x) -> (ret[1] = x[1]; nothing),
        jacobian_structure = [(1, 1)],
        eval_jacobian = (ret, x) -> (ret[1] = 1.0; nothing),
        hessian_lagrangian_structure = Tuple{Int,Int}[],
        eval_hessian_lagrangian = (ret, x, mu) -> nothing,
    )
    second_oracle = @constraint(model, [x] in set)
    return model, x, legacy, first_oracle, second_oracle
end

@testset "Legacy NLP block and vector-oracle duals" begin
    for sense in (MOI.MIN_SENSE, MOI.MAX_SENSE)
        model, x, legacy, first_oracle, second_oracle = mixed_model(sense)
        sign = sense == MOI.MIN_SENSE ? 1.0 : -1.0
        optimize!(model)
        @test termination_status(model) == MOI.LOCALLY_SOLVED
        @test value(x) ≈ 3.0 atol = 1e-6
        # Both legacy constraints are strictly slack. The active oracle's
        # multiplier must not be reported as either legacy constraint's dual.
        @test dual.(legacy) ≈ zeros(2) atol = 1e-5
        @test MOI.get(backend(model), MOI.NLPBlockDual()) ≈ zeros(2) atol = 1e-5
        @test MOI.get(backend(model), MOI.LagrangeMultiplier(), index(first_oracle)) ≈
              [2.0, 0.0] atol = 1e-5
        @test MOI.get(backend(model), MOI.LagrangeMultiplier(), index(second_oracle)) ≈
              [0.0] atol = 1e-5

        optimizer = unsafe_backend(model)
        affine = only(
            MOI.get(
                optimizer,
                MOI.ListOfConstraintIndices{
                    MOI.ScalarAffineFunction{Float64},
                    MOI.GreaterThan{Float64},
                }(),
            ),
        )
        oracles = MOI.get(
            optimizer,
            MOI.ListOfConstraintIndices{
                MOI.VectorOfVariables,
                MOI.VectorNonlinearOracle{Float64},
            }(),
        )
        MOI.set(optimizer, MOI.ConstraintDualStart(), affine, 23.0)
        MOI.set(optimizer, MOI.LagrangeMultiplierStart(), oracles[1], [11.0, -19.0])
        MOI.set(optimizer, MOI.LagrangeMultiplierStart(), oracles[2], [-17.0])
        MOI.set(optimizer, MOI.RawOptimizerAttribute("warm_start_init_point"), "yes")
        MOI.set(optimizer, MOI.RawOptimizerAttribute("presolve"), "no")
        inputs = Vector{Float64}[]
        MOI.set(
            optimizer,
            Pounce.CallbackFunction(),
            (args...) -> begin
                # IpoptSolve writes its output multipliers after the callbacks.
                # Observe the Julia-owned input buffer at the actual C boundary.
                push!(inputs, copy(optimizer.native.problem.mult_g))
                return false
            end,
        )
        for (nlp_start, expected) in (
            ([-7.0, -13.0], [-23.0, -11.0, 19.0, 17.0, 7.0, 13.0]),
            (nothing, [-23.0, -11.0, 19.0, 17.0, 0.0, 0.0]),
        )
            MOI.set(optimizer, MOI.NLPBlockDualStart(), nlp_start)
            empty!(inputs)
            MOI.optimize!(optimizer)
            @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.INTERRUPTED
            @test length(inputs) == 1
            @test only(inputs) == sign .* expected
        end
        # Clear oracle starts too, then reuse the existing native problem.
        # No component may silently inherit the previous solve's multipliers.
        for ci in oracles
            MOI.set(optimizer, MOI.LagrangeMultiplierStart(), ci, nothing)
        end
        empty!(inputs)
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) == MOI.INTERRUPTED
        @test length(inputs) == 1
        @test only(inputs) == sign .* [-23.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end
end

end # module
