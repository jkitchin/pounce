module TestComplementarity
using Test, JuMP
import Pounce
const MOI = JuMP.MOI
const Ext = Base.get_extension(Pounce, :PounceMathOptInterfaceExt)

function scalar_model(lower, upper; nonlinear = false, cached = false)
    m = cached ? Model(Pounce.Optimizer) : direct_model(Pounce.Optimizer())
    set_silent(m)
    @variable(m, x, start = 0.5)
    isfinite(lower) && set_lower_bound(x, lower)
    isfinite(upper) && set_upper_bound(x, upper)
    c =
        nonlinear ? @constraint(m, complements(x^3 - 0.125, x)) :
        @constraint(m, complements(2x - 1, x))
    return m, x, c
end

@testset "Complementarity capability" begin
    m = Pounce.Optimizer()
    for F in (
        MOI.VectorOfVariables,
        MOI.VectorAffineFunction{Float64},
        MOI.VectorQuadraticFunction{Float64},
        MOI.VectorNonlinearFunction,
    )
        @test MOI.supports_constraint(m, F, MOI.Complements) ==
              Pounce.supports_complementarity()
    end
    @test_throws MOI.GetAttributeNotAllowed MOI.get(m, Pounce.ComplementarityStatistics())
    if !Pounce.supports_complementarity()
        x = MOI.add_variable(m)
        @test_throws MOI.UnsupportedConstraint MOI.add_constraint(
            m,
            MOI.VectorOfVariables([x, x]),
            MOI.Complements(2),
        )
        MOI.optimize!(m)
        @test MOI.get(m, Pounce.ComplementarityStatistics()) === nothing
        @test isempty(MOI.get(m, Pounce.ComplementaritySolves()))
        @test MOI.get(m, Pounce.ComplementarityText()) === nothing
    end
end

if Pounce.supports_complementarity()
    @testset "All five MOI bound cases; direct/cached and affine/nonlinear" begin
        for cached in (false, true),
            nonlinear in (false, true),
            (lower, upper) in
            ((0.0, Inf), (-Inf, 1.0), (0.0, 1.0), (-Inf, Inf), (0.5, 0.5))

            m, x, c = scalar_model(lower, upper; cached, nonlinear)
            optimize!(m)
            @test termination_status(m) in (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED)
            @test primal_status(m) in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
            @test value(x) ≈ 0.5 atol = 1e-5
            @test abs(value(x) - clamp(value(x) - value(c)[1], lower, upper)) < 1e-6
            @test length(dual(c)) == 2
            @test dual(c)[2] == 0
            @test dual_status(m) == MOI.UNKNOWN_RESULT_STATUS
            stats = MOI.get(m, Pounce.ComplementarityStatistics())
            if lower == upper || (isinf(lower) && isinf(upper))
                @test stats === nothing
                @test isempty(MOI.get(m, Pounce.ComplementaritySolves()))
            else
                @test stats.n_pairs == 1
                @test stats.natural_residual < 1e-6
                @test stats.total_iterations >= barrier_iterations(m)
                solves = MOI.get(m, Pounce.ComplementaritySolves())
                @test length(solves) == stats.n_solves
                @test all(s -> !isempty(s.stage), solves)
                @test MOI.get(m, Pounce.ComplementarityText()) isa String
                @test occursin("complementarity", MOI.get(m, Pounce.SolveReport()))
            end
        end
    end

    @testset "Declaration validation, atomic refusal and owned functions" begin
        m = Pounce.Optimizer()
        x = MOI.add_variables(m, 3)
        f = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[2])),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[1])),
            ],
            [0.0, 0.0],
        )
        c = MOI.add_constraint(m, f, MOI.Complements(2))
        original = deepcopy(f)
        f.constants[1] = 5
        got = MOI.get(m, MOI.ConstraintFunction(), c)
        @test got ≈ original
        got.constants[1] = 7
        @test MOI.get(m, MOI.ConstraintFunction(), c) ≈ original
        @test MOI.get(m, MOI.NumberOfConstraints{typeof(f),MOI.Complements}()) == 1
        @test MOI.get(m, MOI.ListOfConstraintIndices{typeof(f),MOI.Complements}()) == [c]
        @test (typeof(f), MOI.Complements) in MOI.get(m, MOI.ListOfConstraintTypesPresent())
        @test MOI.dimension(MOI.get(m, MOI.ConstraintSet(), c)) == 2
        @test !MOI.is_valid(
            m,
            MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.Complements}(c.value),
        )
        @test_throws MOI.AddConstraintNotAllowed MOI.add_constraint(
            m,
            MOI.VectorOfVariables([x[3], x[1]]),
            MOI.Complements(2),
        )
        @test_throws MOI.AddConstraintNotAllowed MOI.add_constraint(
            m,
            MOI.VectorOfVariables([x[1], x[2], x[3], x[3]]),
            MOI.Complements(4),
        )
        @test_throws DimensionMismatch MOI.add_constraint(
            m,
            MOI.VectorOfVariables(x),
            MOI.Complements(2),
        )
        invalid = MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(2.0, x[2]))],
            [0.0, 0.0],
        )
        @test_throws MOI.AddConstraintNotAllowed MOI.add_constraint(
            m,
            invalid,
            MOI.Complements(2),
        )
        @test length(m.data.complementarity_constraints) == 1
        @test_throws MOI.InvalidIndex MOI.add_constraint(
            m,
            MOI.VectorOfVariables([MOI.VariableIndex(99), x[2]]),
            MOI.Complements(2),
        )
        MOI.empty!(m)
        @test MOI.is_empty(m)
        @test !MOI.is_valid(m, c)
    end

    @testset "Bounds, parameters, objective sense and result lifetime" begin
        m, x, c = scalar_model(0.0, Inf)
        optimize!(m)
        opt = unsafe_backend(m)
        previous = opt.native.problem
        set_upper_bound(x, 1.0) # lower-only -> mixed, rebuilding from original declarations
        @test result_count(m) == 0
        @test_throws MOI.GetAttributeNotAllowed MOI.get(
            opt,
            Pounce.ComplementarityStatistics(),
        )
        optimize!(m)
        @test opt.native.problem !== previous
        @test value(x) ≈ 0.5 atol = 1e-6
        delete_lower_bound(x)
        delete_upper_bound(x)
        optimize!(m)
        @test MOI.get(m, Pounce.ComplementarityStatistics()) === nothing
        @test value(x) ≈ 0.5 atol = 1e-6
        fix(x, 0.75)
        optimize!(m)
        @test value(x) ≈ 0.75
        @test value(c)[1] ≈ 0.5 # fixed variable permits arbitrary F

        m = direct_model(Pounce.Optimizer())
        set_silent(m)
        @variable(m, y >= 0, start = 0.5)
        @variable(m, p in Parameter(0.25))
        @constraint(m, cp, complements(y^2 - p, y))
        optimize!(m)
        old = unsafe_backend(m).native.problem
        @test value(y) ≈ 0.5 atol = 1e-6
        set_parameter_value(p, 0.36)
        @test result_count(m) == 0
        optimize!(m)
        @test unsafe_backend(m).native.problem === old
        @test value(y) ≈ 0.6 atol = 1e-6
        @test value(cp) ≈ [0, 0.6] atol = 1e-6
        # A success status alone must not certify the declared complementarity.
        prob = unsafe_backend(m).result.problem
        prob.x[1] = 1.0
        prob.g[end] = 0.64
        @test primal_status(m) == MOI.INFEASIBLE_POINT
    end

    @testset "Large interior values do not hide complementarity infeasibility" begin
        for (start, magnitude) in ((1e16, 1.0), (1e12, 1e-5)),
            f in (-magnitude, magnitude),
            (lower, upper) in ((-Inf, Inf), (0.0, Inf), (-Inf, 2start), (0.0, 2start))

            m = direct_model(Pounce.Optimizer())
            set_silent(m)
            @variable(m, x, start = start)
            isfinite(lower) && set_lower_bound(x, lower)
            isfinite(upper) && set_upper_bound(x, upper)
            c = @constraint(m, complements(f, x))
            set_optimizer_attribute(m, "max_iter", 0)
            optimize!(m)
            @test termination_status(m) == MOI.ITERATION_LIMIT
            @test value(x) == start
            @test value(c) == [f, start]
            # Every returned point is interior, so the original declaration
            # requires F = 0. Both magnitudes exceed the feasibility tolerances.
            @test primal_status(m) == MOI.INFEASIBLE_POINT
            @test Ext._complementarity_violation(unsafe_backend(m)) == magnitude
        end
    end

    @testset "Complementarity feasibility at bounds and nonfinite results" begin
        cases = (
            (-Inf, Inf, ((0.0, 1.0, 1.0), (0.0, -1.0, 1.0), (1e16, 0.0, 0.0))),
            (0.0, Inf, ((0.0, 1.0, 0.0), (0.0, -1.0, 1.0), (-1.0, 1.0, 1.0))),
            (-Inf, 1.0, ((1.0, -1.0, 0.0), (1.0, 1.0, 1.0), (2.0, -1.0, 1.0))),
            (
                0.0,
                1.0,
                (
                    (0.0, 1.0, 0.0),
                    (0.0, -1.0, 1.0),
                    (1.0, -1.0, 0.0),
                    (1.0, 1.0, 1.0),
                    (0.5, 2.0, 0.5),
                    (0.5, -2.0, 0.5),
                ),
            ),
            (0.5, 0.5, ((0.5, 1.0, 0.0), (0.5, -1.0, 0.0), (0.0, -1.0, 0.5))),
        )
        for (lower, upper, points) in cases
            m, _, _ = scalar_model(lower, upper)
            optimize!(m)
            opt = unsafe_backend(m)
            prob = opt.result.problem
            # Exercise the result getter with controlled returned x/F values,
            # independently of the native solver's bound pushes or convergence.
            for (x, f, violation) in points
                prob.x[1], prob.g[end] = x, f
                @test Ext._complementarity_violation(opt) == violation
                expected = violation == 0 ? MOI.FEASIBLE_POINT : MOI.INFEASIBLE_POINT
                @test primal_status(m) == expected
            end
            for (x, f) in ((NaN, 0.0), (Inf, 0.0), (0.5, NaN), (0.5, Inf))
                prob.x[1], prob.g[end] = x, f
                @test Ext._complementarity_violation(opt) == Inf
                @test primal_status(m) == MOI.INFEASIBLE_POINT
            end
        end
    end

    @testset "Signed multipliers, maximum sense, and no double counting" begin
        for maximize in (false, true), upper in (false, true)
            m = direct_model(Pounce.Optimizer())
            set_silent(m)
            @variable(m, 0 <= x <= 1, start = upper ? 1 : 0)
            @variable(m, y, start = 2)
            z = upper ? 1 - x : x
            @constraint(m, c, complements(upper ? z - y : y - z, x))
            f = (z - 0.5)^2 + (y - 2)^2
            @objective(m, Min, f)
            maximize && @objective(m, Max, -f)
            optimize!(m)
            @test value(z) ≈ 0 atol = 1e-6
            @test value(y) ≈ 2 atol = 1e-6
            @test dual(c)[2] == 0
            # The paired side is negative in the native z convention. MOI reverses
            # the upper-side sign, while retaining the actual side and sense rule.
            @test dual(upper ? UpperBoundRef(x) : LowerBoundRef(x)) ≈ (upper ? 1 : -1) atol =
                1e-5
            @test abs(dual(upper ? LowerBoundRef(x) : UpperBoundRef(x))) < 1e-6
            grad_x = upper ? 1.0 : -1.0
            jac_x = upper ? -1.0 : -1.0
            @test abs(
                grad_x - jac_x * dual(c)[1] - dual(LowerBoundRef(x)) -
                dual(UpperBoundRef(x)),
            ) < 1e-5
            @test abs(2(value(y) - 2) - (upper ? -1 : 1) * dual(c)[1]) < 1e-5
        end
    end

    @testset "Biactive pair and mixed ordinary rows" begin
        m = direct_model(Pounce.Optimizer())
        set_silent(m)
        @variable(m, x >= 0, start = 0)
        @variable(m, y >= 0, start = 0)
        @variable(m, z, start = 1)
        @constraint(m, linear, z >= 0.5)
        @constraint(m, quadratic, z^2 <= 2)
        @constraint(m, nonlinear, sin(z) >= 0)
        @constraint(m, c, complements(y, x))
        @objective(m, Min, x + y + (z - 1)^2)
        optimize!(m)
        @test maximum(abs.([value(x), value(y)])) < 1e-6
        @test value(z) ≈ 1 atol = 1e-5
        @test MOI.get(m, Pounce.ComplementarityStatistics()).n_biactive == 1
        @test value(linear) ≈ value(z)
        @test value(quadratic) ≈ value(z)^2
        @test value(nonlinear) ≈ sin(value(z))
        @test length(MOI.get(m, MOI.NLPBlockDual())) == 1
        @test length(value(c)) == 2
        @test isfinite(dual(nonlinear))
    end

    @testset "Oracle and legacy blocks keep their row and derivative offsets" begin
        for approximation in ("exact", "limited-memory", "finite-difference")
            m = Model(Pounce.Optimizer)
            set_silent(m)
            set_optimizer_attribute(m, "hessian_approximation", approximation)
            @variable(m, x >= 0, start = 0.5)
            @variable(m, y, start = 1)
            @NLconstraint(m, legacy, y^2 <= 4)
            set = MOI.VectorNonlinearOracle(;
                dimension = 1,
                l = [1.0],
                u = [Inf],
                eval_f = (ret, x) -> (ret[1] = x[1]; nothing),
                jacobian_structure = [(1, 1)],
                eval_jacobian = (ret, x) -> (ret[1] = 1.0; nothing),
                hessian_lagrangian_structure = Tuple{Int,Int}[],
                eval_hessian_lagrangian = (ret, x, mu) -> nothing,
            )
            @constraint(m, oracle, [y] in set)
            @constraint(m, c, complements(x^3 - 0.125, x))
            @objective(m, Min, y^2)
            optimize!(m)
            @test value(x) ≈ 0.5 atol = 1e-5
            @test value(y) ≈ 1 atol = 1e-5
            @test value(c) ≈ [0, 0.5] atol = 1e-5
            @test abs(dual(legacy)) < 1e-5
            @test MOI.get(backend(m), MOI.LagrangeMultiplier(), index(oracle)) ≈ [2.0] atol =
                1e-5
            @test length(MOI.get(m, MOI.NLPBlockDual())) == 1
        end
    end

    @testset "Registered operators and callback guards" begin
        m = direct_model(Pounce.Optimizer())
        set_silent(m)
        @variable(m, x >= 0, start = 0.5)
        square = (x) -> x^2
        MOI.set(
            backend(m),
            MOI.UserDefinedFunction(:cc_square, 1),
            (square, x -> 2x, x -> 2.0),
        )
        c = MOI.add_constraint(
            backend(m),
            MOI.VectorNonlinearFunction([
                MOI.ScalarNonlinearFunction(
                    :-,
                    Any[MOI.ScalarNonlinearFunction(:cc_square, Any[index(x)]), 0.25],
                ),
                index(x),
            ]),
            MOI.Complements(2),
        )
        optimize!(m)
        @test value(x) ≈ 0.5 atol = 1e-6
        calls = Ref(0)
        opt = unsafe_backend(m)
        MOI.set(
            m,
            Pounce.CallbackFunction(),
            function (args...)
                calls[] += 1
                @test_throws ErrorException Pounce.ClearPounceComplementarityPairs(
                    opt.native.problem,
                )
                @test_throws ErrorException Pounce.GetPounceComplementarityStats(
                    opt.native.problem,
                )
                @test_throws ErrorException MOI.add_constraint(
                    opt,
                    MOI.VectorOfVariables([index(x), index(x)]),
                    MOI.Complements(2),
                )
                return false
            end,
        )
        optimize!(m)
        @test calls[] > 0
    end

    @testset "MOI quadratic complementarity example with explicit objective sense" begin
        # MOI.Test.test_nonlinear_qp_complementarity_constraint omits ObjectiveSense.
        # Its expected minimizer only applies when MIN_SENSE is explicitly set.
        m = direct_model(Pounce.Optimizer())
        set_silent(m)
        @variable(m, x[1:8] >= 0, start = 0)
        @objective(m, Min, (x[1] - 5)^2 + (2x[2] + 1)^2)
        @constraint(m, -1.5x[1] + 2x[2] + x[3] + 0.5x[4] + x[5] == 2)
        @constraint(m, 3x[1] - x[2] - x[6] == 3)
        @constraint(m, -x[1] + 0.5x[2] - x[7] == -4)
        @constraint(m, -x[1] - x[2] - x[8] == -7)
        @constraint(m, complements(x[3:5], x[6:8]))
        optimize!(m)
        @test termination_status(m) == MOI.LOCALLY_SOLVED
        @test value.(x) ≈ [1, 0, 3.5, 0, 0, 0, 3, 6] atol = 1e-5
        @test objective_value(m) ≈ 17 atol = 1e-5
        @test maximum(abs.(min.(value.(x[3:5]), value.(x[6:8])))) < 1e-6
        # Keep feasibility sense faithful to MOI, even when an objective is stored.
        set_objective_sense(m, MOI.FEASIBILITY_SENSE)
        optimize!(m)
        @test objective_value(m) == 0
        @test primal_status(m) in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
    end

    @testset "Native batch errors preserve declarations and report one-based position" begin
        m = direct_model(Pounce.Optimizer())
        set_silent(m)
        @variable(m, 0 <= x[1:2] <= 1, start = 0.5)
        @constraint(m, complements(2 .* x .- 1, x))
        opt = unsafe_backend(m)
        Ext._setup_model(opt)
        prob = Ext._setup_inner(opt)
        before = Pounce.GetPounceComplementarityPairs(prob)
        err = try
            Pounce.SetPounceComplementarityPairs(prob, [1, 2], [1, 1])
        catch e
            e
        end
        @test err isa Pounce.NativeAPIError
        @test err.code == Pounce.POUNCE_ERR_VAR_ALREADY_PAIRED
        @test err.bad_index == 2
        @test Pounce.GetPounceComplementarityPairs(prob) == before
        optimize!(m)
        @test value.(x) ≈ [0.5, 0.5] atol = 1e-6
    end

    @testset "C API, finalized flags ABI, errors and owned diagnostics" begin
        m, x, _ = scalar_model(0.0, 1.0)
        opt = unsafe_backend(m)
        Ext._setup_model(opt)
        prob = Ext._setup_inner(opt)
        @test Pounce.GetPounceComplementarityPairCount(prob) == 1
        @test Pounce.GetPounceComplementarityPairs(prob) == (rows = [1], variables = [1])
        @test_throws ErrorException Pounce.GetPounceComplementarityStats(prob)
        @test_throws ArgumentError Pounce.AddPounceComplementsConstraint(prob, 0, 1)
        err = try
            Pounce.AddPounceComplementsConstraint(prob, 1, 1)
        catch e
            e
        end
        @test err isa Pounce.NativeAPIError
        @test err.code == Pounce.POUNCE_ERR_VAR_ALREADY_PAIRED
        @test occursin("already", sprint(showerror, err))
        @test Pounce.PounceErrorString(-999) == "unknown error code"
        @test_throws DimensionMismatch Pounce.SetPounceComplementarityPairs(
            prob,
            [1],
            Int[],
        )
        @test_throws ArgumentError Pounce.SetPounceComplementarityPairs(prob, [1], [2])
        # Clear/redeclare native pairs: this preserves the previously lowered row.
        Pounce.ClearPounceComplementarityPairs(prob)
        @test Pounce.GetPounceComplementarityPairCount(prob) == 0
        @test Pounce.SetPounceComplementarityPairs(prob, [1], [1]) == 0
        optimize!(m)
        stats = Pounce.GetPounceComplementarityStats(prob)
        @test stats.n_solves > 0
        @test Pounce.GetPounceComplementaritySolve(prob, stats.n_solves + 1) === nothing
        @test_throws ArgumentError Pounce.GetPounceComplementaritySolve(prob, 0)
        @test Pounce.GetPounceComplementarityText(prob, -99) === nothing
        @test_throws MOI.ResultIndexBoundsError MOI.get(
            opt,
            MOI.ConstraintDual(2),
            MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64},MOI.Complements}(1),
        )
        Pounce.FreeIpoptProblem(prob)
        @test stats.n_solves > 0 # returned values own their memory
        @test_throws ErrorException Pounce.GetPounceComplementarityStats(prob)
        @test_throws ErrorException Pounce.ClearPounceComplementarityPairs(prob)
    end
end # capability
end # module
