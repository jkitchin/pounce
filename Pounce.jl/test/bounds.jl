module TestBounds
using Test, JuMP
import Pounce
const MOI = JuMP.MOI
const Ext = Base.get_extension(Pounce, :PounceMathOptInterfaceExt)

function scalar_model(kind, upper; cached = false)
    m = cached ? Model(Pounce.Optimizer) : direct_model(Pounce.Optimizer())
    set_silent(m)
    @variable(m, x >= 0, start = 1)
    f = kind == :variable ? x : kind == :affine ? 2x : kind == :quadratic ? x^2 : exp(x)
    c = if kind == :variable
        set_upper_bound(x, upper)
        UpperBoundRef(x)
    else
        @constraint(m, f <= upper)
    end
    @objective(m, Max, x)
    return m, x, c
end

@testset "Coalesced bounds match a rebuilt model" begin
    for kind in (:variable, :affine, :quadratic, :nonlinear), cached in (false, true)
        m, x, c = scalar_model(kind, 4.0; cached)
        optimize!(m)
        opt = unsafe_backend(m)
        old, evaluator = opt.native.problem, opt.evaluator.nlp_data.evaluator
        old_bounds = Pounce.supports_bound_updates() ? Pounce.GetPounceBounds(old) : nothing
        offset = MOI.get(m, MOI.ConstraintSet(), c).upper - 4.0
        for upper in (8.0, 9.0)
            MOI.set(m, MOI.ConstraintSet(), c, MOI.LessThan(upper + offset))
        end
        @test result_count(m) == 0
        @test MOI.get(opt, MOI.PrimalStatus()) == MOI.NO_SOLUTION
        @test_throws MOI.GetAttributeNotAllowed MOI.get(opt, Pounce.NativeResiduals())
        if Pounce.supports_bound_updates()
            @test opt.native.problem === old
            @test length(opt.native.pending_x_bounds) +
                  length(opt.native.pending_g_bounds) == 1
            @test Pounce.GetPounceBounds(old) == old_bounds # no native write before solve
        else
            @test old.ipopt_problem == C_NULL
        end
        optimize!(m)
        @test (opt.native.problem === old) == Pounce.supports_bound_updates()
        @test opt.evaluator.nlp_data.evaluator === evaluator
        fresh, y, d = scalar_model(kind, 9.0; cached)
        optimize!(fresh)
        @test termination_status(m) == termination_status(fresh) == MOI.LOCALLY_SOLVED
        @test value(x) ≈ value(y) atol = 1e-6
        @test dual(c) ≈ dual(d) atol = 1e-6
        @test isempty(opt.native.pending_x_bounds)
        @test isempty(opt.native.pending_g_bounds)
    end
end

@testset "Bounds coexist with parameters, options and structural changes" begin
    m = direct_model(Pounce.Optimizer())
    set_silent(m)
    @variable(m, 0 <= x <= 3, start = 1)
    @variable(m, p in Parameter(2.0))
    @objective(m, Min, (x - p)^2)
    set_optimizer_attribute(m, "hessian_approximation", "limited-memory")
    optimize!(m)
    opt = unsafe_backend(m)
    old = opt.native.problem
    set_upper_bound(x, 1.0)
    set_parameter_value(p, 2.5) # must not lose the pending bound edit
    optimize!(m)
    @test value(x) ≈ 1 atol = 1e-6
    @test (opt.native.problem === old) == Pounce.supports_bound_updates()
    @test Ext._nonlinear_variables(opt) == [1]
    old = opt.native.problem
    delete_upper_bound(x)
    set_upper_bound(x, 2.0)
    set_optimizer_attribute(m, "tol", 1e-9) # rebuild must incorporate pending bounds
    optimize!(m)
    @test value(x) ≈ 2 atol = 1e-6
    @test opt.native.problem !== old
    @test old.ipopt_problem == C_NULL
    old = opt.native.problem
    fix(x, 1.5; force = true)
    optimize!(m)
    @test value(x) ≈ 1.5
    @test (opt.native.problem === old) == Pounce.supports_bound_updates()
    unfix(x)
    @constraint(m, x <= 1.25) # bound addition reuses an ordinary NLP handle
    @variable(m, y == 0) # structural change discards pending edits
    optimize!(m)
    @test value(x) ≈ 1.25 atol = 1e-6
    @test value(y) ≈ 0
    @test old.ipopt_problem == C_NULL
end

if Pounce.supports_bound_updates()
    @testset "Explicit starts survive handle-preserving bound updates" begin
        m = direct_model(Pounce.Optimizer())
        set_silent(m)
        set_optimizer_attribute(m, "presolve", "no")
        set_optimizer_attribute(m, "warm_start_init_point", "yes")
        @variable(m, 0 <= x <= 4, start = 0.5)
        c = @constraint(m, 2x <= 10)
        @objective(m, Min, (x - 2)^2)
        optimize!(m)
        opt = unsafe_backend(m)
        old = opt.native.problem
        set_upper_bound(x, 1.5)
        set_start_value(x, 0.75)
        set_dual_start_value(LowerBoundRef(x), 10.0)
        set_dual_start_value(UpperBoundRef(x), -20.0)
        set_dual_start_value(c, -30.0)
        inputs = NTuple{4,Vector{Float64}}[]
        # Native output buffers are written after callbacks return; observe the
        # explicit Julia input buffers at the C boundary, including MOI dual signs.
        MOI.set(
            opt,
            Pounce.CallbackFunction(),
            (args...) -> begin
                p = opt.native.problem
                push!(
                    inputs,
                    (copy(p.x), copy(p.mult_x_L), copy(p.mult_x_U), copy(p.mult_g)),
                )
                false
            end,
        )
        optimize!(m)
        @test !isempty(inputs)
        @test first(inputs) == ([0.75], [10.0], [20.0], [30.0])
        @test opt.native.problem === old
        @test Pounce.GetPounceBounds(old).x_upper == [1.5]
    end

    @testset "Failed native updates retain the batch for correction" begin
        m, x, c = scalar_model(:affine, 4.0)
        optimize!(m)
        opt = unsafe_backend(m)
        old = opt.native.problem
        before = Pounce.GetPounceBounds(old)
        set_lower_bound(x, 0.5)
        MOI.set(m, MOI.ConstraintSet(), c, MOI.LessThan(NaN))
        @test_throws Pounce.NativeAPIError optimize!(m)
        @test result_count(m) == 0
        @test opt.native.problem === old
        @test Pounce.GetPounceBounds(old) == before
        MOI.set(m, MOI.ConstraintSet(), c, MOI.LessThan(6.0))
        optimize!(m)
        @test opt.native.problem === old
        @test value(x) ≈ 3 atol = 1e-6
        @test Pounce.GetPounceBounds(old).x_lower == [0.5]
    end

    @testset "Crossed ordinary bounds are reported by solve and can recover" begin
        m, x, _ = scalar_model(:variable, 4.0)
        optimize!(m)
        old = unsafe_backend(m).native.problem
        set_lower_bound(x, 5.0)
        optimize!(m)
        fresh, y, _ = scalar_model(:variable, 4.0)
        set_lower_bound(y, 5.0)
        optimize!(fresh)
        @test termination_status(m) == termination_status(fresh)
        @test primal_status(m) == primal_status(fresh)
        @test unsafe_backend(m).native.problem === old
        set_lower_bound(x, 0.0)
        optimize!(m)
        @test termination_status(m) == MOI.LOCALLY_SOLVED
        @test value(x) ≈ 4 atol = 1e-6
        @test unsafe_backend(m).native.problem === old
    end

    @testset "Native bound API ownership, sparse/dense reads and atomic refusals" begin
        m = direct_model(Pounce.Optimizer())
        set_silent(m)
        @variable(m, 0 <= x[1:2] <= 4, start = 1)
        @constraint(m, x[1] + x[2] <= 6)
        @constraint(m, x[1] - x[2] <= 3)
        @objective(m, Min, sum((x .- 1) .^ 2))
        optimize!(m)
        prob = unsafe_backend(m).native.problem
        before = Pounce.GetPounceBounds(prob)
        @test before.x_lower == [0, 0]
        @test before.g_upper == [6, 3]
        @test Pounce.SetPounceBounds(prob; x_upper = [3.0, 2.0]) == 0
        @test Pounce.SetPounceBounds(
            prob;
            x_indices = [2],
            x_lower = [0.5],
            g_indices = [2],
            g_upper = [2.5],
        ) == 0
        got = Pounce.GetPounceBounds(prob; x_indices = [2, 1, 2], g_indices = [2, 1, 2])
        @test got.x_lower == [0.5, 0, 0.5]
        @test got.x_upper == [2, 3, 2]
        @test got.g_upper == [2.5, 6, 2.5]
        @test before.x_upper == [4, 4] # owned arrays
        @test isempty(
            Pounce.GetPounceBounds(prob; x_indices = Int[], g_indices = Int[]).g_lower,
        )
        @test Pounce.SetPounceBounds(prob) == 0
        before = Pounce.GetPounceBounds(prob)
        for (args, code, bad) in (
            ((x_indices = [1, 1], x_lower = [1.0, 2.0]), Pounce.POUNCE_ERR_DUPLICATE, 2),
            (
                (x_indices = [1], x_upper = [2.0], g_indices = [2], g_lower = [NaN]),
                Pounce.POUNCE_ERR_NAN,
                2,
            ),
        )
            err = try
                Pounce.SetPounceBounds(prob; args...)
            catch e
                e
            end
            @test err isa Pounce.NativeAPIError
            @test err.code == code
            @test err.bad_index == bad
            @test Pounce.GetPounceBounds(prob) == before
        end
        @test_throws DimensionMismatch Pounce.SetPounceBounds(prob; x_lower = [0.0])
        @test_throws ArgumentError Pounce.SetPounceBounds(prob; x_indices = [0])
        @test_throws ArgumentError Pounce.GetPounceBounds(prob; g_indices = [3])
        @test_throws Pounce.NativeAPIError Pounce.GetPounceBounds(prob; view = -1)
        Pounce.FreeIpoptProblem(prob)
        @test_throws ErrorException Pounce.GetPounceBounds(prob)
        @test_throws ErrorException Pounce.SetPounceBounds(prob)
    end

    @testset "Callback bound access is refused before mutation" begin
        m, x, c = scalar_model(:nonlinear, 4.0)
        opt = unsafe_backend(m)
        calls = Ref(0)
        MOI.set(
            opt,
            Pounce.CallbackFunction(),
            function (args...)
                calls[] += 1
                @test_throws ErrorException Pounce.SetPounceBounds(
                    opt.native.problem;
                    x_upper = [0.5],
                )
                @test_throws ErrorException Pounce.GetPounceBounds(opt.native.problem)
                @test_throws ErrorException MOI.set(
                    m,
                    MOI.ConstraintSet(),
                    c,
                    MOI.LessThan(9.0),
                )
                return false
            end,
        )
        optimize!(m)
        @test calls[] > 0
        @test MOI.get(m, MOI.ConstraintSet(), c) == MOI.LessThan(0.0)
    end
else
    @testset "Older libraries reject optional bound calls" begin
        m, _, _ = scalar_model(:affine, 4.0)
        optimize!(m)
        prob = unsafe_backend(m).native.problem
        @test_throws ErrorException Pounce.GetPounceBounds(prob)
        @test_throws ErrorException Pounce.SetPounceBounds(prob)
    end
end

if Pounce.supports_bound_updates() && Pounce.supports_complementarity()
    function complementarity_transition_model(bounds)
        m = Pounce.Optimizer()
        MOI.set(m, MOI.Silent(), true)
        x, b = MOI.add_constrained_variable(m, MOI.Interval(bounds...))
        MOI.set(m, MOI.VariablePrimalStart(), x, 0.5)
        f = MOI.VectorAffineFunction(
            [
                MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x)),
                MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x)),
            ],
            [-1.0, 0.0],
        )
        MOI.add_constraint(m, f, MOI.Complements(2))
        return m, x, b
    end
    @testset "Every complementarity class transition versus a fresh declaration" begin
        classes = [(-1.0, Inf), (-Inf, 3.0), (-1.0, 3.0), (-Inf, Inf), (0.5, 0.5)]
        for from in classes, to in classes
            m, x, b = complementarity_transition_model(from)
            MOI.optimize!(m)
            old = m.native.problem
            evaluator = m.evaluator.complementarity
            MOI.set(m, MOI.ConstraintSet(), b, MOI.Interval(to...))
            @test MOI.get(m, MOI.ResultCount()) == 0
            MOI.optimize!(m)
            fresh, y, _ = complementarity_transition_model(to)
            MOI.optimize!(fresh)
            @test MOI.get(m, MOI.VariablePrimal(), x) ≈
                  MOI.get(fresh, MOI.VariablePrimal(), y) atol = 1e-6
            @test MOI.get(m, MOI.PrimalStatus()) == MOI.get(fresh, MOI.PrimalStatus())
            @test (m.native.problem === old) == (from == to)
            @test m.evaluator.complementarity === evaluator
            @test Pounce.GetPounceBounds(m.native.problem) ==
                  Pounce.GetPounceBounds(fresh.native.problem)
            @test Pounce.GetPounceBounds(
                m.native.problem;
                view = Pounce.POUNCE_BOUNDS_VIEW_EFFECTIVE,
            ) == Pounce.GetPounceBounds(
                fresh.native.problem;
                view = Pounce.POUNCE_BOUNDS_VIEW_EFFECTIVE,
            )
            MOI.empty!(m)
            MOI.empty!(fresh)
        end
    end

    function mixed_model(final, approximation)
        m = direct_model(Pounce.Optimizer())
        set_silent(m)
        set_optimizer_attribute(m, "hessian_approximation", approximation)
        @variable(m, 0 <= x[1:4] <= 5, start = 0.5)
        @variable(m, p in Parameter(final ? 0.75 : 0.5))
        a = @constraint(m, x[1] <= (final ? 3.0 : 2.0))
        q = @constraint(m, x[3]^2 <= (final ? 9.0 : 4.0))
        oracle = MOI.VectorNonlinearOracle(;
            dimension = 1,
            l = [-Inf],
            u = [4.0],
            eval_f = (ret, z) -> (ret[1] = z[1]; nothing),
            jacobian_structure = [(1, 1)],
            eval_jacobian = (ret, z) -> (ret[1] = 1.0; nothing),
            hessian_lagrangian_structure = Tuple{Int,Int}[],
            eval_hessian_lagrangian = (ret, z, mu) -> nothing,
        )
        @constraint(m, [x[3]] in oracle)
        n = @constraint(m, exp(x[2]) <= (final ? exp(1.0) : exp(2.0)))
        @constraint(m, complements(x[4] - p, x[4]))
        @objective(m, Max, x[1] + x[2] + x[3])
        return m, x, p, a, q, n
    end

    @testset "Mixed row offsets, parameter updates and pending variable bounds" begin
        for approximation in ("exact", "limited-memory")
            m, x, p, a, q, n = mixed_model(false, approximation)
            optimize!(m)
            opt = unsafe_backend(m)
            old = opt.native.problem
            snapshot = Pounce.GetPounceBounds(old)
            set_normalized_rhs(a, 3.0)
            set_normalized_rhs(q, 9.0)
            s = MOI.get(m, MOI.ConstraintSet(), n)
            MOI.set(m, MOI.ConstraintSet(), n, MOI.LessThan(s.upper + exp(1.0) - exp(2.0)))
            set_upper_bound(x[4], 4.0) # same boxed complementarity class
            set_parameter_value(p, 0.75)
            optimize!(m)
            @test opt.native.problem === old
            fresh, y, _, fa, fq, fn = mixed_model(true, approximation)
            set_upper_bound(y[4], 4.0)
            optimize!(fresh)
            @test termination_status(m) == termination_status(fresh) == MOI.LOCALLY_SOLVED
            @test value.(x) ≈ [3.0, 1.0, 3.0, 0.75] atol = 1e-5
            @test value.(x) ≈ value.(y) atol = 1e-5
            @test dual.([a, q, n]) ≈ dual.([fa, fq, fn]) atol = 1e-5
            current = Pounce.GetPounceBounds(old)
            @test current.g_upper[3] == snapshot.g_upper[3] == 4 # oracle untouched
            @test current.g_lower[end] == -Inf # complementarity USER row stays free
            @test current.g_upper[end] == Inf
            @test current.x_upper[4] == 4
        end
    end

    @testset "Complementarity class changes rebuild; same-class edits reuse" begin
        m = direct_model(Pounce.Optimizer())
        set_silent(m)
        @variable(m, x >= 0, start = 0.5)
        @constraint(m, complements(2x - 1, x))
        optimize!(m)
        opt = unsafe_backend(m)
        old = opt.native.problem
        @test Pounce.GetPounceBounds(old).g_lower == [-Inf]
        @test Pounce.GetPounceBounds(old; view = Pounce.POUNCE_BOUNDS_VIEW_EFFECTIVE).g_lower ==
              [0]
        set_lower_bound(x, -0.25)
        optimize!(m)
        @test opt.native.problem === old
        @test value(x) ≈ 0.5 atol = 1e-6
        # A transient class change disappears before the batch is applied.
        set_upper_bound(x, 2.0)
        delete_upper_bound(x)
        optimize!(m)
        @test opt.native.problem === old
        for kind in (:boxed, :upper, :free, :fixed)
            old = opt.native.problem
            if kind == :boxed
                set_upper_bound(x, 2.0)
            elseif kind == :upper
                delete_lower_bound(x)
            elseif kind == :free
                delete_upper_bound(x)
            else
                fix(x, 0.75)
            end
            @test result_count(m) == 0
            optimize!(m)
            @test opt.native.problem !== old
            @test old.ipopt_problem == C_NULL
            @test value(x) ≈ (kind == :fixed ? 0.75 : 0.5) atol = 1e-6
        end
        old = opt.native.problem
        fix(x, 0.8)
        optimize!(m)
        @test opt.native.problem === old
        @test value(x) ≈ 0.8
    end
end
end # module
