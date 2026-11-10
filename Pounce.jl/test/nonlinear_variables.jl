module TestNonlinearVariables

using Test, JuMP
import Pounce
const MOI = JuMP.MOI
const Ext = Base.get_extension(Pounce, :PounceMathOptInterfaceExt)

function nonlinear_variables(model)
    optimizer = unsafe_backend(model)
    Ext._setup_model(optimizer)
    return Ext._nonlinear_variables(optimizer)
end

# The existing native nonlinear_variables_mask.rs fixture, with the nonlinear
# variables permuted into noncontiguous Julia columns 2 and 5.
function mixed_model(; approximation = "limited-memory", automatic = true, cached = false)
    model = cached ? Model(Pounce.Optimizer) : direct_model(Pounce.Optimizer())
    set_silent(model)
    set_optimizer_attribute(model, "hessian_approximation", approximation)
    set_optimizer_attribute(model, "tol", 1e-9)
    if !automatic
        set_optimizer_attribute(model, "num_linear_variables", 0)
    end
    @variable(model, -5 <= x[1:6] <= 5)
    set_start_value.(x, [1, 0, 1, 0, 0, 0])
    @objective(
        model,
        Min,
        (1 - x[2])^2 + 100 * (x[5] - x[2]^2)^2 + 2 * x[1] + 3 * x[3] + x[4] + x[6]
    )
    @constraint(model, x[2] + x[5] + x[1] + x[3] == 4)
    @constraint(model, x[2]^2 + x[5]^2 - x[4] - x[6] <= 1)
    return model, x
end

function solve_with_trace(model)
    trace = Tuple[]
    MOI.set(model, Pounce.CallbackFunction(), function (args...)
        push!(trace, args[2:5]) # iteration, objective, primal and dual residuals
        return true
    end)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    return trace
end

@testset "Nonlinear-variable C bindings" begin
    @test Pounce.supports_nonlinear_variable_hints()
    model, _ = mixed_model(; automatic = false)
    optimizer = unsafe_backend(model)
    Ext._setup_model(optimizer)
    inner = Ext._setup_inner(optimizer)
    positions = [5, 2]
    @test Pounce.IpoptSetNonlinearVariables(inner, positions) === nothing
    positions .= 0 # The native declaration owns a copy.
    for invalid in ([0], [-1], [7], [2, 2], fill(1, 7), [typemax(UInt)])
        @test_throws ArgumentError Pounce.IpoptSetNonlinearVariables(inner, invalid)
    end
    hinted = solve_with_trace(model)
    @test Pounce.IpoptClearNonlinearVariables(inner) === nothing
    cleared = solve_with_trace(model)
    reference, _ = mixed_model(; automatic = false)
    full = solve_with_trace(reference)
    @test cleared == full
    @test hinted != full # The native declaration changes the approximation space.
    @test Pounce.IpoptSetNonlinearVariables(inner, 1:6) === nothing
    @test solve_with_trace(model) == full
    calls = Ref(0)
    MOI.set(
        optimizer,
        Pounce.CallbackFunction(),
        function (args...)
            calls[] += 1
            @test_throws ErrorException Pounce.IpoptSetNonlinearVariables(inner, [2, 5])
            @test_throws ErrorException Pounce.IpoptClearNonlinearVariables(inner)
            return false
        end,
    )
    optimize!(model)
    @test termination_status(model) == MOI.INTERRUPTED
    @test calls[] == 1
    Pounce.FreeIpoptProblem(inner)
    @test_throws ErrorException Pounce.IpoptSetNonlinearVariables(inner, [2, 5])
    @test_throws ErrorException Pounce.IpoptClearNonlinearVariables(inner)
end

@testset "Automatic hints reach the native L-BFGS solve" begin
    auto, x = mixed_model()
    @test nonlinear_variables(auto) == [2, 5]
    auto_trace = solve_with_trace(auto)
    manual, y = mixed_model(; automatic = false)
    optimizer = unsafe_backend(manual)
    Ext._setup_model(optimizer)
    inner = Ext._setup_inner(optimizer)
    Pounce.IpoptSetNonlinearVariables(inner, [2, 5])
    @test solve_with_trace(manual) == auto_trace
    @test value.(x) == value.(y)
    full, z = mixed_model(; automatic = false)
    @test solve_with_trace(full) != auto_trace
    @test value.(x) ≈ value.(z) atol = 1e-5
    @test objective_value(auto) ≈ objective_value(full) atol = 1e-5
    cached, cached_x = mixed_model(; cached = true)
    @test !isempty(solve_with_trace(cached))
    @test Ext._nonlinear_variables(unsafe_backend(cached)) == [2, 5]
    # Cache copying may reorder expression evaluation on older MOI versions.
    @test value.(cached_x) ≈ value.(x) atol = 1e-5
    # No automatic declaration is made outside limited-memory mode.
    for approximation in ("exact", "finite-difference")
        plain, _ = mixed_model(; approximation, automatic = false)
        detected, _ = mixed_model(; approximation)
        @test solve_with_trace(plain) == solve_with_trace(detected)
        @test objective_value(auto) ≈ objective_value(plain) atol = 1e-5
    end
end

@testset "Structural sources, not numerical Hessian values" begin
    model = direct_model(Pounce.Optimizer())
    @variable(model, x[1:6])
    @objective(model, Min, x[2] * x[5] + x[1])
    @constraint(model, x[3]^2 <= 1)
    @test nonlinear_variables(model) == [2, 3, 5] # Both off-diagonal endpoints.

    legacy = direct_model(Pounce.Optimizer())
    @variable(legacy, y[1:4])
    @NLobjective(legacy, Min, y[2]^4 + y[1])
    @NLconstraint(legacy, sin(y[4]) <= 0)
    # JuMP normally transfers the legacy NLP block immediately before optimize!.
    MOI.set(unsafe_backend(legacy), MOI.NLPBlock(), MOI.NLPBlockData(NLPEvaluator(legacy)))
    @test nonlinear_variables(legacy) == [2, 4] # Hessian is zero at x == 0.

    for second_order in (true, false)
        oracle = direct_model(Pounce.Optimizer())
        @variable(oracle, z[1:5])
        @objective(oracle, Min, z[1]^2)
        set = MOI.VectorNonlinearOracle(;
            dimension = 2,
            l = [-Inf],
            u = [1.0],
            eval_f = (ret, x) -> (ret[1] = x[1] * x[2]; nothing),
            jacobian_structure = [(1, 1), (1, 2)],
            eval_jacobian = (ret, x) -> (ret .= (x[2], x[1]); nothing),
            hessian_lagrangian_structure = second_order ? [(2, 1)] : Tuple{Int,Int}[],
            eval_hessian_lagrangian = second_order ?
                                      ((ret, x, mu) -> (ret[1] = mu[1]; nothing)) :
                                      nothing,
        )
        @constraint(oracle, [z[5], z[2]] in set)
        @test nonlinear_variables(oracle) == (second_order ? [1, 2, 5] : nothing)
    end
end

struct FirstOrderEvaluator <: MOI.AbstractNLPEvaluator end
MOI.features_available(::FirstOrderEvaluator) = [:Grad, :Jac]
function MOI.initialize(::FirstOrderEvaluator, features)
    @test :Hess ∉ features
    return
end
MOI.eval_objective(::FirstOrderEvaluator, x) = (x[2] - 1)^2
function MOI.eval_objective_gradient(::FirstOrderEvaluator, g, x)
    fill!(g, 0.0)
    g[2] = 2 * (x[2] - 1)
    return
end
MOI.jacobian_structure(::FirstOrderEvaluator) = Tuple{Int,Int}[]
MOI.eval_constraint(::FirstOrderEvaluator, g, x) = nothing
MOI.eval_constraint_jacobian(::FirstOrderEvaluator, J, x) = nothing
MOI.hessian_lagrangian_structure(::FirstOrderEvaluator) = error("No Hessian available")

@testset "Unknown NLP structure retains the full space" begin
    traces = []
    for automatic in (true, false)
        model = direct_model(Pounce.Optimizer())
        set_silent(model)
        @variable(model, -5 <= x[1:3] <= 5)
        MOI.set(
            unsafe_backend(model),
            MOI.NLPBlock(),
            MOI.NLPBlockData(MOI.NLPBoundsPair[], FirstOrderEvaluator(), true),
        )
        MOI.set(unsafe_backend(model), MOI.ObjectiveSense(), MOI.MIN_SENSE)
        if !automatic
            set_optimizer_attribute(model, "num_linear_variables", 0)
        end
        @test nonlinear_variables(model) === nothing
        push!(traces, solve_with_trace(model))
        @test value(x[2]) ≈ 1 atol = 1e-6
    end
    @test traces[1] == traces[2]
end

@testset "Empty hints, parameters and structural rebuilds" begin
    model = direct_model(Pounce.Optimizer())
    set_silent(model)
    set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    @variable(model, 0 <= x[1:3] <= 4)
    @objective(model, Min, sum(x))
    @test nonlinear_variables(model) == Int[]
    solve_with_trace(model)
    @test value.(x) ≈ zeros(3) atol = 1e-6
    old = unsafe_backend(model).native.problem
    @objective(model, Min, x[1] + (x[3] - 2)^2)
    @test nonlinear_variables(model) == [3]
    solve_with_trace(model)
    @test old.ipopt_problem == C_NULL
    @test value(x[3]) ≈ 2 atol = 1e-6

    @variable(model, p in Parameter(0.0))
    @objective(model, Min, x[1] + p * (x[2] - 1)^2)
    # The combined Hessian also retains the now-inactive quadratic objective's
    # structural slot. Including it is conservative; omitting column 2 is not.
    @test nonlinear_variables(model) == [2, 3]
    solve_with_trace(model)
    inner = unsafe_backend(model).native.problem
    set_parameter_value(p, 1.0)
    solve_with_trace(model)
    @test unsafe_backend(model).native.problem === inner
    @test Ext._nonlinear_variables(unsafe_backend(model)) == [2, 3]
    @test value(x[2]) ≈ 1 atol = 1e-6
    set_upper_bound(x[2], 0.5)
    solve_with_trace(model)
    @test (inner.ipopt_problem != C_NULL) == Pounce.supports_bound_updates()
    @test Ext._nonlinear_variables(unsafe_backend(model)) == [2, 3]
    @test value(x[2]) ≈ 0.5 atol = 1e-6
end

@testset "Missing optional symbols retain the full space" begin
    reference, _ = mixed_model(; automatic = false)
    full = solve_with_trace(reference)
    # Simulate a library without the optional entry points. Existing required
    # bindings still use libpounce, so the fallback is tested with a real solve.
    original = Pounce._library_handle[]
    missing = Pounce.Libdl.dlopen(Pounce.Libdl.dlpath("libjulia"))
    try
        Pounce._library_handle[] = missing
        @test !Pounce.supports_nonlinear_variable_hints()
        model, _ = mixed_model()
        @test solve_with_trace(model) == full
        inner = unsafe_backend(model).native.problem
        @test_throws ErrorException Pounce.IpoptSetNonlinearVariables(inner, [2, 5])
        @test_throws ErrorException Pounce.IpoptClearNonlinearVariables(inner)
    finally
        Pounce._library_handle[] = original
        Pounce.Libdl.dlclose(missing)
    end
end

@testset "Fixed columns and empty reduced nonlinear space" begin
    # Native compression must drop fixed columns from the declared subset,
    # including the case where no nonlinear columns remain.
    for columns in ([2], [2, 5], [1, 2, 5])
        points = Vector{Float64}[]
        objectives = Float64[]
        for automatic in (true, false)
            model, x = mixed_model(; automatic)
            for i in columns
                fix(x[i], 1.0; force = true)
            end
            solve_with_trace(model)
            push!(points, value.(x))
            push!(objectives, objective_value(model))
            @test abs(sum(value.(x[[2, 5, 1, 3]])) - 4) < 1e-6
            @test value(x[2])^2 + value(x[5])^2 - value(x[4]) - value(x[6]) <= 1 + 1e-6
        end
        @test objectives[1] ≈ objectives[2] atol = 1e-6
        @test points[1] ≈ points[2] atol = 1e-5
    end
end

@testset "Option changes release previous declarations" begin
    model, _ = mixed_model()
    solve_with_trace(model)
    for approximation in ("exact", "limited-memory", "finite-difference")
        old = unsafe_backend(model).native.problem
        set_optimizer_attribute(model, "hessian_approximation", approximation)
        trace = solve_with_trace(model)
        reference, _ = mixed_model(; approximation)
        @test trace == solve_with_trace(reference)
        @test old.ipopt_problem == C_NULL
    end
    set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
    solve_with_trace(model)
    set_optimizer_attribute(model, "num_linear_variables", 0)
    trace = solve_with_trace(model)
    full, _ = mixed_model(; automatic = false)
    @test trace == solve_with_trace(full)
end

function add_unknown_oracle(model, x)
    # The row is linear, but its oracle does not advertise Hessian structure.
    oracle = MOI.VectorNonlinearOracle(;
        dimension = 1,
        l = [-Inf],
        u = [100.0],
        eval_f = (ret, x) -> (ret[1] = x[1]; nothing),
        jacobian_structure = [(1, 1)],
        eval_jacobian = (ret, x) -> (ret[1] = 1.0; nothing),
    )
    @constraint(model, [x[2]] in oracle)
end

@testset "Known-to-unknown structure discards subset" begin
    model, x = mixed_model()
    solve_with_trace(model)
    old = unsafe_backend(model).native.problem
    add_unknown_oracle(model, x)
    trace = solve_with_trace(model)
    full, y = mixed_model(; automatic = false)
    add_unknown_oracle(full, y)
    @test trace == solve_with_trace(full)
    @test Ext._nonlinear_variables(unsafe_backend(model)) === nothing
    @test old.ipopt_problem == C_NULL
end

end # module
