module TestCallbacks

using Test
import JuMP
import Pounce
const MOI = JuMP.MOI

function model_with_variables()
    model = JuMP.direct_model(Pounce.Optimizer())
    JuMP.set_silent(model)
    JuMP.@variable(model, 0 <= x[1:2] <= 5, start = 0.5)
    JuMP.@variable(model, p in JuMP.Parameter(1.0))
    JuMP.@objective(model, Min, (x[1] - p)^2 + (x[2] - 2)^2)
    return model, x, p
end

function check_statistics(model)
    inner = JuMP.backend(model).inner
    native = MOI.get(model, Pounce.NativeSolveTimeSec())
    @test isfinite(native) && native >= 0.0
    @test MOI.get(model, MOI.SolveTimeSec()) >= native
    @test native == Pounce.GetIpoptSolveTime(inner)
    @test MOI.get(model, MOI.BarrierIterations()) == Pounce.GetIpoptIterCount(inner)
end

@testset "Native statistics without callbacks" begin
    model, x, _ = model_with_variables()
    @test MOI.is_set_by_optimize(Pounce.NativeSolveTimeSec())
    @test isnan(MOI.get(JuMP.backend(model), Pounce.NativeSolveTimeSec()))
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.LOCALLY_SOLVED
    @test JuMP.value.(x) ≈ [1, 2] atol = 1e-6
    # No native callback is installed merely for iteration counting.
    @test JuMP.backend(model).inner.intermediate === nothing
    @test MOI.get(model, MOI.BarrierIterations()) > 0
    check_statistics(model)

    JuMP.set_optimizer_attribute(model, "max_iter", 0)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.ITERATION_LIMIT
    @test MOI.get(JuMP.backend(model), MOI.BarrierIterations()) == 0
    check_statistics(model)

    JuMP.set_optimizer_attribute(model, "not_an_option", 1)
    @test_throws ErrorException JuMP.optimize!(model)
    @test JuMP.result_count(model) == 0
    @test isnan(MOI.get(JuMP.backend(model), Pounce.NativeSolveTimeSec()))
    @test MOI.get(JuMP.backend(model), MOI.BarrierIterations()) == 0
    @test isfinite(MOI.get(JuMP.backend(model), MOI.SolveTimeSec()))
    MOI.empty!(JuMP.backend(model))
    @test isnan(MOI.get(JuMP.backend(model), Pounce.NativeSolveTimeSec()))
    MOI.optimize!(JuMP.backend(model))
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INVALID_MODEL
    @test isnan(MOI.get(JuMP.backend(model), Pounce.NativeSolveTimeSec()))

    # The custom attribute must also be forwarded through JuMP's cache/bridges.
    cached = JuMP.Model(Pounce.Optimizer)
    JuMP.set_silent(cached)
    JuMP.@variable(cached, y >= 1)
    JuMP.@objective(cached, Min, y^2)
    JuMP.optimize!(cached)
    @test MOI.get(cached, Pounce.NativeSolveTimeSec()) >= 0
end

@testset "Callback installation, snapshots, replacement, and removal" begin
    model, x, p = model_with_variables()
    optimizer = JuMP.backend(model)
    xs = JuMP.index.(x)
    attr = MOI.CallbackVariablePrimal(optimizer)
    @test_throws MOI.GetAttributeNotAllowed MOI.get(optimizer, attr, xs[1])
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.LOCALLY_SOLVED
    snapshots = Vector{Float64}[]
    callback = function (args...)
        if args[1] != 0
            @test_throws MOI.GetAttributeNotAllowed MOI.get(optimizer, attr, xs[1])
            return true
        end
        values = MOI.get(optimizer, attr, xs)
        @test MOI.get(optimizer, attr, reverse(xs)) == reverse(values)
        raw = zeros(length(xs))
        Pounce.GetIpoptCurrentIterate(
            optimizer.inner,
            false,
            length(xs),
            raw,
            C_NULL,
            C_NULL,
            optimizer.inner.m,
            C_NULL,
            C_NULL,
        )
        @test values == raw
        @test MOI.get(optimizer, attr, JuMP.index(p)) == JuMP.parameter_value(p)
        @test_throws MOI.InvalidIndex MOI.get(optimizer, attr, MOI.VariableIndex(99))
        # A cached snapshot must not bypass the native callback's scope.
        task = @async try
            MOI.get(optimizer, attr, xs[1])
        catch err
            err
        end
        @test fetch(task) isa MOI.GetAttributeNotAllowed
        @test_throws ErrorException MOI.set(optimizer, Pounce.CallbackFunction(), nothing)
        @test_throws ErrorException MOI.optimize!(optimizer)
        @test MOI.get(optimizer, attr, xs) == values
        push!(snapshots, values)
        return args[2] < 2
    end
    MOI.set(model, Pounce.CallbackFunction(), callback)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.INTERRUPTED
    @test length(snapshots) >= 2
    @test first(snapshots) != last(snapshots)
    @test_throws MOI.GetAttributeNotAllowed MOI.get(optimizer, attr, xs[1])
    check_statistics(model)

    # A new solve's first snapshot must reflect its new start, not the old cache.
    empty!(snapshots)
    JuMP.set_start_value.(x, [3.0, 4.0])
    JuMP.optimize!(model)
    @test first(snapshots) ≈ [3.0, 4.0] atol = 1e-6

    replacement_calls = Ref(0)
    MOI.set(
        model,
        Pounce.CallbackFunction(),
        (args...) -> (replacement_calls[] += 1; false),
    )
    JuMP.optimize!(model)
    @test replacement_calls[] == 1
    @test JuMP.termination_status(model) == MOI.INTERRUPTED
    check_statistics(model)

    MOI.set(model, Pounce.CallbackFunction(), nothing)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.LOCALLY_SOLVED
    @test JuMP.value.(x) ≈ [1, 2] atol = 1e-6
    @test replacement_calls[] == 1
    @test optimizer.inner.intermediate === nothing
    check_statistics(model)

    MOI.set(model, Pounce.CallbackFunction(), function (args...)
        MOI.get(optimizer, attr, xs[1]) # Populate the cache before throwing.
        error("snapshot callback failure")
    end)
    @test_throws ErrorException("snapshot callback failure") JuMP.optimize!(model)
    @test_throws MOI.GetAttributeNotAllowed MOI.get(optimizer, attr, xs[1])
    @test JuMP.result_count(model) == 0
    @test isnan(MOI.get(JuMP.backend(model), Pounce.NativeSolveTimeSec()))
    @test MOI.get(JuMP.backend(model), MOI.BarrierIterations()) == 0
    @test isfinite(MOI.get(JuMP.backend(model), MOI.SolveTimeSec()))

    # Exercise restoration dispatch independently of a solver's choice to
    # enter restoration. This is a callback contract test, not a solver fixture.
    MOI.set(
        model,
        Pounce.CallbackFunction(),
        function (args...)
            @test_throws MOI.GetAttributeNotAllowed MOI.get(optimizer, attr, xs[1])
            return true
        end,
    )
    @test Pounce._dispatch_callback(optimizer, Cint(1))
    @test_throws MOI.GetAttributeNotAllowed MOI.get(optimizer, attr, xs[1])
    MOI.set(model, Pounce.CallbackFunction(), nothing)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.LOCALLY_SOLVED
    check_statistics(model)
end

end # module
