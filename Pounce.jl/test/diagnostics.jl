module TestDiagnostics
using Test
import JSON
import JuMP
import Pounce
const MOI = JuMP.MOI

const getters = (
    Pounce.GetIpoptPrimalInf,
    Pounce.GetIpoptDualInf,
    Pounce.GetIpoptComplInf,
    Pounce.GetPounceRestorationStats,
    Pounce.GetPounceFdHessianStats,
    Pounce.GetPounceLinearSolverStats,
)
const attributes = (
    Pounce.NativeResiduals(),
    Pounce.RestorationStatistics(),
    Pounce.FiniteDifferenceHessianStatistics(),
    Pounce.LinearSolverStatistics(),
    Pounce.SolveReport(),
)

function native_report(prob; detail = "full")
    return mktempdir() do dir
        path = joinpath(dir, "résultat.json")
        @test Pounce.IpoptWriteSolveReport(prob, path; detail) === nothing
        return JSON.parsefile(path)
    end
end

function check_native_report(prob, report)
    @test report["schema"] == "pounce.solve-report/v1"
    @test report["problem"]["n_variables"] == prob.n
    @test report["solution"]["objective"] ≈ prob.obj_val
    @test report["solution"]["x"] ≈ prob.x
    stats = report["statistics"]
    @test Pounce.GetIpoptPrimalInf(prob) ≈ stats["final_constr_viol"]
    @test Pounce.GetIpoptDualInf(prob) ≈ stats["final_dual_inf"]
    @test Pounce.GetIpoptComplInf(prob) ≈ stats["final_compl"]
    r = Pounce.GetPounceRestorationStats(prob)
    for field in keys(r)
        @test r[field] ≈ stats["restoration_"*string(field)]
    end
    linear = Pounce.GetPounceLinearSolverStats(prob)
    if haskey(report, "linear_solver")
        @test linear !== nothing
        native = report["linear_solver"]
        for field in (
            :solver_name,
            :n_factors,
            :n_pattern_reuse,
            :n_pattern_changes,
            :max_fill_ratio,
            :min_abs_pivot,
            :max_abs_pivot,
            :last_nnz_a,
            :last_nnz_l,
        )
            if haskey(native, string(field))
                @test linear[field] == native[string(field)]
            else
                @test linear[field] == -1 || isnan(linear[field])
            end
        end
        @test [
            linear.last_inertia_positive,
            linear.last_inertia_negative,
            linear.last_inertia_zero,
        ] == get(native, "last_inertia", [-1, -1, -1])
    else
        @test linear === nothing
    end
end

@testset "Native diagnostic bindings and report errors" begin
    prob = Main.hs071_problem()
    try
        for getter in getters
            @test_throws ErrorException getter(prob)
        end
        mktempdir() do dir
            path = joinpath(dir, "report.json")
            @test_throws ErrorException Pounce.IpoptWriteSolveReport(prob, path)
            @test !isfile(path)
        end
        @test Pounce.IpoptSolve(prob) == 0
        check_native_report(prob, native_report(prob))
        @test !haskey(native_report(prob), "iterations")
        fd = Pounce.GetPounceFdHessianStats(prob)
        @test fd.pattern_used == -1
        @test fd.nnz == fd.n == fd.groups == fd.rho_max == 0
        @test !fd.coloring_fell_back && !fd.objective_clique_widened
        mktempdir() do dir
            path = joinpath(dir, "report.json")
            @test_throws ArgumentError Pounce.IpoptWriteSolveReport(prob, path * "\0suffix")
            @test !isfile(path)
            @test_throws ArgumentError Pounce.IpoptWriteSolveReport(
                prob,
                path;
                detail = "full\0suffix",
            )
            @test_throws ArgumentError Pounce.IpoptWriteSolveReport(
                prob,
                path;
                detail = "unknown",
            )
            @test !isfile(path)
            @test_throws ErrorException Pounce.IpoptWriteSolveReport(
                prob,
                joinpath(dir, "missing", "report.json"),
            )
            write(path, "old content")
            Pounce.IpoptWriteSolveReport(prob, path)
            @test JSON.parsefile(path)["schema"] == "pounce.solve-report/v1"
        end
        # Capture is opt-in and does not retroactively alter the old result.
        Pounce.IpoptEnableIterHistory(prob)
        @test !haskey(native_report(prob), "iterations")
        prob.x .= [1.0, 5.0, 5.0, 1.0]
        Pounce.AddIpoptStrOption(prob, "start_with_resto", "yes")
        @test Pounce.IpoptSolve(prob) == 0
        full = native_report(prob)
        check_native_report(prob, full)
        @test !isempty(full["iterations"])
        @test Pounce.GetPounceRestorationStats(prob).calls > 0
        @test !haskey(native_report(prob; detail = "summary"), "iterations")
        Pounce.AddIpoptStrOption(prob, "start_with_resto", "no")
        Pounce.AddIpoptStrOption(prob, "hessian_approximation", "finite-difference")
        prob.x .= [1.0, 5.0, 5.0, 1.0]
        @test Pounce.IpoptSolve(prob) == 0
        fd = Pounce.GetPounceFdHessianStats(prob)
        @test fd.pattern_used == 0
        @test fd.n == 4
        @test fd.nnz == 10
        @test 0 < fd.groups <= fd.n
        @test fd.coloring_fell_back isa Bool
        @test fd.objective_clique_widened isa Bool
        check_native_report(prob, native_report(prob))
        Pounce.SetIntermediateCallback(
            prob,
            function (args...)
                for getter in getters
                    @test_throws ErrorException getter(prob)
                end
                @test_throws ErrorException Pounce.IpoptEnableIterHistory(prob)
                mktempdir() do dir
                    @test_throws ErrorException Pounce.IpoptWriteSolveReport(
                        prob,
                        joinpath(dir, "during.json"),
                    )
                end
                error("diagnostics callback sentinel")
            end,
        )
        @test_throws ErrorException("diagnostics callback sentinel") Pounce.IpoptSolve(prob)
        for getter in getters
            @test_throws ErrorException getter(prob)
        end
        mktempdir() do dir
            @test_throws ErrorException Pounce.IpoptWriteSolveReport(
                prob,
                joinpath(dir, "failed.json"),
            )
        end
        Pounce.SetIntermediateCallback(prob, nothing)
        Pounce.AddIpoptIntOption(prob, "max_iter", 0)
        prob.x .= [1.0, 5.0, 5.0, 1.0]
        @test Pounce.IpoptSolve(prob) == -1
        check_native_report(prob, native_report(prob))
    finally
        Pounce.FreeIpoptProblem(prob)
    end
    for getter in getters
        @test_throws ErrorException getter(prob)
    end
    @test_throws ErrorException Pounce.IpoptEnableIterHistory(prob)
    # Inconsistent bounds fail before a backend summary is produced. Native
    # GetPounceLinearSolverStats leaves its output buffer untouched in this case.
    invalid = Pounce.CreateIpoptProblem(
        1,
        [2.0],
        [1.0],
        0,
        Float64[],
        Float64[],
        0,
        0,
        x -> x[1],
        (x, g) -> nothing,
        (x, g) -> (g[1] = 1.0),
        (args...) -> nothing,
        (args...) -> nothing,
    )
    try
        Pounce.AddIpoptIntOption(invalid, "print_level", 0)
        @test Pounce.IpoptSolve(invalid) == -11
        @test Pounce.GetPounceLinearSolverStats(invalid) === nothing
    finally
        Pounce.FreeIpoptProblem(invalid)
    end
end

@testset "MOI diagnostics, history toggles, and result lifecycle" begin
    for direct in (true, false)
        model =
            direct ? JuMP.direct_model(Pounce.Optimizer()) : JuMP.Model(Pounce.Optimizer)
        JuMP.set_silent(model)
        JuMP.@variable(model, 0 <= x <= 5, start = 0.5)
        JuMP.@variable(model, p in JuMP.Parameter(1.0))
        JuMP.@objective(model, Min, (x - p)^2)
        @test !MOI.get(model, Pounce.RecordIterationHistory())
        @test_throws ArgumentError Pounce.SolveReport("invalid")
        for attr in attributes
            @test MOI.is_set_by_optimize(attr)
            @test_throws JuMP.OptimizeNotCalled MOI.get(model, attr)
        end
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == MOI.LOCALLY_SOLVED
        plain = JSON.parse(MOI.get(model, Pounce.SolveReport("full")))
        @test !haskey(plain, "iterations")
        residuals = MOI.get(model, Pounce.NativeResiduals())
        @test all(isfinite, residuals)
        @test residuals.primal_infeasibility == plain["statistics"]["final_constr_viol"]
        @test MOI.get(model, Pounce.RestorationStatistics()).calls >= 0
        @test MOI.get(model, Pounce.FiniteDifferenceHessianStatistics()).pattern_used == -1
        @test MOI.get(model, Pounce.LinearSolverStatistics()).solver_name == "feral"
        MOI.set(model, Pounce.RecordIterationHistory(), true)
        @test MOI.get(model, Pounce.RecordIterationHistory())
        @test MOI.get(JuMP.backend(model), Pounce.NativeResiduals()) == residuals
        @test !haskey(
            JSON.parse(MOI.get(JuMP.backend(model), Pounce.SolveReport("full"))),
            "iterations",
        )
        JuMP.optimize!(model)
        @test !isempty(JSON.parse(MOI.get(model, Pounce.SolveReport("full")))["iterations"])
        @test !haskey(JSON.parse(MOI.get(model, Pounce.SolveReport())), "iterations")
        JuMP.set_parameter_value(p, 1.5)
        @test_throws JuMP.OptimizeNotCalled MOI.get(model, Pounce.SolveReport("full"))
        JuMP.optimize!(model)
        repeat_report = JSON.parse(MOI.get(model, Pounce.SolveReport("full")))
        @test repeat_report["solution"]["x"][1] ≈ 1.5 atol = 1e-6
        @test !isempty(repeat_report["iterations"])
        JuMP.set_upper_bound(x, 4.5) # Rebuild the handle while capture stays enabled.
        JuMP.optimize!(model)
        @test !isempty(JSON.parse(MOI.get(model, Pounce.SolveReport("full")))["iterations"])
        MOI.set(model, Pounce.RecordIterationHistory(), false)
        @test !isempty(
            JSON.parse(MOI.get(JuMP.backend(model), Pounce.SolveReport("full")))["iterations"],
        )
        JuMP.optimize!(model)
        @test !haskey(JSON.parse(MOI.get(model, Pounce.SolveReport("full"))), "iterations")
        JuMP.set_parameter_value(p, 2.0)
        for attr in attributes
            @test_throws JuMP.OptimizeNotCalled MOI.get(model, attr)
        end
        JuMP.optimize!(model)
        @test JuMP.value(x) ≈ 2.0 atol = 1e-6
        @test JSON.parse(MOI.get(model, Pounce.SolveReport()))["solution"]["x"][1] ≈ 2.0 atol =
            1e-6
        MOI.set(
            model,
            Pounce.CallbackFunction(),
            function (args...)
                for attr in attributes
                    @test_throws JuMP.OptimizeNotCalled MOI.get(model, attr)
                end
                @test_throws ErrorException MOI.set(
                    model,
                    Pounce.RecordIterationHistory(),
                    true,
                )
                error("MOI diagnostics sentinel")
            end,
        )
        @test_throws ErrorException("MOI diagnostics sentinel") JuMP.optimize!(model)
        for attr in attributes
            @test_throws JuMP.OptimizeNotCalled MOI.get(model, attr)
        end
        MOI.set(model, Pounce.CallbackFunction(), nothing)
        JuMP.optimize!(model)
        JuMP.set_upper_bound(x, 4.0)
        @test_throws JuMP.OptimizeNotCalled MOI.get(model, Pounce.NativeResiduals())
        JuMP.optimize!(model)
        JuMP.set_optimizer_attribute(model, "not_an_option", 1)
        @test_throws ErrorException JuMP.optimize!(model)
        @test_throws JuMP.OptimizeNotCalled MOI.get(model, Pounce.SolveReport())
    end
    optimizer = Pounce.Optimizer()
    MOI.set(optimizer, Pounce.RecordIterationHistory(), true)
    MOI.empty!(optimizer)
    @test MOI.get(optimizer, Pounce.RecordIterationHistory())
    @test_throws MOI.GetAttributeNotAllowed MOI.get(optimizer, Pounce.SolveReport())
end
end # module
