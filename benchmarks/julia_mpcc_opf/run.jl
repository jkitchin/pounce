# Run in an environment containing Pounce, JuMP, ADNLPModels, NLPModels and
# MPCCModels. The model checkout is an explicit argument; see README.md.
using JuMP, Pounce, NLPModels, ADNLPModels, MPCCModels, Test, SHA, Dates, Pkg
const MODEL_REVISION = "19a4b275c8ebb3fb5f7a051cfe36c45ad1a948de"
const MODEL_ROOT = abspath(only(ARGS))
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
@assert readchomp(`git -C $MODEL_ROOT rev-parse HEAD`) == MODEL_REVISION
@assert success(`git -C $MODEL_ROOT diff --quiet $MODEL_REVISION -- examples/opf src`)
include(joinpath(MODEL_ROOT, "examples", "opf", "balanced-ac-opf.jl"))
using .BalancedACOPF

# Adapt the original NLP callbacks directly; all declared pairs pass through
# MOI.Complements in Pounce.jl. No hand-transcribed network equations.
struct OriginalEvaluator{T} <: MOI.AbstractNLPEvaluator
    nlp::T
end
MOI.features_available(::OriginalEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(::OriginalEvaluator, ::Any) = nothing
MOI.eval_objective(e::OriginalEvaluator, x) = obj(e.nlp, x)
MOI.eval_objective_gradient(e::OriginalEvaluator, g, x) = (grad!(e.nlp, x, g); nothing)
MOI.eval_constraint(e::OriginalEvaluator, g, x) = (cons!(e.nlp, x, g); nothing)
MOI.jacobian_structure(e::OriginalEvaluator) = collect(zip(jac_structure(e.nlp)...))
MOI.eval_constraint_jacobian(e::OriginalEvaluator, j, x) =
    (jac_coord!(e.nlp, x, j); nothing)
MOI.hessian_lagrangian_structure(e::OriginalEvaluator) =
    collect(zip(hess_structure(e.nlp)...))
MOI.eval_hessian_lagrangian(e::OriginalEvaluator, h, x, sigma, mu) =
    (hess_coord!(e.nlp, x, mu, h; obj_weight = sigma); nothing)

function make_model(mpcc, layout, approximation)
    nlp = mpcc.nlp
    m = direct_model(Pounce.Optimizer())
    set_silent(m)
    set_optimizer_attribute(m, "tol", 1e-8)
    set_optimizer_attribute(m, "hessian_approximation", approximation)
    @variable(m, x[1:get_nvar(nlp)])
    for i in eachindex(x)
        isfinite(get_lvar(nlp)[i]) && set_lower_bound(x[i], get_lvar(nlp)[i])
        isfinite(get_uvar(nlp)[i]) && set_upper_bound(x[i], get_uvar(nlp)[i])
        set_start_value(x[i], get_x0(nlp)[i])
    end
    bounds = [MOI.NLPBoundsPair(l, u) for (l, u) in zip(get_lcon(nlp), get_ucon(nlp))]
    MOI.set(m, MOI.NLPBlock(), MOI.NLPBlockData(bounds, OriginalEvaluator(nlp), true))
    set_objective_sense(m, MOI.MIN_SENSE)
    @constraint(m, complements(x[layout.s], x[layout.r]))
    return m, x
end

println("UTC: ", now(UTC))
println("Wrapper: ", readchomp(`git -C $REPO rev-parse HEAD`))
println("Wrapper worktree: ", readchomp(`git -C $REPO status --porcelain`))
println("Native source: ", ENV["POUNCE_NATIVE_REVISION"])
println("Library: ", Pounce.library_path())
println("Library SHA256: ", bytes2hex(sha256(read(Pounce.library_path()))))
println("MPCCModels: ", MODEL_REVISION)
println("Julia: ", VERSION)
Pkg.status()
flush(stdout)
@testset "Original MPCCModels volt-var/volt-watt cases" begin
    for case in (:vv_lower_knee, :vv_injection_20, :vw_ramp_50, :curtailment),
        approximation in ("exact", "limited-memory")

        data = balanced_ac_opf_case(case)
        mpcc = BalancedACOPFModel(data)
        layout = opf_layout(data)
        m, x = make_model(mpcc, layout, approximation)
        optimize!(m)
        point = value.(x)
        residuals = opf_residuals(mpcc, point, data)
        stats = MOI.get(m, Pounce.ComplementarityStatistics())
        println((
            case = case,
            approximation = approximation,
            status = termination_status(m),
            objective = objective_value(m),
            residuals = residuals,
            certificate = stats.certificate_class,
        ))
        flush(stdout)
        # Independent original-model checks, including min(G,H) and direct laws.
        @test all(isfinite, values(residuals))
        @test maximum(values(residuals)) <= 1e-6
        @test primal_status(m) in (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
        @test stats.n_pairs == length(layout.r)
        @test length(MOI.get(m, MOI.NLPBlockDual())) == get_ncon(mpcc.nlp)
        # Compare the same model evaluated directly through NLPModels.
        @test objective_value(m) ≈ obj(mpcc.nlp, point) atol = 1e-8
    end
end
