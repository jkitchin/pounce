# Run setup.jl first. Each solver gets a fresh Julia process for first-call timing.
using BMOPFTools, JuMP, Pounce, Ipopt, JSON3, SHA, Pkg, Dates, TOML
include("physics.jl")

const MODEL_REVISION = "542a48ba7f4c95e50b8acdc790a5323a4ef24930"
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const FIXTURE =
    joinpath(pkgdir(BMOPFTools), "test", "data", "case_development", "feeder.json")

function build_case(solver, factor)
    net = parse_bmopf(FIXTURE)
    model = Model(solver == "pounce" ? Pounce.Optimizer : Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "tol", 1e-8)
    set_optimizer_attribute(model, "max_iter", 500)
    set_optimizer_attribute(model, "linear_solver", solver == "ipopt" ? "mumps" : "feral")
    @variable(model, demand in Parameter(factor))
    providers = Dict{OpfCoefficientKey,OpfCoefficientProvider}()
    for (id, load) in net["load"],
        field in (:p_nom, :q_nom),
        i in eachindex(load[string(field)])

        key = OpfCoefficientKey(:load, :load, id, field, i)
        providers[key] = OpfCoefficientProvider(
            :PounceValidation,
            (ctx, key, default) -> default * demand,
        )
    end
    ctx = build_opf_model(
        net;
        model,
        per_unit = true,
        build_spec = OpfBuildSpec(coefficient_providers = providers),
    )
    enforce_kcl!(ctx)
    return (; ctx, model, demand, net)
end

function checked_solve(case, solver, label, factor; build_seconds = nothing)
    m = case.model
    elapsed = @elapsed optimize!(m)
    status = termination_status(m)
    @assert status in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL) (label, status)
    violations = primal_feasibility_report(m; atol = 0.0)
    residual = maximum(values(violations); init = 0.0)
    @assert isfinite(residual) && residual <= 1e-6 (label, residual)
    result = extract_result(case.ctx)
    # Verify against the scenario's SI powers, not the unchanged build snapshot.
    scenario = deepcopy(case.net)
    for load in values(scenario["load"]), key in ("p_nom", "q_nom")
        load[key] .*= factor
    end
    report = profile_solution(scenario, result)
    @assert isempty(errors(report)) (label, errors(report))
    independent_physics = check_fixture_physics(scenario, result)
    row = Dict{String,Any}(
        "label" => label,
        "factor" => factor,
        "status" => string(status),
        "build_seconds" => build_seconds,
        "optimize_elapsed_seconds" => elapsed,
        "moi_solve_seconds" => solve_time(m),
        "native_solve_seconds" =>
            solver == "pounce" ? MOI.get(m, Pounce.NativeSolveTimeSec()) : nothing,
        "iterations" => MOI.get(m, MOI.BarrierIterations()),
        "objective" => objective_value(m),
        "max_model_violation" => residual,
        "physical_errors" => length(errors(report)),
        "physical_warnings" => length(warnings(report)),
        "physics" => report.results[:solution],
        "independent_physics" => independent_physics,
        "variables" => num_variables(m),
        "constraints" => num_constraints(m; count_variable_in_set_constraints = true),
    )
    return row
end

function capture_starts(m)
    # Collect all values before any mutation invalidates JuMP result access.
    primals = Dict(v => value(v) for v in all_variables(m) if !is_parameter(v))
    duals = Dict(
        c => dual(c) for (F, S) in list_of_constraint_types(m) if !(S <: MOI.Parameter)
        for c in all_constraints(m, F, S)
    )
    for (v, val) in primals
        set_start_value(v, val)
    end
    for (c, val) in duals
        set_dual_start_value(c, val)
    end
    set_optimizer_attribute(m, "warm_start_init_point", "yes")
    return (; primals = length(primals), duals = length(duals))
end

function run(solver)
    solver in ("pounce", "ipopt") || error("solver must be pounce or ipopt")
    info = Pkg.dependencies()[Base.UUID("5f9a3b62-8f63-4a39-b67b-af9b51e6a7c3")]
    @assert info.git_revision == MODEL_REVISION
    native_revision = ENV["POUNCE_NATIVE_REVISION"]
    rows = Dict{String,Any}[]
    build = @elapsed case = build_case(solver, 1.0)
    push!(rows, checked_solve(case, solver, "first", 1.0; build_seconds = build))
    push!(rows, checked_solve(case, solver, "repeat", 1.0))
    starts = capture_starts(case.model)
    push!(rows, checked_solve(case, solver, "explicit_primal_dual_starts", 1.0))
    # The normal optimize! path exercises wrapper parameter invalidation. Using
    # BMOPFTools.resolve_opf! would forcibly replace the cached optimizer instead.
    for factor in (0.5, 1.4, 1.0)
        set_parameter_value(case.demand, factor)
        row = checked_solve(case, solver, "parameter_update", factor)
        build = @elapsed fresh = build_case(solver, factor)
        control =
            checked_solve(fresh, solver, "fresh_control", factor; build_seconds = build)
        @assert isapprox(row["objective"], control["objective"]; rtol = 1e-6, atol = 1e-6)
        push!(rows, row, control)
    end
    # Force an unsuccessful attempt on this real application model, then recover.
    for v in all_variables(case.model)
        is_parameter(v) || set_start_value(v, nothing)
    end
    set_optimizer_attribute(case.model, "warm_start_init_point", "no")
    set_optimizer_attribute(case.model, "max_iter", 0)
    optimize!(case.model)
    failed_status = termination_status(case.model)
    @assert failed_status == MOI.ITERATION_LIMIT failed_status
    set_optimizer_attribute(case.model, "max_iter", 500)
    push!(rows, checked_solve(case, solver, "recovered_after_iteration_limit", 1.0))

    # Separate diagnostic on a disposable model, after timed solves. These
    # pinned private setup functions initialize AD/sparsity, not a native solve.
    # This number is not subtracted from or added to any solve timer above.
    diagnostic = build_case(solver, 1.0)
    copy_time = @elapsed MOI.Utilities.attach_optimizer(backend(diagnostic.model))
    optimizer = unsafe_backend(diagnostic.model)
    ext = Base.get_extension(
        solver == "pounce" ? Pounce : Ipopt,
        solver == "pounce" ? :PounceMathOptInterfaceExt : :IpoptMathOptInterfaceExt,
    )
    ad_time = @elapsed ext._setup_model(optimizer)
    deps = Dict(
        p.name =>
            Dict("version" => string(p.version), "tree_hash" => string(p.tree_hash)) for
        p in values(Pkg.dependencies())
    )
    wrapper_path = pkgdir(Pounce)
    candidate_file = joinpath(dirname(wrapper_path), "Candidate.toml")
    candidate = isfile(candidate_file) ? TOML.parsefile(candidate_file) : nothing
    wrapper_commit =
        candidate === nothing ? readchomp(`git -C $wrapper_path rev-parse HEAD`) :
        candidate["wrapper_commit"]
    wrapper_working_tree =
        candidate === nothing ? readchomp(`git -C $wrapper_path status --porcelain`) :
        candidate["working_tree"]
    return Dict(
        "schema" => "pounce.julia-bmopf/v1",
        "timestamp_utc" => string(now(UTC)),
        "wrapper_commit" => wrapper_commit,
        "working_tree" => wrapper_working_tree,
        "wrapper_package_path" => wrapper_path,
        "release_candidate" => candidate,
        "harness_commit" => readchomp(`git -C $REPO rev-parse HEAD`),
        "harness_working_tree" => readchomp(`git -C $REPO status --porcelain`),
        "native_commit" => native_revision,
        "pounce_version" => string(Pounce.GetIpoptVersion()),
        "ipopt_version" => string(Ipopt.GetIpoptVersion()),
        "native_library" => Pounce.library_path(),
        "native_library_sha256" => bytes2hex(sha256(read(Pounce.library_path()))),
        "fixture_sha256" => bytes2hex(sha256(read(FIXTURE))),
        "model_commit" => MODEL_REVISION,
        "julia" => string(VERSION),
        "machine" => Sys.MACHINE,
        "cpu" => Sys.CPU_NAME,
        "julia_threads" => Threads.nthreads(),
        "solver" => solver,
        "options" => Dict(
            "tol" => 1e-8,
            "max_iter" => 500,
            "linear_solver" => solver == "ipopt" ? "mumps" : "feral",
        ),
        "dependencies" => deps,
        "starts" => Dict(pairs(starts)),
        "failure_status" => string(failed_status),
        "rows" => rows,
        "diagnostic_copy_seconds" => copy_time,
        "diagnostic_ad_setup_seconds" => ad_time,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("Usage: run.jl pounce|ipopt output.json")
    result = run(ARGS[1])
    open(ARGS[2], "w") do io
        JSON3.pretty(io, result)
        println(io)
    end
    println("Validated ", length(result["rows"]), " solves with ", ARGS[1])
end
