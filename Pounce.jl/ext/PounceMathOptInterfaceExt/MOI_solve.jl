# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

### MOI.optimize!

function _eval_jac_g_cb(model, x, rows, cols, values)
    if values === nothing
        for i = 1:length(model.evaluator.jacobian_sparsity)
            rows[i], cols[i] = model.evaluator.jacobian_sparsity[i]
        end
    else
        MOI.eval_constraint_jacobian(model, values, x)
    end
    return
end

function _eval_h_cb(model, x, rows, cols, obj_factor, lambda, values)
    if values === nothing
        for (i, v) in enumerate(model.evaluator.hessian_sparsity::Vector{Tuple{Int,Int}})
            rows[i], cols[i] = v
        end
    else
        MOI.eval_hessian_lagrangian(model, values, x, obj_factor, lambda)
    end
    return
end

function _setup_inner(model::Optimizer)::Pounce.IpoptProblem
    if model.native.problem !== nothing &&
       model.native.revision == model.data.revision &&
       model.native.options_revision == model.options_revision
        return model.native.problem
    end
    g_L, g_U = copy(model.data.qp_data.g_L), copy(model.data.qp_data.g_U)
    for (_, s) in model.data.vector_nonlinear_oracle_constraints
        append!(g_L, s.set.l)
        append!(g_U, s.set.u)
    end
    for bound in model.evaluator.nlp_data.constraint_bounds
        push!(g_L, bound.lower)
        push!(g_U, bound.upper)
    end
    function eval_h_cb(x, rows, cols, obj_factor, lambda, values)
        return _eval_h_cb(model, x, rows, cols, obj_factor, lambda, values)
    end
    has_hessian = model.evaluator.hessian_sparsity !== nothing
    _release_native!(model)
    model.native.problem = Pounce.CreateIpoptProblem(
        length(model.data.variables.lower),
        model.data.variables.lower,
        model.data.variables.upper,
        length(g_L),
        g_L,
        g_U,
        length(model.evaluator.jacobian_sparsity),
        has_hessian ? length(model.evaluator.hessian_sparsity) : 0,
        (x) -> MOI.eval_objective(model, x),
        (x, g) -> MOI.eval_constraint(model, g, x),
        (x, grad_f) -> MOI.eval_objective_gradient(model, grad_f, x),
        (x, rows, cols, values) -> _eval_jac_g_cb(model, x, rows, cols, values),
        has_hessian ? eval_h_cb : nothing,
    )
    inner = model.native.problem::Pounce.IpoptProblem
    if model.data.sense == MOI.MIN_SENSE
        Pounce.AddIpoptNumOption(inner, "obj_scaling_factor", 1.0)
    elseif model.data.sense == MOI.MAX_SENSE
        Pounce.AddIpoptNumOption(inner, "obj_scaling_factor", -1.0)
    end
    # POUNCE refuses check_derivatives_for_naninf=yes (gh#483).
    # The Julia callbacks check finiteness before returning to Rust instead.
    if !has_hessian
        Pounce.AddIpoptStrOption(inner, "hessian_approximation", "limited-memory")
    end
    if model.evaluator.has_only_linear_constraints
        Pounce.AddIpoptStrOption(inner, "jac_c_constant", "yes")
        Pounce.AddIpoptStrOption(inner, "jac_d_constant", "yes")
        if !model.evaluator.nlp_data.has_objective
            Pounce.AddIpoptStrOption(inner, "hessian_constant", "yes")
        end
    end
    model.native.revision = model.data.revision
    model.native.options_revision = model.options_revision
    return model.native.problem
end

function MOI.optimize!(model::Optimizer)
    if model.native.problem !== nothing && model.native.problem.solving
        error("A POUNCE optimizer cannot be solved recursively")
    end
    start_time = time_ns()
    _reset_result!(model)
    for (_, cache) in model.data.vector_nonlinear_oracle_constraints
        cache.eval_f_timer = 0.0
        cache.eval_jacobian_timer = 0.0
        cache.eval_hessian_lagrangian_timer = 0.0
    end
    try
        _optimize!(model)
    catch
        model.result.problem = nothing
        model.result.invalid_model = false
        model.result.barrier_iterations = 0
        model.result.native_solve_time = NaN
        rethrow()
    finally
        model.result.solve_time = (time_ns() - start_time) / 1e9
    end
    return
end

function _optimize!(model::Optimizer)
    if model.evaluator.structure_revision != model.data.structure_revision
        _setup_model(model)
    elseif model.evaluator.bounds_revision != model.data.bounds_revision
        if model.data.nlp_model !== nothing
            # Reuse the initialized evaluator; only its bound snapshot changes.
            model.evaluator.nlp_data = MOI.NLPBlockData(model.evaluator.nlp_data.evaluator)
        end
        model.evaluator.bounds_revision = model.data.bounds_revision
    end
    if model.result.invalid_model
        return
    end
    inner = _setup_inner(model)
    _setup_callback!(model, inner)
    if model.data.nlp_model !== nothing
        empty!(model.data.qp_data.parameters)
        for (p, index) in model.data.parameters
            model.data.qp_data.parameters[p.value] = model.data.nlp_model[index]
        end
    end
    for (name, value) in model.options
        if value isa String
            Pounce.AddIpoptStrOption(inner, name, value)
        elseif value isa Integer
            Pounce.AddIpoptIntOption(inner, name, value)
        elseif value isa Float64
            Pounce.AddIpoptNumOption(inner, name, value)
        else
            error(
                "Unable to add option `\"$name\"` with the value " *
                "`$value::$(typeof(value))`. The value must be a `::String`, " *
                "`::Integer`, or `::Float64`.",
            )
        end
    end
    if model.silent
        Pounce.AddIpoptIntOption(inner, "print_level", 0)
    end
    # Initialize the starting point, projecting variables from 0 onto their
    # bounds if VariablePrimalStart is not provided.
    for i = 1:length(model.data.variable_primal_start)
        inner.x[i] = something(
            model.data.variable_primal_start[i],
            clamp(0.0, model.data.variables.lower[i], model.data.variables.upper[i]),
        )
    end
    for (i, start) in enumerate(model.data.qp_data.mult_g)
        inner.mult_g[i] = _dual_start(model, start, -1)
    end
    offset = length(model.data.qp_data.mult_g)
    inner.mult_g[(offset+1):end] .= 0.0
    # Oracle rows precede the NLP block, and their starts are independent of
    # whether the NLP block has an explicit start vector.
    for (_, cache) in model.data.vector_nonlinear_oracle_constraints
        if cache.start !== nothing
            for i = 1:cache.set.output_dimension
                inner.mult_g[offset+i] = _dual_start(model, cache.start[i], -1)
            end
        end
        offset += cache.set.output_dimension
    end
    if model.data.nlp_dual_start === nothing
        for (key, val) in model.data.mult_g_nlp
            inner.mult_g[offset+key.value] = _dual_start(model, val, -1)
        end
    else
        for (i, start) in enumerate(model.data.nlp_dual_start::Vector{Float64})
            inner.mult_g[offset+i] = _dual_start(model, start, -1)
        end
    end
    for i = 1:inner.n
        inner.mult_x_L[i] = _dual_start(model, model.data.mult_x_L[i])
        inner.mult_x_U[i] = _dual_start(model, model.data.mult_x_U[i], -1)
    end
    model.result.tol = get(model.options, "tol", 1e-8)
    model.result.acceptable_tol = get(model.options, "acceptable_tol", 1e-6)
    Pounce.IpoptSolve(inner)
    model.result.problem = inner
    model.result.barrier_iterations = Pounce.GetIpoptIterCount(inner)
    model.result.native_solve_time = Pounce.GetIpoptSolveTime(inner)
    return
end
