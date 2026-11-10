# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

### Eval_F_CB

function MOI.eval_objective(model::Optimizer, x)
    # TODO(odow): FEASIBILITY_SENSE could produce confusing solver output if
    # a nonzero objective is set.
    if model.data.sense == MOI.FEASIBILITY_SENSE
        return 0.0
    elseif model.evaluator.nlp_data.has_objective
        return MOI.eval_objective(model.evaluator.nlp_data.evaluator, x)::Float64
    end
    return MOI.eval_objective(model.data.qp_data, x)
end

### Eval_Grad_F_CB

function MOI.eval_objective_gradient(model::Optimizer, grad, x)
    if model.data.sense == MOI.FEASIBILITY_SENSE
        grad .= zero(eltype(grad))
    elseif model.evaluator.nlp_data.has_objective
        MOI.eval_objective_gradient(model.evaluator.nlp_data.evaluator, grad, x)
    else
        MOI.eval_objective_gradient(model.data.qp_data, grad, x)
    end
    return
end

### Eval_G_CB

function _eval_constraint(
    g::AbstractVector,
    offset::Int,
    x::AbstractVector,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracleCache,
)
    for i = 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
    end
    ret = view(g, offset .+ (1:s.set.output_dimension))
    s.eval_f_timer += @elapsed s.set.eval_f(ret, s.x)
    return offset + s.set.output_dimension
end

function MOI.eval_constraint(model::Optimizer, g, x)
    MOI.eval_constraint(model.data.qp_data, g, x)
    offset = length(model.data.qp_data)
    for (f, s) in model.data.vector_nonlinear_oracle_constraints
        offset = _eval_constraint(g, offset, x, f, s)
    end
    stop = _complementarity_offset(model)
    g_nlp = view(g, (offset+1):stop)
    MOI.eval_constraint(model.evaluator.nlp_data.evaluator, g_nlp, x)
    MOI.eval_constraint(model.evaluator.complementarity, view(g, (stop+1):length(g)), x)
    return
end

### Eval_Jac_G_CB

function _jacobian_structure(
    ret::AbstractVector,
    row_offset::Int,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracleCache,
)
    for (i, j) in s.set.jacobian_structure
        push!(ret, (row_offset + i, f.variables[j].value))
    end
    return row_offset + s.set.output_dimension
end

function MOI.jacobian_structure(model::Optimizer)
    J = MOI.jacobian_structure(model.data.qp_data)
    offset = length(model.data.qp_data)
    for (f, s) in model.data.vector_nonlinear_oracle_constraints
        offset = _jacobian_structure(J, offset, f, s)
    end
    if length(model.evaluator.nlp_data.constraint_bounds) > 0
        J_nlp = MOI.jacobian_structure(
            model.evaluator.nlp_data.evaluator,
        )::Vector{Tuple{Int64,Int64}}
        for (row, col) in J_nlp
            push!(J, (row + offset, col))
        end
    end
    offset = _complementarity_offset(model)
    for (row, col) in MOI.jacobian_structure(model.evaluator.complementarity)
        push!(J, (row + offset, col))
    end
    return J
end

function _eval_constraint_jacobian(
    values::AbstractVector,
    offset::Int,
    x::AbstractVector,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracleCache,
)
    for i = 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
    end
    nnz = length(s.set.jacobian_structure)
    s.eval_jacobian_timer +=
        @elapsed s.set.eval_jacobian(view(values, offset .+ (1:nnz)), s.x)
    return offset + nnz
end

function MOI.eval_constraint_jacobian(model::Optimizer, values, x)
    offset = MOI.eval_constraint_jacobian(model.data.qp_data, values, x)
    offset -= 1  # .qp_data returns one-indexed offset
    for (f, s) in model.data.vector_nonlinear_oracle_constraints
        offset = _eval_constraint_jacobian(values, offset, x, f, s)
    end
    stop = length(values) - model.evaluator.complementarity_jacobian_nnz
    nlp_values = view(values, (offset+1):stop)
    MOI.eval_constraint_jacobian(model.evaluator.nlp_data.evaluator, nlp_values, x)
    MOI.eval_constraint_jacobian(
        model.evaluator.complementarity,
        view(values, (stop+1):length(values)),
        x,
    )
    return
end

### Eval_H_CB

function _hessian_lagrangian_structure(
    ret::AbstractVector,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracleCache,
)
    for (i, j) in s.set.hessian_lagrangian_structure
        push!(ret, (f.variables[i].value, f.variables[j].value))
    end
    return
end

function MOI.hessian_lagrangian_structure(model::Optimizer)
    H = MOI.hessian_lagrangian_structure(model.data.qp_data)
    for (f, s) in model.data.vector_nonlinear_oracle_constraints
        _hessian_lagrangian_structure(H, f, s)
    end
    append!(H, MOI.hessian_lagrangian_structure(model.evaluator.nlp_data.evaluator))
    append!(H, MOI.hessian_lagrangian_structure(model.evaluator.complementarity))
    return H
end

function _eval_hessian_lagrangian(
    H::AbstractVector,
    H_offset::Int,
    x::AbstractVector,
    μ::AbstractVector,
    μ_offset::Int,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracleCache,
)
    for i = 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
    end
    H_nnz = length(s.set.hessian_lagrangian_structure)
    H_view = view(H, H_offset .+ (1:H_nnz))
    μ_view = view(μ, μ_offset .+ (1:s.set.output_dimension))
    s.eval_hessian_lagrangian_timer +=
        @elapsed s.set.eval_hessian_lagrangian(H_view, s.x, μ_view)
    return H_offset + H_nnz, μ_offset + s.set.output_dimension
end

function MOI.eval_hessian_lagrangian(model::Optimizer, H, x, σ, μ)
    # Match eval_objective/gradient: a nonlinear objective overrides the QP
    # objective, and feasibility mode has no objective curvature at all.
    σ = model.data.sense == MOI.FEASIBILITY_SENSE ? 0.0 : σ
    qp_σ = model.evaluator.nlp_data.has_objective ? 0.0 : σ
    offset = MOI.eval_hessian_lagrangian(model.data.qp_data, H, x, qp_σ, μ)
    offset -= 1  # .qp_data returns one-indexed offset
    μ_offset = length(model.data.qp_data)
    for (f, s) in model.data.vector_nonlinear_oracle_constraints
        offset, μ_offset = _eval_hessian_lagrangian(H, offset, x, μ, μ_offset, f, s)
    end
    stop = length(H) - model.evaluator.complementarity_hessian_nnz
    μ_stop = _complementarity_offset(model)
    H_nlp = view(H, (offset+1):stop)
    μ_nlp = view(μ, (μ_offset+1):μ_stop)
    nlp_σ = model.evaluator.nlp_data.has_objective ? σ : 0.0
    MOI.eval_hessian_lagrangian(model.evaluator.nlp_data.evaluator, H_nlp, x, nlp_σ, μ_nlp)
    MOI.eval_hessian_lagrangian(
        model.evaluator.complementarity,
        view(H, (stop+1):length(H)),
        x,
        0.0,
        view(μ, (μ_stop+1):length(μ)),
    )
    return
end

### MOI.AutomaticDifferentiationBackend

MOI.supports(::Optimizer, ::MOI.AutomaticDifferentiationBackend) = true

function MOI.get(model::Optimizer, ::MOI.AutomaticDifferentiationBackend)
    return model.data.ad_backend
end

function MOI.set(
    model::Optimizer,
    ::MOI.AutomaticDifferentiationBackend,
    backend::MOI.Nonlinear.AbstractAutomaticDifferentiation,
)
    _assert_not_solving(model)
    # Setting the backend will invalidate the model if it is different. But we
    # don't requrire == for `::MOI.Nonlinear.AutomaticDifferentiationBackend` so
    # act defensive and invalidate regardless.
    _structure_changed!(model)
    model.data.ad_backend = backend
    return
end


function _setup_model(model::Optimizer)
    model.result.invalid_model_reason = nothing
    if isempty(model.data.variables.lower)
        # A zero-variable model is not supported by this optimizer.
        model.result.invalid_model_reason = "The model has no variable"
        return
    end
    if model.data.nlp_model !== nothing
        vars = MOI.get(model.data.variables, MOI.ListOfVariableIndices())
        model.evaluator.nlp_data = MOI.NLPBlockData(
            MOI.Nonlinear.Evaluator(model.data.nlp_model, model.data.ad_backend, vars),
        )
    end
    _setup_complementarity_evaluator!(model)
    empty!(model.evaluator.nlp_rows)
    if model.data.nlp_model !== nothing
        for (row, ci) in enumerate(keys(model.data.nlp_model.constraints))
            model.evaluator.nlp_rows[ci] = row
        end
    end
    has_quadratic_constraints =
        any(isequal(_kFunctionTypeScalarQuadratic), model.data.qp_data.function_type)
    has_nlp_constraints =
        !isempty(model.evaluator.nlp_data.constraint_bounds) ||
        !isempty(model.data.vector_nonlinear_oracle_constraints)
    has_complements = !isempty(model.data.complementarity_constraints)
    has_hessian =
        :Hess in MOI.features_available(model.evaluator.nlp_data.evaluator) &&
        :Hess in MOI.features_available(model.evaluator.complementarity)
    for (_, s) in model.data.vector_nonlinear_oracle_constraints
        if s.set.eval_hessian_lagrangian === nothing
            has_hessian = false
            break
        end
    end
    init_feat = [:Grad]
    if has_hessian
        push!(init_feat, :Hess)
    end
    if has_nlp_constraints
        push!(init_feat, :Jac)
    end
    MOI.initialize(model.evaluator.nlp_data.evaluator, init_feat)
    comp_feat = has_complements ? [:Jac] : Symbol[]
    has_hessian && push!(comp_feat, :Hess)
    MOI.initialize(model.evaluator.complementarity, comp_feat)
    model.evaluator.complementarity_jacobian_nnz =
        length(MOI.jacobian_structure(model.evaluator.complementarity))
    model.evaluator.complementarity_hessian_nnz =
        has_hessian ?
        length(MOI.hessian_lagrangian_structure(model.evaluator.complementarity)) : 0
    model.evaluator.jacobian_sparsity = MOI.jacobian_structure(model)
    model.evaluator.hessian_sparsity = nothing
    if has_hessian
        model.evaluator.hessian_sparsity = MOI.hessian_lagrangian_structure(model)
    end
    model.evaluator.has_only_linear_constraints =
        !has_complements && !has_nlp_constraints && !has_quadratic_constraints
    model.evaluator.structure_revision = model.data.structure_revision
    model.evaluator.bounds_revision = model.data.bounds_revision
    return
end
