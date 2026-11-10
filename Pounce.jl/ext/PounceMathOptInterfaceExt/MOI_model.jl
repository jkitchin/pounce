# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function MOI.empty!(model::Optimizer)
    _assert_not_solving(model)
    backend = model.data.ad_backend
    revision = model.data.revision + 1
    _release_native!(model)
    model.data = _ModelData(revision = revision, ad_backend = backend)
    model.evaluator = _EvaluatorCache()
    _reset_result!(model)
    model.callback = nothing
    model.callback_state = _CallbackState()
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.data.variables) &&
           isempty(model.data.parameters) &&
           isempty(model.data.variable_primal_start) &&
           isempty(model.data.mult_x_L) &&
           isempty(model.data.mult_x_U) &&
           model.evaluator.nlp_data.evaluator isa _EmptyNLPEvaluator &&
           model.data.sense == MOI.FEASIBILITY_SENSE &&
           isempty(model.data.vector_nonlinear_oracle_constraints) &&
           isempty(model.data.complementarity_constraints) &&
           length(model.data.qp_data) == 0 &&
           model.data.nlp_model === nothing &&
           model.native.problem === nothing
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(model, src)
end

MOI.get(::Optimizer, ::MOI.SolverName) = "POUNCE"

function MOI.supports_add_constrained_variable(::Optimizer, ::Type{MOI.Parameter{Float64}})
    return true
end

function _init_nlp_model(model)
    if model.data.nlp_model === nothing
        if !(model.evaluator.nlp_data.evaluator isa _EmptyNLPEvaluator)
            error("Cannot mix the new and legacy nonlinear APIs")
        end
        model.data.nlp_model = MOI.Nonlinear.Model()
    end
    return
end

function MOI.add_constrained_variable(model::Optimizer, set::MOI.Parameter{Float64})
    _assert_not_solving(model)
    _structure_changed!(model)
    _init_nlp_model(model)
    p = MOI.VariableIndex(_PARAMETER_OFFSET + length(model.data.parameters))
    push!(model.data.list_of_variable_indices, p)
    model.data.parameters[p] = MOI.Nonlinear.add_parameter(model.data.nlp_model, set.value)
    ci = MOI.ConstraintIndex{MOI.VariableIndex,typeof(set)}(p.value)
    return p, ci
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{Float64}},
)
    return haskey(model.data.parameters, MOI.VariableIndex(ci.value))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{F,S},
) where {F<:MOI.VariableIndex,S<:MOI.Parameter{Float64}}
    ret = [MOI.ConstraintIndex{F,S}(p.value) for p in keys(model.data.parameters)]
    sort!(ret; by = x -> x.value)
    return ret
end

function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{MOI.VariableIndex,MOI.Parameter{Float64}},
)
    return length(model.data.parameters)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{Float64}},
    set::MOI.Parameter{Float64},
)
    _assert_not_solving(model)
    MOI.throw_if_not_valid(model, ci)
    p = model.data.parameters[MOI.VariableIndex(ci.value)]
    model.data.nlp_model[p] = set.value
    # Parameters update the evaluator in place, but the previous solution no
    # longer describes the current model. Keep its derivative caches reusable.
    _parameters_changed!(model)
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    p = model.data.parameters[MOI.VariableIndex(ci.value)]
    return MOI.Parameter(model.data.nlp_model[p])
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    return MOI.VariableIndex(ci.value)
end

_replace_parameters(model::Optimizer, f) = f

function _replace_parameters(model::Optimizer, f::MOI.VariableIndex)
    if _is_parameter(f)
        return model.data.parameters[f]
    end
    return f
end

function _replace_parameters(model::Optimizer, f::MOI.ScalarAffineFunction)
    if any(_is_parameter, f.terms)
        g = convert(MOI.ScalarNonlinearFunction, f)
        return _replace_parameters(model, g)
    end
    return f
end

function _replace_parameters(model::Optimizer, f::MOI.ScalarQuadraticFunction)
    if any(_is_parameter, f.affine_terms) || any(_is_parameter, f.quadratic_terms)
        g = convert(MOI.ScalarNonlinearFunction, f)
        return _replace_parameters(model, g)
    end
    return f
end

function _replace_parameters(model::Optimizer, f::MOI.ScalarNonlinearFunction)
    for (i, arg) in enumerate(f.args)
        f.args[i] = _replace_parameters(model, arg)
    end
    return f
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{
        <:Union{
            MOI.VariableIndex,
            MOI.ScalarAffineFunction{Float64},
            MOI.ScalarQuadraticFunction{Float64},
            MOI.ScalarNonlinearFunction,
        },
    },
    ::Type{<:_SETS},
)
    return true
end

### MOI.ListOfConstraintTypesPresent

_add_scalar_nonlinear_constraints(ret, ::Nothing) = nothing

function _add_scalar_nonlinear_constraints(ret, nlp_model::MOI.Nonlinear.Model)
    for v in values(nlp_model.constraints)
        F, S = MOI.ScalarNonlinearFunction, typeof(v.set)
        if !((F, S) in ret)
            push!(ret, (F, S))
        end
    end
    return
end

function MOI.get(model::Optimizer, attr::MOI.ListOfConstraintTypesPresent)
    ret = MOI.get(model.data.variables, attr)
    append!(ret, MOI.get(model.data.qp_data, attr))
    _add_scalar_nonlinear_constraints(ret, model.data.nlp_model)
    for c in model.data.complementarity_constraints
        pair = (typeof(c.func), MOI.Complements)
        pair in ret || push!(ret, pair)
    end
    if !isempty(model.data.vector_nonlinear_oracle_constraints)
        push!(ret, (MOI.VectorOfVariables, MOI.VectorNonlinearOracle{Float64}))
    end
    if !isempty(model.data.parameters)
        push!(ret, (MOI.VariableIndex, MOI.Parameter{Float64}))
    end
    return ret
end

### MOI.Name

MOI.supports(::Optimizer, ::MOI.Name) = true

function MOI.set(model::Optimizer, ::MOI.Name, value::String)
    _assert_not_solving(model)
    model.name = value
    return
end

MOI.get(model::Optimizer, ::MOI.Name) = model.name

### MOI.Silent

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value)
    _assert_not_solving(model)
    model.silent = value
    _options_changed!(model)
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

### MOI.TimeLimitSec

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    _assert_not_solving(model)
    MOI.set(model, MOI.RawOptimizerAttribute("max_wall_time"), Float64(value))
    return
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    _assert_not_solving(model)
    delete!(model.options, "max_wall_time")
    _options_changed!(model)
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return get(model.options, "max_wall_time", nothing)
end

### MOI.RawOptimizerAttribute

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.set(model::Optimizer, p::MOI.RawOptimizerAttribute, value)
    _assert_not_solving(model)
    model.options[p.name] = value
    _options_changed!(model)
    # No need to reset model.native.problem because this gets handled in optimize!.
    return
end

function MOI.get(model::Optimizer, p::MOI.RawOptimizerAttribute)
    if !haskey(model.options, p.name)
        msg = "RawOptimizerAttribute with name $(p.name) is not already set."
        throw(MOI.GetAttributeNotAllowed(p, msg))
    end
    return model.options[p.name]
end

### Variables

"""
    Pounce.column(x::MOI.VariableIndex)

Return the column associated with a variable.
"""
Pounce.column(x::MOI.VariableIndex) = x.value

function MOI.add_variable(model::Optimizer)
    _assert_not_solving(model)
    push!(model.data.variable_primal_start, nothing)
    push!(model.data.mult_x_L, nothing)
    push!(model.data.mult_x_U, nothing)
    _structure_changed!(model)
    x = MOI.add_variable(model.data.variables)
    push!(model.data.list_of_variable_indices, x)
    return x
end

function MOI.is_valid(model::Optimizer, x::MOI.VariableIndex)
    if _is_parameter(x)
        return haskey(model.data.parameters, x)
    end
    return MOI.is_valid(model.data.variables, x)
end

function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return model.data.list_of_variable_indices
end

function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    return length(model.data.list_of_variable_indices)
end

function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS})
    return MOI.is_valid(model.data.variables, ci)
end

function MOI.get(
    model::Optimizer,
    attr::Union{
        MOI.NumberOfConstraints{MOI.VariableIndex,<:_SETS},
        MOI.ListOfConstraintIndices{MOI.VariableIndex,<:_SETS},
    },
)
    return MOI.get(model.data.variables, attr)
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ConstraintFunction,MOI.ConstraintSet},
    c::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    return MOI.get(model.data.variables, attr, c)
end

function MOI.add_constraint(model::Optimizer, x::MOI.VariableIndex, set::_SETS)
    _assert_not_solving(model)
    index = MOI.add_constraint(model.data.variables, x, set)
    _bounds_changed!(model)
    return index
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
    set::S,
) where {S<:_SETS}
    _assert_not_solving(model)
    MOI.set(model.data.variables, MOI.ConstraintSet(), ci, set)
    _bounds_changed!(model)
    return
end

function MOI.delete(model::Optimizer, ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS})
    _assert_not_solving(model)
    MOI.delete(model.data.variables, ci)
    _bounds_changed!(model)
    return
end

### ScalarAffineFunction and ScalarQuadraticFunction constraints

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F,<:_SETS},
) where {F<:Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}}}
    return MOI.is_valid(model.data.qp_data, ci)
end

function MOI.add_constraint(
    model::Optimizer,
    func::Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}},
    set::_SETS,
)
    _assert_not_solving(model)
    index = MOI.add_constraint(model.data.qp_data, func, set)
    _structure_changed!(model)
    return index
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.NumberOfConstraints{F,S},MOI.ListOfConstraintIndices{F,S}},
) where {
    F<:Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}},
    S<:_SETS,
}
    return MOI.get(model.data.qp_data, attr)
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ConstraintFunction,MOI.ConstraintSet},
    c::MOI.ConstraintIndex{F,<:_SETS},
) where {F<:Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}}}
    return MOI.get(model.data.qp_data, attr, c)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{F,S},
    set::S,
) where {
    F<:Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}},
    S<:_SETS,
}
    _assert_not_solving(model)
    MOI.set(model.data.qp_data, MOI.ConstraintSet(), ci, set)
    _bounds_changed!(model)
    return
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{<:MOI.ConstraintIndex{F,<:_SETS}},
) where {F<:Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}}}
    return true
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDualStart,
    c::MOI.ConstraintIndex{F,<:_SETS},
) where {F<:Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}}}
    return MOI.get(model.data.qp_data, attr, c)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{F,<:_SETS},
    value::Union{Real,Nothing},
) where {F<:Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}}}
    _assert_not_solving(model)
    MOI.throw_if_not_valid(model, ci)
    MOI.set(model.data.qp_data, attr, ci, value)
    # No need to reset model.native.problem, because this gets handled in optimize!.
    return
end

### ScalarNonlinearFunction

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,<:_SETS},
)
    if model.data.nlp_model === nothing
        return false
    end
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    return MOI.is_valid(model.data.nlp_model, index)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ListOfConstraintIndices{F,S},
) where {F<:MOI.ScalarNonlinearFunction,S<:_SETS}
    ret = MOI.ConstraintIndex{F,S}[]
    if model.data.nlp_model === nothing
        return ret
    end
    for (k, v) in model.data.nlp_model.constraints
        if v.set isa S
            push!(ret, MOI.ConstraintIndex{F,S}(k.value))
        end
    end
    return ret
end

function MOI.get(
    model::Optimizer,
    attr::MOI.NumberOfConstraints{F,S},
) where {F<:MOI.ScalarNonlinearFunction,S<:_SETS}
    if model.data.nlp_model === nothing
        return 0
    end
    return count(v.set isa S for v in values(model.data.nlp_model.constraints))
end

function MOI.add_constraint(model::Optimizer, f::MOI.ScalarNonlinearFunction, s::_SETS)
    _assert_not_solving(model)
    _init_nlp_model(model)
    if !isempty(model.data.parameters)
        _replace_parameters(model, f)
    end
    index = MOI.Nonlinear.add_constraint(model.data.nlp_model, f, s)
    _structure_changed!(model)
    return MOI.ConstraintIndex{typeof(f),typeof(s)}(index.value)
end

function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction})
    return true
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
    func::MOI.ScalarNonlinearFunction,
)
    _assert_not_solving(model)
    _init_nlp_model(model)
    if !isempty(model.data.parameters)
        _replace_parameters(model, func)
    end
    MOI.Nonlinear.set_objective(model.data.nlp_model, func)
    _structure_changed!(model)
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,<:_SETS},
)
    MOI.throw_if_not_valid(model, ci)
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    return model.data.nlp_model[index].set
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,S},
    set::S,
) where {S<:_SETS}
    _assert_not_solving(model)
    MOI.throw_if_not_valid(model, ci)
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    func = model.data.nlp_model[index].expression
    model.data.nlp_model.constraints[index] = MOI.Nonlinear.Constraint(func, set)
    # Refresh the NLP bound snapshot on the next solve; derivatives remain valid.
    _bounds_changed!(model)
    return
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{<:MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,<:_SETS}},
)
    return true
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,<:_SETS},
)
    MOI.throw_if_not_valid(model, ci)
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    return get(model.data.mult_g_nlp, index, nothing)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,<:_SETS},
    value::Union{Real,Nothing},
)
    _assert_not_solving(model)
    MOI.throw_if_not_valid(model, ci)
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    if value === nothing
        delete!(model.data.mult_g_nlp, index)
    else
        model.data.mult_g_nlp[index] = convert(Float64, value)
    end
    # No need to reset model.native.problem, because this gets handled in optimize!.
    return
end

### MOI.VectorOfVariables in MOI.VectorNonlinearOracle{Float64}

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorOfVariables},
    ::Type{MOI.VectorNonlinearOracle{Float64}},
)
    return true
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables,MOI.VectorNonlinearOracle{Float64}},
)
    return 1 <= ci.value <= length(model.data.vector_nonlinear_oracle_constraints)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ListOfConstraintIndices{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    n = length(model.data.vector_nonlinear_oracle_constraints)
    return MOI.ConstraintIndex{F,S}.(1:n)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.NumberOfConstraints{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    return length(model.data.vector_nonlinear_oracle_constraints)
end

function MOI.add_constraint(
    model::Optimizer,
    f::F,
    s::S,
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    _assert_not_solving(model)
    _structure_changed!(model)
    cache = _VectorNonlinearOracleCache(s)
    push!(model.data.vector_nonlinear_oracle_constraints, (f, cache))
    n = length(model.data.vector_nonlinear_oracle_constraints)
    return MOI.ConstraintIndex{F,S}(n)
end

function row(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    offset = length(model.data.qp_data)
    for i = 1:(ci.value-1)
        _, s = model.data.vector_nonlinear_oracle_constraints[i]
        offset += s.set.output_dimension
    end
    _, s = model.data.vector_nonlinear_oracle_constraints[ci.value]
    return offset .+ (1:s.set.output_dimension)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    f, _ = model.data.vector_nonlinear_oracle_constraints[ci.value]
    return MOI.get.(model, MOI.VariablePrimal(attr.result_index), f.variables)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    sign = -_dual_multiplier(model)
    f, s = model.data.vector_nonlinear_oracle_constraints[ci.value]
    λ = model.result.problem.mult_g[row(model, ci)]
    J = Tuple{Int,Int}[]
    _jacobian_structure(J, 0, f, s)
    J_val = zeros(length(J))
    _eval_constraint_jacobian(J_val, 0, model.result.problem.x, f, s)
    dual = zeros(MOI.dimension(s.set))
    # dual = λ' * J(x)
    col_to_index = Dict(x.value => j for (j, x) in enumerate(f.variables))
    for ((row, col), J_rc) in zip(J, J_val)
        dual[col_to_index[col]] += sign * J_rc * λ[row]
    end
    return dual
end

function MOI.get(
    model::Optimizer,
    attr::MOI.LagrangeMultiplier,
    ci::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return -_dual_multiplier(model) * model.result.problem.mult_g[row(model, ci)]
end

function MOI.supports(
    ::Optimizer,
    ::MOI.LagrangeMultiplierStart,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    return true
end

function MOI.get(
    model::Optimizer,
    attr::MOI.LagrangeMultiplierStart,
    ci::MOI.ConstraintIndex{F,S},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    _, cache = model.data.vector_nonlinear_oracle_constraints[ci.value]
    return cache.start
end

function MOI.set(
    model::Optimizer,
    attr::MOI.LagrangeMultiplierStart,
    ci::MOI.ConstraintIndex{F,S},
    start::Union{Nothing,Vector{Float64}},
) where {F<:MOI.VectorOfVariables,S<:MOI.VectorNonlinearOracle{Float64}}
    _assert_not_solving(model)
    _, cache = model.data.vector_nonlinear_oracle_constraints[ci.value]
    cache.start = start
    return
end

### UserDefinedFunction

MOI.supports(model::Optimizer, ::MOI.UserDefinedFunction) = true

function MOI.set(model::Optimizer, attr::MOI.UserDefinedFunction, args)
    _assert_not_solving(model)
    _init_nlp_model(model)
    MOI.Nonlinear.register_operator(model.data.nlp_model, attr.name, attr.arity, args...)
    _structure_changed!(model)
    return
end

### ListOfSupportedNonlinearOperators

function MOI.get(model::Optimizer, attr::MOI.ListOfSupportedNonlinearOperators)
    _init_nlp_model(model)
    return MOI.get(model.data.nlp_model, attr)
end

### MOI.VariablePrimalStart

function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex})
    return true
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimalStart, vi::MOI.VariableIndex)
    if _is_parameter(vi)
        throw(MOI.GetAttributeNotAllowed(attr, "Variable is a Parameter"))
    end
    MOI.throw_if_not_valid(model, vi)
    return model.data.variable_primal_start[Pounce.column(vi)]
end

function MOI.set(
    model::Optimizer,
    attr::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    _assert_not_solving(model)
    if _is_parameter(vi)
        throw(MOI.SetAttributeNotAllowed(attr, "Variable is a Parameter"))
    end
    MOI.throw_if_not_valid(model, vi)
    model.data.variable_primal_start[Pounce.column(vi)] = value
    # No need to reset model.native.problem, because this gets handled in optimize!.
    return
end

### MOI.ConstraintDualStart

_dual_start(::Optimizer, ::Nothing, ::Int = 1) = 0.0

function _dual_start(model::Optimizer, value::Real, scale::Int = 1)
    return _dual_multiplier(model) * value * scale
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{MOI.ConstraintIndex{MOI.VariableIndex,S}},
) where {S<:_SETS}
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
    value::Union{Real,Nothing},
)
    _assert_not_solving(model)
    MOI.throw_if_not_valid(model, ci)
    model.data.mult_x_L[ci.value] = value
    # No need to reset model.native.problem, because this gets handled in optimize!.
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    return model.data.mult_x_L[ci.value]
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
    value::Union{Real,Nothing},
)
    _assert_not_solving(model)
    MOI.throw_if_not_valid(model, ci)
    model.data.mult_x_U[ci.value] = value
    # No need to reset model.native.problem, because this gets handled in optimize!.
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    return model.data.mult_x_U[ci.value]
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
    value::Union{Real,Nothing},
) where {S<:Union{MOI.EqualTo{Float64},MOI.Interval{Float64}}}
    _assert_not_solving(model)
    MOI.throw_if_not_valid(model, ci)
    if value === nothing
        model.data.mult_x_L[ci.value] = nothing
        model.data.mult_x_U[ci.value] = nothing
    elseif value >= 0.0
        model.data.mult_x_L[ci.value] = value
        model.data.mult_x_U[ci.value] = 0.0
    else
        model.data.mult_x_L[ci.value] = 0.0
        model.data.mult_x_U[ci.value] = value
    end
    # No need to reset model.native.problem, because this gets handled in optimize!.
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
) where {S<:Union{MOI.EqualTo{Float64},MOI.Interval{Float64}}}
    MOI.throw_if_not_valid(model, ci)
    l = model.data.mult_x_L[ci.value]
    u = model.data.mult_x_U[ci.value]
    return (l === u === nothing) ? nothing : (l + u)
end

### MOI.NLPBlockDualStart

MOI.supports(::Optimizer, ::MOI.NLPBlockDualStart) = true

function MOI.set(model::Optimizer, ::MOI.NLPBlockDualStart, values::Union{Nothing,Vector})
    _assert_not_solving(model)
    model.data.nlp_dual_start = values
    # No need to reset model.native.problem, because this gets handled in optimize!.
    return
end

MOI.get(model::Optimizer, ::MOI.NLPBlockDualStart) = model.data.nlp_dual_start

### MOI.NLPBlock

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

# This may also be set by `optimize!` and contain the block created from
# ScalarNonlinearFunction
MOI.get(model::Optimizer, ::MOI.NLPBlock) = model.evaluator.nlp_data

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    _assert_not_solving(model)
    if model.data.nlp_model !== nothing
        error("Cannot mix the new and legacy nonlinear APIs")
    end
    model.evaluator.nlp_data = nlp_data
    _structure_changed!(model)
    return
end

### ObjectiveSense

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.set(model::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    _assert_not_solving(model)
    model.data.sense = sense
    _model_changed!(model)
    return
end

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.data.sense

### ObjectiveFunction

function MOI.get(model::Optimizer, attr::MOI.ObjectiveFunctionType)
    if model.data.nlp_model !== nothing && model.data.nlp_model.objective !== nothing
        return MOI.ScalarNonlinearFunction
    end
    return MOI.get(model.data.qp_data, attr)
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{
        <:Union{
            MOI.VariableIndex,
            MOI.ScalarAffineFunction{Float64},
            MOI.ScalarQuadraticFunction{Float64},
        },
    },
)
    return true
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{F},
) where {
    F<:Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
}
    return convert(F, MOI.get(model.data.qp_data, attr))
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{F},
    func::F,
) where {
    F<:Union{
        MOI.VariableIndex,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
    },
}
    _assert_not_solving(model)
    MOI.set(model.data.qp_data, attr, func)
    if model.data.nlp_model !== nothing
        MOI.Nonlinear.set_objective(model.data.nlp_model, nothing)
    end
    _structure_changed!(model)
    return
end
