# Copyright (c) 2026: Frederik Geth
# Licensed under MIT; see ../../LICENSE.md.

const _ComplementarityFunction = Union{
    MOI.VectorOfVariables,
    MOI.VectorAffineFunction{Float64},
    MOI.VectorQuadraticFunction{Float64},
    MOI.VectorNonlinearFunction,
}

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:_ComplementarityFunction},
    ::Type{MOI.Complements},
)
    return Pounce.supports_complementarity()
end

function _complements_variable(f)
    f isa MOI.VariableIndex && return f
    # Canonicalize polynomial representations, but never approximate coefficients.
    if f isa Union{MOI.ScalarAffineFunction,MOI.ScalarQuadraticFunction}
        return convert(MOI.VariableIndex, MOI.Utilities.canonical(f))
    elseif f isa MOI.ScalarNonlinearFunction && f.head == :+ && length(f.args) == 1
        return _complements_variable(only(f.args))
    end
    throw(ArgumentError("The second half of Complements must contain single variables"))
end

function MOI.add_constraint(
    model::Optimizer,
    f::F,
    set::MOI.Complements,
) where {F<:_ComplementarityFunction}
    _assert_not_solving(model)
    MOI.supports_constraint(model, F, MOI.Complements) ||
        throw(MOI.UnsupportedConstraint{F,MOI.Complements}())
    MOI.output_dimension(f) == MOI.dimension(set) ||
        throw(DimensionMismatch("Complements function and set dimensions differ"))
    scalars = collect(MOI.Utilities.eachscalar(f))
    n = length(scalars) ÷ 2
    vars = MOI.VariableIndex[]
    used = BitSet()
    for component in scalars[(n+1):end]
        var = try
            _complements_variable(component)
        catch err
            err isa Union{InexactError,ArgumentError} || rethrow()
            throw(
                MOI.AddConstraintNotAllowed{F,MOI.Complements}(
                    "The second half of Complements must contain single variables",
                ),
            )
        end
        MOI.throw_if_not_valid(model, var)
        _is_parameter(var) && throw(
            MOI.AddConstraintNotAllowed{F,MOI.Complements}(
                "A paired variable cannot be a Parameter",
            ),
        )
        if var.value in used || var.value in model.data.complementarity_variables
            throw(
                MOI.AddConstraintNotAllowed{F,MOI.Complements}(
                    "A variable may occur in only one complementarity pair; repeated variables need an explicit reformulation",
                ),
            )
        end
        push!(vars, var)
        push!(used, var.value)
    end
    # Validate all references without mutating the supplied function or model.
    MOI.Utilities.map_indices(f) do var
        MOI.throw_if_not_valid(model, var)
        return var
    end
    first_row = 1 + _num_complementarity_rows(model)
    push!(
        model.data.complementarity_constraints,
        _ComplementarityConstraint(deepcopy(f), vars, first_row),
    )
    union!(model.data.complementarity_variables, used)
    _structure_changed!(model)
    return MOI.ConstraintIndex{F,MOI.Complements}(
        length(model.data.complementarity_constraints),
    )
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F,MOI.Complements},
) where {F}
    return 1 <= ci.value <= length(model.data.complementarity_constraints) &&
           model.data.complementarity_constraints[ci.value].func isa F
end
function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{F,MOI.Complements},
) where {F}
    return [
        MOI.ConstraintIndex{F,MOI.Complements}(i) for
        (i, c) in enumerate(model.data.complementarity_constraints) if c.func isa F
    ]
end
MOI.get(model::Optimizer, ::MOI.NumberOfConstraints{F,MOI.Complements}) where {F} =
    length(MOI.get(model, MOI.ListOfConstraintIndices{F,MOI.Complements}()))
function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    ci::MOI.ConstraintIndex{<:Any,MOI.Complements},
)
    MOI.throw_if_not_valid(model, ci)
    return deepcopy(model.data.complementarity_constraints[ci.value].func)
end
function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{<:Any,MOI.Complements},
)
    MOI.throw_if_not_valid(model, ci)
    return MOI.Complements(
        2 * length(model.data.complementarity_constraints[ci.value].variables),
    )
end

_num_complementarity_rows(model) =
    sum((length(c.variables) for c in model.data.complementarity_constraints); init = 0)
_complementarity_offset(model) =
    _nlp_constraint_offset(model) + length(model.evaluator.nlp_data.constraint_bounds)
function _complementarity_rows(model, c)
    return _complementarity_offset(model) .+
           (c.first_row:(c.first_row+length(c.variables)-1))
end

function _setup_complementarity_evaluator!(model)
    if isempty(model.data.complementarity_constraints)
        model.evaluator.complementarity = _EmptyNLPEvaluator()
        return
    end
    nlp = MOI.Nonlinear.Model()
    if model.data.nlp_model !== nothing
        # Share the parameter vector so MOI.Parameter updates preserve both
        # initialized evaluators. Operators are rebuilt on structural changes.
        nlp.parameters = model.data.nlp_model.parameters
        nlp.operators = model.data.nlp_model.operators
    end
    for c in model.data.complementarity_constraints
        scalars = collect(MOI.Utilities.eachscalar(deepcopy(c.func)))
        for f in scalars[1:length(c.variables)]
            MOI.Nonlinear.add_constraint(
                nlp,
                _replace_parameters(model, f),
                MOI.Interval(-Inf, Inf),
            )
        end
    end
    vars = MOI.get(model.data.variables, MOI.ListOfVariableIndices())
    model.evaluator.complementarity =
        MOI.Nonlinear.Evaluator(nlp, model.data.ad_backend, vars)
    return
end

function _declare_complements!(model, inner)
    for c in model.data.complementarity_constraints
        for (row, var) in zip(_complementarity_rows(model, c), c.variables)
            Pounce.AddPounceComplementsConstraint(inner, row, var.value)
        end
    end
    return
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{<:Any,MOI.Complements},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    c = model.data.complementarity_constraints[ci.value]
    prob = model.result.problem
    return vcat(
        prob.g[_complementarity_rows(model, c)],
        [prob.x[v.value] for v in c.variables],
    )
end

# Multipliers use the same objective-sense convention as ordinary rows. Native
# paired-bound multipliers stay on the explicit variable-bound constraints (see
# MOI_results.jl); the x half is zero to avoid counting them twice. These are
# stationarity multipliers, not a convex dual certificate, and need not be unique.
function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{<:Any,MOI.Complements},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    c = model.data.complementarity_constraints[ci.value]
    return vcat(
        -_dual_multiplier(model) .*
        model.result.problem.mult_g[_complementarity_rows(model, c)],
        zeros(length(c.variables)),
    )
end

function _is_paired_variable(model, i)
    lower, upper = model.data.variables.lower[i], model.data.variables.upper[i]
    # Fixed/free Complements lower to ordinary NLP rows, without a native pair.
    return lower != upper &&
           (lower > -1e19 || upper < 1e19) &&
           i in model.data.complementarity_variables
end

function _complementarity_violation(model)
    prob = model.result.problem
    residual = 0.0
    for c in model.data.complementarity_constraints
        for (row, var) in zip(_complementarity_rows(model, c), c.variables)
            i = var.value
            lower, upper = model.data.variables.lower[i], model.data.variables.upper[i]
            lower = lower <= -1e19 ? -Inf : lower
            upper = upper >= 1e19 ? Inf : upper
            # Natural map for mixed complementarity. It is zero for every F
            # when x is fixed, and |F| when x is free. Bounds are checked too.
            x, f = prob.x[i], prob.g[row]
            isfinite(x) && isfinite(f) || return Inf
            # Equivalent to x - clamp(x - f, lower, upper), but without
            # subtracting F from a large x and rounding an interior residual
            # to zero. In particular, a free variable retains |F| exactly.
            residual = max(residual, abs(clamp(f, x - upper, x - lower)))
        end
    end
    return residual
end
