# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

#!format:off
const _STATUS_CODES = Dict{
    Pounce.ApplicationReturnStatus,         Tuple{MOI.TerminationStatusCode, MOI.ResultStatusCode}
}(
    Pounce.Solve_Succeeded                    => (MOI.LOCALLY_SOLVED,        MOI.FEASIBLE_POINT),
    Pounce.Solved_To_Acceptable_Level         => (MOI.ALMOST_LOCALLY_SOLVED, MOI.NEARLY_FEASIBLE_POINT),
    Pounce.Infeasible_Problem_Detected        => (MOI.LOCALLY_INFEASIBLE,    MOI.INFEASIBLE_POINT),
    Pounce.Search_Direction_Becomes_Too_Small => (MOI.SLOW_PROGRESS,         MOI.UNKNOWN_RESULT_STATUS),
    Pounce.Diverging_Iterates                 => (MOI.NORM_LIMIT,            MOI.UNKNOWN_RESULT_STATUS),
    Pounce.User_Requested_Stop                => (MOI.INTERRUPTED,           MOI.UNKNOWN_RESULT_STATUS),
    Pounce.Feasible_Point_Found               => (MOI.LOCALLY_SOLVED,        MOI.FEASIBLE_POINT),
    Pounce.Maximum_Iterations_Exceeded        => (MOI.ITERATION_LIMIT,       MOI.UNKNOWN_RESULT_STATUS),
    Pounce.Restoration_Failed                 => (MOI.OTHER_ERROR,           MOI.UNKNOWN_RESULT_STATUS),
    Pounce.Error_In_Step_Computation          => (MOI.NUMERICAL_ERROR,       MOI.UNKNOWN_RESULT_STATUS),
    Pounce.Maximum_CpuTime_Exceeded           => (MOI.TIME_LIMIT,            MOI.UNKNOWN_RESULT_STATUS),
    Pounce.Maximum_WallTime_Exceeded          => (MOI.TIME_LIMIT,            MOI.UNKNOWN_RESULT_STATUS),
    Pounce.Not_Enough_Degrees_Of_Freedom      => (MOI.INVALID_MODEL,         MOI.UNKNOWN_RESULT_STATUS),
    Pounce.Invalid_Problem_Definition         => (MOI.INVALID_MODEL,         MOI.UNKNOWN_RESULT_STATUS),
    Pounce.Invalid_Option                     => (MOI.INVALID_OPTION,        MOI.UNKNOWN_RESULT_STATUS),
    Pounce.Invalid_Number_Detected            => (MOI.INVALID_MODEL,         MOI.UNKNOWN_RESULT_STATUS),
    Pounce.Unrecoverable_Exception            => (MOI.OTHER_ERROR,           MOI.UNKNOWN_RESULT_STATUS),
    Pounce.NonIpopt_Exception_Thrown          => (MOI.OTHER_ERROR,           MOI.UNKNOWN_RESULT_STATUS),
    Pounce.Insufficient_Memory                => (MOI.MEMORY_LIMIT,          MOI.UNKNOWN_RESULT_STATUS),
    Pounce.Internal_Error                     => (MOI.OTHER_ERROR,           MOI.UNKNOWN_RESULT_STATUS),
)
#!format:on

### MOI.ResultCount

# Setup or callback failures must not expose uninitialized results.
function MOI.get(model::Optimizer, ::MOI.ResultCount)
    inner = model.result.problem
    return _has_result(model) && inner.status > -10 && isfinite(inner.obj_val) ? 1 : 0
end

### MOI.TerminationStatus

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.result.invalid_model_reason !== nothing
        return MOI.INVALID_MODEL
    elseif !_has_result(model)
        return MOI.OPTIMIZE_NOT_CALLED
    end
    status, _ = _STATUS_CODES[Pounce.ApplicationReturnStatus(model.result.problem.status)]
    return status
end

### MOI.RawStatusString

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.result.invalid_model_reason !== nothing
        return model.result.invalid_model_reason
    elseif !_has_result(model)
        return "Optimize not called"
    end
    return string(Pounce.ApplicationReturnStatus(model.result.problem.status))
end

### MOI.PrimalStatus

function _manually_evaluated_primal_status(model::Optimizer)
    x, g = model.result.problem.x, model.result.problem.g
    x_L, x_U = model.data.variables.lower, model.data.variables.upper
    g_L, g_U = copy(model.data.qp_data.g_L), copy(model.data.qp_data.g_U)
    for (_, cache) in model.data.vector_nonlinear_oracle_constraints
        append!(g_L, cache.set.l)
        append!(g_U, cache.set.u)
    end
    for bound in model.evaluator.nlp_data.constraint_bounds
        push!(g_L, bound.lower)
        push!(g_U, bound.upper)
    end
    m, n = length(g_L), length(x)
    # 1e-8 is the default tolerance
    tol = model.result.tol
    cc_violation = _complementarity_violation(model)
    if cc_violation <= tol &&
       all(x_L[i] - tol <= x[i] <= x_U[i] + tol for i = 1:n) &&
       all(g_L[i] - tol <= g[i] <= g_U[i] + tol for i = 1:m)
        return MOI.FEASIBLE_POINT
    end
    # 1e-6 is the default acceptable tolerance
    atol = model.result.acceptable_tol
    if cc_violation <= atol &&
       all(x_L[i] - atol <= x[i] <= x_U[i] + atol for i = 1:n) &&
       all(g_L[i] - atol <= g[i] <= g_U[i] + atol for i = 1:m)
        return MOI.NEARLY_FEASIBLE_POINT
    end
    return MOI.INFEASIBLE_POINT
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    _, status = _STATUS_CODES[Pounce.ApplicationReturnStatus(model.result.problem.status)]
    if status == MOI.UNKNOWN_RESULT_STATUS ||
       !isempty(model.data.complementarity_constraints)
        # Not sure. RestorationFailure can terminate at a feasible (but
        # non-stationary) point.
        return _manually_evaluated_primal_status(model)
    end
    return status
end

### MOI.DualStatus

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    isempty(model.data.complementarity_constraints) || return MOI.UNKNOWN_RESULT_STATUS
    _, status = _STATUS_CODES[Pounce.ApplicationReturnStatus(model.result.problem.status)]
    return status
end

### MOI.SolveTimeSec

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.result.solve_time

"""
    NativeSolveTimeSec()

Native solver wall time from the most recent `optimize!`, excluding Julia-side
setup. `NaN` before solving or when the attempt throws or never reaches the
native solve. A native return without statistics reports zero. In contrast,
`MOI.SolveTimeSec` measures the entire optimizer call, including setup.
"""
struct NativeSolveTimeSec <: MOI.AbstractModelAttribute end

MOI.is_set_by_optimize(::NativeSolveTimeSec) = true
MOI.get(model::Optimizer, ::NativeSolveTimeSec) = model.result.native_solve_time

### MOI.BarrierIterations

MOI.get(model::Optimizer, ::MOI.BarrierIterations) = model.result.barrier_iterations

### MOI.ObjectiveValue

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.result.problem.obj_val
end

### MOI.VariablePrimal

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    if _is_parameter(vi)
        p = model.data.parameters[vi]
        return model.data.nlp_model[p]
    end
    return model.result.problem.x[Pounce.column(vi)]
end

### MOI.ConstraintPrimal

function row(
    model::Optimizer,
    ci::MOI.ConstraintIndex{F},
) where {F<:Union{MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64}}}
    return ci.value
end

function _nlp_constraint_offset(model::Optimizer)
    offset = length(model.data.qp_data)
    for (_, s) in model.data.vector_nonlinear_oracle_constraints
        offset += s.set.output_dimension
    end
    return offset
end

function row(model::Optimizer, ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction})
    return _nlp_constraint_offset(model) + ci.value
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{F,<:_SETS},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarNonlinearFunction,
    },
}
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.result.problem.g[row(model, ci)]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.result.problem.x[ci.value]
end

### MOI.ConstraintDual

_dual_multiplier(model::Optimizer) = model.data.sense == MOI.MIN_SENSE ? 1.0 : -1.0

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{F,<:_SETS},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarNonlinearFunction,
    },
}
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    s = -_dual_multiplier(model)
    return s * model.result.problem.mult_g[row(model, ci)]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.result.problem.mult_x_L[ci.value] - model.result.problem.mult_x_U[ci.value]
    if _is_paired_variable(model, ci.value)
        return -_dual_multiplier(model) * model.result.problem.mult_x_U[ci.value]
    end
    return min(0.0, _dual_multiplier(model) * rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.result.problem.mult_x_L[ci.value] - model.result.problem.mult_x_U[ci.value]
    if _is_paired_variable(model, ci.value)
        return _dual_multiplier(model) * model.result.problem.mult_x_L[ci.value]
    end
    return max(0.0, _dual_multiplier(model) * rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.result.problem.mult_x_L[ci.value] - model.result.problem.mult_x_U[ci.value]
    return _dual_multiplier(model) * rc
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Interval{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.result.problem.mult_x_L[ci.value] - model.result.problem.mult_x_U[ci.value]
    return _dual_multiplier(model) * rc
end

### MOI.NLPBlockDual

function MOI.get(model::Optimizer, attr::MOI.NLPBlockDual)
    MOI.check_result_index_bounds(model, attr)
    s = -_dual_multiplier(model)
    return s .* model.result.problem.mult_g[(_nlp_constraint_offset(
        model,
    )+1):_complementarity_offset(model)]
end
