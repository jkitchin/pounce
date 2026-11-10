# Copyright (c) 2026: Frederik Geth
# Licensed under MIT; see ../../LICENSE.md.

"""
Native scaled primal infeasibility, dual infeasibility and bound/multiplier
complementarity from the completed solve, returned as a named tuple. These are
solver diagnostics, not independently recomputed feasibility or MPCC certificates.
"""
struct NativeResiduals <: MOI.AbstractModelAttribute end

"""Restoration calls, inner/outer iterations and wall seconds from the completed solve."""
struct RestorationStatistics <: MOI.AbstractModelAttribute end

"""Finite-difference Hessian census; see `Pounce.GetPounceFdHessianStats`."""
struct FiniteDifferenceHessianStatistics <: MOI.AbstractModelAttribute end

"""Linear-solver summary, or `nothing` when the backend did not provide one."""
struct LinearSolverStatistics <: MOI.AbstractModelAttribute end

"""
    SolveReport(detail = "summary")

The completed native solve report as a JSON string, using the native
`pounce.solve-report/v1` schema. `detail` must be `"summary"` or `"full"`.
Full reports include a trajectory if `RecordIterationHistory()` was enabled
before solving. Serialization uses a temporary file, removed before returning.
No JSON parser dependency is loaded by the wrapper.
"""
struct SolveReport <: MOI.AbstractModelAttribute
    detail::String
    function SolveReport(detail::AbstractString = "summary")
        detail in ("summary", "full") ||
            throw(ArgumentError("Report detail must be summary or full"))
        return new(String(detail))
    end
end

"""MPCC outcome statistics, or nothing if the completed solve ran no MPCC driver."""
struct ComplementarityStatistics <: MOI.AbstractModelAttribute end
"""Owned statistics for each lowered solve in driver order (empty without a driver)."""
struct ComplementaritySolves <: MOI.AbstractModelAttribute end
"""Owned MPCC driver text, or nothing if unavailable. `which` is POUNCE_CC_TEXT_*."""
struct ComplementarityText <: MOI.AbstractModelAttribute
    which::Int
    ComplementarityText(which::Integer = Pounce.POUNCE_CC_TEXT_SUMMARY) = new(which)
end

const _DiagnosticAttribute = Union{
    ComplementarityStatistics,
    ComplementaritySolves,
    ComplementarityText,
    NativeResiduals,
    RestorationStatistics,
    FiniteDifferenceHessianStatistics,
    LinearSolverStatistics,
    SolveReport,
}
MOI.is_set_by_optimize(::_DiagnosticAttribute) = true
# These values contain scalars, strings and counts, never MOI indices. In
# particular the JSON is the native report, not a JuMP-variable-indexed object.
MOI.Utilities.map_indices(::Any, ::_DiagnosticAttribute, value) = value

function _diagnostic_problem(model::Optimizer, attr::_DiagnosticAttribute)
    _has_result(model) ||
        throw(MOI.GetAttributeNotAllowed(attr, "No completed solve is available"))
    return model.result.problem::Pounce.IpoptProblem
end

function MOI.get(model::Optimizer, attr::NativeResiduals)
    prob = _diagnostic_problem(model, attr)
    return (
        primal_infeasibility = Pounce.GetIpoptPrimalInf(prob),
        dual_infeasibility = Pounce.GetIpoptDualInf(prob),
        complementarity = Pounce.GetIpoptComplInf(prob),
    )
end

for (attr, getter) in (
    (RestorationStatistics, :GetPounceRestorationStats),
    (FiniteDifferenceHessianStatistics, :GetPounceFdHessianStats),
    (LinearSolverStatistics, :GetPounceLinearSolverStats),
)
    @eval MOI.get(model::Optimizer, attr::$attr) =
        Pounce.$getter(_diagnostic_problem(model, attr))
end

function MOI.get(model::Optimizer, attr::SolveReport)
    prob = _diagnostic_problem(model, attr)
    return mktempdir() do dir
        path = joinpath(dir, "solve.json")
        Pounce.IpoptWriteSolveReport(prob, path; detail = attr.detail)
        return read(path, String)
    end
end

"""
    RecordIterationHistory()

Boolean optimizer setting, false by default. Capture a trajectory for full
solve reports. Changes apply to the next solve and preserve the last result at the MOI optimizer
level. JuMP may additionally mark a cached model dirty when settings change.
Turning it off rebuilds the native problem at the next solve because the C API
only offers an enable operation. Retained by `MOI.empty!`, like solver options.
"""
struct RecordIterationHistory <: MOI.AbstractOptimizerAttribute end
MOI.supports(::Optimizer, ::RecordIterationHistory) = true
MOI.get(model::Optimizer, ::RecordIterationHistory) = model.record_iteration_history
function MOI.set(model::Optimizer, ::RecordIterationHistory, value::Bool)
    _assert_not_solving(model)
    if model.record_iteration_history != value
        model.record_iteration_history = value
        _options_changed!(model)
    end
    return
end

function MOI.get(model::Optimizer, attr::ComplementarityStatistics)
    prob = _diagnostic_problem(model, attr)
    Pounce.supports_complementarity() || return nothing
    return Pounce.GetPounceComplementarityStats(prob)
end
function MOI.get(model::Optimizer, attr::ComplementaritySolves)
    prob = _diagnostic_problem(model, attr)
    Pounce.supports_complementarity() || return NamedTuple[]
    stats = Pounce.GetPounceComplementarityStats(prob)
    stats === nothing && return NamedTuple[]
    return [Pounce.GetPounceComplementaritySolve(prob, k) for k = 1:stats.n_solves]
end
function MOI.get(model::Optimizer, attr::ComplementarityText)
    prob = _diagnostic_problem(model, attr)
    Pounce.supports_complementarity() || return nothing
    return Pounce.GetPounceComplementarityText(prob, attr.which)
end
