# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

const _PARAMETER_OFFSET = 0x00f0000000000000

_is_parameter(x::MOI.VariableIndex) = x.value >= _PARAMETER_OFFSET

_is_parameter(term::MOI.ScalarAffineTerm) = _is_parameter(term.variable)

function _is_parameter(term::MOI.ScalarQuadraticTerm)
    return _is_parameter(term.variable_1) || _is_parameter(term.variable_2)
end

mutable struct _VectorNonlinearOracleCache
    set::MOI.VectorNonlinearOracle{Float64}
    x::Vector{Float64}
    start::Union{Nothing,Vector{Float64}}
    eval_f_timer::Float64
    eval_jacobian_timer::Float64
    eval_hessian_lagrangian_timer::Float64

    function _VectorNonlinearOracleCache(set::MOI.VectorNonlinearOracle{Float64})
        return new(set, zeros(set.input_dimension), nothing, 0.0, 0.0, 0.0)
    end
end

mutable struct _CallbackState
    task::Union{Nothing,Task}
    mode::Cint
    primal::Vector{Float64}
    has_primal::Bool
end

_CallbackState() = _CallbackState(nothing, Cint(0), Float64[], false)

struct _ComplementarityConstraint
    func::MOI.AbstractVectorFunction
    variables::Vector{MOI.VariableIndex}
    first_row::Int
end

Base.@kwdef mutable struct _ModelData
    revision::UInt = 1
    structure_revision::UInt = 1
    bounds_revision::UInt = 1
    sense::MOI.OptimizationSense = MOI.FEASIBILITY_SENSE
    parameters::Dict{MOI.VariableIndex,MOI.Nonlinear.ParameterIndex} =
        Dict{MOI.VariableIndex,MOI.Nonlinear.ParameterIndex}()
    variables::MOI.Utilities.VariablesContainer{Float64} =
        MOI.Utilities.VariablesContainer{Float64}()
    list_of_variable_indices::Vector{MOI.VariableIndex} = MOI.VariableIndex[]
    variable_primal_start::Vector{Union{Nothing,Float64}} = Union{Nothing,Float64}[]
    mult_x_L::Vector{Union{Nothing,Float64}} = Union{Nothing,Float64}[]
    mult_x_U::Vector{Union{Nothing,Float64}} = Union{Nothing,Float64}[]
    nlp_dual_start::Union{Nothing,Vector{Float64}} = nothing
    mult_g_nlp::Dict{MOI.Nonlinear.ConstraintIndex,Float64} =
        Dict{MOI.Nonlinear.ConstraintIndex,Float64}()
    complementarity_constraints::Vector{_ComplementarityConstraint} =
        _ComplementarityConstraint[]
    complementarity_variables::BitSet = BitSet()
    qp_data::QPBlockData{Float64} = QPBlockData{Float64}()
    nlp_model::Union{Nothing,MOI.Nonlinear.Model} = nothing
    ad_backend::MOI.Nonlinear.AbstractAutomaticDifferentiation =
        MOI.Nonlinear.SparseReverseMode()
    vector_nonlinear_oracle_constraints::Vector{
        Tuple{MOI.VectorOfVariables,_VectorNonlinearOracleCache},
    } = Tuple{MOI.VectorOfVariables,_VectorNonlinearOracleCache}[]
end

Base.@kwdef mutable struct _EvaluatorCache
    structure_revision::UInt = 0
    bounds_revision::UInt = 0
    nlp_data::MOI.NLPBlockData = MOI.NLPBlockData([], _EmptyNLPEvaluator(), false)
    jacobian_sparsity::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[]
    hessian_sparsity::Union{Nothing,Vector{Tuple{Int,Int}}} = nothing
    has_only_linear_constraints::Bool = false
    complementarity::MOI.AbstractNLPEvaluator = _EmptyNLPEvaluator()
    complementarity_jacobian_nnz::Int = 0
    complementarity_hessian_nnz::Int = 0
    nlp_rows::Dict{MOI.Nonlinear.ConstraintIndex,Int} =
        Dict{MOI.Nonlinear.ConstraintIndex,Int}()
end

Base.@kwdef mutable struct _NativeCache
    problem::Union{Nothing,Pounce.IpoptProblem} = nothing
    revision::UInt = 0
    options_revision::UInt = 0
    pending_x_bounds::Set{Int} = Set{Int}()
    pending_g_bounds::Set{Int} = Set{Int}()
end

Base.@kwdef mutable struct _SolveResult
    # Keep the Julia result buffers alive without copying them. The next solve
    # clears this reference before it can overwrite any buffers.
    problem::Union{Nothing,Pounce.IpoptProblem} = nothing
    revision::UInt = 0
    invalid_model_reason::Union{Nothing,String} = nothing
    solve_time::Float64 = NaN
    native_solve_time::Float64 = NaN
    barrier_iterations::Int = 0
    tol::Float64 = 1e-8
    acceptable_tol::Float64 = 1e-6
end

"""
    Optimizer()

Create a POUNCE optimizer backed by the C library.
"""
Base.@kwdef mutable struct Optimizer <: MOI.AbstractOptimizer
    data::_ModelData = _ModelData()
    evaluator::_EvaluatorCache = _EvaluatorCache()
    native::_NativeCache = _NativeCache()
    result::_SolveResult = _SolveResult()
    name::String = ""
    silent::Bool = false
    record_iteration_history::Bool = false
    options::Dict{String,Any} = Dict{String,Any}()
    options_revision::UInt = 1
    callback::Union{Nothing,Function} = nothing
    callback_state::_CallbackState = _CallbackState()
end

function _assert_not_solving(model::Optimizer)
    if model.native.problem !== nothing && model.native.problem.solving
        error("Cannot modify or recursively solve a POUNCE optimizer during a solve")
    end
    return
end

function _reset_result!(model::Optimizer)
    model.result = _SolveResult(revision = model.data.revision)
    return
end

function _release_native!(model::Optimizer)
    if model.native.problem !== nothing
        Pounce.FreeIpoptProblem(model.native.problem)
    end
    model.native = _NativeCache()
    return
end

function _model_changed!(
    model::Optimizer;
    structure = false,
    bounds = false,
    rebuild_native = true,
)
    _assert_not_solving(model)
    model.data.revision += 1
    if structure
        model.data.structure_revision += 1
    end
    if structure || bounds
        model.data.bounds_revision += 1
    end
    _reset_result!(model)
    if rebuild_native
        _release_native!(model)
    elseif model.native.problem !== nothing
        model.native.revision = model.data.revision
    end
    return
end

_structure_changed!(model::Optimizer) = _model_changed!(model; structure = true)
function _bounds_changed!(model::Optimizer; variable = nothing, constraint = nothing)
    _model_changed!(model; bounds = true, rebuild_native = !Pounce.supports_bound_updates())
    if model.native.problem !== nothing
        variable === nothing || push!(model.native.pending_x_bounds, variable)
        constraint === nothing || push!(model.native.pending_g_bounds, constraint)
    end
    return
end
_parameters_changed!(model::Optimizer) = _model_changed!(model; rebuild_native = false)

function _options_changed!(model::Optimizer)
    # Retain the completed result until another solve starts. Rebuilding the
    # handle then also removes sticky options (e.g. a cleared time limit).
    model.options_revision += 1
    return
end

_has_result(model::Optimizer) =
    model.result.problem !== nothing && model.result.revision == model.data.revision

const _SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

MOI.get(::Optimizer, ::MOI.SolverVersion) = string(Pounce.GetIpoptVersion())

### _EmptyNLPEvaluator

struct _EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end

MOI.features_available(::_EmptyNLPEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(::_EmptyNLPEvaluator, ::Any) = nothing
MOI.eval_constraint(::_EmptyNLPEvaluator, g, x) = nothing
MOI.jacobian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.hessian_lagrangian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.eval_constraint_jacobian(::_EmptyNLPEvaluator, J, x) = nothing
MOI.eval_hessian_lagrangian(::_EmptyNLPEvaluator, H, x, σ, μ) = nothing
