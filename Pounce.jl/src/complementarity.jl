# Copyright (c) 2026: Frederik Geth
# Licensed under MIT; see ../LICENSE.md.

# New-in-0.17 entry points are optional, so older libraries still load. Requiring
# PounceErrorString also excludes the development ABI before #1161's flags change.
const _complementarity_symbols = (
    :PounceErrorString,
    :AddPounceComplementsConstraint,
    :AddPounceComplementarityPair,
    :SetPounceComplementarityPairs,
    :ClearPounceComplementarityPairs,
    :GetPounceComplementarityPairCount,
    :GetPounceComplementarityPairs,
    :GetPounceComplementarityStats,
    :GetPounceComplementaritySolve,
    :GetPounceComplementarityText,
)

"""Whether the loaded library provides the finalized native complementarity API."""
supports_complementarity() =
    all(s -> Libdl.dlsym_e(_library_handle[], s) != C_NULL, _complementarity_symbols)

function _complementarity_symbol(prob::IpoptProblem, name::Symbol)
    prob.solving && error("Complementarity APIs are only available between solves")
    Base.unsafe_convert(Ptr{Cvoid}, prob)
    supports_complementarity() || error(
        "The loaded POUNCE library does not support the finalized complementarity API",
    )
    return Libdl.dlsym(_library_handle[], name)
end

"""A refused native extension operation; `bad_index` is one-based, or nothing."""
struct NativeAPIError <: Exception
    operation::Symbol
    code::Int
    message::String
    bad_index::Union{Nothing,Int}
end
function Base.showerror(io::IO, err::NativeAPIError)
    print(io, err.operation, ": ", err.message, " (", err.code, ")")
    err.bad_index === nothing || print(io, "; batch entry ", err.bad_index)
end

"""Copy the native extension error message. Nonnegative outcomes are function-specific."""
function PounceErrorString(code::Integer)
    symbol = Libdl.dlsym_e(_library_handle[], :PounceErrorString)
    symbol == C_NULL && error("The loaded POUNCE library has no extension error messages")
    return unsafe_string(ccall(symbol, Cstring, (Cint,), code))
end
function _check_native_error(operation::Symbol, code::Integer, bad_index::Integer = -1)
    code >= 0 || throw(
        NativeAPIError(
            operation,
            Int(code),
            PounceErrorString(code),
            bad_index < 0 ? nothing : Int(bad_index) + 1,
        ),
    )
    return Int(code)
end

const POUNCE_OK = 0
const POUNCE_ERR_NULL = -1
const POUNCE_ERR_COUNT = -2
const POUNCE_ERR_OUT_OF_RANGE = -3
const POUNCE_ERR_DUPLICATE = -4
const POUNCE_ERR_NAN = -5
const POUNCE_ERR_FLAGS = -6
const POUNCE_ERR_ARGUMENT = -7
const POUNCE_ERR_INTERNAL = -8
const POUNCE_ERR_VAR_ALREADY_PAIRED = -20
const POUNCE_ERR_ROW_ALREADY_PAIRED = -21
const POUNCE_ERR_SHAPE = -22
const POUNCE_ERR_ROW_NOT_FREE = -23
const POUNCE_ERR_VAR_BOUNDS = -24
const POUNCE_ERR_PAIR_CHANGED = -40
const POUNCE_ERR_COMPLEMENTS_CLASS = -41
const POUNCE_ERR_WORKING_SET = -42
const POUNCE_ERR_INCOMPLETE = -60
const POUNCE_ERR_CONES = -61
const POUNCE_ERR_OPTIONS = -62
const POUNCE_ERR_INDEFINITE = -63
const POUNCE_COMPLEMENTS_PAIR_MIXED = 0
const POUNCE_COMPLEMENTS_PAIR_LOWER = 1
const POUNCE_COMPLEMENTS_PAIR_UPPER = 2
const POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR = 3
const POUNCE_COMPLEMENTS_VAR_FIXED_NO_PAIR = 4
const POUNCE_CC_CLASS_S = 0
const POUNCE_CC_CLASS_W = 1
const POUNCE_CC_CLASS_NONE = 2
const POUNCE_CC_CLASS_NOT_SEARCHED = 3
const POUNCE_CC_CLASS_ABSENT = -1
const POUNCE_CC_REPORTED_ANSWER = 0
const POUNCE_CC_REPORTED_WEAK_FALLBACK = 1
const POUNCE_CC_REPORTED_FEASIBLE_UNCERTIFIED = 2
const POUNCE_CC_REPORTED_VERDICT = 3
const POUNCE_CC_B_VERDICT_B = 0
const POUNCE_CC_B_VERDICT_NOT_B = 1
const POUNCE_CC_B_VERDICT_NOT_CHECKED = 2
const POUNCE_CC_B_VERDICT_ABSENT = -1
const POUNCE_CC_TEXT_SUMMARY = 0
const POUNCE_CC_TEXT_POLISH_REASON = 1
const POUNCE_CC_TEXT_B_REASON = 2

function _check_pair_indices(prob, row, var)
    1 <= row <= prob.m ||
        throw(ArgumentError("Complementarity row must lie in 1:$(prob.m)"))
    1 <= var <= prob.n ||
        throw(ArgumentError("Complementarity variable must lie in 1:$(prob.n)"))
    return
end

"""
Declare MOI's [F, x] semantics using a dedicated free row and one-based indices.
Returns the native outcome (0–4); negative errors throw NativeAPIError. Native
row edits persist after clearing pairs: recreate the handle to undo lowering.
"""
function AddPounceComplementsConstraint(prob::IpoptProblem, row::Integer, var::Integer)
    symbol = _complementarity_symbol(prob, :AddPounceComplementsConstraint)
    _check_pair_indices(prob, row, var)
    return _check_native_error(
        :AddPounceComplementsConstraint,
        ccall(symbol, Cint, (Ptr{Cvoid}, Cint, Cint), prob, row, var),
    )
end

"""Declare a native pair on an already bounded row, using one-based indices."""
function AddPounceComplementarityPair(prob::IpoptProblem, row::Integer, var::Integer)
    symbol = _complementarity_symbol(prob, :AddPounceComplementarityPair)
    _check_pair_indices(prob, row, var)
    return _check_native_error(
        :AddPounceComplementarityPair,
        ccall(symbol, Cint, (Ptr{Cvoid}, Cint, Cint), prob, row, var),
    )
end

"""Replace native pairs atomically. Rows/variables are one-based; reserved flags are zero."""
function SetPounceComplementarityPairs(
    prob::IpoptProblem,
    rows::AbstractVector{<:Integer},
    vars::AbstractVector{<:Integer},
)
    symbol = _complementarity_symbol(prob, :SetPounceComplementarityPairs)
    length(rows) == length(vars) ||
        throw(DimensionMismatch("Pair arrays must have equal length"))
    length(rows) <= min(prob.n, prob.m) ||
        throw(ArgumentError("Too many complementarity pairs"))
    for (row, var) in zip(rows, vars)
        _check_pair_indices(prob, row, var)
    end
    r, v, bad = Cint[rows...], Cint[vars...], Ref{Cint}(-1)
    code = ccall(
        symbol,
        Cint,
        (Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cint}, Cuint, Ref{Cint}),
        prob,
        length(r),
        r,
        v,
        0,
        bad,
    )
    return _check_native_error(:SetPounceComplementarityPairs, code, bad[])
end

"""Clear pairs, but not the row edits or reservations made by the MOI declaration API."""
function ClearPounceComplementarityPairs(prob::IpoptProblem)
    symbol = _complementarity_symbol(prob, :ClearPounceComplementarityPairs)
    ccall(symbol, Bool, (Ptr{Cvoid},), prob) ||
        error("Could not clear complementarity pairs")
    return
end
function GetPounceComplementarityPairCount(prob::IpoptProblem)
    symbol = _complementarity_symbol(prob, :GetPounceComplementarityPairCount)
    return Int(ccall(symbol, Cint, (Ptr{Cvoid},), prob))
end
"""Return owned one-based row and variable arrays in declaration order."""
function GetPounceComplementarityPairs(prob::IpoptProblem)
    n = GetPounceComplementarityPairCount(prob)
    rows, vars = zeros(Cint, n), zeros(Cint, n)
    symbol = _complementarity_symbol(prob, :GetPounceComplementarityPairs)
    ccall(symbol, Bool, (Ptr{Cvoid}, Ptr{Cint}, Ptr{Cint}), prob, rows, vars) ||
        error("Could not read complementarity pairs")
    return (rows = Int.(rows), variables = Int.(vars))
end

struct _CComplementarityStats
    struct_size::Cint
    strategy::NTuple{16,UInt8}
    reported::Cint
    accepted_by::NTuple{32,UInt8}
    fallback::NTuple{32,UInt8}
    n_solves::Cint
    total_iterations::Cint
    max_abs_product::Cdouble
    certificate_class::Cint
    n_pairs::Cint
    n_biactive::Cint
    n_neither::Cint
    multipliers_unique::Cint
    residual::Cdouble
    resid_tol::Cdouble
    tight::Cint
    feasible::Cint
    primal_violation::Cdouble
    b_verdict::Cint
    b_n_biactive::Cint
    b_branches_solved::Cint
    b_enumerated::Cint
    escapes::Cint
    lifted::Cint
    natural_residual::Cdouble
    polish_ran::Cint
    polish_taken::Cint
    polish_residual_before::Cdouble
    polish_residual_after::Cdouble
    polish_threshold::Cdouble
end

struct _CComplementaritySolve
    struct_size::Cint
    stage::NTuple{32,UInt8}
    relaxed::Cint
    tau::Cdouble
    status::Cint
    iterations::Cint
    max_abs_product::Cdouble
    seconds::Cdouble
end

# Initialize the complete isbits value, including struct_size, before every call.
_zero_c(::Type{T}) where {T<:Number} = zero(T)
_zero_c(::Type{NTuple{N,UInt8}}) where {N} = ntuple(_ -> UInt8(0), N)
function _sized_c(::Type{T}) where {T}
    return T(Cint(sizeof(T)), (_zero_c(fieldtype(T, i)) for i = 2:fieldcount(T))...)
end
_c_value(x) = x
_c_value(x::Cint) = Int(x)
function _c_value(x::NTuple{N,UInt8}) where {N}
    stop = findfirst(iszero, x)
    return String(collect(x)[1:(stop===nothing ? N : stop-1)])
end
function _owned_c(x::T) where {T}
    names = fieldnames(T)[2:end]
    return NamedTuple{names}(Tuple(_c_value(getfield(x, name)) for name in names))
end

"""Native MPCC outcome, or nothing when the last solve did not run the driver.
Sentinels (-1, NaN, empty strings) are preserved; certificates are not solver statuses.
"""
function GetPounceComplementarityStats(prob::IpoptProblem)
    symbol = _complementarity_symbol(prob, :GetPounceComplementarityStats)
    _require_completed_solve(prob)
    out = Ref(_sized_c(_CComplementarityStats))
    ok = ccall(symbol, Bool, (Ptr{Cvoid}, Ref{_CComplementarityStats}), prob, out)
    return ok ? _owned_c(out[]) : nothing
end

"""Get driver solve k (one-based), or nothing when k/the driver outcome is unavailable."""
function GetPounceComplementaritySolve(prob::IpoptProblem, k::Integer)
    symbol = _complementarity_symbol(prob, :GetPounceComplementaritySolve)
    _require_completed_solve(prob)
    1 <= k <= typemax(Cint) ||
        throw(ArgumentError("Solve index must be positive and fit Cint"))
    out = Ref(_sized_c(_CComplementaritySolve))
    ok = ccall(
        symbol,
        Bool,
        (Ptr{Cvoid}, Cint, Ref{_CComplementaritySolve}),
        prob,
        k - 1,
        out,
    )
    return ok ? _owned_c(out[]) : nothing
end

"""Copy a driver text (POUNCE_CC_TEXT_*), or nothing if unavailable."""
function GetPounceComplementarityText(
    prob::IpoptProblem,
    which::Integer = POUNCE_CC_TEXT_SUMMARY,
)
    symbol = _complementarity_symbol(prob, :GetPounceComplementarityText)
    _require_completed_solve(prob)
    n = ccall(symbol, Cint, (Ptr{Cvoid}, Cint, Ptr{UInt8}, Cint), prob, which, C_NULL, 0)
    n < 0 && return nothing
    n < typemax(Cint) || error("Complementarity text is too large")
    buffer = Vector{UInt8}(undef, n + 1)
    written = ccall(
        symbol,
        Cint,
        (Ptr{Cvoid}, Cint, Ptr{UInt8}, Cint),
        prob,
        which,
        buffer,
        length(buffer),
    )
    written == n || error("Complementarity text changed while reading")
    resize!(buffer, n)
    return String(buffer)
end
