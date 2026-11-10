# Copyright (c) 2026: Frederik Geth
# Licensed under MIT; see ../LICENSE.md.

const POUNCE_BOUNDS_VIEW_USER = 0
const POUNCE_BOUNDS_VIEW_EFFECTIVE = 1

"""Whether the loaded library provides the finalized in-place bound APIs."""
supports_bound_updates() = all(
    s -> Libdl.dlsym_e(_library_handle[], s) != C_NULL,
    (:PounceErrorString, :SetPounceBounds, :GetPounceBounds),
)

function _bounds_symbol(prob::IpoptProblem, name::Symbol)
    prob.solving && error("Bound APIs are only available between solves")
    Base.unsafe_convert(Ptr{Cvoid}, prob)
    supports_bound_updates() ||
        error("The loaded POUNCE library does not support in-place bound updates")
    return Libdl.dlsym(_library_handle[], name)
end

function _bound_indices(indices, dimension)
    indices === nothing && return nothing
    length(indices) <= typemax(Cint) || throw(ArgumentError("Bound count must fit Cint"))
    all(i -> 1 <= i <= dimension, indices) ||
        throw(ArgumentError("Bound indices must lie in 1:$dimension"))
    return Vector{Cint}(indices)
end

function _bound_update(indices, lower, upper, dimension)
    idx = _bound_indices(indices, dimension)
    idx === nothing ||
        length(idx) <= dimension ||
        throw(ArgumentError("Too many bound indices"))
    count =
        idx === nothing ? (lower === nothing && upper === nothing ? 0 : dimension) :
        length(idx)
    for values in (lower, upper)
        values === nothing ||
            length(values) == count ||
            throw(DimensionMismatch("Bound arrays must match the selected index count"))
    end
    lo = lower === nothing ? nothing : Vector{Float64}(lower)
    up = upper === nothing ? nothing : Vector{Float64}(upper)
    return count, idx, lo, up
end

_bound_ptr(values) = values === nothing ? C_NULL : values

"""
    SetPounceBounds(prob; x_indices=nothing, x_lower=nothing, x_upper=nothing,
                         g_indices=nothing, g_lower=nothing, g_upper=nothing)

Atomically update selected variable/row bounds. Indices are one-based. Omit
indices for a dense update; omit either bound array to leave that side unchanged.
Omitting both arrays and indices leaves that block unchanged. Negative native
codes throw NativeAPIError; its bad_index counts variables then rows, one-based.
Complementarity bound-class changes require a new problem. The previous solve's
low-level diagnostics remain readable, describing that solve, until the next solve.
"""
function SetPounceBounds(
    prob::IpoptProblem;
    x_indices::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    x_lower::Union{Nothing,AbstractVector{<:Real}} = nothing,
    x_upper::Union{Nothing,AbstractVector{<:Real}} = nothing,
    g_indices::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    g_lower::Union{Nothing,AbstractVector{<:Real}} = nothing,
    g_upper::Union{Nothing,AbstractVector{<:Real}} = nothing,
)
    symbol = _bounds_symbol(prob, :SetPounceBounds)
    nx, xi, xl, xu = _bound_update(x_indices, x_lower, x_upper, prob.n)
    ng, gi, gl, gu = _bound_update(g_indices, g_lower, g_upper, prob.m)
    bad = Ref{Cint}(-1)
    code = ccall(
        symbol,
        Cint,
        (
            Ptr{Cvoid},
            Cint,
            Ptr{Cint},
            Ptr{Cdouble},
            Ptr{Cdouble},
            Cint,
            Ptr{Cint},
            Ptr{Cdouble},
            Ptr{Cdouble},
            Cuint,
            Ref{Cint},
        ),
        prob,
        nx,
        _bound_ptr(xi),
        _bound_ptr(xl),
        _bound_ptr(xu),
        ng,
        _bound_ptr(gi),
        _bound_ptr(gl),
        _bound_ptr(gu),
        0,
        bad,
    )
    return _check_native_error(:SetPounceBounds, code, bad[])
end

"""
    GetPounceBounds(prob; x_indices=nothing, g_indices=nothing,
                         view=POUNCE_BOUNDS_VIEW_USER)

Return owned x_lower/x_upper/g_lower/g_upper arrays. Omitted indices select the
whole block; an empty vector selects none. Duplicate indices are allowed. USER
bounds precede native Complements lowering; EFFECTIVE bounds include its row edits.
"""
function GetPounceBounds(
    prob::IpoptProblem;
    x_indices::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    g_indices::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    view::Integer = POUNCE_BOUNDS_VIEW_USER,
)
    symbol = _bounds_symbol(prob, :GetPounceBounds)
    xi, gi = _bound_indices(x_indices, prob.n), _bound_indices(g_indices, prob.m)
    nx, ng = xi === nothing ? prob.n : length(xi), gi === nothing ? prob.m : length(gi)
    xl, xu, gl, gu = zeros(nx), zeros(nx), zeros(ng), zeros(ng)
    bad = Ref{Cint}(-1)
    code = ccall(
        symbol,
        Cint,
        (
            Ptr{Cvoid},
            Cint,
            Ptr{Cint},
            Ptr{Cdouble},
            Ptr{Cdouble},
            Cint,
            Ptr{Cint},
            Ptr{Cdouble},
            Ptr{Cdouble},
            Cint,
            Cuint,
            Ref{Cint},
        ),
        prob,
        nx,
        _bound_ptr(xi),
        xl,
        xu,
        ng,
        _bound_ptr(gi),
        gl,
        gu,
        view,
        0,
        bad,
    )
    _check_native_error(:GetPounceBounds, code, bad[])
    return (x_lower = xl, x_upper = xu, g_lower = gl, g_upper = gu)
end
