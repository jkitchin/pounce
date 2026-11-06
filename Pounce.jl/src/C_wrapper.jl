# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
# Adapted for POUNCE; see ../NOTICE.md and ../LICENSE.md.
#
# C names intentionally match pounce.h. These bindings use POUNCE's one-byte
# Bool ABI; they are not interchangeable with arbitrary Ipopt builds.

mutable struct IpoptProblem
    ipopt_problem::Ptr{Cvoid}
    n::Int
    m::Int
    x::Vector{Float64}
    g::Vector{Float64}
    mult_g::Vector{Float64}
    mult_x_L::Vector{Float64}
    mult_x_U::Vector{Float64}
    obj_val::Float64
    status::Cint
    eval_f::Function
    eval_g::Function
    eval_grad_f::Function
    eval_jac_g::Function
    eval_h::Union{Function,Nothing}
    hessian_approximation::String
    intermediate::Union{Function,Nothing}
    callback_error::Any
    solving::Bool
    solved::Bool
end

function Base.unsafe_convert(::Type{Ptr{Cvoid}}, p::IpoptProblem)
    p.ipopt_problem != C_NULL || error("POUNCE problem has been freed")
    return p.ipopt_problem
end

_problem(ptr::Ptr{Cvoid}) = unsafe_pointer_to_objref(ptr)::IpoptProblem

# No exception may unwind through Rust. Domain errors reject a trial point;
# other exceptions are retained and rethrown only after IpoptSolve returns.
function _guard_callback(f, prob::IpoptProblem; domain_failure = true)
    prob.callback_error === nothing || return UInt8(0)
    try
        return UInt8(reenable_sigint(f))
    catch err
        if !(domain_failure && err isa DomainError)
            prob.callback_error = CapturedException(err, catch_backtrace())
        end
        return UInt8(0)
    end
end

function _array(ptr::Ptr{T}, n::Integer) where {T}
    n == 0 && return T[]
    ptr != C_NULL || error("POUNCE passed a null callback buffer")
    return unsafe_wrap(Vector{T}, ptr, Int(n); own = false)
end

function _Eval_F_CB(
    n::Cint,
    xptr::Ptr{Float64},
    ::UInt8,
    out::Ptr{Float64},
    data::Ptr{Cvoid},
)::UInt8
    prob = _problem(data)
    return _guard_callback(prob) do
        x = _array(xptr, n)
        value = Float64(prob.eval_f(x))
        unsafe_store!(out, value)
        return isfinite(value)
    end
end

function _Eval_Grad_F_CB(
    n::Cint,
    xptr::Ptr{Float64},
    ::UInt8,
    out::Ptr{Float64},
    data::Ptr{Cvoid},
)::UInt8
    prob = _problem(data)
    return _guard_callback(prob) do
        values = _array(out, n)
        prob.eval_grad_f(_array(xptr, n), values)
        return all(isfinite, values)
    end
end

function _Eval_G_CB(
    n::Cint,
    xptr::Ptr{Float64},
    ::UInt8,
    m::Cint,
    out::Ptr{Float64},
    data::Ptr{Cvoid},
)::UInt8
    prob = _problem(data)
    return _guard_callback(prob) do
        values = _array(out, m)
        prob.eval_g(_array(xptr, n), values)
        return all(isfinite, values)
    end
end

function _Eval_Jac_G_CB(
    n::Cint,
    xptr::Ptr{Float64},
    ::UInt8,
    ::Cint,
    nnz::Cint,
    rowptr::Ptr{Cint},
    colptr::Ptr{Cint},
    out::Ptr{Float64},
    data::Ptr{Cvoid},
)::UInt8
    prob = _problem(data)
    return _guard_callback(prob) do
        if out == C_NULL
            # x is unspecified for a sparsity query. Never dereference it.
            prob.eval_jac_g(Float64[], _array(rowptr, nnz), _array(colptr, nnz), nothing)
            return true
        end
        values = _array(out, nnz)
        prob.eval_jac_g(_array(xptr, n), Cint[], Cint[], values)
        return all(isfinite, values)
    end
end

function _Eval_H_CB(
    n::Cint,
    xptr::Ptr{Float64},
    ::UInt8,
    factor::Float64,
    m::Cint,
    lambdaptr::Ptr{Float64},
    ::UInt8,
    nnz::Cint,
    rowptr::Ptr{Cint},
    colptr::Ptr{Cint},
    out::Ptr{Float64},
    data::Ptr{Cvoid},
)::UInt8
    prob = _problem(data)
    return _guard_callback(prob) do
        prob.eval_h === nothing && return false
        if out == C_NULL
            prob.eval_h(
                Float64[],
                _array(rowptr, nnz),
                _array(colptr, nnz),
                factor,
                Float64[],
                nothing,
            )
            return true
        end
        values = _array(out, nnz)
        prob.eval_h(_array(xptr, n), Cint[], Cint[], factor, _array(lambdaptr, m), values)
        return all(isfinite, values)
    end
end

function _Intermediate_CB(
    alg::Cint,
    iter::Cint,
    obj::Float64,
    inf_pr::Float64,
    inf_du::Float64,
    mu::Float64,
    d_norm::Float64,
    reg::Float64,
    alpha_du::Float64,
    alpha_pr::Float64,
    ls::Cint,
    data::Ptr{Cvoid},
)::UInt8
    prob = _problem(data)
    return _guard_callback(prob; domain_failure = false) do
        prob.intermediate === nothing && return true
        return prob.intermediate(
            alg,
            iter,
            obj,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            reg,
            alpha_du,
            alpha_pr,
            ls,
        )::Bool
    end
end

"""
    CreateIpoptProblem(n, x_L, x_U, m, g_L, g_U, nnz_jac, nnz_hess,
                       eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)

Create a POUNCE problem. Sparse indices are one-based; the Hessian is lower
triangular. Jacobian/Hessian callbacks receive `values === nothing` to request
the structure (x/lambda are empty), otherwise rows/cols are empty and only
values must be filled. Callback arrays are borrowed and must not be retained.
With `eval_h === nothing`, select `hessian_approximation=limited-memory` or
`finite-difference` before solving. An explicit callback with zero declared
entries represents a genuinely zero Hessian.
Use `FreeIpoptProblem` for deterministic cleanup; a finalizer is also installed.
"""
function CreateIpoptProblem(
    n::Int,
    x_L::Vector{Float64},
    x_U::Vector{Float64},
    m::Int,
    g_L::Vector{Float64},
    g_U::Vector{Float64},
    nele_jac::Int,
    nele_hess::Int,
    eval_f,
    eval_g,
    eval_grad_f,
    eval_jac_g,
    eval_h,
)
    all(v -> 0 <= v <= typemax(Cint), (n, m, nele_jac, nele_hess)) ||
        throw(ArgumentError("Dimensions and nonzero counts must fit nonnegative Cint"))
    _check_length(x_L, n)
    _check_length(x_U, n)
    _check_length(g_L, m)
    _check_length(g_U, m)
    f = @cfunction(_Eval_F_CB, UInt8, (Cint, Ptr{Float64}, UInt8, Ptr{Float64}, Ptr{Cvoid}))
    g = @cfunction(
        _Eval_G_CB,
        UInt8,
        (Cint, Ptr{Float64}, UInt8, Cint, Ptr{Float64}, Ptr{Cvoid})
    )
    grad = @cfunction(
        _Eval_Grad_F_CB,
        UInt8,
        (Cint, Ptr{Float64}, UInt8, Ptr{Float64}, Ptr{Cvoid})
    )
    jac = @cfunction(
        _Eval_Jac_G_CB,
        UInt8,
        (
            Cint,
            Ptr{Float64},
            UInt8,
            Cint,
            Cint,
            Ptr{Cint},
            Ptr{Cint},
            Ptr{Float64},
            Ptr{Cvoid},
        )
    )
    hess = @cfunction(
        _Eval_H_CB,
        UInt8,
        (
            Cint,
            Ptr{Float64},
            UInt8,
            Float64,
            Cint,
            Ptr{Float64},
            UInt8,
            Cint,
            Ptr{Cint},
            Ptr{Cint},
            Ptr{Float64},
            Ptr{Cvoid},
        )
    )
    # None and a supplied, structurally zero Hessian are different contracts.
    # Rust must see None when exact second derivatives are unavailable.
    eval_h === nothing && (hess = C_NULL)
    ptr = @ccall libpounce.CreateIpoptProblem(
        n::Cint,
        x_L::Ptr{Cdouble},
        x_U::Ptr{Cdouble},
        m::Cint,
        g_L::Ptr{Cdouble},
        g_U::Ptr{Cdouble},
        nele_jac::Cint,
        nele_hess::Cint,
        1::Cint,
        f::Ptr{Cvoid},
        g::Ptr{Cvoid},
        grad::Ptr{Cvoid},
        jac::Ptr{Cvoid},
        hess::Ptr{Cvoid},
    )::Ptr{Cvoid}
    ptr != C_NULL || error("POUNCE could not create the problem")
    prob = IpoptProblem(
        ptr,
        n,
        m,
        zeros(n),
        zeros(m),
        zeros(m),
        zeros(n),
        zeros(n),
        NaN,
        Cint(-199),
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        eval_h,
        "exact",
        nothing,
        nothing,
        false,
        false,
    )
    # FreeIpoptProblem only drops the owned IpoptProblemInfo graph. Its Rc
    # aliases remain inside that graph after IpoptSolve returns; no live solver
    # or thread-local bridge remains, and dropping it does not invoke callbacks.
    # GC.@preserve in IpoptSolve prevents finalization while Rust borrows it.
    # Thus the idle handle may be finalized on a different thread. Concurrent
    # access to the same problem is unsupported; prefer explicit cleanup.
    finalizer(FreeIpoptProblem, prob)
    return prob
end

function _check_length(values::AbstractVector, n::Integer)
    length(values) == n ||
        throw(DimensionMismatch("Expected $n elements, got $(length(values))"))
    return
end

function _check_length(ptr::Ptr{Cvoid}, ::Integer)
    ptr == C_NULL || throw(ArgumentError("Only C_NULL or a Vector{Float64} is accepted"))
    return
end

function FreeIpoptProblem(prob::IpoptProblem)
    prob.solving && error("Cannot free a POUNCE problem during a solve")
    if prob.ipopt_problem != C_NULL
        @ccall libpounce.FreeIpoptProblem(prob.ipopt_problem::Ptr{Cvoid})::Cvoid
        prob.ipopt_problem = C_NULL
        prob.solved = false
    end
    return
end

function AddIpoptStrOption(prob::IpoptProblem, keyword::String, value::String)
    if !(isascii(keyword) && isascii(value))
        error("POUNCE: Non ASCII parameters not supported")
    end
    ret = @ccall libpounce.AddIpoptStrOption(
        prob::Ptr{Cvoid},
        keyword::Cstring,
        value::Cstring,
    )::Bool
    if !ret
        error("POUNCE: Couldn't set option '$keyword' to value '$value'.")
    end
    if lowercase(keyword) == "hessian_approximation"
        prob.hessian_approximation = lowercase(value)
    end
    return
end

function AddIpoptNumOption(prob::IpoptProblem, keyword::String, value::Float64)
    if !isascii(keyword)
        error("POUNCE: Non ASCII parameters not supported")
    end
    ret = @ccall libpounce.AddIpoptNumOption(
        prob::Ptr{Cvoid},
        keyword::Cstring,
        value::Cdouble,
    )::Bool
    if !ret
        error("POUNCE: Couldn't set option '$keyword' to value '$value'.")
    end
    return
end

function AddIpoptIntOption(prob::IpoptProblem, keyword::String, value::Integer)
    if !isascii(keyword)
        error("POUNCE: Non ASCII parameters not supported")
    end
    ret = @ccall libpounce.AddIpoptIntOption(
        prob::Ptr{Cvoid},
        keyword::Cstring,
        value::Cint,
    )::Bool
    if !ret
        error(
            "POUNCE: Couldn't set option '$keyword' to value '$value'::Int32. " *
            "Note that `Num` options need to be explictly passed as " *
            "`Float64($value)` instead of their integer equivalents.",
        )
    end
    return
end

function OpenIpoptOutputFile(prob::IpoptProblem, file_name::String, print_level::Int)
    if !isascii(file_name)
        error("POUNCE: Non ASCII parameters not supported")
    end
    ret = @ccall libpounce.OpenIpoptOutputFile(
        prob::Ptr{Cvoid},
        file_name::Cstring,
        print_level::Cint,
    )::Bool
    if !ret
        error("POUNCE: Couldn't open output file.")
    end
    return
end

function SetIpoptProblemScaling(
    prob::IpoptProblem,
    obj_scaling::Float64,
    x_scaling::Union{Ptr{Cvoid},Vector{Float64}},
    g_scaling::Union{Ptr{Cvoid},Vector{Float64}},
)
    _check_length(x_scaling, prob.n)
    _check_length(g_scaling, prob.m)
    ret = @ccall libpounce.SetIpoptProblemScaling(
        prob::Ptr{Cvoid},
        obj_scaling::Cdouble,
        x_scaling::Ptr{Cdouble},
        g_scaling::Ptr{Cdouble},
    )::Bool
    ret || error("POUNCE rejected the operation")
    return
end

"""
    SetIntermediateCallback(prob, callback::Union{Function,Nothing})

Install a per-iteration callback, or remove it with `nothing`. Change callbacks
only between solves. Returning `false` requests a user stop.
"""
function SetIntermediateCallback(prob::IpoptProblem, intermediate::Union{Function,Nothing})
    prob.solving && error("Cannot change the intermediate callback during a solve")
    intermediate_cb = if intermediate === nothing
        C_NULL
    else
        @cfunction(
            _Intermediate_CB,
            UInt8,
            (
                Cint,
                Cint,
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
                Float64,
                Cint,
                Ptr{Cvoid},
            ),
        )
    end
    ret = @ccall libpounce.SetIntermediateCallback(
        prob::Ptr{Cvoid},
        intermediate_cb::Ptr{Cvoid},
    )::Bool
    ret || error("POUNCE rejected the operation")
    prob.intermediate = intermediate
    return
end

"""
    GetIpoptIterCount(prob)

Return the native iteration count for the most recent solve, or zero before
the first solve. Unavailable during a solve or after the problem is freed.
"""
function GetIpoptIterCount(prob::IpoptProblem)
    prob.solving && error("Native statistics are only available between solves")
    return Int(@ccall libpounce.GetIpoptIterCount(prob::Ptr{Cvoid})::Cint)
end

"""
    GetIpoptSolveTime(prob)

Return the native solver's reported wall time in seconds, or zero before the
first solve. This includes callbacks invoked by the solver and excludes
Julia-side problem setup. Unavailable during a solve or after the problem
is freed.
"""
function GetIpoptSolveTime(prob::IpoptProblem)
    prob.solving && error("Native statistics are only available between solves")
    return @ccall libpounce.GetIpoptSolveTime(prob::Ptr{Cvoid})::Cdouble
end

# Rust's bridge and iterate snapshots are thread-local. Julia does not promise
# to pin tasks across ordinary foreign calls, so pin the whole native solve,
# including derivative callbacks. Restore the caller's scheduling policy on exit.
# See https://docs.julialang.org/en/v1/base/parallel/#Core.Task
const _solve_threads = Dict{Int,Task}()
const _solve_threads_lock = ReentrantLock()

function _with_solve_thread(f)
    task = current_task()
    sticky = task.sticky
    task.sticky = true
    tid = Threads.threadid()
    registered = false
    try
        lock(_solve_threads_lock) do
            owner = get(_solve_threads, tid, nothing)
            if owner !== nothing && owner !== task
                error("Another task has an active POUNCE solve on this thread")
            end
            if owner === nothing
                _solve_threads[tid] = task
                registered = true
            end
        end
        return f()
    finally
        if registered
            lock(_solve_threads_lock) do
                return delete!(_solve_threads, tid)
            end
        end
        task.sticky = sticky
    end
end

function IpoptSolve(prob::IpoptProblem)
    prob.solving && error("A POUNCE problem cannot be solved recursively")
    for v in (prob.x, prob.mult_x_L, prob.mult_x_U)
        _check_length(v, prob.n)
    end
    _check_length(prob.g, prob.m)
    _check_length(prob.mult_g, prob.m)
    prob.callback_error = nothing
    prob.solved = false
    if prob.eval_h === nothing && prob.hessian_approximation == "exact"
        throw(
            ArgumentError(
                "Exact Hessians require an eval_h callback; supply one or select " *
                "hessian_approximation=limited-memory or finite-difference",
            ),
        )
    end
    prob.solving = true
    p_objval = Ref{Cdouble}(NaN)
    try
        GC.@preserve prob begin
            _with_solve_thread() do
                disable_sigint() do
                    prob.status = @ccall libpounce.IpoptSolve(
                        prob::Ptr{Cvoid},
                        prob.x::Ptr{Cdouble},
                        prob.g::Ptr{Cdouble},
                        p_objval::Ptr{Cdouble},
                        prob.mult_g::Ptr{Cdouble},
                        prob.mult_x_L::Ptr{Cdouble},
                        prob.mult_x_U::Ptr{Cdouble},
                        pointer_from_objref(prob)::Ptr{Cvoid},
                    )::Cint
                end
            end
        end
    finally
        prob.solving = false
    end
    prob.obj_val = p_objval[]
    if prob.callback_error !== nothing
        throw(prob.callback_error.ex)
    end
    prob.solved = true
    return prob.status
end

function GetIpoptCurrentIterate(
    prob::IpoptProblem,
    scaled::Bool,
    n::Integer,
    x::Union{Ptr{Cvoid},Vector{Float64}},
    z_L::Union{Ptr{Cvoid},Vector{Float64}},
    z_U::Union{Ptr{Cvoid},Vector{Float64}},
    m::Integer,
    g::Union{Ptr{Cvoid},Vector{Float64}},
    lambda::Union{Ptr{Cvoid},Vector{Float64}},
)
    n == prob.n && m == prob.m || throw(DimensionMismatch("Problem dimensions differ"))
    for v in (x, z_L, z_U)
        _check_length(v, n)
    end
    for v in (g, lambda)
        _check_length(v, m)
    end
    ret = @ccall libpounce.GetIpoptCurrentIterate(
        prob::Ptr{Cvoid},
        scaled::Bool,
        n::Cint,
        x::Ptr{Cdouble},
        z_L::Ptr{Cdouble},
        z_U::Ptr{Cdouble},
        m::Cint,
        g::Ptr{Cdouble},
        lambda::Ptr{Cdouble},
    )::Bool
    if !ret
        error("POUNCE: Something went wrong getting the current iterate.")
    end
    return
end

function GetIpoptCurrentViolations(
    prob::IpoptProblem,
    scaled::Bool,
    n::Integer,
    x_L_violation::Union{Ptr{Cvoid},Vector{Float64}},
    x_U_violation::Union{Ptr{Cvoid},Vector{Float64}},
    compl_x_L::Union{Ptr{Cvoid},Vector{Float64}},
    compl_x_U::Union{Ptr{Cvoid},Vector{Float64}},
    grad_lag_x::Union{Ptr{Cvoid},Vector{Float64}},
    m::Integer,
    nlp_constraint_violation::Union{Ptr{Cvoid},Vector{Float64}},
    compl_g::Union{Ptr{Cvoid},Vector{Float64}},
)
    n == prob.n && m == prob.m || throw(DimensionMismatch("Problem dimensions differ"))
    for v in (x_L_violation, x_U_violation, compl_x_L, compl_x_U, grad_lag_x)
        _check_length(v, n)
    end
    for v in (nlp_constraint_violation, compl_g)
        _check_length(v, m)
    end
    ret = @ccall libpounce.GetIpoptCurrentViolations(
        prob::Ptr{Cvoid},
        scaled::Bool,
        n::Cint,
        x_L_violation::Ptr{Cdouble},
        x_U_violation::Ptr{Cdouble},
        compl_x_L::Ptr{Cdouble},
        compl_x_U::Ptr{Cdouble},
        grad_lag_x::Ptr{Cdouble},
        m::Cint,
        nlp_constraint_violation::Ptr{Cdouble},
        compl_g::Ptr{Cdouble},
    )::Bool
    if !ret
        error("POUNCE: Something went wrong getting the current violations.")
    end
    return
end

function GetIpoptVersion()
    major, minor, patch = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
    @ccall libpounce.GetIpoptVersion(
        major::Ptr{Cint},
        minor::Ptr{Cint},
        patch::Ptr{Cint},
    )::Cvoid
    return VersionNumber(major[], minor[], patch[])
end

# https://github.com/coin-or/Ipopt/blob/8f2b8efcd53d93518984597808db05dce43e348f/src/Interfaces/IpReturnCodes_inc.h#L13-L38
#!format:off
@enum(
   ApplicationReturnStatus,
   Solve_Succeeded                    = 0,
   Solved_To_Acceptable_Level         = 1,
   Infeasible_Problem_Detected        = 2,
   Search_Direction_Becomes_Too_Small = 3,
   Diverging_Iterates                 = 4,
   User_Requested_Stop                = 5,
   Feasible_Point_Found               = 6,

   Maximum_Iterations_Exceeded        = -1,
   Restoration_Failed                 = -2,
   Error_In_Step_Computation          = -3,
   Maximum_CpuTime_Exceeded           = -4,
   Maximum_WallTime_Exceeded          = -5,   # @since 3.14.0

   Not_Enough_Degrees_Of_Freedom      = -10,
   Invalid_Problem_Definition         = -11,
   Invalid_Option                     = -12,
   Invalid_Number_Detected            = -13,

   Unrecoverable_Exception            = -100,
   NonIpopt_Exception_Thrown          = -101,
   Insufficient_Memory                = -102,
   Internal_Error                     = -199
)
#!format:on
