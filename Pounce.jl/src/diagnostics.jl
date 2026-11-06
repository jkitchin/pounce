# Copyright (c) 2026: Frederik Geth
# Licensed under MIT; see ../LICENSE.md.

function _require_completed_solve(prob::IpoptProblem)
    prob.ipopt_problem != C_NULL || error("POUNCE problem has been freed")
    prob.solving && error("Diagnostics are only available between solves")
    prob.solved || error("No completed POUNCE solve is available")
    return
end

for (name, field) in (
    (:GetIpoptPrimalInf, "primal infeasibility"),
    (:GetIpoptDualInf, "dual infeasibility"),
    (:GetIpoptComplInf, "bound/multiplier complementarity error"),
)
    doc = """
            $name(prob)

        Return native $field after a completed solve. This is the solver's
        internally scaled statistic, not an independent user-space certificate.
        Native code returns zero if the completed call produced no statistics.
        Unavailable before solving, during a solve, after a thrown solve, or
        after freeing the problem. Complementarity here is not an MPCC residual.
        """
    @eval begin
        @doc $doc function $name(prob::IpoptProblem)
            _require_completed_solve(prob)
            return ccall(($(QuoteNode(name)), libpounce), Cdouble, (Ptr{Cvoid},), prob)
        end
    end
end

"""Return restoration calls, inner/outer iterations and wall seconds as a named tuple."""
function GetPounceRestorationStats(prob::IpoptProblem)
    _require_completed_solve(prob)
    calls, inner, outer, seconds = Ref{Cint}(), Ref{Cint}(), Ref{Cint}(), Ref{Cdouble}()
    @ccall libpounce.GetPounceRestorationStats(
        prob::Ptr{Cvoid},
        calls::Ref{Cint},
        inner::Ref{Cint},
        outer::Ref{Cint},
        seconds::Ref{Cdouble},
    )::Cvoid
    return (
        calls = Int(calls[]),
        inner_iters = Int(inner[]),
        outer_iters = Int(outer[]),
        wall_secs = seconds[],
    )
end

"""
Return the finite-difference Hessian census as a named tuple. `pattern_used` is
-1 when the mode did not run, 0 for declared sparsity, and 1 for Jacobian-derived
sparsity. Counts are zero when unavailable; the two flags are booleans.
"""
function GetPounceFdHessianStats(prob::IpoptProblem)
    _require_completed_solve(prob)
    pattern, nnz, n, groups, rho, fallback, widened = ntuple(_ -> Ref{Cint}(), 7)
    @ccall libpounce.GetPounceFdHessianStats(
        prob::Ptr{Cvoid},
        pattern::Ref{Cint},
        nnz::Ref{Cint},
        n::Ref{Cint},
        groups::Ref{Cint},
        rho::Ref{Cint},
        fallback::Ref{Cint},
        widened::Ref{Cint},
    )::Cvoid
    return (
        pattern_used = Int(pattern[]),
        nnz = Int(nnz[]),
        n = Int(n[]),
        groups = Int(groups[]),
        rho_max = Int(rho[]),
        coloring_fell_back = fallback[] != 0,
        objective_clique_widened = widened[] != 0,
    )
end

# Mirrors the version-gated pounce.h layout. Julia supplies C-compatible padding.
struct _CLinearSolverStats
    solver_name::NTuple{32,UInt8}
    n_factors::Cint
    n_pattern_reuse::Cint
    n_pattern_changes::Cint
    max_fill_ratio::Cdouble
    min_abs_pivot::Cdouble
    max_abs_pivot::Cdouble
    last_inertia_positive::Cint
    last_inertia_negative::Cint
    last_inertia_zero::Cint
    last_nnz_a::Cint
    last_nnz_l::Cint
end

"""
Return linear-solver statistics as an owned named tuple, or `nothing` if no
backend supplied a summary. Native optional values retain their sentinels:
`NaN` for unreported floating-point values and -1 for unreported counts.
"""
function GetPounceLinearSolverStats(prob::IpoptProblem)
    _require_completed_solve(prob)
    out = Ref{_CLinearSolverStats}()
    ok = @ccall libpounce.GetPounceLinearSolverStats(
        prob::Ptr{Cvoid},
        out::Ref{_CLinearSolverStats},
    )::Bool
    ok || return nothing # C leaves the output buffer untouched on failure.
    stats = out[]
    name = collect(stats.solver_name)
    stop = findfirst(iszero, name)
    stop === nothing || resize!(name, stop - 1)
    return (
        solver_name = String(name),
        n_factors = Int(stats.n_factors),
        n_pattern_reuse = Int(stats.n_pattern_reuse),
        n_pattern_changes = Int(stats.n_pattern_changes),
        max_fill_ratio = stats.max_fill_ratio,
        min_abs_pivot = stats.min_abs_pivot,
        max_abs_pivot = stats.max_abs_pivot,
        last_inertia_positive = Int(stats.last_inertia_positive),
        last_inertia_negative = Int(stats.last_inertia_negative),
        last_inertia_zero = Int(stats.last_inertia_zero),
        last_nnz_a = Int(stats.last_nnz_a),
        last_nnz_l = Int(stats.last_nnz_l),
    )
end

"""
Enable iteration capture for future solves on this low-level handle. The C API
has no disable operation; recreate the handle to turn capture off. Does not
retroactively populate the most recent solve's report.
"""
function IpoptEnableIterHistory(prob::IpoptProblem)
    prob.solving && error("Cannot enable iteration history during a solve")
    ok = @ccall libpounce.IpoptEnableIterHistory(prob::Ptr{Cvoid})::Bool
    ok || error("POUNCE could not enable iteration history")
    return
end

"""
    IpoptWriteSolveReport(prob, path; detail = "summary")

Write the completed native solve as `pounce.solve-report/v1` JSON. `detail` is
`"summary"` or `"full"`; full reports include history only when capture was
previously enabled. Existing files are overwritten. UTF-8 paths are supported;
embedded NULs, invalid detail values and failed writes raise errors.
"""
function IpoptWriteSolveReport(
    prob::IpoptProblem,
    path::AbstractString;
    detail::AbstractString = "summary",
)
    _require_completed_solve(prob)
    detail in ("summary", "full") ||
        throw(ArgumentError("Report detail must be summary or full"))
    ok = @ccall libpounce.IpoptWriteSolveReport(
        prob::Ptr{Cvoid},
        String(path)::Cstring,
        String(detail)::Cstring,
    )::Bool
    ok || error("POUNCE could not write the solve report to $path")
    return
end
