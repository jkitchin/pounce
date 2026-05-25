"""Curated glossary + citations backing the `explain` and `citations` tools.

Static data so the MCP server can answer "what is `inf_pr`?" or
"what should I cite for the restoration phase?" without round-tripping
to the docs site or the .bib file every call. The bibliography is
loaded once on first access from the repo's `.crucible/references.bib`.

Keep entries terse — Claude can elaborate at call time. The job here is
to surface canonical names, numeric ranges, and the right paper keys.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# Per-iter columns emitted by `convergence_trace` (and the derived
# log10_* fields produced by `get_iterate`).
COLUMNS: dict[str, dict[str, Any]] = {
    "iter": {
        "definition": "Zero-based iteration index of the outer interior-point loop.",
        "typical_range": "0 to a few hundred for well-scaled problems.",
        "what_abnormal_means": "Hitting `max_iter` without converging usually points at scaling or degeneracy.",
        "see_also": ["wachter2006"],
    },
    "objective": {
        "definition": "Current objective value f(x_k) at the iterate.",
        "typical_range": "Problem-dependent. For well-scaled problems on the order of 1.",
        "what_abnormal_means": "Wild swings (especially after restoration) can signal bad scaling.",
        "see_also": [],
    },
    "inf_pr": {
        "definition": "Primal infeasibility: max-norm of constraint violation c(x_k).",
        "typical_range": "Drops monotonically toward `tol` (default 1e-8) at convergence.",
        "what_abnormal_means": "Stalling at large inf_pr → likely infeasible or restoration-stuck.",
        "see_also": ["wachter2006", "byrd2010"],
    },
    "inf_du": {
        "definition": "Dual infeasibility: max-norm of the gradient of the Lagrangian.",
        "typical_range": "Drops toward `tol` alongside inf_pr; sometimes lags.",
        "what_abnormal_means": "inf_du much larger than inf_pr → multipliers ill-conditioned or scaling bad.",
        "see_also": ["wachter2006"],
    },
    "mu": {
        "definition": "Barrier parameter for the log-barrier homotopy.",
        "typical_range": "Starts ~0.1, decreases toward 1e-9 as iterations progress.",
        "what_abnormal_means": "Mu stuck at one value across many iterations → see finding `mu_stuck`.",
        "see_also": ["wachter2006", "hinder2018"],
    },
    "d_norm": {
        "definition": "Norm of the Newton search direction d_k.",
        "typical_range": "Decreases as the iterate approaches the solution.",
        "what_abnormal_means": "d_norm growing → search direction quality degrading; check regularization.",
        "see_also": ["wachter2006"],
    },
    "regularization": {
        "definition": (
            "Diagonal Hessian regularization δ added to make the KKT "
            "matrix have the correct inertia."
        ),
        "typical_range": "0 for convex problems; small positive values near saddle points.",
        "what_abnormal_means": "Repeated large δ → Hessian is indefinite, problem is non-convex; finding `hessian_regularized` fires.",
        "see_also": ["wachter2006"],
    },
    "alpha_dual": {
        "definition": "Step length applied to the dual variables (bound multipliers).",
        "typical_range": "(0, 1]. Often 1.0 near the solution.",
        "what_abnormal_means": "Persistently tiny alpha_dual → fraction-to-boundary biting; bounds may be active.",
        "see_also": ["wachter2006"],
    },
    "alpha_primal": {
        "definition": "Step length applied to the primal variables x.",
        "typical_range": "(0, 1]. Tiny values point at line-search difficulty.",
        "what_abnormal_means": "Repeated tiny alpha_primal → finding `heavy_line_search` fires.",
        "see_also": ["wachter2006"],
    },
    "alpha_primal_char": {
        "definition": (
            "Single-character tag for what the line search did this iter: "
            "`f` filter-accepted, `h` Armijo, `r` restoration, `s` second-order correction, "
            "`R` restoration entry, `-` rejected."
        ),
        "typical_range": "Mostly `f` on a healthy solve.",
        "what_abnormal_means": "Runs of `r` are restoration windows; consecutive `R` entries = `restoration_loop`.",
        "see_also": ["wachter2006"],
    },
    "ls_trials": {
        "definition": "Number of line-search trials in this iteration.",
        "typical_range": "1-3 for healthy solves.",
        "what_abnormal_means": "Persistently high → curvature mismatch; consider regularization or restart.",
        "see_also": ["wachter2006"],
    },
    "log10_mu": {
        "definition": "log10(mu); convenience for plotting the barrier homotopy.",
        "typical_range": "Decreases from ~-1 to ~-9 over a typical solve.",
        "what_abnormal_means": "Flat trace → mu_stuck.",
        "see_also": ["wachter2006"],
    },
    "log10_inf_pr": {
        "definition": "log10(inf_pr); convenience for spotting stalls.",
        "typical_range": "Monotone descent to ~-8.",
        "what_abnormal_means": "Plateaus over many iters → `find_stalls` will flag the window.",
        "see_also": [],
    },
    "log10_inf_du": {
        "definition": "log10(inf_du); convenience for spotting stalls.",
        "typical_range": "Monotone descent to ~-8.",
        "what_abnormal_means": "Plateaus → check Hessian regularization and step quality.",
        "see_also": [],
    },
    # Linear-solver summary fields (surfaced by `linear_solver_summary`).
    # Populated by the FERAL backend's self-instrumentation.
    "n_factors": {
        "definition": "Total successful symmetric factorisations across the solve.",
        "typical_range": "≈ iteration_count for filter-line-search runs.",
        "what_abnormal_means": "Much larger than iter count → repeated regularization retries.",
        "see_also": ["n_pattern_reuse", "n_pattern_changes"],
    },
    "n_pattern_reuse": {
        "definition": (
            "Factors that reused the prior symbolic factorisation "
            "(sparsity pattern unchanged → cheap)."
        ),
        "typical_range": "Should dominate n_factors after iter 1.",
        "what_abnormal_means": (
            "Low share → matrix structure changing per iter; analyse() runs "
            "repeatedly, hurting throughput."
        ),
        "see_also": ["n_pattern_changes"],
    },
    "n_pattern_changes": {
        "definition": "Factors that required a fresh symbolic factorisation.",
        "typical_range": "1 (the first factor) for a healthy solve.",
        "what_abnormal_means": (
            "> 1 → KKT structure shifting; check inertia-correction regularization "
            "policy or active-set churn."
        ),
        "see_also": ["n_pattern_reuse"],
    },
    "max_fill_ratio": {
        "definition": "Max nnz(L) / nnz(A) observed across factors.",
        "typical_range": "1–10 for well-ordered KKT systems.",
        "what_abnormal_means": (
            ">> 10 → AMD/METIS ordering struggled; expect memory + time spikes."
        ),
        "see_also": ["last_nnz_a", "last_nnz_l"],
    },
    "min_abs_pivot": {
        "definition": "Smallest absolute pivot encountered during factorisation.",
        "typical_range": "1e-8 .. 1e+6 depending on problem scaling.",
        "what_abnormal_means": (
            "Approaching working precision floor (~1e-16) → matrix near-singular; "
            "regularization is probably kicking in."
        ),
        "see_also": ["max_abs_pivot", "regularization"],
    },
    "max_abs_pivot": {
        "definition": "Largest absolute pivot encountered during factorisation.",
        "typical_range": "Within ~6 orders of magnitude of min_abs_pivot.",
        "what_abnormal_means": (
            "max/min >> 1e8 → catastrophic conditioning; consider nlp_scaling_method."
        ),
        "see_also": ["min_abs_pivot"],
    },
    "last_inertia": {
        "definition": (
            "(positive, negative, zero) eigenvalue counts of the final factorisation, "
            "from the LDLᵀ pivots."
        ),
        "typical_range": "(n, m, 0) at a converged primal-dual KKT system.",
        "what_abnormal_means": (
            "zero > 0 → singular; positive < n → indefinite, inertia correction failed."
        ),
        "see_also": ["regularization"],
    },
    "last_nnz_a": {
        "definition": "nnz(A) at the final factorisation's input KKT matrix.",
        "typical_range": "Problem-dependent.",
        "what_abnormal_means": "n/a — informational.",
        "see_also": ["last_nnz_l", "max_fill_ratio"],
    },
    "last_nnz_l": {
        "definition": "nnz(L) at the final factorisation.",
        "typical_range": "Problem-dependent.",
        "what_abnormal_means": "n/a — informational; combine with last_nnz_a for fill.",
        "see_also": ["last_nnz_a", "max_fill_ratio"],
    },
}


# Codes emitted by `diagnose`. Source of truth:
# `crates/pounce-studio-core/src/analysis.rs`.
FINDINGS: dict[str, dict[str, Any]] = {
    "converged": {
        "severity": "info",
        "meaning": "Solver reached the convergence tolerance on both primal and dual infeasibility.",
        "what_to_try": "Nothing — this is the success path.",
        "see_also": ["wachter2006"],
    },
    "max_iter_exceeded": {
        "severity": "error",
        "meaning": "Solver hit `max_iter` without satisfying tolerances.",
        "what_to_try": (
            "Inspect the convergence trace: is residual still decreasing (raise max_iter), "
            "stalled (loosen tol, improve scaling), or oscillating (regularize)?"
        ),
        "see_also": ["wachter2006"],
    },
    "restoration_used": {
        "severity": "info",
        "meaning": "Restoration phase was entered at least once during the solve.",
        "what_to_try": (
            "Often benign on hard problems. Check restoration_windows; a single "
            "short entry is fine, repeated entries suggest a deeper feasibility issue."
        ),
        "see_also": ["wachter2006", "byrd2010"],
    },
    "mu_stuck": {
        "severity": "warning",
        "meaning": (
            "Barrier parameter μ failed to decrease across a window of iterations. "
            "Usually a degenerate active set or a poorly-scaled barrier."
        ),
        "what_to_try": (
            "Try `mu_strategy=adaptive`, tighten `bound_relax_factor`, or check that "
            "bound values are sensible (no infs masking effective bounds)."
        ),
        "see_also": ["wachter2006", "hinder2018"],
    },
    "heavy_line_search": {
        "severity": "warning",
        "meaning": "Line search needed many trials on average — search direction is low-quality.",
        "what_to_try": (
            "Often Hessian-related: enable second-order-correction, or "
            "investigate regularization values."
        ),
        "see_also": ["wachter2006"],
    },
    "hessian_regularized": {
        "severity": "warning",
        "meaning": (
            "Hessian needed inertia-correction (added δ on the diagonal) frequently — "
            "the problem is non-convex or has near-singular Hessian."
        ),
        "what_to_try": (
            "Consider tightening tolerances on `min_hessian_perturbation`, providing "
            "an analytic Hessian if you have one, or reformulating to convexify."
        ),
        "see_also": ["wachter2006"],
    },
    "restoration_loop": {
        "severity": "error",
        "meaning": "Restoration phase was entered repeatedly and never exited cleanly.",
        "what_to_try": (
            "Strong signal of local infeasibility. Try a different start point, "
            "relax tight constraints, or run feasibility diagnostics."
        ),
        "see_also": ["wachter2006", "byrd2010", "leyffer2003"],
    },
    "convergence_stall": {
        "severity": "warning",
        "meaning": "log10(inf_pr|inf_du) barely moved across a window — solver is grinding.",
        "what_to_try": (
            "Check `find_stalls` for the window, then `get_iterate` at its midpoint to "
            "inspect μ, alpha_primal, and regularization. Common cause: bad scaling."
        ),
        "see_also": ["wachter2006"],
    },
}


# Topic → list of citation keys, ordered most-relevant first. Used by
# `citations(topic=...)`. Topics are subsystem-aligned.
TOPICS: dict[str, list[str]] = {
    "interior_point": ["wachter2006", "byrd1999", "hinder2018"],
    "filter_line_search": ["wachter2006"],
    "restoration": ["wachter2006", "byrd2010"],
    "regularization": ["wachter2006"],
    "trust_region": ["byrd2000", "waltz2006"],
    "inexact_step": ["curtis2010"],
    "infeasibility_detection": ["byrd2010", "leyffer2003"],
    "mu_strategy": ["wachter2006", "hinder2018"],
    "sensitivity": ["zavala2009"],
    "knitro": ["byrd2006"],
}


# ---- bibliography loading -------------------------------------------

_BIB_CACHE: dict[str, dict[str, str]] | None = None


def _find_bib() -> Path | None:
    """Locate the repo's references.bib by walking up from this file."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".crucible" / "references.bib"
        if candidate.exists():
            return candidate
        if parent == parent.parent:
            break
    return None


_ENTRY_RE = re.compile(
    r"@(?P<type>\w+)\s*\{\s*(?P<key>[^,\s]+)\s*,(?P<body>.*?)\n\}",
    re.DOTALL,
)
_FIELD_RE = re.compile(r"(\w+)\s*=\s*\{([^{}]*)\}", re.DOTALL)


def load_bib() -> dict[str, dict[str, str]]:
    """Parse `.crucible/references.bib` into {key: {field: value, ...}}.

    Cached after first call. Returns an empty dict if the file isn't
    found (e.g. the package was installed outside the source tree).
    Tolerant of unknown fields; entries that fail to parse are dropped.
    """
    global _BIB_CACHE
    if _BIB_CACHE is not None:
        return _BIB_CACHE
    path = _find_bib()
    if path is None:
        _BIB_CACHE = {}
        return _BIB_CACHE
    text = path.read_text(errors="replace")
    out: dict[str, dict[str, str]] = {}
    for m in _ENTRY_RE.finditer(text):
        fields = {
            name.lower(): value.strip()
            for name, value in _FIELD_RE.findall(m.group("body"))
        }
        fields["entry_type"] = m.group("type").lower()
        out[m.group("key")] = fields
    _BIB_CACHE = out
    return out


def fuzzy_suggest(term: str, candidates: list[str], limit: int = 3) -> list[str]:
    """Return up to `limit` candidates closest to `term` (substring + prefix)."""
    term_l = term.lower()
    scored: list[tuple[int, str]] = []
    for c in candidates:
        cl = c.lower()
        if cl == term_l:
            scored.append((0, c))
        elif cl.startswith(term_l) or term_l.startswith(cl):
            scored.append((1, c))
        elif term_l in cl or cl in term_l:
            scored.append((2, c))
    scored.sort()
    return [c for _, c in scored[:limit]]
