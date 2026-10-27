"""Python interface to POUNCE — a pure-Rust interior-point optimization solver.

POUNCE began as a port of Ipopt and has grown into a family of solvers
sharing one numerical backbone. The nonlinear-programming surface is
intentionally cyipopt-compatible: Problem class construction,
``add_option``, and ``solve`` accept the same arguments and return the
same shape of result, with a scipy-style ``minimize`` facade alongside.
Convex and conic programs (LP, QP, SOCP, exponential / power cones, small
SDP) are exposed through ``solve_qp`` / ``solve_socp``; polynomial global
optimization through ``sos_minimize``. JAX integration (autodiff-built
derivatives, implicit differentiation through ``x*(p)``) lives in the
``pounce.jax`` submodule and is only imported on demand to avoid pulling
in JAX when it is not installed.
"""

from ._pounce import (
    Problem, Solver, NlProblem, NlExpr, read_nl, parse_nl_text,
    build_nl_problem, classify_working_set, print_banner,
    __version__,
)
# Installs the Problem.solve(warm_start=...) wrapper as an import side
# effect — keep this import directly after ._pounce.
from ._warm_start import (
    WarmStart,
    TransferContext,
    ProblemSignature,
    WarmStartCompatibilityError,
    WarmStartCompatibilityWarning,
    WarmStartLegacyWarning,
    WarmStartOrderingUnverifiedWarning,
)
# Everything below loads on first use (PEP 562), not at `import pounce`
# (gh #1086). The solver facades pull in scipy, and `import pounce` took
# ~175 ms, nearly all of it in `_minimize`'s scipy.optimize import. That is
# paid by every process that only needs part of the package: the GAMS link
# runs once per GAMS solve, and the `pounce` console script used to run
# through this file twice per Pyomo solve. `_pounce` and `_warm_start` above
# stay eager: the second installs the Problem.solve wrapper as a side
# effect. Names resolve exactly as before (`from pounce import minimize`,
# `pounce.minimize`, `from pounce import *`, `pounce.qp`), on first access.
_LAZY = {
    # name -> submodule it lives in
    "Continuation": "_continuation",
    "ContinuationStep": "_continuation",
    "ContinuationTrace": "_continuation",
    "StepController": "_continuation",
    "kkt_residual_monitor": "_continuation",
    "minimize": "_minimize",
    "OptimizeResult": "_minimize",
    "solve_nlp_batch": "_nlp_batch",
    "curve_fit": "_curve_fit",
    "curve_fit_minima": "_curve_fit",
    "curve_fit_streaming": "_curve_fit",
    "CurveFitResult": "_curve_fit",
    "find_minima": "_minima",
    "MinimaResult": "_minima",
    "trf_minimize": "trf",
    "TRFResult": "trf",
    "TRFConfig": "trf",
    "preflight": "_preflight",
    "PreflightReport": "_preflight",
    "ProjectionReport": "_starts",
    "RaceCandidate": "_starts",
    "RaceReport": "_starts",
    "RaceRound": "_starts",
    "generate_starts": "_starts",
    "project_to_feasible": "_starts",
    "race_starts": "_starts",
    "find_critical_points": "_critical",
    "find_saddles": "_critical",
    "reaction_network": "_critical",
    "CriticalPoint": "_critical",
    "CriticalPointResult": "_critical",
    "Connection": "_critical",
    "ReactionNetwork": "_critical",
    "ActiveSet": "qp",
    "QpResult": "qp",
    "QpFactorization": "qp",
    "QpSensitivity": "qp",
    "ReducedHessian": "qp",
    "solve_qp": "qp",
    "solve_socp": "qp",
    "solve_qp_batch": "qp",
    "solve_qp_multi_rhs": "qp",
    "sos_minimize": "sos",
    "SosResult": "sos",
    "solve_bvp": "bvp",
    "BVPResult": "bvp",
    "solve_bvp_constrained": "bvp",
    "solve_ivp": "ode",
    "solve_dae": "ode",
    "OdeResult": "ode",
}

# Submodules the eager imports used to bind as attributes as a side effect,
# so `import pounce; pounce.qp.solve_qp(...)` kept working without an
# explicit `import pounce.qp`.
_LAZY_SUBMODULES = frozenset(
    {"_continuation", "_minimize", "_nlp_batch", "_curve_fit", "_minima", "trf",
     "_preflight", "_starts", "_critical", "qp", "sos", "bvp", "ode"}
)


def __getattr__(name):
    import importlib

    if name in _LAZY:
        value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    elif name in _LAZY_SUBMODULES:
        value = importlib.import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | _LAZY_SUBMODULES)


__all__ = [
    # Nonlinear programming (cyipopt-compatible)
    "Problem",
    "Solver",
    "NlProblem",
    "NlExpr",
    "read_nl",
    "parse_nl_text",
    "build_nl_problem",
    "print_banner",
    "solve_nlp_batch",
    "minimize",
    "OptimizeResult",
    "curve_fit",
    "curve_fit_minima",
    "curve_fit_streaming",
    "CurveFitResult",
    "find_minima",
    "MinimaResult",
    # Glass box / black box (trust-region filter)
    "trf_minimize",
    "TRFResult",
    "TRFConfig",
    "find_critical_points",
    "find_saddles",
    "reaction_network",
    "CriticalPoint",
    "CriticalPointResult",
    "Connection",
    "ReactionNetwork",
    "classify_working_set",
    # Starting-point preflight, warm starts, and generation/repair
    # (see docs: initialization.md)
    "preflight",
    "PreflightReport",
    "WarmStart",
    "TransferContext",
    "ProblemSignature",
    "WarmStartCompatibilityError",
    "WarmStartCompatibilityWarning",
    "WarmStartLegacyWarning",
    "WarmStartOrderingUnverifiedWarning",
    # Predictor--corrector continuation over a repeated-NLP sequence
    # (see docs: continuation.md)
    "Continuation",
    "ContinuationStep",
    "ContinuationTrace",
    "StepController",
    "kkt_residual_monitor",
    "generate_starts",
    "project_to_feasible",
    "ProjectionReport",
    # Adaptive successive-halving racing (see docs: initialization.md)
    "race_starts",
    "RaceReport",
    "RaceRound",
    "RaceCandidate",
    # Convex QP / SOCP (the same solvers also live under ``pounce.qp``)
    "ActiveSet",
    "QpResult",
    "QpFactorization",
    "QpSensitivity",
    "ReducedHessian",
    "solve_qp",
    "solve_socp",
    "solve_qp_batch",
    "solve_qp_multi_rhs",
    # Polynomial global optimization (SOS / Lasserre)
    "sos_minimize",
    "SosResult",
    # Boundary value problems (SciPy-compatible)
    "solve_bvp",
    "BVPResult",
    "solve_bvp_constrained",
    # Stiff ODE / DAE initial value problems (SciPy-compatible)
    "solve_ivp",
    "solve_dae",
    "OdeResult",
    "__version__",
]
