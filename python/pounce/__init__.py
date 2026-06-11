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
    Problem, Solver, NlProblem, read_nl, classify_working_set, __version__,
)
from ._minimize import minimize, OptimizeResult
from ._nlp_batch import solve_nlp_batch
from ._curve_fit import (
    curve_fit,
    curve_fit_minima,
    curve_fit_streaming,
    CurveFitResult,
)
from ._minima import find_minima, MinimaResult
from ._critical import (
    find_critical_points, find_saddles, reaction_network,
    CriticalPoint, CriticalPointResult, Connection, ReactionNetwork,
)
from .qp import (
    QpResult,
    QpFactorization,
    QpSensitivity,
    ReducedHessian,
    solve_qp,
    solve_socp,
    solve_qp_batch,
    solve_qp_multi_rhs,
)
from .sos import sos_minimize, SosResult

__all__ = [
    # Nonlinear programming (cyipopt-compatible)
    "Problem",
    "Solver",
    "NlProblem",
    "read_nl",
    "solve_nlp_batch",
    "minimize",
    "OptimizeResult",
    "curve_fit",
    "curve_fit_minima",
    "curve_fit_streaming",
    "CurveFitResult",
    "find_minima",
    "MinimaResult",
    "find_critical_points",
    "find_saddles",
    "reaction_network",
    "CriticalPoint",
    "CriticalPointResult",
    "Connection",
    "ReactionNetwork",
    "classify_working_set",
    # Convex QP / SOCP (the same solvers also live under ``pounce.qp``)
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
    "__version__",
]
