"""Python interface to POUNCE — a pure-Rust port of Ipopt.

The public surface is intentionally cyipopt-compatible: Problem class
construction, ``add_option``, and ``solve`` accept the same arguments
and return the same shape of result. A scipy-style ``minimize`` facade
is also provided. JAX integration (autodiff-built derivatives, implicit
differentiation through ``x*(p)``) lives in the ``pounce.jax``
submodule and is only imported on demand to avoid pulling in JAX when
it is not installed.
"""

from ._pounce import (
    Problem, Solver, NlProblem, read_nl, classify_working_set, __version__,
)
from ._minimize import minimize, OptimizeResult
from ._minima import find_minima, MinimaResult
from ._critical import (
    find_critical_points, find_saddles, reaction_network,
    CriticalPoint, CriticalPointResult, Connection, ReactionNetwork,
)

__all__ = [
    "Problem",
    "Solver",
    "NlProblem",
    "read_nl",
    "minimize",
    "OptimizeResult",
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
    "__version__",
]
