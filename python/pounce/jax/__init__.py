"""JAX integration for pounce.

Three layers, each independently useful:

1. :func:`from_jax` — build a :class:`pounce.Problem` from JAX-traced
   ``f(x)`` and ``g(x)``. Gradient, Jacobian (with sparsity detected
   from a one-shot dense probe and stored as ``(rows, cols)`` per
   cyipopt convention), and the Lagrangian Hessian are auto-derived
   with ``jax.grad`` / ``jax.jacrev`` / ``jax.hessian`` and JIT-compiled.

2. :func:`solve` — wraps ``Problem.solve`` in ``jax.custom_vjp`` so a
   ``loss(p)`` that calls into the solver can be differentiated with
   ``jax.grad``. The backward uses the implicit-function theorem on
   the KKT system at ``x*``.

3. :func:`vmap_solve` — convenience helper that runs a batch of solves
   over a leading axis. Because the solver itself is single-threaded
   and stateful, we register a manual batching rule that loops over the
   batch rather than literal ``vmap`` (which would lift the impure
   callback in unsupported ways).

JAX is an optional dependency. Importing this module without JAX
installed raises a useful error.
"""

from __future__ import annotations

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pounce.jax requires JAX; install with `pip install pounce[jax]`."
    ) from e

# pounce is a double-precision solver, and the implicit-diff and
# `from_jax` paths assume float64 throughout (Newton convergence stalls
# in float32, and the KKT solve in `solve`'s VJP needs the extra
# precision to give a meaningful gradient). Enable x64 globally on
# first import — it's a no-op if already on.
from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from ._build import from_jax
from ._diff import solve, solve_with_warm, vmap_solve, vmap_solve_parallel
from ._problem import AnchorState, JaxProblem
from ._path import PathFollower, PathTrace, inverse_map_rhs
from ._qp import QpLayer, solve_qp, solve_qp_batch, solve_socp

__all__ = [
    "from_jax",
    "solve",
    "solve_with_warm",
    "vmap_solve",
    "vmap_solve_parallel",
    "JaxProblem",
    "AnchorState",
    "PathFollower",
    "PathTrace",
    "inverse_map_rhs",
    "solve_qp",
    "solve_qp_batch",
    "solve_socp",
    "QpLayer",
]
