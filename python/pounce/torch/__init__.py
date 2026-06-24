"""PyTorch integration for pounce (pounce#109).

A PyTorch frontend for pounce's differentiable solver, mirroring the
:mod:`pounce.jax` subpackage. A solve is a :class:`torch.autograd.Function`
you can drop inside a learned model and backprop through, with the same
constraint-satisfaction guarantee the JAX path gives.

This is a **frontend/adapter**, not a second solver. The numerical core
(the Rust IPM, exposed via :class:`pounce._pounce.Problem`) and the
implicit-function-theorem backward math are autodiff-framework-agnostic;
only a thin wrapper layer differs. Because PyTorch is eager, that layer
is *smaller* than the JAX one (no ``pure_callback`` / ``ShapeDtypeStruct``
machinery) — the forward calls ``problem.solve(...)`` directly and the
backward runs the KKT solve in-framework with ``torch.linalg.solve``.

Layers, each independently useful:

1. :func:`from_torch` — build a :class:`pounce.Problem` from
   PyTorch-traced ``f(x)`` and ``g(x)``. Gradient / Jacobian (with
   detected sparsity) / Lagrangian Hessian are auto-derived with
   ``torch.func.grad`` / ``jacrev`` / ``jacfwd`` / ``hessian``.

2. :func:`solve` / :func:`solve_with_warm` — wrap ``Problem.solve`` in a
   ``torch.autograd.Function`` so a ``loss(p)`` that calls into the
   solver can be differentiated. The backward uses the implicit-function
   theorem on the KKT system at ``x*``.

3. :func:`vmap_solve` / :func:`vmap_solve_parallel` — batched solves
   (sequential loop / threadpool).

4. :class:`TorchProblem` — a build-once, solve-many stateful handle that
   caches the compiled AD artefacts, sparsity, and (optionally) reuses the
   IPM's converged KKT factor for the backward.

5. :func:`solve_qp` / :func:`solve_qp_batch` / :func:`solve_socp` /
   :class:`QpLayer` — differentiable conic layers (feasible-by-construction).

6. :class:`PathFollower` / :func:`inverse_map_rhs` — predictor–corrector
   path following over a :class:`TorchProblem`.

PyTorch is an optional dependency. Importing this module without PyTorch
installed raises a useful error. ``torch.func`` requires torch ≥ 2.0; the
``pounce[torch]`` extra pins ≥ 2.2 for a stable surface.

pounce is a double-precision solver, and the implicit-diff and
``from_torch`` paths assume float64 throughout (Newton convergence stalls
in float32, and the KKT solve in the VJP needs the extra precision). The
adapters request float64 tensors explicitly — there is no global flag as
in JAX's ``jax_enable_x64``.
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pounce.torch requires PyTorch; install with "
        "`pip install pounce[torch]`."
    ) from e

from ._build import from_torch
from ._diff import solve, solve_with_warm, vmap_solve, vmap_solve_parallel
from ._problem import AnchorState, TorchProblem
from ._path import PathFollower, PathTrace, inverse_map_rhs
from ._qp import QpLayer, solve_qp, solve_qp_batch, solve_socp
from ._bvp import solve_bvp, TorchBVPSolution
from ._ode import odeint, TorchODESolution
from ._dae import daeint

__all__ = [
    "from_torch",
    "solve",
    "solve_with_warm",
    "vmap_solve",
    "vmap_solve_parallel",
    "solve_bvp",
    "TorchBVPSolution",
    "odeint",
    "TorchODESolution",
    "daeint",
    "TorchProblem",
    "AnchorState",
    "PathFollower",
    "PathTrace",
    "inverse_map_rhs",
    "solve_qp",
    "solve_qp_batch",
    "solve_socp",
    "QpLayer",
]
