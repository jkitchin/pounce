"""Differentiable boundary value problem solver (PyTorch frontend).

``pounce.torch.solve_bvp`` is the eager-mode mirror of
``pounce.jax.solve_bvp``: identical fixed-mesh Hermite--Simpson
collocation, the same feasibility-NLP formulation (``min 0`` s.t.
``R(z, theta) = 0``) routed through :func:`pounce.torch.solve`, and the
same implicit-function-theorem sensitivity ``dz*/dtheta =
-(dR/dz)^{-1} (dR/dtheta)``. The collocation residual itself is shared
verbatim with the NumPy and JAX paths (:mod:`pounce.bvp._core`).

``fun(x, y, p, theta)`` / ``bc(ya, yb, p, theta)`` (drop ``p`` when there
are no unknown parameters) must be written with torch ops. ``theta`` is a
tensor you can backprop into; the returned ``y`` / ``p`` participate in the
autograd graph through pounce.torch's ``solve`` Function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from . import solve as _pounce_solve
from ..bvp import _core


def _cat(parts):
    return torch.cat(parts)


@dataclass
class TorchBVPSolution:
    """Differentiable BVP solution (PyTorch).

    ``y`` ``(n, m)`` and ``p`` ``(k,)`` are tensors in the autograd graph;
    call ``.backward()`` on a downstream scalar to get ``theta.grad``.
    ``sol`` is a (detached) cubic Hermite interpolant for evaluation.
    """

    y: torch.Tensor
    p: Any
    z: torch.Tensor
    yp: torch.Tensor
    sol: Callable


def solve_bvp(fun, bc, x, y, p=None, theta=None, *, tol=1e-8, options=None):
    """Solve a BVP differentiably w.r.t. ``theta`` with PyTorch + pounce.

    See :func:`pounce.jax.solve_bvp` for the argument contract; this is the
    torch-tensor equivalent.
    """
    if theta is None:
        raise ValueError(
            "pounce.torch.solve_bvp requires `theta` (the differentiation "
            "parameter). For a plain non-differentiable solve use "
            "pounce.bvp.solve_bvp."
        )

    x = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float64)
    n, m = y.shape
    uses_p = p is not None
    if uses_p:
        p0 = torch.as_tensor(np.asarray(p), dtype=torch.float64).reshape(-1)
    else:
        p0 = torch.zeros(0, dtype=torch.float64)
    k = int(p0.shape[0])
    N = _core.num_unknowns(n, m, k)

    def f(z, th):
        return 0.0 * torch.sum(z)

    def g(z, th):
        nfun, nbc = _core._make_normalized(fun, bc, theta=th, uses_p=uses_p)
        Y, pp = _core.unpack_z(z, n, m)
        return _core.collocation_residual(nfun, nbc, x, Y, pp, _cat)

    z0 = _core.pack_z(y, p0, _cat)
    cl = torch.zeros(N, dtype=torch.float64)
    cu = torch.zeros(N, dtype=torch.float64)

    opts = {"tol": float(tol), "print_level": 0}
    if options:
        opts.update(options)

    z_star = _pounce_solve(
        theta, f=f, g=g, x0=z0, n=N, m=N, cl=cl, cu=cu, options=opts,
    )

    Y_star, p_star = _core.unpack_z(z_star, n, m)
    nfun, _ = _core._make_normalized(fun, bc, theta=theta, uses_p=uses_p)
    yp = nfun(x, Y_star, p_star)

    sol = _make_spline(x, Y_star, yp)

    return TorchBVPSolution(
        y=Y_star,
        p=(p_star if uses_p else None),
        z=z_star,
        yp=yp,
        sol=sol,
    )


def _make_spline(x, y, yp):
    """Lazily-built cubic Hermite interpolant over detached values."""
    cache = {}

    def sol(xq):
        if "spline" not in cache:
            from scipy.interpolate import CubicHermiteSpline

            xn = x.detach().cpu().numpy()
            yn = y.detach().cpu().numpy()
            ypn = yp.detach().cpu().numpy()
            cache["spline"] = CubicHermiteSpline(xn, yn.T, ypn.T)
        return cache["spline"](xq).T

    return sol
