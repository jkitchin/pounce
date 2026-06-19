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
from ..bvp._solve import _make_spline


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


def _newton_autograd_fn(fun, bc, x, n, m, k, uses_p, z0, tol):
    """First-order-differentiable feral-Newton solve as a torch Function.

    Forward: damped Newton on ``R(z, theta) = 0`` via FERAL's sparse LU.
    Backward: the implicit-function-theorem VJP — ``R_z^T u = grad_out`` via
    ``SparseLU.solve_transpose``, then ``-(dR/dtheta)^T u`` through torch
    autograd of the residual at ``z*``. Both directions stay on the ``N``
    system (no ``2N`` saddle); first-order only. Mirrors the JAX path.
    """
    import numpy as _np
    from ..bvp._jac import CollocationJacobian
    from ..bvp._newton import newton_solve
    from ..bvp._solve import ift_solve_transpose

    x_np = _np.asarray(x.detach().cpu().numpy(), dtype=_np.float64)
    z0_np = _np.asarray(z0.detach().cpu().numpy(), dtype=_np.float64)

    def _np_normalized(theta_t):
        nfun_t, nbc_t = _core._make_normalized(fun, bc, theta=theta_t, uses_p=uses_p)
        nfun = lambda xx, YY, pp: _np.asarray(
            nfun_t(torch.as_tensor(xx, dtype=torch.float64),
                   torch.as_tensor(YY, dtype=torch.float64),
                   torch.as_tensor(pp, dtype=torch.float64)).detach().cpu().numpy(),
            dtype=_np.float64,
        )
        nbc = lambda ya, yb, pp: _np.asarray(
            nbc_t(torch.as_tensor(ya, dtype=torch.float64),
                  torch.as_tensor(yb, dtype=torch.float64),
                  torch.as_tensor(pp, dtype=torch.float64)).detach().cpu().numpy(),
            dtype=_np.float64,
        )
        return nfun, nbc

    class _Newton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, theta):
            nfun, nbc = _np_normalized(theta.detach())

            def residual_fn(z):
                return _core.residual_of_z(z, nfun, nbc, x_np, n, m, k, _np.concatenate)

            jac = CollocationJacobian(nfun, nbc, x_np, n, m, k)
            z_star, _it, _ok, _rn = newton_solve(
                residual_fn, jac, z0_np, n, m, k, tol=float(tol),
            )
            z_star = _np.asarray(z_star, dtype=_np.float64)
            ctx.save_for_backward(theta)
            ctx._z_star = z_star
            return torch.as_tensor(z_star, dtype=torch.float64)

        @staticmethod
        def backward(ctx, grad_out):
            (theta,) = ctx.saved_tensors
            z_star = ctx._z_star
            nfun, nbc = _np_normalized(theta.detach())
            # Shared host-side IFT back-solve: R_z^T u = grad_out at z*.
            u = ift_solve_transpose(
                nfun, nbc, x_np, n, m, k, z_star,
                _np.asarray(grad_out.detach().cpu().numpy(), _np.float64),
            )
            u_t = torch.as_tensor(u, dtype=torch.float64)
            # dL/dtheta = -(dR/dtheta)^T u via torch autograd of the residual.
            # backward() runs with grad disabled by default — re-enable it so
            # the residual graph w.r.t. theta is tracked.
            with torch.enable_grad():
                th = theta.detach().clone().requires_grad_(True)
                z_t = torch.as_tensor(z_star, dtype=torch.float64)
                nfun_t, nbc_t = _core._make_normalized(fun, bc, theta=th, uses_p=uses_p)
                Yt, ppt = _core.unpack_z(z_t, n, m)
                R = _core.collocation_residual(nfun_t, nbc_t, x, Yt, ppt, _cat)
                (dtheta,) = torch.autograd.grad(R, th, grad_outputs=-u_t)
            return dtheta

    return _Newton.apply


def solve_bvp(
    fun, bc, x, y, p=None, theta=None, *, tol=1e-8, options=None, method="newton",
):
    """Solve a BVP differentiably w.r.t. ``theta`` with PyTorch + pounce.

    See :func:`pounce.jax.solve_bvp` for the argument contract; this is the
    torch-tensor equivalent. ``method="newton"`` (default) uses the fast
    FERAL sparse-LU Newton forward with an implicit-function-theorem
    backward; ``method="ipm"`` routes through :func:`pounce.torch.solve`.
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

    if method == "newton":
        solve_root = _newton_autograd_fn(fun, bc, x, n, m, k, uses_p, z0, tol)
        z_star = solve_root(theta)
    elif method == "ipm":
        z_star = _pounce_solve(
            theta, f=f, g=g, x0=z0, n=N, m=N, cl=cl, cu=cu, options=opts,
        )
    else:
        raise ValueError(f"unknown method {method!r}; use 'newton' or 'ipm'.")

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
