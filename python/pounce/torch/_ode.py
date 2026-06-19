"""Differentiable ODE integration on a fixed mesh (PyTorch frontend).

Eager-mode mirror of :func:`pounce.jax.odeint`: integrates
``dy/dt = f(t, y, theta)`` on a fixed mesh ``t`` and returns the trajectory
differentiably w.r.t. the ODE parameters ``theta`` **and** the initial
condition ``y0``.

An IVP on a fixed mesh is a boundary value problem with ``bc(ya, yb) =
ya - y0``, so this reuses pounce's Hermite--Simpson collocation core
(:mod:`pounce.bvp._core`) and the implicit-function-theorem back-solve
(``R_z^T u = grad_out`` via FERAL's sparse LU) shared with the BVP layer.
``f`` must be written with torch ops; ``theta`` / ``y0`` are tensors you can
backprop into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from ..bvp import _core
from ..bvp._solve import _make_spline, ift_solve_transpose
from ..ode._solve import mesh_initial_guess


def _cat(parts):
    return torch.cat(parts)


@dataclass
class TorchODESolution:
    """Differentiable IVP solution on the mesh ``t`` (PyTorch).

    ``t`` ``(m,)`` and ``y`` ``(n, m)`` (SciPy ``solve_ivp`` layout). ``y``
    is in the autograd graph; ``.backward()`` on a downstream scalar fills
    ``theta.grad`` / ``y0.grad``. ``yp`` (``dy/dt`` at the nodes) and ``sol``
    (a cubic-Hermite interpolant) are **non-differentiable** diagnostics —
    both detached, so only ``y`` should appear in a loss.
    """

    t: torch.Tensor
    y: torch.Tensor
    yp: torch.Tensor
    sol: Callable


def _vec_rhs(fun, xx, YY, th, has_theta):
    """Column-wise scalar-``t`` RHS over the mesh -> ``(n, m)``."""
    cols = []
    for j in range(YY.shape[1]):
        yj, xj = YY[:, j], xx[j]
        cols.append(fun(xj, yj, th) if has_theta else fun(xj, yj))
    return torch.stack(cols, dim=1)


def odeint(fun, y0, t, theta=None, *, tol=1e-8):
    """Differentiably integrate ``dy/dt = f(t, y, theta)`` on the mesh ``t``.

    See :func:`pounce.jax.odeint` for the argument contract; this is the
    torch-tensor equivalent. ``fun(t, y, theta) -> dy/dt`` (or ``fun(t, y)``
    when ``theta`` is ``None``) is written with torch ops. ``y0`` ``(n,)``
    and ``theta`` are tensors; the returned ``y`` ``(n, m)`` is
    differentiable w.r.t. both.
    """
    y0 = torch.as_tensor(np.asarray(y0) if not torch.is_tensor(y0) else y0,
                         dtype=torch.float64).reshape(-1)
    n = int(y0.shape[0])
    tt = torch.as_tensor(np.asarray(t) if not torch.is_tensor(t) else t,
                         dtype=torch.float64)
    m = int(tt.shape[0])
    t_np = tt.detach().cpu().numpy().astype(np.float64)

    has_theta = theta is not None
    if has_theta:
        theta = torch.as_tensor(
            np.asarray(theta) if not torch.is_tensor(theta) else theta,
            dtype=torch.float64,
        )
        theta_shape = tuple(theta.shape)
        theta_flat = theta.reshape(-1)
        ntheta = int(theta_flat.shape[0])
    else:
        theta_shape = ()
        theta_flat = torch.zeros(0, dtype=torch.float64)
        ntheta = 0

    combined = torch.cat([theta_flat, y0])

    def _split(p):
        th = p[:ntheta].reshape(theta_shape) if has_theta else None
        return th, p[ntheta:]

    def _np_callables(p_t):
        th, y0v = _split(p_t)
        y0v_np = y0v.detach().cpu().numpy().astype(np.float64)

        def nfun(xx, YY, pp):
            xt = torch.as_tensor(xx, dtype=torch.float64)
            Yt = torch.as_tensor(YY, dtype=torch.float64)
            out = _vec_rhs(fun, xt, Yt, th, has_theta)
            return out.detach().cpu().numpy().astype(np.float64)

        def nbc(ya, yb, pp):
            return np.asarray(ya, dtype=np.float64) - y0v_np
        return nfun, nbc, y0v_np

    class _Solve(torch.autograd.Function):
        @staticmethod
        def forward(ctx, p):
            from ..bvp._jac import CollocationJacobian
            from ..bvp._newton import newton_solve, STATUS_CONVERGED
            nfun, nbc, y0v_np = _np_callables(p.detach())

            def fun_np(ti, yi):
                return nfun(np.array([ti]), np.asarray(yi)[:, None], None)[:, 0]

            Yg = mesh_initial_guess(fun_np, t_np, y0v_np, n, m)
            z0 = Yg.reshape(-1)

            def residual_fn(z):
                return _core.residual_of_z(z, nfun, nbc, t_np, n, m, 0,
                                           np.concatenate)

            jac = CollocationJacobian(nfun, nbc, t_np, n, m, 0)
            z_star, _it, status, rnorm = newton_solve(
                residual_fn, jac, z0, n, m, 0, tol=float(tol),
            )
            # A non-converged z* gives a wrong trajectory and IFT gradients
            # about a point where R(z) != 0; there is no status field on the
            # solution, so fail loudly rather than return silently-wrong data.
            if status != STATUS_CONVERGED:
                raise RuntimeError(
                    "pounce.torch.odeint: collocation Newton did not converge "
                    f"(status={status}, ||R||={rnorm:.3e}). The fixed mesh `t` "
                    "is likely too coarse to resolve the dynamics — refine it."
                )
            z_star = np.asarray(z_star, dtype=np.float64)
            ctx.save_for_backward(p)
            ctx._z_star = z_star
            return torch.as_tensor(z_star, dtype=torch.float64)

        @staticmethod
        def backward(ctx, grad_out):
            (p,) = ctx.saved_tensors
            z_star = ctx._z_star
            nfun, nbc, _ = _np_callables(p.detach())
            u = ift_solve_transpose(
                nfun, nbc, t_np, n, m, 0, z_star,
                grad_out.detach().cpu().numpy().astype(np.float64),
            )
            u_t = torch.as_tensor(u, dtype=torch.float64)
            with torch.enable_grad():
                pp = p.detach().clone().requires_grad_(True)
                th, y0v = _split(pp)
                z_t = torch.as_tensor(z_star, dtype=torch.float64)
                Yt = z_t[: n * m].reshape(n, m)
                nfun_t = lambda xx, YY, q: _vec_rhs(fun, xx, YY, th, has_theta)
                nbc_t = lambda ya, yb, q: ya - y0v
                R = _core.collocation_residual(
                    nfun_t, nbc_t, tt, Yt, z_t[n * m:], _cat)
                (dp,) = torch.autograd.grad(R, pp, grad_outputs=-u_t)
            return dp

    z_star = _Solve.apply(combined)
    Y_star = z_star.reshape(n, m)
    th, _ = _split(combined)
    # yp / sol are non-differentiable diagnostics (detached): yp is recomputed
    # outside the autograd.Function boundary, so attached it would carry a
    # spurious direct df/dtheta term inconsistent with the true converged
    # sensitivity (which flows only through Y_star).
    yp = _vec_rhs(fun, tt, Y_star.detach(), th.detach() if has_theta else th,
                  has_theta)
    sol = _make_spline(tt, Y_star.detach(), yp)

    return TorchODESolution(t=tt, y=Y_star, yp=yp, sol=sol)
