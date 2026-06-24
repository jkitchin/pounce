"""Differentiable fully-implicit DAE integration on a fixed mesh (PyTorch).

Eager-mode mirror of :func:`pounce.jax._dae.daeint`. ``daeint(F, y0, t, theta)``
integrates ``F(t, y, y', theta) = 0`` on the fixed mesh ``t`` with backward-Euler
collocation (see ``pounce.ode._dae_collocation``) and returns the node
trajectory, differentiable w.r.t. ``theta`` and ``y0`` via the
implicit-function theorem. ``F`` must be torch-traceable.
"""

from __future__ import annotations

import numpy as np
import torch

from ..ode import _dae_collocation as C


def daeint(F, y0, t, theta, *, tol=1e-10):
    y0 = torch.as_tensor(y0, dtype=torch.float64)
    theta = torch.as_tensor(theta, dtype=torch.float64)
    t_t = torch.as_tensor(t, dtype=torch.float64)
    t_np = t_t.detach().cpu().numpy()
    n = y0.shape[0]
    m = t_np.shape[0]
    N = n * (m - 1)
    tshape = theta.shape

    def _np_F(th_np):
        th = torch.as_tensor(th_np, dtype=torch.float64)
        def f(ti, yi, ypi):
            r = F(float(ti), torch.as_tensor(yi, dtype=torch.float64),
                  torch.as_tensor(ypi, dtype=torch.float64), th)
            return r.detach().cpu().numpy()
        return f

    def _full_from_flat(zflat_np, y0_np):
        Yint = zflat_np.reshape(m - 1, n).T
        return np.concatenate([y0_np[:, None], Yint], axis=1)

    class _Solve(torch.autograd.Function):
        @staticmethod
        def forward(ctx, theta_, y0_):
            th_np = theta_.detach().cpu().numpy()
            y0_np = y0_.detach().cpu().numpy()
            Y = C.be_forward(_np_F(th_np), t_np, y0_np, tol=tol)
            z = np.ascontiguousarray(Y[:, 1:].T.reshape(-1))
            ctx.save_for_backward(theta_, y0_)
            ctx._z = z
            return torch.as_tensor(z, dtype=torch.float64)

        @staticmethod
        def backward(ctx, grad_z):
            theta_, y0_ = ctx.saved_tensors
            z = ctx._z
            th_np = theta_.detach().cpu().numpy()
            y0_np = y0_.detach().cpu().numpy()
            Yfull = _full_from_flat(z, y0_np)
            # R_Y^T u = grad_z  (host, FERAL sparse LU)
            u = C.be_transpose_solve(_np_F(th_np), t_np, Yfull, y0_np,
                                     grad_z.detach().cpu().numpy())
            u_t = torch.as_tensor(u, dtype=torch.float64)
            # IFT: dL/dp = -(dR/dp)^T u, via torch autodiff of the residual at z*.
            # Function.backward runs under no_grad, so build the residual graph
            # explicitly with enable_grad.
            with torch.enable_grad():
                th = theta_.detach().clone().requires_grad_(True)
                y0v = y0_.detach().clone().requires_grad_(True)
                R = _residual_torch(torch.as_tensor(z, dtype=torch.float64),
                                    th, y0v)
            gth, gy0 = torch.autograd.grad(R, (th, y0v), grad_outputs=-u_t)
            return gth, gy0

    def _residual_torch(zflat, th, y0v):
        Yint = zflat.reshape(m - 1, n).T                       # (n, m-1)
        Yfull = torch.cat([y0v[:, None], Yint], dim=1)         # (n, m)
        rows = []
        for j in range(m - 1):
            h = t_t[j + 1] - t_t[j]
            w = Yfull[:, j + 1]
            wp = (w - Yfull[:, j]) / h
            rows.append(F(float(t_np[j + 1]), w, wp, th))
        return torch.cat(rows)

    z = _Solve.apply(theta, y0)
    Yint = z.reshape(m - 1, n).T
    return torch.cat([y0[:, None], Yint], dim=1)               # (n, m)
