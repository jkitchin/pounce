"""Differentiable fully-implicit DAE integration on a fixed mesh (JAX).

``daeint(F, y0, t, theta)`` integrates ``F(t, y, y', theta) = 0`` on the fixed
mesh ``t`` with backward-Euler collocation and returns the node trajectory
differentiable w.r.t. ``theta`` and ``y0``, via the implicit-function theorem on
the collocation system (the same scheme as ``pounce.ode._dae_collocation``).

Forward solve and the ``R_Y^T`` back-solve run on the host (numpy + FERAL sparse
LU); the parameter VJP ``(dR/dp)^T u`` is taken by JAX autodiff of a traced
residual at the converged nodes. ``F`` must therefore be JAX-traceable.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..ode import _dae_collocation as C


def daeint(F, y0, t, theta, *, order=2, tol=1e-10):
    y0 = jnp.asarray(y0, dtype=jnp.float64)
    t = jnp.asarray(t, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    n = y0.shape[0]
    m = t.shape[0]
    k = theta.size
    N = n * (m - 1)
    # ``t`` is a callback operand, never read as numpy at trace time, so it
    # may be a tracer (built under ``jax.jit``, a jitted argument, vmapped);
    # it is also differentiable through the stencil in ``_residual_jax``.

    def _split(p):
        return p[:k].reshape(theta.shape), p[k:]

    # numpy residual callable F(t,y,yp) closed over a concrete theta
    def _np_F(th):
        return lambda ti, yi, ypi: np.asarray(F(ti, yi, ypi, th), dtype=np.float64)

    def _full_from_flat_np(zflat, y0v):
        Yint = np.asarray(zflat, np.float64).reshape(m - 1, n).T   # (n, m-1)
        return np.concatenate([np.asarray(y0v)[:, None], Yint], axis=1)

    def _host_solve(p_np, t_np):
        p_np = np.asarray(p_np, np.float64)
        t_np = np.asarray(t_np, np.float64)
        th, y0v = _split(p_np)
        Y = C.collocation_forward(_np_F(np.asarray(th)), t_np, np.asarray(y0v),
                                  order=order, tol=tol)
        return np.ascontiguousarray(Y[:, 1:].T.reshape(-1))   # node-major flat

    def _host_btran(z_np, p_np, t_np, v_np):
        p_np = np.asarray(p_np, np.float64)
        t_np = np.asarray(t_np, np.float64)
        th, y0v = _split(p_np)
        Y = _full_from_flat_np(z_np, y0v)
        return C.collocation_transpose_solve(
            _np_F(np.asarray(th)), t_np, Y, np.asarray(v_np), order=order)

    def _wp(Yfull, j, t_):
        # backward-difference y' stencil matching the collocation core; the
        # coefficients are functions of the (fixed, not adapted) mesh.
        h = t_[j + 1] - t_[j]
        if order == 1 or j == 0:
            return (Yfull[:, j + 1] - Yfull[:, j]) / h
        hm = t_[j] - t_[j - 1]; rho = h / hm
        c_w = (1.0 + 2.0 * rho) / (1.0 + rho) / h
        c_k = -(1.0 + rho) / h
        c_km1 = rho * rho / (1.0 + rho) / h
        return c_w * Yfull[:, j + 1] + c_k * Yfull[:, j] + c_km1 * Yfull[:, j - 1]

    # JAX-traced residual R(zflat, p) for the parameter VJP
    def _residual_jax(zflat, p, t_):
        th, y0v = _split(p)
        Yint = zflat.reshape(m - 1, n).T                      # (n, m-1)
        Yfull = jnp.concatenate([y0v[:, None], Yint], axis=1)  # (n, m)
        rows = [F(t_[j + 1], Yfull[:, j + 1], _wp(Yfull, j, t_), th)
                for j in range(m - 1)]
        return jnp.concatenate(rows)

    def _forward(p, t_):
        return jax.pure_callback(
            _host_solve, jax.ShapeDtypeStruct((N,), jnp.float64), p, t_,
            vmap_method="sequential")

    @jax.custom_vjp
    def solve(p, t_):
        return _forward(p, t_)

    def _fwd(p, t_):
        z = _forward(p, t_)
        return z, (z, p, t_)

    def _bwd(res, v):
        z, p, t_ = res
        u = jax.pure_callback(
            _host_btran, jax.ShapeDtypeStruct((N,), jnp.float64),
            z, p, t_, v, vmap_method="sequential")
        # IFT: dL/d(p, t) = -(dR/d(p, t))^T u, with R_Y^T u = v already
        # solved on host.
        _, vjp = jax.vjp(lambda pp, tt: _residual_jax(z, pp, tt), p, t_)
        return vjp(-u)

    solve.defvjp(_fwd, _bwd)

    p = jnp.concatenate([theta.reshape(-1), y0])
    z = solve(p, t)
    Yint = z.reshape(m - 1, n).T
    return jnp.concatenate([y0[:, None], Yint], axis=1)        # (n, m)
