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


def daeint(F, y0, t, theta, *, tol=1e-10):
    y0 = jnp.asarray(y0, dtype=jnp.float64)
    t = jnp.asarray(t, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    n = y0.shape[0]
    m = t.shape[0]
    k = theta.size
    N = n * (m - 1)
    t_np = np.asarray(t, dtype=np.float64)

    def _split(p):
        return p[:k].reshape(theta.shape), p[k:]

    # numpy residual callable F(t,y,yp) closed over a concrete theta
    def _np_F(th):
        return lambda ti, yi, ypi: np.asarray(F(ti, yi, ypi, th), dtype=np.float64)

    def _full_from_flat_np(zflat, y0v):
        Yint = np.asarray(zflat, np.float64).reshape(m - 1, n).T   # (n, m-1)
        return np.concatenate([np.asarray(y0v)[:, None], Yint], axis=1)

    def _host_solve(p_np):
        p_np = np.asarray(p_np, np.float64)
        th, y0v = _split(p_np)
        Y = C.be_forward(_np_F(np.asarray(th)), t_np, np.asarray(y0v), tol=tol)
        return np.ascontiguousarray(Y[:, 1:].T.reshape(-1))   # node-major flat

    def _host_btran(z_np, p_np, v_np):
        p_np = np.asarray(p_np, np.float64)
        th, y0v = _split(p_np)
        Y = _full_from_flat_np(z_np, y0v)
        return C.be_transpose_solve(_np_F(np.asarray(th)), t_np, Y,
                                    np.asarray(y0v), np.asarray(v_np))

    # JAX-traced residual R(zflat, p) for the parameter VJP
    def _residual_jax(zflat, p):
        th, y0v = _split(p)
        Yint = zflat.reshape(m - 1, n).T                      # (n, m-1)
        Yfull = jnp.concatenate([y0v[:, None], Yint], axis=1)  # (n, m)
        rows = []
        for j in range(m - 1):
            h = t[j + 1] - t[j]
            w = Yfull[:, j + 1]
            wp = (w - Yfull[:, j]) / h
            rows.append(F(t[j + 1], w, wp, th))
        return jnp.concatenate(rows)

    @jax.custom_vjp
    def solve(p):
        return jax.pure_callback(
            _host_solve, jax.ShapeDtypeStruct((N,), jnp.float64), p,
            vmap_method="sequential")

    def _fwd(p):
        z = jax.pure_callback(
            _host_solve, jax.ShapeDtypeStruct((N,), jnp.float64), p,
            vmap_method="sequential")
        return z, (z, p)

    def _bwd(res, v):
        z, p = res
        u = jax.pure_callback(
            _host_btran, jax.ShapeDtypeStruct((N,), jnp.float64),
            z, p, v, vmap_method="sequential")
        # IFT: dL/dp = -(dR/dp)^T u, with R_Y^T u = v already solved on host.
        _, vjp = jax.vjp(lambda pp: _residual_jax(z, pp), p)
        (g,) = vjp(-u)
        return (g,)

    solve.defvjp(_fwd, _bwd)

    p = jnp.concatenate([theta.reshape(-1), y0])
    z = solve(p)
    Yint = z.reshape(m - 1, n).T
    return jnp.concatenate([y0[:, None], Yint], axis=1)        # (n, m)
