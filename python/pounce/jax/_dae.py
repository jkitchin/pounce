"""Differentiable fully-implicit DAE integration on a fixed mesh (JAX).

``daeint(F, y0, t, theta)`` integrates ``F(t, y, y', theta) = 0`` on the fixed
mesh ``t`` with backward-Euler collocation and returns the node trajectory
differentiable w.r.t. ``theta``, ``y0`` and ``t``, via the implicit-function
theorem on the collocation system (the same scheme as
``pounce.ode._dae_collocation``).

The rule is a ``jax.custom_jvp`` (gh#1229), so forward mode (``jax.jvp`` /
``jax.jacfwd``) and reverse mode (``jax.grad`` / ``jax.jacrev``, by
transposition) both work, as does ``jax.vmap``. The forward solve and the
``R_Y`` / ``R_Y^T`` linear solves run on the host (numpy + FERAL sparse LU);
the parameter terms ``dR/dp`` are taken by JAX autodiff of a traced residual
at the converged nodes. ``F`` must therefore be JAX-traceable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..bvp._solve import solve_by_primal_lanes
from ..ode import _dae_collocation as C
from ._host import _UNTRACEABLE, host_jit
from ._ift import make_ift_root


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

    # The host Newton loop and its finite-difference Jacobian evaluate F
    # outside any enclosing jit trace; compile it once per solve instead of
    # dispatching op by op (gh#1241).
    F_host = host_jit(F)

    # numpy residual callable F(t,y,yp) closed over a concrete theta
    def _np_F(th):
        return lambda ti, yi, ypi: np.asarray(F_host(ti, yi, ypi, th),
                                              dtype=np.float64)

    # Node Jacobians (dF/dy, dF/dy') by forward-mode autodiff, built once per
    # solve with theta an argument. The collocation core's own jacfwd probe
    # cannot see through the numpy wrapper in ``_np_F`` and silently fell
    # back to central differences (2n residuals per node per Newton step,
    # sqrt(eps)-accurate). ``host_jit`` compiles it; if F branches in Python
    # on a concrete value, jit cannot trace it but un-jitted jacfwd can (its
    # JVP tracers carry concrete primals), so the fallback is still exact.
    JF_host = host_jit(jax.jacfwd(lambda *a: jnp.asarray(F(*a)),
                                  argnums=(1, 2)))

    def _np_jac(th):
        def jac(ti, yi, ypi):
            try:
                Fy, Fyp = JF_host(ti, yi, ypi, th)
            except _UNTRACEABLE as e:
                raise TypeError(
                    "daeint: F must be JAX-traceable (jax.jacfwd of F "
                    "failed even without jit): " + str(e)) from e
            return np.asarray(Fy, np.float64), np.asarray(Fyp, np.float64)
        return jac

    def _full_from_flat_np(zflat, y0v):
        Yint = np.asarray(zflat, np.float64).reshape(m - 1, n).T   # (n, m-1)
        return np.concatenate([np.asarray(y0v)[:, None], Yint], axis=1)

    def _host_solve(p_np, t_np):
        p_np = np.asarray(p_np, np.float64)
        t_np = np.asarray(t_np, np.float64)
        th, y0v = _split(p_np)
        th = np.asarray(th)
        Y = C.collocation_forward(_np_F(th), t_np, np.asarray(y0v),
                                  order=order, jac=_np_jac(th), tol=tol)
        return np.ascontiguousarray(Y[:, 1:].T.reshape(-1))   # node-major flat

    def _host_lin(z_np, p_np, t_np, v_np, transpose):
        # broadcast_all: lanes may share one primal (jacfwd / jacrev) or carry
        # distinct ones (an outer vmap); group them by primal (gh#1049).
        def group(z_i, p_i, t_i, V):
            th, y0v = _split(p_i)
            Y = _full_from_flat_np(z_i, y0v)
            th = np.asarray(th)
            return C.collocation_lin_solve(
                _np_F(th), t_i, Y, V, order=order, jac=_np_jac(th),
                transpose=transpose)
        return solve_by_primal_lanes(group, v_np, (z_np, p_np, t_np), (1, 1, 1))

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

    def _lin_solve(z, args, b, transpose):
        p, t_ = args
        return jax.pure_callback(
            lambda zz, pp, tt, bb: _host_lin(zz, pp, tt, bb, transpose),
            jax.ShapeDtypeStruct((N,), jnp.float64),
            z, p, t_, b, vmap_method="broadcast_all")

    # Implicit-function custom_jvp (gh#1229): jvp / jacfwd directly,
    # grad / jacrev by transposition, R_Y solves on the host sparse LU.
    solve = make_ift_root(_forward, _residual_jax, _lin_solve)

    p = jnp.concatenate([theta.reshape(-1), y0])
    z = solve(p, t)
    Yint = z.reshape(m - 1, n).T
    return jnp.concatenate([y0[:, None], Yint], axis=1)        # (n, m)
