"""Differentiable ODE integration on a fixed mesh (JAX frontend).

``pounce.jax.odeint`` integrates ``dy/dt = f(t, y, theta)`` and returns the
trajectory on a user-supplied mesh ``t`` differentiably w.r.t. the ODE
parameters ``theta`` **and** the initial condition ``y0``.

An initial value problem on a fixed mesh is just a boundary value problem
whose boundary condition pins the left end, ``bc(ya, yb) = ya - y0``, so this
reuses pounce's 4th-order Hermite--Simpson collocation and its
implicit-function-theorem differentiation verbatim (see
:mod:`pounce.bvp._core`). The converged node states solve the *square*
collocation root-find ``R(z, p) = 0`` with ``p = [theta; y0]``; gradients
fall out of ``dz*/dp = -(dR/dz)^{-1}(dR/dp)`` with the same FERAL sparse-LU
back-solve used by the BVP layer — no per-step adjoint, no unrolled tape.

This is the *differentiable* counterpart to the adaptive, stiff/DAE
:func:`pounce.ode.solve_ivp`: the result here is the collocation solution on
the mesh you pass (make it fine enough to resolve the dynamics), and its
gradients are exact for that discretisation. ``f`` must be JAX-traceable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..bvp import _core
from ..bvp._solve import _make_spline, ift_solve_transpose
from ..ode._solve import mesh_initial_guess


@dataclass
class JaxODESolution:
    """Differentiable IVP solution on the mesh ``t``.

    ``t`` ``(m,)`` and ``y`` ``(n, m)`` (SciPy ``solve_ivp`` layout). ``y``
    carries the custom-VJP back to ``theta`` and ``y0``; differentiate
    through it with ``jax.grad`` / ``jax.jacobian``. ``yp`` (``dy/dt`` at the
    nodes) and ``sol`` (a cubic-Hermite interpolant) are **non-differentiable**
    diagnostics — both are detached, so only ``y`` should appear in a loss.
    """

    t: jnp.ndarray
    y: jnp.ndarray
    yp: jnp.ndarray
    sol: Callable


def odeint(fun, y0, t, theta=None, *, tol=1e-8):
    """Differentiably integrate ``dy/dt = f(t, y, theta)`` on the mesh ``t``.

    Parameters
    ----------
    fun : callable
        ``fun(t, y, theta) -> dy/dt`` (``(n,)``), or ``fun(t, y)`` when
        ``theta`` is ``None``. Scalar ``t``, state ``y`` ``(n,)``.
        JAX-traceable.
    y0 : array (n,)
        Initial state. Differentiable.
    t : array (m,)
        Output / collocation mesh (monotonic). The solution is returned at
        these points; refine it to resolve the dynamics.
    theta : array-like or None
        ODE parameters threaded into ``fun``. Differentiable.
    tol : float
        Collocation Newton tolerance.

    Returns
    -------
    JaxODESolution
        ``y`` ``(n, m)`` differentiable w.r.t. ``theta`` and ``y0``.
    """
    y0 = jnp.asarray(y0, dtype=jnp.float64).ravel()
    n = int(y0.shape[0])
    tj = jnp.asarray(t, dtype=jnp.float64)
    m = int(tj.shape[0])
    t_np = np.asarray(tj, dtype=np.float64)

    has_theta = theta is not None
    if has_theta:
        theta = jnp.asarray(theta, dtype=jnp.float64)
        theta_shape = theta.shape
        theta_flat = theta.ravel()
        ntheta = int(theta_flat.shape[0])
    else:
        theta_shape = ()
        theta_flat = jnp.zeros(0, dtype=jnp.float64)
        ntheta = 0

    combined = jnp.concatenate([theta_flat, y0])

    def _split(p):
        th = p[:ntheta].reshape(theta_shape) if has_theta else None
        return th, p[ntheta:]

    def _vec_rhs(xx, YY, th):
        # xx (m',), YY (n, m'); evaluate the scalar-t RHS column-wise.
        def col(xi, yi):
            return fun(xi, yi, th) if has_theta else fun(xi, yi)
        return jax.vmap(col, in_axes=(0, 1), out_axes=1)(xx, YY)

    def residual_jax(z, p):
        th, y0v = _split(p)
        nfun = lambda xx, YY, pp: _vec_rhs(xx, YY, th)
        nbc = lambda ya, yb, pp: ya - y0v
        Y = z[: n * m].reshape(n, m)
        pp = z[n * m:]
        return _core.collocation_residual(nfun, nbc, tj, Y, pp, jnp.concatenate)

    def _np_callables(p_np):
        th, y0v = _split(p_np)
        th = None if not has_theta else np.asarray(th)
        y0v = np.asarray(y0v, dtype=np.float64)

        def nfun(xx, YY, pp):
            return np.asarray(_vec_rhs(xx, YY, th), dtype=np.float64)

        def nbc(ya, yb, pp):
            return np.asarray(ya, dtype=np.float64) - y0v
        return nfun, nbc, y0v

    def _host_solve(p_np):
        from ..bvp._jac import CollocationJacobian
        from ..bvp._newton import newton_solve, STATUS_CONVERGED
        p_np = np.asarray(p_np, dtype=np.float64)
        nfun, nbc, y0v = _np_callables(p_np)

        def fun_np(ti, yi):
            return nfun(np.array([ti]), np.asarray(yi)[:, None], None)[:, 0]

        Yg = mesh_initial_guess(fun_np, t_np, y0v, n, m)
        z0 = Yg.reshape(-1)

        def residual_fn(z):
            return _core.residual_of_z(z, nfun, nbc, t_np, n, m, 0, np.concatenate)

        jac = CollocationJacobian(nfun, nbc, t_np, n, m, 0)
        z_star, _it, status, rnorm = newton_solve(
            residual_fn, jac, z0, n, m, 0, tol=float(tol),
        )
        # Returning a non-converged z* would yield a wrong trajectory *and*
        # IFT gradients about a point where R(z) != 0. There is no status
        # surface on the solution object, so fail loudly instead.
        if status != STATUS_CONVERGED:
            raise RuntimeError(
                "pounce.jax.odeint: collocation Newton did not converge "
                f"(status={status}, ||R||={rnorm:.3e}). The fixed mesh `t` is "
                "likely too coarse to resolve the dynamics — refine it."
            )
        return np.asarray(z_star, dtype=np.float64)

    def _host_btran(z_star_np, p_np, v_np):
        z_star_np = np.asarray(z_star_np, np.float64)
        p_np = np.asarray(p_np, np.float64)
        v_np = np.asarray(v_np, np.float64)
        if z_star_np.ndim == 2:
            z_star_np = z_star_np[0]
        if p_np.ndim == 2:
            p_np = p_np[0]
        nfun, nbc, _ = _np_callables(p_np)
        return ift_solve_transpose(nfun, nbc, t_np, n, m, 0, z_star_np, v_np)

    N = n * m

    @jax.custom_vjp
    def solve_fn(p):
        return jax.pure_callback(
            _host_solve, jax.ShapeDtypeStruct((N,), jnp.float64), p,
            vmap_method="sequential",
        )

    def fwd(p):
        z_star = jax.pure_callback(
            _host_solve, jax.ShapeDtypeStruct((N,), jnp.float64), p,
            vmap_method="sequential",
        )
        return z_star, (z_star, p)

    def bwd(res, v):
        z_star, p = res
        u = jax.pure_callback(
            _host_btran, jax.ShapeDtypeStruct((N,), jnp.float64),
            z_star, p, v, vmap_method="broadcast_all",
        )
        _, vjp_p = jax.vjp(lambda pp: residual_jax(z_star, pp), p)
        (dp,) = vjp_p(-u)
        return (dp,)

    solve_fn.defvjp(fwd, bwd)

    z_star = solve_fn(combined)
    Y_star = z_star.reshape(n, m)
    th, _ = _split(combined)
    # yp / sol are non-differentiable diagnostics. yp is recomputed outside
    # the custom-VJP boundary, so left attached it would carry a spurious
    # direct df/dtheta term inconsistent with the true converged sensitivity
    # (which flows only through Y_star); stop_gradient makes that explicit.
    yp = jax.lax.stop_gradient(_vec_rhs(tj, Y_star, th))
    sol = _make_spline(tj, jax.lax.stop_gradient(Y_star), yp)

    return JaxODESolution(t=tj, y=Y_star, yp=yp, sol=sol)
