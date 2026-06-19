"""Differentiable boundary value problem solver (JAX frontend).

``pounce.jax.solve_bvp`` discretises a BVP with the same fixed-mesh
Hermite--Simpson collocation as :func:`pounce.bvp.solve_bvp`, but poses the
square collocation root-find ``R(z, theta) = 0`` as a pounce *feasibility*
NLP (``min 0`` s.t. ``R = 0``) routed through :func:`pounce.jax.solve`. The
converged node states ``y`` and unknown parameters ``p`` are therefore
differentiable w.r.t. ``theta`` — any quantity ``fun`` / ``bc`` close over
(a physical coefficient, a boundary value, ...) — via the
implicit-function theorem on the collocation KKT system:

    dz*/dtheta = -(dR/dz)^{-1} (dR/dtheta).

For the square, all-equality feasibility problem the generic pounce.jax
implicit-diff backward collapses to exactly this Newton sensitivity (the
objective Hessian is absent, every constraint row is an active equality,
and there are no variable bounds), so ``jax.grad`` / ``jax.jacobian``
through the returned ``y`` / ``p`` give the right gradients with no
BVP-specific backward code.

``fun`` and ``bc`` take ``theta`` as a trailing argument
(``fun(x, y, p, theta)`` / ``bc(ya, yb, p, theta)``, or without ``p`` when
there are no unknown parameters) and must be JAX-traceable. The mesh,
initial guess, and pounce options are static.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import solve as _pounce_solve
from ..bvp import _core


@dataclass
class JaxBVPSolution:
    """Differentiable BVP solution.

    ``y`` ``(n, m)`` and ``p`` ``(k,)`` are JAX arrays carrying the
    custom-VJP back to ``theta``; differentiate through them with
    ``jax.grad`` / ``jax.jacobian``. ``z`` is the flat unknown vector and
    ``yp`` the node derivatives. ``sol`` is a (non-differentiable) cubic
    Hermite interpolant over concrete values for plotting / evaluation.
    """

    y: jnp.ndarray
    p: Any
    z: jnp.ndarray
    yp: jnp.ndarray
    sol: Callable


def solve_bvp(fun, bc, x, y, p=None, theta=None, *, tol=1e-8, options=None):
    """Solve a BVP differentiably w.r.t. ``theta`` with JAX + pounce.

    Parameters
    ----------
    fun, bc : callable
        ``fun(x, y, p, theta) -> (n, m)`` and
        ``bc(ya, yb, p, theta) -> (n + k,)`` (drop ``p`` if there are no
        unknown parameters). JAX-traceable.
    x : array (m,)
        Fixed mesh.
    y : array (n, m)
        Initial guess for the node states (not differentiated through).
    p : array (k,) or None
        Initial guess for unknown parameters, or ``None``.
    theta : pytree
        The differentiation parameter threaded into ``fun`` / ``bc``.
    tol : float
        pounce convergence tolerance.
    options : dict or None
        Extra pounce options.

    Returns
    -------
    JaxBVPSolution
    """
    if theta is None:
        raise ValueError(
            "pounce.jax.solve_bvp requires `theta` (the differentiation "
            "parameter). For a plain non-differentiable solve use "
            "pounce.bvp.solve_bvp."
        )

    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    n, m = y.shape
    uses_p = p is not None
    p0 = jnp.asarray(p, dtype=jnp.float64).ravel() if uses_p else jnp.zeros(0)
    k = int(p0.shape[0])
    N = _core.num_unknowns(n, m, k)

    def f(z, th):
        # Pure feasibility: zero objective (kept dependent on z so the
        # trace has the right shape; grad is identically zero).
        return 0.0 * jnp.sum(z)

    def g(z, th):
        nfun, nbc = _core._make_normalized(fun, bc, theta=th, uses_p=uses_p)
        Y, pp = _core.unpack_z(z, n, m)
        return _core.collocation_residual(nfun, nbc, x, Y, pp, jnp.concatenate)

    z0 = _core.pack_z(y, p0, jnp.concatenate)
    cl = jnp.zeros(N, dtype=jnp.float64)
    cu = jnp.zeros(N, dtype=jnp.float64)

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

    return JaxBVPSolution(
        y=Y_star,
        p=(p_star if uses_p else None),
        z=z_star,
        yp=yp,
        sol=sol,
    )


def _make_spline(x, y, yp):
    """Lazily-built cubic Hermite interpolant.

    Construction is deferred to first call so a ``solve_bvp`` invoked
    *inside* a JAX trace (``jax.grad`` / ``jit``), where ``y`` / ``yp`` are
    tracers, does not eagerly force them to concrete NumPy arrays. The
    spline materialises (once, cached) when the user evaluates ``sol`` on
    a concrete solution.
    """
    cache = {}

    def sol(xq):
        if "spline" not in cache:
            from scipy.interpolate import CubicHermiteSpline

            cache["spline"] = CubicHermiteSpline(
                np.asarray(x), np.asarray(y).T, np.asarray(yp).T
            )
        return cache["spline"](xq).T

    return sol
