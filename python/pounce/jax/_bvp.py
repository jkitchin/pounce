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


def _make_newton_vjp(fun, bc, x, n, m, k, uses_p, z0, tol):
    """First-order-differentiable feral-Newton solve (``method="newton"``).

    Forward: damped Newton on ``R(z, theta) = 0`` factoring the ``N x N``
    collocation Jacobian with FERAL's sparse LU (host ``pure_callback``).
    Backward: the implicit-function-theorem VJP. For a cotangent ``v`` on
    ``z*``, ``dL/dtheta = -(dR/dtheta)^T u`` where ``R_z^T u = v`` is solved
    with the *same* sparse LU (``solve_transpose``), and the
    ``(dR/dtheta)^T`` product is taken by JAX VJP over the traceable
    residual at the converged ``z*``. Both directions stay on the ``N``
    system — no ``2N`` saddle — so it is the fast path. First-order only
    (the forward is an opaque callback); use ``method="ipm",
    second_order=True`` for higher-order derivatives.
    """
    import numpy as np
    from ..bvp._jac import CollocationJacobian
    from ..bvp._newton import newton_solve
    from ..bvp._solve import _BvpNlp  # noqa: F401  (kept for parity / reuse)
    from .._pounce import SparseLU

    x_np = np.asarray(x, dtype=np.float64)
    z0_np = np.asarray(z0, dtype=np.float64)

    def residual_jax(z, theta):
        nfun, nbc = _core._make_normalized(fun, bc, theta=theta, uses_p=uses_p)
        Y, pp = _core.unpack_z(z, n, m)
        return _core.collocation_residual(nfun, nbc, x, Y, pp, jnp.concatenate)

    def _np_normalized(theta_np):
        nfun_j, nbc_j = _core._make_normalized(fun, bc, theta=theta_np, uses_p=uses_p)
        nfun = lambda xx, YY, pp: np.asarray(nfun_j(xx, YY, pp), dtype=np.float64)
        nbc = lambda ya, yb, pp: np.asarray(nbc_j(ya, yb, pp), dtype=np.float64)
        return nfun, nbc

    def _host_newton(theta_np):
        nfun, nbc = _np_normalized(theta_np)

        def residual_fn(z):
            return _core.residual_of_z(z, nfun, nbc, x_np, n, m, k, np.concatenate)

        jac = CollocationJacobian(nfun, nbc, x_np, n, m, k)
        z_star, _it, _ok, _rn = newton_solve(
            residual_fn, jac, z0_np, n, m, k, tol=float(tol),
        )
        return np.asarray(z_star, dtype=np.float64)

    def _host_btran(z_star_np, theta_np, v_np):
        nfun, nbc = _np_normalized(theta_np)
        jac = CollocationJacobian(nfun, nbc, x_np, n, m, k)
        rows, cols = jac.structure()
        lu = SparseLU(n * m + k, np.asarray(rows, np.int64), np.asarray(cols, np.int64))
        Y = np.asarray(z_star_np)[: n * m].reshape(n, m)
        pp = np.asarray(z_star_np)[n * m :]
        lu.factor(jac.values(Y, pp))
        # Solve R_z^T u = v.
        return np.asarray(lu.solve_transpose(np.asarray(v_np, np.float64)), np.float64)

    N = n * m + k

    @jax.custom_vjp
    def solve_fn(theta):
        return jax.pure_callback(
            _host_newton, jax.ShapeDtypeStruct((N,), jnp.float64), theta,
            vmap_method="sequential",
        )

    def fwd(theta):
        z_star = jax.pure_callback(
            _host_newton, jax.ShapeDtypeStruct((N,), jnp.float64), theta,
            vmap_method="sequential",
        )
        return z_star, (z_star, theta)

    def bwd(res, v):
        z_star, theta = res
        # jax.jacobian vmaps the VJP over output basis vectors, which vmaps
        # this callback — declare a vmap_method so JAX loops it (each cotangent
        # is a fresh R_z^T solve).
        u = jax.pure_callback(
            _host_btran, jax.ShapeDtypeStruct((N,), jnp.float64),
            z_star, theta, v, vmap_method="sequential",
        )
        # dL/dtheta = -(dR/dtheta)^T u, via VJP of the residual at z*.
        _, vjp_theta = jax.vjp(lambda th: residual_jax(z_star, th), theta)
        (dtheta,) = vjp_theta(-u)
        return (dtheta,)

    solve_fn.defvjp(fwd, bwd)
    return solve_fn


def solve_bvp(
    fun, bc, x, y, p=None, theta=None, *,
    tol=1e-8, options=None, method="newton", second_order=False,
):
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
    second_order : bool
        When ``False`` (default) the solve routes through
        :func:`pounce.jax.solve`'s first-order ``custom_vjp`` — efficient
        (the backward reuses the forward's converged duals, no re-solve),
        but ``jax.grad`` only (``jax.grad(jax.grad(...))`` / ``jax.hessian``
        raise, because that path's forward crosses a ``pure_callback`` with
        no JVP rule). When ``True`` the solve is wrapped in a ``custom_jvp``
        whose tangent rule re-applies the implicit-function theorem to the
        collocation root-find ``dz/dtheta = -(dR/dz)^{-1} (dR/dtheta)`` and
        recomputes the solution through the *same* custom-ruled primitive,
        so JAX recurses for arbitrary differentiation order (``jax.hessian``
        works). The cost is one extra forward solve per differentiation
        level (the rule re-solves to recover ``z*``); the opaque forward is
        only ever evaluated for primal values, never differentiated. Use
        this for second derivatives / Newton-type outer loops; leave it off
        for plain gradient-based training.

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

    if method == "newton":
        if second_order:
            raise ValueError(
                "second_order=True is only supported with method='ipm' "
                "(the Newton path's forward is an opaque callback, so it is "
                "first-order differentiable)."
            )
        solve_root = _make_newton_vjp(fun, bc, x, n, m, k, uses_p, z0, tol)
        z_star = solve_root(theta)
    elif method == "ipm":
        if second_order:
            solve_root = _make_root_solver_jvp(f, g, z0, N, cl, cu, opts)
            z_star = solve_root(theta)
        else:
            z_star = _pounce_solve(
                theta, f=f, g=g, x0=z0, n=N, m=N, cl=cl, cu=cu, options=opts,
            )
    else:
        raise ValueError(f"unknown method {method!r}; use 'newton' or 'ipm'.")

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


def _make_root_solver_jvp(f, g, z0, N, cl, cu, opts):
    """Build a ``custom_jvp`` collocation root-solver, differentiable to
    arbitrary order w.r.t. ``theta``.

    The collocation system is the **square** root-find ``R(z, theta) = 0``
    (``g`` is the residual), so the implicit-function theorem gives the
    tangent ``z_dot = -(dR/dz)^{-1} (dR/dtheta . theta_dot)`` directly — no
    active-set / bound bookkeeping (there are none here). Crucially the
    rule recovers ``z*`` by calling ``solve_root`` again, i.e. through the
    *same* ``custom_jvp`` primitive, so differentiating the rule re-enters
    the rule and JAX composes derivatives to any order. The forward
    ``pure_callback`` (inside :func:`pounce.jax.solve`) is only evaluated
    for primal values; it is never asked for a tangent, which is why this
    sidesteps the "pure callbacks do not support JVP" limitation that
    blocks second-order through the plain :func:`pounce.jax.solve` path.
    """

    @jax.custom_jvp
    def solve_root(theta):
        return _pounce_solve(
            theta, f=f, g=g, x0=z0, n=N, m=N, cl=cl, cu=cu, options=opts,
        )

    @solve_root.defjvp
    def solve_root_jvp(primals, tangents):
        (theta,), (theta_dot,) = primals, tangents
        z = solve_root(theta)
        # R_z (N x N) and the directional derivative R_theta . theta_dot.
        Rz = jax.jacfwd(lambda zz: g(zz, theta))(z)
        _, R_theta_dot = jax.jvp(lambda th: g(z, th), (theta,), (theta_dot,))
        z_dot = -jnp.linalg.solve(Rz, R_theta_dot)
        return z, z_dot

    return solve_root


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
