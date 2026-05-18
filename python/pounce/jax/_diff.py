"""Differentiate through the solver via implicit differentiation.

Setup. For a parametric NLP

    min_x  f(x, p)
    s.t.   g(x, p) = 0
           x_L <= x <= x_U,

the KKT conditions at the optimum ``x*(p)`` are

    ∇_x L(x*, λ*, p) = 0      with  L = f + λᵀ g    (active-set L)
              g(x*, p)  = 0

(plus complementarity on the bound multipliers — we treat the active
set as locally fixed; this is the standard implicit-function
assumption). Differentiating w.r.t. ``p`` and grouping into a 2×2 KKT
block,

    ⎡ H_xx   J_gxᵀ ⎤ ⎡ dx/dp ⎤     ⎡ ∂_p ∇_x L ⎤
    ⎣ J_gx     0   ⎦ ⎣ dλ/dp ⎦ = - ⎣ ∂_p g     ⎦.

For a cotangent ``v`` on ``x*``, the VJP w.r.t. ``p`` is computed by
solving the *transposed* KKT system, which is symmetric here:

    ⎡ H_xx   J_gxᵀ ⎤ ⎡ u_x ⎤   ⎡ v ⎤
    ⎣ J_gx     0   ⎦ ⎣ u_λ ⎦ = ⎣ 0 ⎦,

then    dL/dp = - u_xᵀ (∂_p ∇_x L) - u_λᵀ (∂_p g).

We assemble the dense KKT matrix from the JAX-AD Hessian and Jacobian
and solve it with ``jax.scipy.linalg.solve``. This keeps the
backward fully traced and itself differentiable (so you can take
second derivatives through the solver if you ever need to). For large
sparse problems the right move is to expose a Rust-side
sensitivity-solve via ``pounce-sensitivity``; that lands as a follow-up
once the JAX prototype is exercising the surface.

Bounds. Active variable bounds reduce dx/dp to zero on the active
coordinates. We detect activity from the optimizer's bound multipliers
``info['mult_x_L']`` / ``info['mult_x_U']`` (above
``active_tol``) and project the cotangent / right-hand-side onto the
inactive set before the KKT solve, then scatter back.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ._build import _JaxProblem
from .._pounce import Problem

_ACTIVE_TOL = 1e-6


def _solve_once(
    f: Callable,
    g: Callable | None,
    p: jnp.ndarray,
    x0: jnp.ndarray,
    n: int,
    m: int,
    lb,
    ub,
    cl,
    cu,
    options: dict | None,
) -> tuple[np.ndarray, dict]:
    """Forward solve. ``p`` is closed over by ``f`` / ``g`` via partial."""

    def f_of_x(x):
        return f(x, p)

    if g is not None:
        def g_of_x(x):
            return g(x, p)
    else:
        g_of_x = None

    obj = _JaxProblem(f=f_of_x, g=g_of_x, n=n, m=m)
    problem = Problem(n=n, m=m, problem_obj=obj, lb=lb, ub=ub, cl=cl, cu=cu)
    if options:
        for k, v in options.items():
            problem.add_option(k, v)
    x_np, info = problem.solve(x0=np.asarray(x0))
    return np.asarray(x_np), info


def _make_solve_custom_vjp(
    f: Callable,
    g: Callable | None,
    n: int,
    m: int,
    lb,
    ub,
    cl,
    cu,
    options: dict | None,
):
    @jax.custom_vjp
    def solve_fn(p, x0):
        # Pure-callback to Python. The forward returns only x*; the
        # backward needs (x*, λ*, mult_x_L, mult_x_U) so we re-pack
        # them via the residual.
        x_star, _info = _pure_callback_solve(f, g, p, x0, n, m, lb, ub, cl, cu, options)
        return x_star

    def fwd(p, x0):
        x_star, info = _pure_callback_solve(f, g, p, x0, n, m, lb, ub, cl, cu, options)
        lam = jnp.asarray(info["mult_g"]) if m > 0 else jnp.zeros(0)
        mult_xL = jnp.asarray(info["mult_x_L"])
        mult_xU = jnp.asarray(info["mult_x_U"])
        return x_star, (p, x_star, lam, mult_xL, mult_xU)

    def bwd(residuals, cotangent_x):
        p, x_star, lam, mult_xL, mult_xU = residuals
        v = cotangent_x

        # Detect active variable bounds (|mult| > tol → bound binds → dx/dp = 0).
        active = (mult_xL > _ACTIVE_TOL) | (mult_xU > _ACTIVE_TOL)
        inactive = ~active

        # AD-build the Lagrangian Hessian and Jacobian at (x*, λ*, p).
        def lagrangian(x, p_):
            base = f(x, p_)
            if g is not None and m > 0:
                base = base + jnp.dot(lam, g(x, p_))
            return base

        H = jax.hessian(lagrangian, argnums=0)(x_star, p)
        # ∂_p ∇_x L  — partial Jacobian of grad-L w.r.t. p.
        grad_L_of_p = lambda p_: jax.grad(lagrangian, argnums=0)(x_star, p_)
        dgradL_dp = jax.jacrev(grad_L_of_p)(p)  # shape (n, *p_shape)

        if g is not None and m > 0:
            J = jax.jacrev(g, argnums=0)(x_star, p)
            dg_dp = jax.jacrev(lambda p_: g(x_star, p_))(p)  # (m, *p_shape)
        else:
            J = jnp.zeros((0, n))
            dg_dp = jnp.zeros((0,) + jnp.shape(p))

        # Project to inactive variables.
        idx = jnp.where(inactive, jnp.arange(n), n)  # n sentinel for masked-out
        keep = jnp.nonzero(inactive, size=n, fill_value=-1)[0]
        # We can't dynamically size arrays inside jit, so do a static
        # version: zero out rows/cols belonging to active vars, replace
        # diagonal with 1 so the system stays invertible, and zero the
        # RHS on those rows. This is the standard "augment with
        # identity on the active set" trick.
        active_mat = jnp.diag(active.astype(H.dtype))
        H_eff = jnp.where(
            active[:, None] | active[None, :], 0.0, H
        ) + active_mat
        J_eff = jnp.where(active[None, :], 0.0, J)
        v_eff = jnp.where(active, 0.0, v)

        # Assemble [[H, Jᵀ], [J, 0]] u = [v; 0].
        if m > 0:
            top = jnp.concatenate([H_eff, J_eff.T], axis=1)
            bot = jnp.concatenate([J_eff, jnp.zeros((m, m))], axis=1)
            K = jnp.concatenate([top, bot], axis=0)
            rhs = jnp.concatenate([v_eff, jnp.zeros(m)])
            u = jnp.linalg.solve(K, rhs)
            u_x, u_lam = u[:n], u[n:]
        else:
            u_x = jnp.linalg.solve(H_eff, v_eff)
            u_lam = jnp.zeros(0)

        # Contract with the parameter sensitivities. The minus sign
        # comes from rearranging dKKT/dp = 0 into the form above.
        # u_x has shape (n,); dgradL_dp has shape (n, *p_shape).
        # u_lam has shape (m,); dg_dp has shape (m, *p_shape).
        dL_dp = -jnp.tensordot(u_x, dgradL_dp, axes=1)
        if m > 0:
            dL_dp = dL_dp - jnp.tensordot(u_lam, dg_dp, axes=1)
        # The x0 input has no sensitivity through x* (the solver is
        # deterministic at optimum); return zeros.
        return dL_dp, jnp.zeros_like(idx, dtype=jnp.float64)

    solve_fn.defvjp(fwd, bwd)
    return solve_fn


def _pure_callback_solve(f, g, p, x0, n, m, lb, ub, cl, cu, options):
    """JAX pure_callback wrapper around :func:`_solve_once`.

    Returns ``(x_star, info)`` where ``info`` is a dict of arrays.
    The shapes are static so JAX can trace through cleanly.
    """
    result_shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        {
            "obj_val": jax.ShapeDtypeStruct((), jnp.float64),
            "status": jax.ShapeDtypeStruct((), jnp.int32),
            "iter_count": jax.ShapeDtypeStruct((), jnp.int32),
            "g": jax.ShapeDtypeStruct((m,), jnp.float64),
            "mult_g": jax.ShapeDtypeStruct((m,), jnp.float64),
            "mult_x_L": jax.ShapeDtypeStruct((n,), jnp.float64),
            "mult_x_U": jax.ShapeDtypeStruct((n,), jnp.float64),
        },
    )

    def host_call(p_h, x0_h):
        x_np, info = _solve_once(
            f=f, g=g,
            p=jnp.asarray(p_h),
            x0=jnp.asarray(x0_h),
            n=n, m=m, lb=lb, ub=ub, cl=cl, cu=cu,
            options=options,
        )
        info_out = {
            "obj_val": np.float64(info["obj_val"]),
            "status": np.int32(info["status"]),
            "iter_count": np.int32(info["iter_count"]),
            "g": np.asarray(info["g"], dtype=np.float64),
            "mult_g": np.asarray(info["mult_g"], dtype=np.float64),
            "mult_x_L": np.asarray(info["mult_x_L"], dtype=np.float64),
            "mult_x_U": np.asarray(info["mult_x_U"], dtype=np.float64),
        }
        return np.asarray(x_np, dtype=np.float64), info_out

    return jax.pure_callback(host_call, result_shapes, p, x0)


def solve(
    p,
    *,
    f: Callable,
    g: Callable | None = None,
    x0,
    n: int,
    m: int = 0,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    options: dict | None = None,
):
    """Parametric solve. ``x* = solve(p, f=..., g=..., x0=..., ...)``.

    Differentiable w.r.t. ``p`` via the implicit-function rule on the
    KKT system at ``x*(p)``. Not differentiable w.r.t. ``x0``.

    ``f`` and ``g`` must take ``(x, p)`` and be JAX-traceable.
    """
    fn = _make_solve_custom_vjp(f, g, n, m, lb, ub, cl, cu, options)
    return fn(p, x0)


def vmap_solve(
    p_batch,
    *,
    f: Callable,
    g: Callable | None = None,
    x0,
    n: int,
    m: int = 0,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    options: dict | None = None,
):
    """Batched solve over the leading axis of ``p_batch``.

    The pounce solver is single-threaded and stateful, so a literal
    ``jax.vmap`` of :func:`solve` would unsafely lift the pure_callback.
    This helper instead loops in Python (or, when JAX provides a
    sequential map primitive, dispatches to that), preserving
    differentiability via :func:`solve`'s ``custom_vjp``.
    """
    p_batch = jnp.asarray(p_batch)
    batch = p_batch.shape[0]

    def one(p_i):
        return solve(
            p_i, f=f, g=g, x0=x0, n=n, m=m,
            lb=lb, ub=ub, cl=cl, cu=cu, options=options,
        )

    # ``jax.lax.map`` runs sequentially under the hood (one element at
    # a time), which is exactly what we want for an impure callback.
    return jax.lax.map(one, p_batch)
