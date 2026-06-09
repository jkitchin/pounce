"""Differentiate through the solver via implicit differentiation
(PyTorch frontend, pounce#109).

PyTorch mirror of :mod:`pounce.jax._diff`. Because PyTorch is eager, the
adapter is *smaller* than the JAX one: there is no ``pure_callback`` /
``ShapeDtypeStruct`` machinery — the forward simply calls
``problem.solve(...)`` directly inside ``torch.autograd.Function.forward``
and the backward runs the KKT implicit-function solve in-framework with
``torch.linalg.solve``.

Setup. For a parametric NLP

    min_x  f(x, p)
    s.t.   g(x, p) = 0
           x_L <= x <= x_U,

the implicit-function theorem on the KKT system at ``x*(p)`` gives, for a
cotangent ``v`` on ``x*``,

    ⎡ H_xx   J_gxᵀ ⎤ ⎡ u_x ⎤   ⎡ v ⎤
    ⎣ J_gx     0   ⎦ ⎣ u_λ ⎦ = ⎣ 0 ⎦,
    then    dL/dp = - u_xᵀ (∂_p ∇_x L) - u_λᵀ (∂_p g).

Active variable bounds reduce ``dx/dp`` to zero on the active
coordinates: we detect activity from the optimizer's bound multipliers
``mult_x_L`` / ``mult_x_U`` and identity-augment the KKT block on the
active set. Slack inequality rows (``cl[i] < cu[i]`` and ``|λ_i|`` below
``active_tol``) are dropped from the block via the same trick — including
a slack row as if it were ``g_i(x) = 0`` over-constrains ``dx*/dp`` and
silently returns the wrong gradient (pounce#73). This is the same
active-set logic as the JAX path, ported line-for-line.

dtype. Inputs must be float64 (the Newton + KKT solves stall in float32);
:func:`solve` validates this and raises a clear error rather than
silently returning a meaningless gradient.

For large sparse problems the right move is the held-factor back-solve
exposed by :class:`pounce.torch.TorchProblem` (factor reuse via the
Rust-side ``Solver.kkt_solve``); this module assembles a dense KKT and
``torch.linalg.solve``s it, matching the top-level JAX ``solve``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import torch
from torch.func import grad, hessian, jacrev

from ._build import _TorchProblem, _DT
from .._pounce import Problem

_ACTIVE_TOL = 1e-6


def _require_f64(name, t):
    if isinstance(t, torch.Tensor) and t.dtype != _DT:
        raise TypeError(
            f"pounce.torch: {name} must be float64 (got {t.dtype}); the "
            "Newton and KKT solves need double precision. Cast with "
            f"`{name}.double()`."
        )


def _solve_once(
    f: Callable,
    g: Callable | None,
    p: torch.Tensor,
    x0: torch.Tensor,
    n: int,
    m: int,
    lb,
    ub,
    cl,
    cu,
    options: dict | None,
) -> tuple[np.ndarray, dict]:
    """Forward solve. ``p`` is closed over by ``f`` / ``g``."""

    def f_of_x(x):
        return f(x, p)

    if g is not None:
        def g_of_x(x):
            return g(x, p)
    else:
        g_of_x = None

    obj = _TorchProblem(f=f_of_x, g=g_of_x, n=n, m=m)
    problem = Problem(n=n, m=m, problem_obj=obj, lb=lb, ub=ub, cl=cl, cu=cu)
    if options:
        for k, v in options.items():
            problem.add_option(k, v)
    x_np, info = problem.solve(x0=np.asarray(x0.detach().cpu(), dtype=np.float64))
    return np.asarray(x_np, dtype=np.float64), info


def _kkt_backward(
    f: Callable,
    g: Callable | None,
    n: int,
    m: int,
    cl,
    cu,
    p: torch.Tensor,
    x_star: torch.Tensor,
    lam: torch.Tensor,
    mult_xL: torch.Tensor,
    mult_xU: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Implicit-function-theorem VJP at a single ``(p, x*, λ*)``.

    Mirror of :func:`pounce.jax._diff` / ``_bwd_single_kkt`` — one shared
    source of truth for the active-set handling (pounce#73 fix), in the
    ``torch`` array namespace. Returns ``dL/dp`` with the shape of ``p``.
    """
    active = (mult_xL > _ACTIVE_TOL) | (mult_xU > _ACTIVE_TOL)

    def lagrangian(x, p_):
        base = f(x, p_)
        if g is not None and m > 0:
            base = base + torch.dot(lam, g(x, p_))
        return base

    H = hessian(lagrangian, argnums=0)(x_star, p)
    grad_L_of_p = lambda p_: grad(lagrangian, argnums=0)(x_star, p_)  # noqa: E731
    dgradL_dp = jacrev(grad_L_of_p)(p)  # shape (n, *p_shape)

    if g is not None and m > 0:
        J = jacrev(g, argnums=0)(x_star, p)
        dg_dp = jacrev(lambda p_: g(x_star, p_))(p)  # (m, *p_shape)
        cl_arr = torch.as_tensor(np.asarray(cl, dtype=np.float64), dtype=H.dtype)
        cu_arr = torch.as_tensor(np.asarray(cu, dtype=np.float64), dtype=H.dtype)
        is_equality = cl_arr == cu_arr
        cons_active = is_equality | (torch.abs(lam) > _ACTIVE_TOL)
        cons_inactive = ~cons_active
    else:
        J = torch.zeros((0, n), dtype=H.dtype)
        dg_dp = torch.zeros((0,) + tuple(p.shape), dtype=H.dtype)
        cons_inactive = torch.zeros((0,), dtype=torch.bool)

    # Identity-augment on the active set: zero rows/cols of active vars,
    # put 1 on their diagonal so the system stays invertible, zero the
    # RHS there. Slack inequality rows drop out via diag(cons_inactive).
    active_mat = torch.diag(active.to(H.dtype))
    H_eff = torch.where(
        active[:, None] | active[None, :],
        torch.zeros((), dtype=H.dtype), H,
    ) + active_mat
    J_eff = torch.where(
        cons_inactive[:, None] | active[None, :],
        torch.zeros((), dtype=H.dtype), J,
    )
    v_eff = torch.where(active, torch.zeros((), dtype=H.dtype), v)

    if m > 0:
        cons_inactive_diag = torch.diag(cons_inactive.to(H.dtype))
        top = torch.cat([H_eff, J_eff.T], dim=1)
        bot = torch.cat([J_eff, cons_inactive_diag], dim=1)
        K = torch.cat([top, bot], dim=0)
        rhs = torch.cat([v_eff, torch.zeros(m, dtype=H.dtype)])
        u = torch.linalg.solve(K, rhs)
        u_x, u_lam = u[:n], u[n:]
    else:
        u_x = torch.linalg.solve(H_eff, v_eff)
        u_lam = torch.zeros(0, dtype=H.dtype)

    dL_dp = -torch.tensordot(u_x, dgradL_dp, dims=1)
    if m > 0:
        dL_dp = dL_dp - torch.tensordot(u_lam, dg_dp, dims=1)
    return dL_dp


def _make_solve_fn(f, g, n, m, lb, ub, cl, cu, options):
    """Build a :class:`torch.autograd.Function` closing over the static
    problem definition; only the tensors ``(p, x0)`` are differentiable
    inputs (``x0`` carries no gradient, like the JAX path)."""

    class _SolveFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, p, x0):
            x_np, info = _solve_once(f, g, p, x0, n, m, lb, ub, cl, cu, options)
            x_star = torch.as_tensor(x_np, dtype=_DT)
            lam = (
                torch.as_tensor(np.asarray(info["mult_g"]), dtype=_DT)
                if m > 0 else torch.zeros(0, dtype=_DT)
            )
            mult_xL = torch.as_tensor(np.asarray(info["mult_x_L"]), dtype=_DT)
            mult_xU = torch.as_tensor(np.asarray(info["mult_x_U"]), dtype=_DT)
            # Save the live ``p`` (not detached) so the in-framework
            # backward stays a function of it.
            ctx.save_for_backward(p, x_star, lam, mult_xL, mult_xU)
            return x_star

        @staticmethod
        def backward(ctx, v):
            p, x_star, lam, mult_xL, mult_xU = ctx.saved_tensors
            dL_dp = _kkt_backward(
                f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v,
            )
            # x0 input has no sensitivity through x* at the optimum.
            return dL_dp, None

    return _SolveFn


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

    Differentiable w.r.t. ``p`` via the implicit-function rule on the KKT
    system at ``x*(p)``. Not differentiable w.r.t. ``x0``.

    ``f`` and ``g`` must take ``(x, p)`` and be ``torch.func``-traceable.
    ``p`` (and ``x0``) must be float64.
    """
    p = torch.as_tensor(p, dtype=_DT) if not isinstance(p, torch.Tensor) else p
    _require_f64("p", p)
    x0 = torch.as_tensor(x0, dtype=_DT) if not isinstance(x0, torch.Tensor) else x0
    fn = _make_solve_fn(f, g, n, m, lb, ub, cl, cu, options)
    return fn.apply(p, x0)


# ----- warm-started solve (dual + barrier-μ threading, pounce#86) -----


def _solve_once_warm(
    f, g, p, x0, n, m, lb, ub, cl, cu, options,
    lam_warm, zL_warm, zU_warm, mu_warm,
):
    """Forward solve with user-supplied dual (and optional μ) warm-start."""

    def f_of_x(x):
        return f(x, p)

    if g is not None:
        def g_of_x(x):
            return g(x, p)
    else:
        g_of_x = None

    obj = _TorchProblem(f=f_of_x, g=g_of_x, n=n, m=m)
    problem = Problem(n=n, m=m, problem_obj=obj, lb=lb, ub=ub, cl=cl, cu=cu)
    merged = dict(options or {})
    merged.setdefault("warm_start_init_point", "yes")
    if np.isfinite(mu_warm):
        # Seed the barrier from the previous solve's converged μ so the
        # corrector resumes near the central path (pounce#86).
        merged.setdefault("mu_init", float(mu_warm))
        merged.setdefault("warm_start_target_mu", float(mu_warm))
    for k, v in merged.items():
        problem.add_option(k, v)
    x_np, info = problem.solve(
        x0=np.asarray(x0.detach().cpu(), dtype=np.float64),
        lagrange=np.asarray(lam_warm, dtype=np.float64),
        zl=np.asarray(zL_warm, dtype=np.float64),
        zu=np.asarray(zU_warm, dtype=np.float64),
    )
    return (
        np.asarray(x_np, dtype=np.float64),
        np.asarray(info["mult_g"], dtype=np.float64),
        np.asarray(info["mult_x_L"], dtype=np.float64),
        np.asarray(info["mult_x_U"], dtype=np.float64),
        float(info["mu"]),
        info,
    )


def _make_solve_with_warm_fn(f, g, n, m, lb, ub, cl, cu, options):
    class _WarmSolveFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, p, x0, lam_warm, zL_warm, zU_warm, mu_warm):
            x_np, lam_out, zL_out, zU_out, mu_out, _info = _solve_once_warm(
                f, g, p, x0, n, m, lb, ub, cl, cu, options,
                np.asarray(lam_warm.detach().cpu()),
                np.asarray(zL_warm.detach().cpu()),
                np.asarray(zU_warm.detach().cpu()),
                float(mu_warm.detach().cpu()),
            )
            x_star = torch.as_tensor(x_np, dtype=_DT)
            lam_t = torch.as_tensor(lam_out, dtype=_DT)
            zL_t = torch.as_tensor(zL_out, dtype=_DT)
            zU_t = torch.as_tensor(zU_out, dtype=_DT)
            mu_t = torch.as_tensor(mu_out, dtype=_DT)
            ctx.save_for_backward(p, x_star, lam_t, zL_t, zU_t)
            # Warm/dual/μ outputs are not differentiable — mark so.
            ctx.mark_non_differentiable(lam_t, zL_t, zU_t, mu_t)
            return x_star, lam_t, zL_t, zU_t, mu_t

        @staticmethod
        def backward(ctx, v, *_drop):
            # Only the x* cotangent carries gradient w.r.t. p; cotangents
            # on (lam, zL, zU, mu) are dropped — same convention the JAX
            # warm path uses (they're consequences of the active set and
            # the barrier homotopy, not inputs to dx*/dp).
            p, x_star, lam, mult_xL, mult_xU = ctx.saved_tensors
            dL_dp = _kkt_backward(
                f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v,
            )
            return dL_dp, None, None, None, None, None

    return _WarmSolveFn


def solve_with_warm(
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
    warm_start: tuple | None = None,
):
    """Parametric solve that consumes and returns dual warm-state.

    Mirror of :func:`pounce.jax.solve_with_warm`:

    * ``warm_start=(lam, zL, zU)`` seeds the solver's dual variables via
      ``warm_start_init_point=yes``. ``None`` starts from zeros.
    * ``warm_start=(lam, zL, zU, mu)`` additionally seeds the barrier μ
      (pounce#86); pass ``mu=None`` inside the 4-tuple to skip the μ seed
      but still receive the converged μ on output.
    * Returns ``(x*, (lam_out, zL_out, zU_out))`` for a 3-tuple / ``None``
      warm-start, or ``(x*, (lam_out, zL_out, zU_out, mu_out))`` for a
      4-tuple — the returned arity matches the input.

    Differentiable w.r.t. ``p`` only; cotangents on the warm duals / μ
    are dropped (zero), matching how :func:`solve` handles ``x0``.
    """
    p = torch.as_tensor(p, dtype=_DT) if not isinstance(p, torch.Tensor) else p
    _require_f64("p", p)
    x0 = torch.as_tensor(x0, dtype=_DT) if not isinstance(x0, torch.Tensor) else x0

    want_mu = warm_start is not None and len(warm_start) == 4
    if warm_start is None:
        lam_warm = torch.zeros(m, dtype=_DT)
        zL_warm = torch.zeros(n, dtype=_DT)
        zU_warm = torch.zeros(n, dtype=_DT)
        mu_warm = torch.as_tensor(float("nan"), dtype=_DT)
    else:
        if want_mu:
            lam_warm, zL_warm, zU_warm, mu_seed = warm_start
        else:
            lam_warm, zL_warm, zU_warm = warm_start
            mu_seed = None
        lam_warm = torch.as_tensor(lam_warm, dtype=_DT)
        zL_warm = torch.as_tensor(zL_warm, dtype=_DT)
        zU_warm = torch.as_tensor(zU_warm, dtype=_DT)
        mu_warm = torch.as_tensor(
            float("nan") if mu_seed is None else float(mu_seed), dtype=_DT,
        )

    fn = _make_solve_with_warm_fn(f, g, n, m, lb, ub, cl, cu, options)
    x_star, lam_out, zL_out, zU_out, mu_out = fn.apply(
        p, x0, lam_warm, zL_warm, zU_warm, mu_warm,
    )
    if want_mu:
        return x_star, (lam_out, zL_out, zU_out, mu_out)
    return x_star, (lam_out, zL_out, zU_out)


# ----- batched solves -----


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
    """Sequential batched solve over the leading axis of ``p_batch``.

    The pounce solver is single-threaded and stateful, so this loops in
    Python over :func:`solve` (which keeps each element differentiable via
    its own ``autograd.Function``) and stacks the results. ``x0`` may be a
    single ``(n,)`` vector (broadcast) or a ``(B, n)`` batch.
    """
    p_batch = torch.as_tensor(p_batch, dtype=_DT)
    _require_f64("p_batch", p_batch)
    B = p_batch.shape[0]
    x0_arr = torch.as_tensor(x0, dtype=_DT)
    if x0_arr.ndim == 1:
        x0_arr = x0_arr.expand(B, n)
    outs = [
        solve(
            p_batch[i], f=f, g=g, x0=x0_arr[i], n=n, m=m,
            lb=lb, ub=ub, cl=cl, cu=cu, options=options,
        )
        for i in range(B)
    ]
    return torch.stack(outs, dim=0)


def _solve_batch_threadpool(
    f, g, p_batch_np, x0_np, n, m, lb, ub, cl, cu, options, workers,
):
    """Dispatch ``B`` independent solves across a ``ThreadPoolExecutor``.

    Each worker builds its own ``Problem`` (no shared state) and runs
    ``Problem.solve``. Genuine parallelism is unlocked by the
    ``py.allow_threads`` block around the IPM iteration in ``pounce-py``
    — the GIL is released across the iteration so threads run
    concurrently on the Rust side. ``torch.func`` callbacks for ``f`` /
    ``g`` reacquire the GIL the usual way; that's serialized but small
    relative to the linear algebra.
    """
    B = p_batch_np.shape[0]
    n_workers = workers or min(B, 8)
    x_out = np.empty((B, n), dtype=np.float64)
    lam_out = np.empty((B, m), dtype=np.float64)
    zL_out = np.empty((B, n), dtype=np.float64)
    zU_out = np.empty((B, n), dtype=np.float64)

    def one(i):
        p_i = torch.as_tensor(p_batch_np[i], dtype=_DT)
        x0_i = torch.as_tensor(
            x0_np[i] if x0_np.ndim == 2 else x0_np, dtype=_DT,
        )
        x_np, info = _solve_once(
            f, g, p_i, x0_i, n, m, lb, ub, cl, cu, options,
        )
        x_out[i] = x_np
        lam_out[i] = (
            np.asarray(info["mult_g"], dtype=np.float64) if m > 0
            else np.zeros(0)
        )
        zL_out[i] = np.asarray(info["mult_x_L"], dtype=np.float64)
        zU_out[i] = np.asarray(info["mult_x_U"], dtype=np.float64)

    if n_workers <= 1 or B <= 1:
        for i in range(B):
            one(i)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(one, range(B)))
    return x_out, lam_out, zL_out, zU_out


def _make_vmap_solve_parallel_fn(f, g, n, m, lb, ub, cl, cu, options, workers):
    class _ParallelSolveFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, p_batch, x0_batch):
            p_np = np.asarray(p_batch.detach().cpu(), dtype=np.float64)
            x0_np = np.asarray(x0_batch.detach().cpu(), dtype=np.float64)
            x_out, lam_out, zL_out, zU_out = _solve_batch_threadpool(
                f, g, p_np, x0_np, n, m, lb, ub, cl, cu, options, workers,
            )
            x_star = torch.as_tensor(x_out, dtype=_DT)
            lam = torch.as_tensor(lam_out, dtype=_DT)
            mxL = torch.as_tensor(zL_out, dtype=_DT)
            mxU = torch.as_tensor(zU_out, dtype=_DT)
            ctx.save_for_backward(p_batch, x_star, lam, mxL, mxU)
            return x_star

        @staticmethod
        def backward(ctx, v_batch):
            p_batch, x_star, lam, mxL, mxU = ctx.saved_tensors
            B = p_batch.shape[0]
            grads = [
                _kkt_backward(
                    f, g, n, m, cl, cu,
                    p_batch[i], x_star[i],
                    lam[i] if m > 0 else lam,
                    mxL[i], mxU[i], v_batch[i],
                )
                for i in range(B)
            ]
            return torch.stack(grads, dim=0), None

    return _ParallelSolveFn


def vmap_solve_parallel(
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
    workers: int | None = None,
):
    """Parallel batched solve. Drop-in for :func:`vmap_solve`.

    Each of the ``B`` elements of ``p_batch`` is dispatched to a worker in
    a ``ThreadPoolExecutor`` of size ``workers`` (default ``min(B, 8)``).
    Each worker owns an independent ``Problem`` so there's no shared
    state, and the ``py.allow_threads`` GIL release around the IPM
    iteration lets threads run concurrently on the Rust side.

    Differentiable w.r.t. ``p_batch`` via per-element implicit function
    theorem. ``x0`` may be a single ``(n,)`` vector (broadcast) or a
    ``(B, n)`` batch.
    """
    p_batch = torch.as_tensor(p_batch, dtype=_DT)
    _require_f64("p_batch", p_batch)
    B = p_batch.shape[0]
    x0_arr = torch.as_tensor(x0, dtype=_DT)
    if x0_arr.ndim == 1:
        x0_arr = x0_arr.expand(B, n).contiguous()
    fn = _make_vmap_solve_parallel_fn(f, g, n, m, lb, ub, cl, cu, options, workers)
    return fn.apply(p_batch, x0_arr)
