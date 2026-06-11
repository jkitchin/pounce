"""Differentiable convex-QP layer (OptNet-style implicit differentiation).

Solves, and differentiates through, the convex QP

.. code-block:: text

    minimize    ½ xᵀP x + cᵀx
    subject to  G x ≤ h
                A x = b

The forward solve calls the ``pounce-convex`` interior-point solver
through a host callback. The backward pass uses the implicit-function
theorem on the KKT system at the optimum (Amos & Kolter, *OptNet*, 2017):
the same KKT matrix that defines the solution also yields its
sensitivities, so a single linear solve gives the cotangents.

Differentiable parameters. Gradients are provided w.r.t. **all** the
parameters that enter the QP linearly through the optimum:

* the linear / right-hand-side vectors ``c``, ``b``, ``h``; and
* the matrices ``P``, ``G``, ``A`` (full OptNet matrix derivatives).

``P`` is differentiated as a **symmetric** matrix — the solver reads its
lower triangle and treats it as symmetric, so ``∇P`` is the symmetrized
gradient ``½(d_x xᵀ + x d_xᵀ)``; perturb ``P`` symmetrically when checking
it against finite differences.

Bounds ``lb ≤ x ≤ ub`` are supported in the *forward* solve by folding
them into ``G``/``h`` before differentiation, so the IFT sees a single
inequality block. The folded bound rows are constants, so they carry no
gradient back to ``lb``/``ub`` (differentiate bound *levels* by passing
them through ``G``/``h`` explicitly instead).

Batching. :func:`solve_qp` is usable under ``jax.vmap`` (each instance is
an independent, sequential host solve). For a *parallel* batch over many
instances that share matrix structure, use :func:`solve_qp_batch`, which
routes the forward solves to the rayon-parallel ``solve_qp_batch`` binding
and differentiates each instance independently.

Warm starting. Pass ``warm_start=`` a previous primal ``x`` to seed the
interior-point iteration on a nearby problem. The core applies a
Mehrotra-style recentering (it keeps the warm primal but pushes the
slacks/multipliers back into the interior with a scale-aware floor, since
a converged point lies on the complementarity boundary — the worst IPM
restart). The warm start is **not** differentiated and never changes the
solution or its gradients; it only reduces the iteration count. For
repeated solves on a *fixed structure*, the host API
:class:`pounce.qp.QpFactorization` additionally reuses the symbolic
factorization (AMD analysis / KKT pattern).
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag

from .. import _pounce

__all__ = ["solve_qp", "solve_qp_batch", "solve_socp", "QpLayer"]

# Active-set tolerance for the backward pass: an inequality counts as
# active when its multiplier is above this (complementarity slackness).
from .._ad_common import ACTIVE_TOL as _ACTIVE_TOL  # single source of truth (DiffHandoff contract)


def _expand_bounds(G, h, lb, ub, n):
    """Fold finite variable bounds into G/h as extra rows.

    Returns ``(G_full, h_full)`` as dense jnp arrays. ``x_i ≤ ub_i`` and
    ``−x_i ≤ −lb_i``."""
    rows = []
    rhs = []
    if G is not None and G.shape[0] > 0:
        rows.append(G)
        rhs.append(h)
    if ub is not None:
        for i in range(n):
            if np.isfinite(float(ub[i])):
                e = jnp.zeros(n).at[i].set(1.0)
                rows.append(e[None, :])
                rhs.append(jnp.asarray(ub[i]).reshape(1))
    if lb is not None:
        for i in range(n):
            if np.isfinite(float(lb[i])):
                e = jnp.zeros(n).at[i].set(-1.0)
                rows.append(e[None, :])
                rhs.append((-jnp.asarray(lb[i])).reshape(1))
    if not rows:
        return jnp.zeros((0, n)), jnp.zeros((0,))
    return jnp.concatenate(rows, axis=0), jnp.concatenate(rhs, axis=0)


def _to_coo_lower(M):
    """COO ``(rows, cols, vals)`` of the lower triangle of dense ``M``."""
    r, cc = np.nonzero(M)
    keep = r >= cc
    return r[keep].tolist(), cc[keep].tolist(), M[r[keep], cc[keep]].tolist()


def _to_coo(M):
    """COO ``(rows, cols, vals)`` of dense ``M``."""
    r, cc = np.nonzero(M)
    return r.tolist(), cc.tolist(), M[r, cc].tolist()


def _build_problem(P, c, G, h, A, b):
    """Assemble a ``_pounce.QpProblem`` from dense numpy arrays."""
    n = c.shape[0]
    pr, pc, pv = _to_coo_lower(np.asarray(P))
    gr, gc, gv = _to_coo(np.asarray(G))
    ar, ac, av = _to_coo(np.asarray(A))
    return _pounce.QpProblem(
        n=n,
        c=np.asarray(c).tolist(),
        p_rows=pr,
        p_cols=pc,
        p_vals=pv,
        a_rows=ar,
        a_cols=ac,
        a_vals=av,
        b=np.asarray(b).tolist(),
        g_rows=gr,
        g_cols=gc,
        g_vals=gv,
        h=np.asarray(h).tolist(),
    )


_SUCCESS_STATUS = "optimal"


def _check_status(status, where):
    """Raise unless the convex solver reached an optimal solution.

    The differentiable layer reads the primal/dual iterate and solves a
    KKT system for the gradient. If the forward solve did not converge
    (``primal_infeasible`` / ``dual_infeasible`` / ``iteration_limit`` /
    ``numerical_failure``), that iterate is not a KKT point and the
    implicit-function gradient is meaningless — so fail loudly rather than
    return silent NaNs/garbage into a downstream optimizer. Use the host
    ``pounce.qp`` API (which surfaces ``QpResult.status``) to inspect the
    failure."""
    if status != _SUCCESS_STATUS:
        raise RuntimeError(
            f"{where}: convex solver returned status {status!r}, not "
            f"{_SUCCESS_STATUS!r}; the differentiable layer cannot produce a "
            f"meaningful gradient for a non-optimal solve."
        )


def _split_duals(d, m_g, m_a):
    """Extract (lam, nu) from a solver result dict, padding empty blocks."""
    lam = (
        np.asarray(d["z"], dtype=np.float64)
        if m_g
        else np.zeros((0,), dtype=np.float64)
    )
    nu = (
        np.asarray(d["y"], dtype=np.float64)
        if m_a
        else np.zeros((0,), dtype=np.float64)
    )
    return lam, nu


def _forward_solve(P, c, G, h, A, b, tol, max_iter, warm_x=None):
    """Host-side forward solve via pounce-convex. Returns (x, lam, nu).

    ``lam`` are the inequality (``G``) multipliers, ``nu`` the equality
    (``A``) multipliers. ``warm_x`` (if its length is ``n``) seeds the
    iteration with that primal; it only affects the iteration count."""
    m_g = G.shape[0]
    m_a = A.shape[0]
    prob = _build_problem(P, c, G, h, A, b)
    warm = None
    if warm_x is not None and np.asarray(warm_x).size == c.shape[0]:
        warm = {"x": np.asarray(warm_x, dtype=np.float64).tolist()}
    d = _pounce.solve_qp(prob, tol=tol, max_iter=max_iter, warm_start=warm)
    _check_status(d["status"], "QpLayer forward solve")
    x = np.asarray(d["x"], dtype=np.float64)
    lam, nu = _split_duals(d, m_g, m_a)
    return x, lam, nu


def _forward_solve_batch(P, cs, G, hs, A, bs, tol, max_iter, warm_xs=None):
    """Parallel host-side batch solve. Shared ``P``/``G``/``A``; per-row
    ``cs``/``hs``/``bs``. Returns stacked (xs, lams, nus). ``warm_xs`` (if
    shaped ``(B, n)``) seeds each instance's primal."""
    m_g = G.shape[0]
    m_a = A.shape[0]
    b_sz = cs.shape[0]
    n = cs.shape[1]
    probs = [_build_problem(P, cs[i], G, hs[i], A, bs[i]) for i in range(b_sz)]
    warms = None
    if warm_xs is not None and np.asarray(warm_xs).shape == (b_sz, n):
        wx = np.asarray(warm_xs, dtype=np.float64)
        warms = [{"x": wx[i].tolist()} for i in range(b_sz)]
    dicts = _pounce.solve_qp_batch(probs, tol=tol, max_iter=max_iter, warm_starts=warms)
    for i, d in enumerate(dicts):
        _check_status(d["status"], f"QpLayer batch forward solve (row {i})")
    xs = np.stack([np.asarray(d["x"], dtype=np.float64) for d in dicts])
    if m_g:
        lams = np.stack([np.asarray(d["z"], dtype=np.float64) for d in dicts])
    else:
        lams = np.zeros((b_sz, 0), dtype=np.float64)
    if m_a:
        nus = np.stack([np.asarray(d["y"], dtype=np.float64) for d in dicts])
    else:
        nus = np.zeros((b_sz, 0), dtype=np.float64)
    return xs, lams, nus


def _kkt_backward(P, G, A, h, x, lam, nu, gx):
    """One OptNet implicit-diff backward (Amos & Kolter 2017, §3).

    At the optimum ``(x, λ, ν)`` of ``min ½xᵀPx+cᵀx s.t. Gx≤h, Ax=b`` the
    KKT differential system is

    .. code-block:: text

        [ P        Gᵀ        Aᵀ ] [d_x]     [ g_x ]
        [ D(λ)G    D(Gx−h)   0  ] [d_λ] = − [  0  ]
        [ A        0         0  ] [d_ν]     [  0  ]

    with ``D(·) = diag(·)``. Solving for ``(d_x, d_λ, d_ν)``, the loss
    gradients are

    .. code-block:: text

        ∇_c = d_x          ∇_P = ½(d_x xᵀ + x d_xᵀ)
        ∇_b = −d_ν         ∇_A = d_ν xᵀ + ν d_xᵀ
        ∇_h = −d_λ         ∇_G = d_λ xᵀ + λ d_xᵀ

    (The matrix forms follow from the standard OptNet result; in this
    scaling ``d_λ`` already absorbs ``D(λ)``, so e.g. ``∇_h = −d_λ`` rather
    than ``−D(λ)d_λ``. All six are checked against finite differences.)
    """
    n = x.shape[0]
    m_g = G.shape[0]
    m_a = A.shape[0]

    slack = G @ x - h  # ≤ 0 at feasibility; 0 on active rows
    dlam_scale = jnp.diag(lam)
    zero_ga = jnp.zeros((m_g, m_a))
    zero_ag = jnp.zeros((m_a, m_g))
    zero_aa = jnp.zeros((m_a, m_a))

    top = jnp.concatenate([P, G.T, A.T], axis=1)
    mid = jnp.concatenate([dlam_scale @ G, jnp.diag(slack), zero_ga], axis=1)
    bot = jnp.concatenate([A, zero_ag, zero_aa], axis=1)
    kkt = jnp.concatenate([top, mid, bot], axis=0)

    rhs = -jnp.concatenate([gx, jnp.zeros(m_g), jnp.zeros(m_a)])
    d = jnp.linalg.solve(kkt, rhs)
    d_x = d[:n]
    d_lam = d[n : n + m_g]
    d_nu = d[n + m_g :]

    grad_c = d_x
    grad_h = -d_lam
    grad_b = -d_nu
    # Matrix gradients (full OptNet). ∇_P symmetrized (P is symmetric).
    grad_P = 0.5 * (jnp.outer(d_x, x) + jnp.outer(x, d_x))
    grad_G = jnp.outer(d_lam, x) + jnp.outer(lam, d_x)
    grad_A = jnp.outer(d_nu, x) + jnp.outer(nu, d_x)
    return grad_P, grad_c, grad_G, grad_h, grad_A, grad_b


def _make_qp_vjp(n, m_g, m_a, tol, max_iter):
    # `warm_x` is a primal input so it threads cleanly through jit/grad,
    # but it never affects the solution (only the iteration count), so its
    # cotangent is zero.
    @jax.custom_vjp
    def qp(P, c, G, h, A, b, warm_x):
        x, _, _ = _pure_forward(P, c, G, h, A, b, warm_x, n, m_g, m_a, tol, max_iter)
        return x

    def fwd(P, c, G, h, A, b, warm_x):
        x, lam, nu = _pure_forward(
            P, c, G, h, A, b, warm_x, n, m_g, m_a, tol, max_iter
        )
        return x, (P, G, A, h, x, lam, nu, warm_x)

    def bwd(res, gx):
        P, G, A, h, x, lam, nu, warm_x = res
        gP, gc, gG, gh, gA, gb = _kkt_backward(P, G, A, h, x, lam, nu, gx)
        return (gP, gc, gG, gh, gA, gb, jnp.zeros_like(warm_x))

    qp.defvjp(fwd, bwd)
    return qp


def _make_qp_batch_vjp(n, m_g, m_a, tol, max_iter):
    """custom_vjp for a parallel batch. Differentiable args are the shared
    ``P``/``G``/``A`` and the per-row ``cs``/``hs``/``bs`` (all leading
    axis ``B``). Matrix gradients sum over the batch; RHS gradients stay
    per-row."""

    @jax.custom_vjp
    def qp(P, cs, G, hs, A, bs, warm_xs):
        xs, _, _ = _pure_forward_batch(
            P, cs, G, hs, A, bs, warm_xs, n, m_g, m_a, tol, max_iter
        )
        return xs

    def fwd(P, cs, G, hs, A, bs, warm_xs):
        xs, lams, nus = _pure_forward_batch(
            P, cs, G, hs, A, bs, warm_xs, n, m_g, m_a, tol, max_iter
        )
        return xs, (P, G, A, hs, xs, lams, nus, warm_xs)

    def bwd(res, gxs):
        P, G, A, hs, xs, lams, nus, warm_xs = res
        per = jax.vmap(
            lambda h, x, lam, nu, gx: _kkt_backward(P, G, A, h, x, lam, nu, gx)
        )(hs, xs, lams, nus, gxs)
        gP, gc, gG, gh, gA, gb = per
        # Shared matrices: sum cotangents over the batch axis. Warm start is
        # not differentiated (start-independent solution).
        return (
            jnp.sum(gP, axis=0),
            gc,
            jnp.sum(gG, axis=0),
            gh,
            jnp.sum(gA, axis=0),
            gb,
            jnp.zeros_like(warm_xs),
        )

    qp.defvjp(fwd, bwd)
    return qp


def _pure_forward(P, c, G, h, A, b, warm_x, n, m_g, m_a, tol, max_iter):
    """custom_vjp-friendly forward via pure_callback. Returns (x, lam, nu).

    ``warm_x`` is an extra (non-differentiated) operand carrying an optional
    warm-start primal; an empty array means cold start."""
    shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((m_g,), jnp.float64),
        jax.ShapeDtypeStruct((m_a,), jnp.float64),
    )

    def host(P_h, c_h, G_h, h_h, A_h, b_h, w_h):
        return _forward_solve(
            np.asarray(P_h),
            np.asarray(c_h),
            np.asarray(G_h),
            np.asarray(h_h),
            np.asarray(A_h),
            np.asarray(b_h),
            tol,
            max_iter,
            warm_x=np.asarray(w_h),
        )

    # `vmap_method="sequential"` lets the layer be used under jax.vmap
    # (each instance is an independent host solve). Older JAX releases
    # don't accept the kwarg, so fall back gracefully.
    try:
        return jax.pure_callback(
            host, shapes, P, c, G, h, A, b, warm_x, vmap_method="sequential"
        )
    except TypeError:
        return jax.pure_callback(host, shapes, P, c, G, h, A, b, warm_x)


def _pure_forward_batch(P, cs, G, hs, A, bs, warm_xs, n, m_g, m_a, tol, max_iter):
    """Parallel-batch forward via a single host callback. Returns stacked
    (xs, lams, nus). ``warm_xs`` is a non-differentiated warm-start operand
    (empty trailing dim ⇒ cold)."""
    b_sz = cs.shape[0]
    shapes = (
        jax.ShapeDtypeStruct((b_sz, n), jnp.float64),
        jax.ShapeDtypeStruct((b_sz, m_g), jnp.float64),
        jax.ShapeDtypeStruct((b_sz, m_a), jnp.float64),
    )

    def host(P_h, cs_h, G_h, hs_h, A_h, bs_h, w_h):
        return _forward_solve_batch(
            np.asarray(P_h),
            np.asarray(cs_h),
            np.asarray(G_h),
            np.asarray(hs_h),
            np.asarray(A_h),
            np.asarray(bs_h),
            tol,
            max_iter,
            warm_xs=np.asarray(w_h),
        )

    return jax.pure_callback(host, shapes, P, cs, G, hs, A, bs, warm_xs)


def _warm_primal(warm_start, n):
    """Extract a warm-start primal ``x`` (length ``n``) from a previous
    solution, returning an empty array (cold start) when absent."""
    if warm_start is None:
        return jnp.zeros((0,))
    wx = getattr(warm_start, "x", None)
    if wx is None:
        wx = warm_start.get("x") if hasattr(warm_start, "get") else warm_start
    if wx is None:
        return jnp.zeros((0,))
    wx = jnp.asarray(wx, dtype=jnp.float64).ravel()
    return wx if wx.shape[0] == n else jnp.zeros((0,))


def solve_qp(
    *,
    P,
    c,
    G=None,
    h=None,
    A=None,
    b=None,
    lb=None,
    ub=None,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    warm_start=None,
):
    """Differentiable convex-QP solve ``x*(P, c, G, h, A, b)``.

    Solves ``min ½xᵀPx+cᵀx s.t. Gx≤h, Ax=b, lb≤x≤ub`` and is
    differentiable w.r.t. ``P``, ``c``, ``G``, ``h``, ``A``, ``b`` via the
    OptNet implicit-function rule (``∇P`` is the symmetric gradient).

    All array args are dense jnp/np arrays. Bounds are folded into the
    inequality block as constant rows (no gradient flows to ``lb``/``ub``;
    pass differentiable bound levels through ``G``/``h`` instead).

    ``warm_start`` (optional) supplies a previous primal ``x`` (an array, or
    anything with an ``x`` attribute/key — e.g. a prior result) to seed the
    interior-point iteration on a nearby problem. It is **not**
    differentiated and does not change the solution or its gradients; it
    only reduces the iteration count. This is the natural fit here, since
    the layer returns the primal — feed the previous output back in.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    n = c.shape[0]
    G0 = jnp.zeros((0, n)) if G is None else jnp.asarray(G, dtype=jnp.float64)
    h0 = jnp.zeros((0,)) if h is None else jnp.asarray(h, dtype=jnp.float64)
    A0 = jnp.zeros((0, n)) if A is None else jnp.asarray(A, dtype=jnp.float64)
    b0 = jnp.zeros((0,)) if b is None else jnp.asarray(b, dtype=jnp.float64)

    # Fold finite bounds into G/h (constants w.r.t. differentiation here).
    G_full, h_full = _expand_bounds(G0, h0, lb, ub, n)
    warm_x = _warm_primal(warm_start, n)

    fn = _make_qp_vjp(n, G_full.shape[0], A0.shape[0], tol, max_iter)
    return fn(P, c, G_full, h_full, A0, b0, warm_x)


def _warm_primal_batch(warm_start, b_sz, n):
    """Extract a ``(B, n)`` warm-start primal from a batch result
    (a ``(B, n)`` array, or a sequence of per-row results/vectors),
    returning an empty ``(B, 0)`` array (cold) when absent or mismatched."""
    if warm_start is None:
        return jnp.zeros((b_sz, 0))
    arr = warm_start
    if isinstance(warm_start, (list, tuple)):
        rows = []
        for w in warm_start:
            wx = getattr(w, "x", None)
            if wx is None:
                wx = w.get("x") if hasattr(w, "get") else w
            rows.append(jnp.asarray(wx, dtype=jnp.float64).ravel())
        arr = jnp.stack(rows) if rows else jnp.zeros((b_sz, 0))
    arr = jnp.asarray(arr, dtype=jnp.float64)
    return arr if arr.shape == (b_sz, n) else jnp.zeros((b_sz, 0))


def solve_qp_batch(
    *,
    P,
    c,
    G=None,
    h=None,
    A=None,
    b=None,
    lb=None,
    ub=None,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    warm_start=None,
):
    """Differentiable **parallel** batch of convex QPs sharing structure.

    ``c`` is required and batched with shape ``(B, n)``. The matrices
    ``P``, ``G``, ``A`` are shared across the batch (2-D). The RHS vectors
    ``h`` and ``b`` may be batched (``(B, ·)``) or shared (``(·,)`` /
    ``None``, broadcast over the batch). Returns ``xs`` of shape
    ``(B, n)``.

    Forward solves run on the rayon-parallel ``solve_qp_batch`` path
    (outer-parallel across instances, serial within). The backward
    differentiates each instance independently: gradients to the shared
    ``P``/``G``/``A`` sum over the batch; gradients to ``c``/``h``/``b``
    stay per-row. ``∇P`` is the symmetric gradient.

    ``warm_start`` (optional) seeds each instance's iteration: a ``(B, n)``
    array of primals (e.g. a previous batch's returned ``xs``) or a
    sequence of per-row results/vectors. It is not differentiated and does
    not change the solution or its gradients — only the iteration count.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    cs = jnp.asarray(c, dtype=jnp.float64)
    if cs.ndim != 2:
        raise ValueError(f"solve_qp_batch: `c` must be 2-D (B, n), got {cs.shape}")
    b_sz, n = cs.shape

    G0 = jnp.zeros((0, n)) if G is None else jnp.asarray(G, dtype=jnp.float64)
    A0 = jnp.zeros((0, n)) if A is None else jnp.asarray(A, dtype=jnp.float64)

    # Fold shared finite bounds into the (shared) inequality block. The
    # per-instance h block only spans the user G rows; the bound rows are
    # constant and broadcast across the batch.
    G_full, h_bounds = _expand_bounds(G0, jnp.zeros((G0.shape[0],)), lb, ub, n)
    m_g = G_full.shape[0]
    n_user_rows = G0.shape[0]
    bound_rows = m_g - n_user_rows

    if h is None:
        hs_user = jnp.zeros((b_sz, n_user_rows))
    else:
        h_arr = jnp.asarray(h, dtype=jnp.float64)
        hs_user = (
            jnp.broadcast_to(h_arr, (b_sz, n_user_rows))
            if h_arr.ndim == 1
            else h_arr
        )
    hs_bounds = jnp.broadcast_to(h_bounds[n_user_rows:], (b_sz, bound_rows))
    hs = jnp.concatenate([hs_user, hs_bounds], axis=1)

    m_a = A0.shape[0]
    if b is None:
        bs = jnp.zeros((b_sz, m_a))
    else:
        b_arr = jnp.asarray(b, dtype=jnp.float64)
        bs = jnp.broadcast_to(b_arr, (b_sz, m_a)) if b_arr.ndim == 1 else b_arr

    warm_xs = _warm_primal_batch(warm_start, b_sz, n)
    fn = _make_qp_batch_vjp(n, m_g, m_a, tol, max_iter)
    return fn(P, cs, G_full, hs, A0, bs, warm_xs)


class QpLayer:
    """A reusable differentiable QP layer with fixed structure.

    Captures ``P, G, A`` (and bounds) once; calling the layer with
    ``c``/``b``/``h`` solves and is differentiable w.r.t. those (and, via
    :func:`solve_qp`, w.r.t. the captured matrices too). Suitable for use
    inside a larger JAX model (``jax.grad`` / ``jacrev`` / ``vmap``).

    Pass ``warm_start=`` (a previous primal ``x``) to ``__call__`` to seed
    the iteration on a nearby problem; for fixed-structure repeated solves,
    :class:`pounce.qp.QpFactorization` (host API) additionally reuses the
    symbolic factorization.
    """

    def __init__(self, P, G=None, A=None, lb=None, ub=None, *, tol=None, max_iter=None):
        self._P = P
        self._G = G
        self._A = A
        self._lb = lb
        self._ub = ub
        self._tol = tol
        self._max_iter = max_iter

    def __call__(self, c, *, b=None, h=None, warm_start=None):
        return solve_qp(
            P=self._P,
            c=c,
            G=self._G,
            h=h,
            A=self._A,
            b=b,
            lb=self._lb,
            ub=self._ub,
            tol=self._tol,
            max_iter=self._max_iter,
            warm_start=warm_start,
        )

    def batch(self, cs, *, b=None, h=None, warm_start=None):
        """Solve a parallel batch (rayon) sharing this layer's structure.

        ``cs`` has shape ``(B, n)``; ``h``/``b`` may be batched or shared.
        Pass ``warm_start`` (a ``(B, n)`` array of primals) to seed each
        instance. Differentiable; see :func:`solve_qp_batch`.
        """
        return solve_qp_batch(
            P=self._P,
            c=cs,
            G=self._G,
            h=h,
            A=self._A,
            b=b,
            lb=self._lb,
            ub=self._ub,
            tol=self._tol,
            max_iter=self._max_iter,
            warm_start=warm_start,
        )


# --- Differentiable SOCP (cone-aware OptNet implicit differentiation) ----
#
# Generalizes the QP backward to a product of nonnegative-orthant and
# second-order cones. The only change in the KKT differential is the
# complementarity row: the orthant's diagonal scalings `diag(z)`,
# `diag(slack)` become the cone's **arrow operators** `Arw(z)`, `Arw(slack)`
# (block-diagonal; an orthant block stays diagonal). The forward solve calls
# the cone-capable `_pounce.solve_socp`.


def _normalize_socp_cones(cones):
    """Coerce cone specs into ``((is_soc, dim), …)`` (static) and the
    ``[(kind, dim), …]`` form the binding wants. Ints are second-order."""
    static = []
    specs = []
    for spec in cones:
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            kind, d = str(spec[0]).lower(), int(spec[1])
        elif isinstance(spec, int):
            kind, d = "soc", int(spec)
        else:
            raise ValueError(f"bad cone spec {spec!r}")
        is_soc = kind in ("soc", "q", "secondorder")
        static.append((is_soc, d))
        specs.append(("soc" if is_soc else "nonneg", d))
    return tuple(static), specs


def _arrow(v):
    """Arrow matrix ``Arw(v) = [[v₀, v₁ᵀ], [v₁, v₀ I]]`` of a cone block."""
    m = v.shape[0]
    if m == 1:
        return v.reshape(1, 1)
    v0, v1 = v[0], v[1:]
    top = jnp.concatenate([v0.reshape(1, 1), v1.reshape(1, -1)], axis=1)
    bot = jnp.concatenate([v1.reshape(-1, 1), v0 * jnp.eye(m - 1)], axis=1)
    return jnp.concatenate([top, bot], axis=0)


def _scaling_blockdiag(v, cones):
    """Block-diagonal cone scaling: ``Arw(v_block)`` for a second-order
    block, ``diag(v_block)`` for an orthant block."""
    blocks = []
    off = 0
    for is_soc, d in cones:
        vb = v[off : off + d]
        blocks.append(_arrow(vb) if is_soc else jnp.diag(vb))
        off += d
    return block_diag(*blocks) if blocks else jnp.zeros((0, 0))


def _socp_backward(P, G, A, h, x, lam, nu, gx, cones):
    """Cone-aware OptNet backward (cf. :func:`_kkt_backward`). The
    complementarity row uses the arrow operators of the cones."""
    n = x.shape[0]
    m_g = G.shape[0]
    m_a = A.shape[0]
    slack = G @ x - h
    arw_z = _scaling_blockdiag(lam, cones)
    arw_slack = _scaling_blockdiag(slack, cones)
    zero_ga = jnp.zeros((m_g, m_a))
    zero_ag = jnp.zeros((m_a, m_g))
    zero_aa = jnp.zeros((m_a, m_a))
    top = jnp.concatenate([P, G.T, A.T], axis=1)
    mid = jnp.concatenate([arw_z @ G, arw_slack, zero_ga], axis=1)
    bot = jnp.concatenate([A, zero_ag, zero_aa], axis=1)
    kkt = jnp.concatenate([top, mid, bot], axis=0)
    rhs = -jnp.concatenate([gx, jnp.zeros(m_g), jnp.zeros(m_a)])
    d = jnp.linalg.solve(kkt, rhs)
    d_x = d[:n]
    d_lam = d[n : n + m_g]
    d_nu = d[n + m_g :]
    grad_c = d_x
    grad_h = -d_lam
    grad_b = -d_nu
    grad_P = 0.5 * (jnp.outer(d_x, x) + jnp.outer(x, d_x))
    grad_G = jnp.outer(d_lam, x) + jnp.outer(lam, d_x)
    grad_A = jnp.outer(d_nu, x) + jnp.outer(nu, d_x)
    return grad_P, grad_c, grad_G, grad_h, grad_A, grad_b


def _forward_solve_socp(P, c, G, h, A, b, specs, tol, max_iter):
    """Host-side SOCP forward via pounce-convex. Returns (x, z, y)."""
    m_g = G.shape[0]
    m_a = A.shape[0]
    prob = _build_problem(P, c, G, h, A, b)
    d = _pounce.solve_socp(prob, specs, tol=tol, max_iter=max_iter)
    _check_status(d["status"], "SOCP differentiable forward solve")
    x = np.asarray(d["x"], dtype=np.float64)
    lam, nu = _split_duals(d, m_g, m_a)
    return x, lam, nu


def _make_socp_vjp(n, m_g, m_a, cones, specs, tol, max_iter):
    shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((m_g,), jnp.float64),
        jax.ShapeDtypeStruct((m_a,), jnp.float64),
    )

    def forward(P, c, G, h, A, b):
        def host(P_h, c_h, G_h, h_h, A_h, b_h):
            return _forward_solve_socp(
                np.asarray(P_h), np.asarray(c_h), np.asarray(G_h),
                np.asarray(h_h), np.asarray(A_h), np.asarray(b_h),
                specs, tol, max_iter,
            )

        return jax.pure_callback(host, shapes, P, c, G, h, A, b)

    @jax.custom_vjp
    def socp(P, c, G, h, A, b):
        x, _, _ = forward(P, c, G, h, A, b)
        return x

    def fwd(P, c, G, h, A, b):
        x, lam, nu = forward(P, c, G, h, A, b)
        return x, (P, G, A, h, x, lam, nu)

    def bwd(res, gx):
        P, G, A, h, x, lam, nu = res
        return _socp_backward(P, G, A, h, x, lam, nu, gx, cones)

    socp.defvjp(fwd, bwd)
    return socp


def solve_socp(*, P, c, G, h, A=None, b=None, cones, tol=None, max_iter=None):
    """Differentiable convex-SOCP solve ``x*(P, c, G, h, A, b)`` over a
    product of cones.

    Solves ``min ½xᵀPx+cᵀx s.t. Gx ⪯_K h, Ax=b`` where the inequality block
    is partitioned by ``cones`` — a sequence of ``(kind, dim)`` specs
    (``"nonneg"``/``"soc"``; an int means a second-order cone). Each slack
    ``s = h − Gx`` block must lie in its cone. Differentiable w.r.t.
    ``P, c, G, h, A, b`` via cone-aware OptNet implicit differentiation
    (``diag`` → the cones' arrow operators).
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    n = c.shape[0]
    G = jnp.asarray(G, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)
    A0 = jnp.zeros((0, n)) if A is None else jnp.asarray(A, dtype=jnp.float64)
    b0 = jnp.zeros((0,)) if b is None else jnp.asarray(b, dtype=jnp.float64)
    static, specs = _normalize_socp_cones(cones)
    fn = _make_socp_vjp(n, G.shape[0], A0.shape[0], static, specs, tol, max_iter)
    return fn(P, c, G, h, A0, b0)
