"""Differentiable convex-QP / SOCP layers (OptNet-style implicit
differentiation), PyTorch frontend (pounce#109).

PyTorch mirror of :mod:`pounce.jax._qp`. Solves, and differentiates
through, the convex QP

.. code-block:: text

    minimize    ½ xᵀP x + cᵀx
    subject to  G x ≤ h
                A x = b

The forward solve calls the ``pounce-convex`` interior-point solver
directly (eager — no host callback needed). The backward pass uses the
implicit-function theorem on the KKT system at the optimum (Amos &
Kolter, *OptNet*, 2017): the same KKT matrix that defines the solution
also yields its sensitivities, so a single ``torch.linalg.solve`` gives
the cotangents.

Differentiable parameters. Gradients are provided w.r.t. **all** the
parameters that enter the QP linearly through the optimum: the vectors
``c``, ``b``, ``h`` and the matrices ``P``, ``G``, ``A`` (full OptNet
matrix derivatives). ``P`` is differentiated as a **symmetric** matrix.

Bounds ``lb ≤ x ≤ ub`` are folded into ``G``/``h`` before differentiation
(the folded rows are constants, so no gradient flows to ``lb``/``ub``).

dtype. All inputs are coerced to float64.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .. import _pounce
from ._build import _DT

__all__ = ["solve_qp", "solve_qp_batch", "solve_socp", "QpLayer"]

# Active-set tolerance kept for parity with the JAX layer (the backward
# here reads the converged multipliers directly via the arrow/diag
# scalings, so no explicit thresholding is needed).
_ACTIVE_TOL = 1e-6


def _t(a, shape=None) -> torch.Tensor:
    out = torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=_DT) \
        if not isinstance(a, torch.Tensor) else a.to(_DT)
    return out


def _np(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy().astype(np.float64)
    return np.asarray(a, dtype=np.float64)


def _expand_bounds(G, h, lb, ub, n):
    """Fold finite variable bounds into G/h as extra rows.

    Returns ``(G_full, h_full)`` as dense tensors. ``x_i ≤ ub_i`` and
    ``−x_i ≤ −lb_i``."""
    rows = []
    rhs = []
    if G is not None and G.shape[0] > 0:
        rows.append(G)
        rhs.append(h)
    if ub is not None:
        ub_np = np.asarray(ub, dtype=np.float64)
        for i in range(n):
            if np.isfinite(float(ub_np[i])):
                e = torch.zeros(n, dtype=_DT)
                e[i] = 1.0
                rows.append(e[None, :])
                rhs.append(_t(ub_np[i]).reshape(1))
    if lb is not None:
        lb_np = np.asarray(lb, dtype=np.float64)
        for i in range(n):
            if np.isfinite(float(lb_np[i])):
                e = torch.zeros(n, dtype=_DT)
                e[i] = -1.0
                rows.append(e[None, :])
                rhs.append((-_t(lb_np[i])).reshape(1))
    if not rows:
        return torch.zeros((0, n), dtype=_DT), torch.zeros((0,), dtype=_DT)
    return torch.cat(rows, dim=0), torch.cat(rhs, dim=0)


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
        p_rows=pr, p_cols=pc, p_vals=pv,
        a_rows=ar, a_cols=ac, a_vals=av,
        b=np.asarray(b).tolist(),
        g_rows=gr, g_cols=gc, g_vals=gv,
        h=np.asarray(h).tolist(),
    )


_SUCCESS_STATUS = "optimal"


def _check_status(status, where):
    """Raise unless the convex solver reached an optimal solution.

    A non-optimal iterate is not a KKT point, so the implicit-function
    gradient would be meaningless — fail loudly rather than feed silent
    NaNs/garbage into a downstream optimizer."""
    if status != _SUCCESS_STATUS:
        raise RuntimeError(
            f"{where}: convex solver returned status {status!r}, not "
            f"{_SUCCESS_STATUS!r}; the differentiable layer cannot produce a "
            f"meaningful gradient for a non-optimal solve."
        )


def _split_duals(d, m_g, m_a):
    lam = (
        np.asarray(d["z"], dtype=np.float64) if m_g
        else np.zeros((0,), dtype=np.float64)
    )
    nu = (
        np.asarray(d["y"], dtype=np.float64) if m_a
        else np.zeros((0,), dtype=np.float64)
    )
    return lam, nu


def _forward_solve(P, c, G, h, A, b, tol, max_iter, warm_x=None):
    """Host-side forward solve via pounce-convex. Returns (x, lam, nu)."""
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
    ``cs``/``hs``/``bs``. Returns stacked (xs, lams, nus)."""
    m_g = G.shape[0]
    m_a = A.shape[0]
    b_sz = cs.shape[0]
    n = cs.shape[1]
    probs = [_build_problem(P, cs[i], G, hs[i], A, bs[i]) for i in range(b_sz)]
    warms = None
    if warm_xs is not None and np.asarray(warm_xs).shape == (b_sz, n):
        wx = np.asarray(warm_xs, dtype=np.float64)
        warms = [{"x": wx[i].tolist()} for i in range(b_sz)]
    dicts = _pounce.solve_qp_batch(
        probs, tol=tol, max_iter=max_iter, warm_starts=warms,
    )
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

    Solving for ``(d_x, d_λ, d_ν)``, the loss gradients are

    .. code-block:: text

        ∇_c = d_x          ∇_P = ½(d_x xᵀ + x d_xᵀ)
        ∇_b = −d_ν         ∇_A = d_ν xᵀ + ν d_xᵀ
        ∇_h = −d_λ         ∇_G = d_λ xᵀ + λ d_xᵀ
    """
    n = x.shape[0]
    m_g = G.shape[0]
    m_a = A.shape[0]

    slack = G @ x - h
    dlam_scale = torch.diag(lam)
    zero_ga = torch.zeros((m_g, m_a), dtype=_DT)
    zero_ag = torch.zeros((m_a, m_g), dtype=_DT)
    zero_aa = torch.zeros((m_a, m_a), dtype=_DT)

    top = torch.cat([P, G.T, A.T], dim=1)
    mid = torch.cat([dlam_scale @ G, torch.diag(slack), zero_ga], dim=1)
    bot = torch.cat([A, zero_ag, zero_aa], dim=1)
    kkt = torch.cat([top, mid, bot], dim=0)

    rhs = -torch.cat([gx, torch.zeros(m_g, dtype=_DT), torch.zeros(m_a, dtype=_DT)])
    d = torch.linalg.solve(kkt, rhs)
    d_x = d[:n]
    d_lam = d[n : n + m_g]
    d_nu = d[n + m_g :]

    grad_c = d_x
    grad_h = -d_lam
    grad_b = -d_nu
    grad_P = 0.5 * (torch.outer(d_x, x) + torch.outer(x, d_x))
    grad_G = torch.outer(d_lam, x) + torch.outer(lam, d_x)
    grad_A = torch.outer(d_nu, x) + torch.outer(nu, d_x)
    return grad_P, grad_c, grad_G, grad_h, grad_A, grad_b


def _make_qp_fn(n, m_g, m_a, tol, max_iter):
    class _QpFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, P, c, G, h, A, b, warm_x):
            x, lam, nu = _forward_solve(
                _np(P), _np(c), _np(G), _np(h), _np(A), _np(b),
                tol, max_iter, warm_x=_np(warm_x),
            )
            x_t = torch.as_tensor(x, dtype=_DT)
            lam_t = torch.as_tensor(lam, dtype=_DT)
            nu_t = torch.as_tensor(nu, dtype=_DT)
            ctx.save_for_backward(P, G, A, h, x_t, lam_t, nu_t, warm_x)
            return x_t

        @staticmethod
        def backward(ctx, gx):
            P, G, A, h, x, lam, nu, warm_x = ctx.saved_tensors
            gP, gc, gG, gh, gA, gb = _kkt_backward(P, G, A, h, x, lam, nu, gx)
            return gP, gc, gG, gh, gA, gb, None

    return _QpFn


def _make_qp_batch_fn(n, m_g, m_a, tol, max_iter):
    class _QpBatchFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, P, cs, G, hs, A, bs, warm_xs):
            xs, lams, nus = _forward_solve_batch(
                _np(P), _np(cs), _np(G), _np(hs), _np(A), _np(bs),
                tol, max_iter, warm_xs=_np(warm_xs),
            )
            xs_t = torch.as_tensor(xs, dtype=_DT)
            lams_t = torch.as_tensor(lams, dtype=_DT)
            nus_t = torch.as_tensor(nus, dtype=_DT)
            ctx.save_for_backward(P, G, A, hs, xs_t, lams_t, nus_t, warm_xs)
            return xs_t

        @staticmethod
        def backward(ctx, gxs):
            P, G, A, hs, xs, lams, nus, warm_xs = ctx.saved_tensors
            B = xs.shape[0]
            gPs, gcs, gGs, ghs, gAs, gbs = [], [], [], [], [], []
            for i in range(B):
                gP, gc, gG, gh, gA, gb = _kkt_backward(
                    P, G, A, hs[i], xs[i], lams[i], nus[i], gxs[i],
                )
                gPs.append(gP); gcs.append(gc); gGs.append(gG)
                ghs.append(gh); gAs.append(gA); gbs.append(gb)
            # Shared matrices: sum cotangents over the batch. RHS stays
            # per-row. Warm start is not differentiated.
            return (
                torch.stack(gPs).sum(dim=0),
                torch.stack(gcs),
                torch.stack(gGs).sum(dim=0),
                torch.stack(ghs),
                torch.stack(gAs).sum(dim=0),
                torch.stack(gbs),
                None,
            )

    return _QpBatchFn


def _warm_primal(warm_start, n):
    """Extract a warm-start primal ``x`` (length ``n``), returning an empty
    tensor (cold start) when absent."""
    if warm_start is None:
        return torch.zeros((0,), dtype=_DT)
    wx = getattr(warm_start, "x", None)
    if wx is None:
        wx = warm_start.get("x") if hasattr(warm_start, "get") else warm_start
    if wx is None:
        return torch.zeros((0,), dtype=_DT)
    wx = _t(wx).reshape(-1)
    return wx if wx.shape[0] == n else torch.zeros((0,), dtype=_DT)


def solve_qp(
    *,
    P, c, G=None, h=None, A=None, b=None, lb=None, ub=None,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    warm_start=None,
):
    """Differentiable convex-QP solve ``x*(P, c, G, h, A, b)``.

    Solves ``min ½xᵀPx+cᵀx s.t. Gx≤h, Ax=b, lb≤x≤ub`` and is
    differentiable w.r.t. ``P``, ``c``, ``G``, ``h``, ``A``, ``b`` via the
    OptNet implicit-function rule (``∇P`` is the symmetric gradient).

    Bounds are folded into the inequality block as constant rows (no
    gradient flows to ``lb``/``ub``). ``warm_start`` supplies a previous
    primal ``x`` to seed the iteration; it is not differentiated and only
    reduces the iteration count.
    """
    P = _t(P)
    c = _t(c)
    n = c.shape[0]
    G0 = torch.zeros((0, n), dtype=_DT) if G is None else _t(G)
    h0 = torch.zeros((0,), dtype=_DT) if h is None else _t(h)
    A0 = torch.zeros((0, n), dtype=_DT) if A is None else _t(A)
    b0 = torch.zeros((0,), dtype=_DT) if b is None else _t(b)

    G_full, h_full = _expand_bounds(G0, h0, lb, ub, n)
    warm_x = _warm_primal(warm_start, n)

    fn = _make_qp_fn(n, G_full.shape[0], A0.shape[0], tol, max_iter)
    return fn.apply(P, c, G_full, h_full, A0, b0, warm_x)


def _warm_primal_batch(warm_start, b_sz, n):
    """Extract a ``(B, n)`` warm-start primal, returning an empty
    ``(B, 0)`` tensor (cold) when absent or mismatched."""
    if warm_start is None:
        return torch.zeros((b_sz, 0), dtype=_DT)
    arr = warm_start
    if isinstance(warm_start, (list, tuple)):
        rows = []
        for w in warm_start:
            wx = getattr(w, "x", None)
            if wx is None:
                wx = w.get("x") if hasattr(w, "get") else w
            rows.append(_t(wx).reshape(-1))
        arr = torch.stack(rows) if rows else torch.zeros((b_sz, 0), dtype=_DT)
    arr = _t(arr)
    return arr if tuple(arr.shape) == (b_sz, n) else torch.zeros((b_sz, 0), dtype=_DT)


def solve_qp_batch(
    *,
    P, c, G=None, h=None, A=None, b=None, lb=None, ub=None,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    warm_start=None,
):
    """Differentiable **parallel** batch of convex QPs sharing structure.

    ``c`` is required and batched with shape ``(B, n)``. The matrices
    ``P``, ``G``, ``A`` are shared across the batch. The RHS vectors ``h``
    and ``b`` may be batched (``(B, ·)``) or shared. Returns ``xs`` of
    shape ``(B, n)``. Gradients to the shared matrices sum over the batch;
    gradients to ``c``/``h``/``b`` stay per-row. ``∇P`` is symmetric.
    """
    P = _t(P)
    cs = _t(c)
    if cs.ndim != 2:
        raise ValueError(f"solve_qp_batch: `c` must be 2-D (B, n), got {tuple(cs.shape)}")
    b_sz, n = cs.shape

    G0 = torch.zeros((0, n), dtype=_DT) if G is None else _t(G)
    A0 = torch.zeros((0, n), dtype=_DT) if A is None else _t(A)

    G_full, h_bounds = _expand_bounds(G0, torch.zeros((G0.shape[0],), dtype=_DT), lb, ub, n)
    m_g = G_full.shape[0]
    n_user_rows = G0.shape[0]
    bound_rows = m_g - n_user_rows

    if h is None:
        hs_user = torch.zeros((b_sz, n_user_rows), dtype=_DT)
    else:
        h_arr = _t(h)
        hs_user = h_arr.expand(b_sz, n_user_rows) if h_arr.ndim == 1 else h_arr
    hs_bounds = h_bounds[n_user_rows:].expand(b_sz, bound_rows)
    hs = torch.cat([hs_user, hs_bounds], dim=1)

    m_a = A0.shape[0]
    if b is None:
        bs = torch.zeros((b_sz, m_a), dtype=_DT)
    else:
        b_arr = _t(b)
        bs = b_arr.expand(b_sz, m_a) if b_arr.ndim == 1 else b_arr

    warm_xs = _warm_primal_batch(warm_start, b_sz, n)
    fn = _make_qp_batch_fn(n, m_g, m_a, tol, max_iter)
    return fn.apply(P, cs, G_full, hs, A0, bs, warm_xs)


class QpLayer:
    """A reusable differentiable QP layer with fixed structure.

    Captures ``P, G, A`` (and bounds) once; calling the layer with
    ``c``/``b``/``h`` solves and is differentiable w.r.t. those (and, via
    :func:`solve_qp`, the captured matrices too). Suitable for use inside
    a ``torch.nn`` model.
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
            P=self._P, c=c, G=self._G, h=h, A=self._A, b=b,
            lb=self._lb, ub=self._ub, tol=self._tol, max_iter=self._max_iter,
            warm_start=warm_start,
        )

    def batch(self, cs, *, b=None, h=None, warm_start=None):
        """Solve a parallel batch sharing this layer's structure."""
        return solve_qp_batch(
            P=self._P, c=cs, G=self._G, h=h, A=self._A, b=b,
            lb=self._lb, ub=self._ub, tol=self._tol, max_iter=self._max_iter,
            warm_start=warm_start,
        )


# --- Differentiable SOCP (cone-aware OptNet implicit differentiation) ----
#
# Generalizes the QP backward to a product of nonnegative-orthant and
# second-order cones. The only change in the KKT differential is the
# complementarity row: the orthant's diagonal scalings become the cone's
# **arrow operators** Arw(z), Arw(slack) (block-diagonal; an orthant block
# stays diagonal). The forward solve calls the cone-capable
# ``_pounce.solve_socp``.


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
    top = torch.cat([v0.reshape(1, 1), v1.reshape(1, -1)], dim=1)
    bot = torch.cat([v1.reshape(-1, 1), v0 * torch.eye(m - 1, dtype=_DT)], dim=1)
    return torch.cat([top, bot], dim=0)


def _block_diag(blocks):
    if not blocks:
        return torch.zeros((0, 0), dtype=_DT)
    return torch.block_diag(*blocks)


def _scaling_blockdiag(v, cones):
    """Block-diagonal cone scaling: ``Arw(v_block)`` for a second-order
    block, ``diag(v_block)`` for an orthant block."""
    blocks = []
    off = 0
    for is_soc, d in cones:
        vb = v[off : off + d]
        blocks.append(_arrow(vb) if is_soc else torch.diag(vb))
        off += d
    return _block_diag(blocks)


def _socp_backward(P, G, A, h, x, lam, nu, gx, cones):
    """Cone-aware OptNet backward (cf. :func:`_kkt_backward`). The
    complementarity row uses the arrow operators of the cones."""
    n = x.shape[0]
    m_g = G.shape[0]
    m_a = A.shape[0]
    slack = G @ x - h
    arw_z = _scaling_blockdiag(lam, cones)
    arw_slack = _scaling_blockdiag(slack, cones)
    zero_ga = torch.zeros((m_g, m_a), dtype=_DT)
    zero_ag = torch.zeros((m_a, m_g), dtype=_DT)
    zero_aa = torch.zeros((m_a, m_a), dtype=_DT)
    top = torch.cat([P, G.T, A.T], dim=1)
    mid = torch.cat([arw_z @ G, arw_slack, zero_ga], dim=1)
    bot = torch.cat([A, zero_ag, zero_aa], dim=1)
    kkt = torch.cat([top, mid, bot], dim=0)
    rhs = -torch.cat([gx, torch.zeros(m_g, dtype=_DT), torch.zeros(m_a, dtype=_DT)])
    d = torch.linalg.solve(kkt, rhs)
    d_x = d[:n]
    d_lam = d[n : n + m_g]
    d_nu = d[n + m_g :]
    grad_c = d_x
    grad_h = -d_lam
    grad_b = -d_nu
    grad_P = 0.5 * (torch.outer(d_x, x) + torch.outer(x, d_x))
    grad_G = torch.outer(d_lam, x) + torch.outer(lam, d_x)
    grad_A = torch.outer(d_nu, x) + torch.outer(nu, d_x)
    return grad_P, grad_c, grad_G, grad_h, grad_A, grad_b


def _forward_solve_socp(P, c, G, h, A, b, specs, tol, max_iter):
    """Host-side SOCP forward via pounce-convex. Returns (x, lam, nu)."""
    m_g = G.shape[0]
    m_a = A.shape[0]
    prob = _build_problem(P, c, G, h, A, b)
    d = _pounce.solve_socp(prob, specs, tol=tol, max_iter=max_iter)
    _check_status(d["status"], "SOCP differentiable forward solve")
    x = np.asarray(d["x"], dtype=np.float64)
    lam, nu = _split_duals(d, m_g, m_a)
    return x, lam, nu


def _make_socp_fn(n, m_g, m_a, cones, specs, tol, max_iter):
    class _SocpFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, P, c, G, h, A, b):
            x, lam, nu = _forward_solve_socp(
                _np(P), _np(c), _np(G), _np(h), _np(A), _np(b),
                specs, tol, max_iter,
            )
            x_t = torch.as_tensor(x, dtype=_DT)
            lam_t = torch.as_tensor(lam, dtype=_DT)
            nu_t = torch.as_tensor(nu, dtype=_DT)
            ctx.save_for_backward(P, G, A, h, x_t, lam_t, nu_t)
            return x_t

        @staticmethod
        def backward(ctx, gx):
            P, G, A, h, x, lam, nu = ctx.saved_tensors
            return _socp_backward(P, G, A, h, x, lam, nu, gx, cones)

    return _SocpFn


def solve_socp(*, P, c, G, h, A=None, b=None, cones, tol=None, max_iter=None):
    """Differentiable convex-SOCP solve over a product of cones.

    Solves ``min ½xᵀPx+cᵀx s.t. Gx ⪯_K h, Ax=b`` where the inequality
    block is partitioned by ``cones`` — a sequence of ``(kind, dim)``
    specs (``"nonneg"``/``"soc"``; an int means a second-order cone).
    Differentiable w.r.t. ``P, c, G, h, A, b`` via cone-aware OptNet
    implicit differentiation (``diag`` → the cones' arrow operators).
    """
    P = _t(P)
    c = _t(c)
    n = c.shape[0]
    G = _t(G)
    h = _t(h)
    A0 = torch.zeros((0, n), dtype=_DT) if A is None else _t(A)
    b0 = torch.zeros((0,), dtype=_DT) if b is None else _t(b)
    static, specs = _normalize_socp_cones(cones)
    fn = _make_socp_fn(n, G.shape[0], A0.shape[0], static, specs, tol, max_iter)
    return fn.apply(P, c, G, h, A0, b0)
