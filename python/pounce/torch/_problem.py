"""Build-once, solve-many PyTorch problem (pounce#109, mirroring
:mod:`pounce.jax._problem`).

The top-level :func:`pounce.torch.solve` / :func:`vmap_solve_parallel` /
:func:`solve_with_warm` rebuild a fresh ``_TorchProblem`` (re-trace of the
``torch.func`` derivatives plus the one-shot random sparsity probe) and a
fresh :class:`pounce.Problem` on every call. For an iterative outer loop
that solves the same-structure problem many times — a differentiable
constrained layer in a training loop, a parametric sweep — that rebuild
dominates wall-clock.

:class:`TorchProblem` is the build-once handle: do the AD tracing and
sparsity probe in ``__init__``, then expose :meth:`solve`,
:meth:`solve_with_warm`, :meth:`vmap_solve`, :meth:`vmap_solve_parallel`,
:meth:`batched_solve` as methods that reuse the prebuilt state.

Eager-mode simplifications over the JAX port. PyTorch runs eagerly, so a
great deal of the JAX ``JaxProblem`` machinery is unnecessary here:

* **No host-callback registry / LRU.** The JAX backward crosses the host
  boundary with ``jax.pure_callback``, so it must stash the converged
  :class:`pounce.Solver` in a bounded-LRU table keyed by an integer id
  (the factor can't ride inside the XLA trace). In eager PyTorch the
  forward and backward are plain Python on the same thread, so the
  forward simply stashes the ``Solver`` on the ``autograd.Function``
  ``ctx`` (or on the :class:`AnchorState`) and the backward reads it
  back. The factor lives exactly as long as the ``ctx`` / handle.

* **No single-thread executor pin (pounce#77).** That existed only
  because XLA dispatches ``pure_callback`` on unstable worker threads and
  ``PySolver`` is ``#[pyclass(unsendable)]``. Eager autograd runs the
  backward on the thread that called ``.backward()``, which is also the
  thread that built the factor in the forward, so no pinning is needed.

* **No ``ShapeDtypeStruct`` plumbing.** Shapes are concrete.

The factor-reuse backward (``factor_reuse=True``, default) reuses the
IPM's converged compound KKT factor (the same one ``k_aug`` uses for
parametric sensitivity) via ``Solver.kkt_solve_many`` — avoiding the
dense ``(n+m)×(n+m)`` back-solve and dropping the explicit active-set
masking (the barrier rows on the bound / slack multipliers already encode
it). ``factor_reuse=False`` falls back to the dense in-framework KKT
solve (:func:`pounce.torch._diff._kkt_backward`), which stays
differentiable for higher-order use.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.func import grad, hessian, jacfwd, jacrev, jvp, vmap

from .._pounce import Problem, Solver
from .._ad_common import (
    _color_columns,
    _detect_pattern_2d_multi,
    _detect_pattern_lower_multi,
)
from ._build import _DT, _seed_matrix, _t, _to_np
from ._diff import _kkt_backward

_ACTIVE_TOL = 1e-6


# ----- cyipopt-shaped problem objects (closures owned by a TorchProblem) ---


class _ReusableTorchNlp:
    """Cyipopt-shaped object whose ``torch.func`` callables are owned by a
    parent :class:`TorchProblem`. Closes over a mutable ``_p`` the parent
    updates between solves so the same Rust :class:`Problem` can serve a
    sequence of solves at different ``p``."""

    __slots__ = ("_tp", "_p")

    def __init__(self, tp: "TorchProblem"):
        self._tp = tp
        self._p = None

    def objective(self, x):
        return float(self._tp._f(_t(x), self._p))

    def gradient(self, x):
        return _to_np(self._tp._grad_f(_t(x), self._p))

    def constraints(self, x):
        if self._tp._m == 0:
            return np.zeros(0, dtype=np.float64)
        return _to_np(self._tp._g(_t(x), self._p))

    def jacobianstructure(self):
        return (self._tp._jac_rows, self._tp._jac_cols)

    def jacobian(self, x):
        if self._tp._m == 0:
            return np.zeros(0, dtype=np.float64)
        if self._tp._sparse:
            comp = _to_np(self._tp._jac_compressed(_t(x), self._p))
            return comp[self._tp._jac_seed_cols, self._tp._jac_rows]
        J = _to_np(self._tp._jac_g_dense(_t(x), self._p))
        return J[self._tp._jac_rows, self._tp._jac_cols]

    def hessianstructure(self):
        return (self._tp._hess_rows, self._tp._hess_cols)

    def hessian(self, x, lam, obj_factor):
        sigma = _t(float(obj_factor))
        if self._tp._sparse:
            if self._tp._m > 0:
                comp = _to_np(
                    self._tp._hess_compressed(_t(x), _t(lam), sigma, self._p)
                )
            else:
                comp = _to_np(self._tp._hess_compressed(_t(x), sigma, self._p))
            return comp[self._tp._hess_seed_cols, self._tp._hess_rows]
        if self._tp._m > 0:
            H = _to_np(self._tp._hess_lag(_t(x), _t(lam), sigma, self._p))
        else:
            H = _to_np(self._tp._hess_lag(_t(x), sigma, self._p))
        return H[self._tp._hess_rows, self._tp._hess_cols]


class _StackedTorchNlp:
    """Cyipopt-shaped object wrapping B replicas of a parent
    :class:`TorchProblem` into one block-diagonal NLP (pounce#76 (A)).

    Variables ``X = [x^(1); …; x^(B)]`` of size ``B*n``, constraints
    ``G(X, P) = concat(g(x^(k), p^(k)))`` of size ``B*m``, objective
    ``F = Σ_k f(x^(k), p^(k))``. The Jacobian and Lagrangian Hessian are
    block-diagonal."""

    __slots__ = ("_tp", "_B", "_P", "_jac_rows_per", "_jac_cols_per",
                 "_hess_rows_per", "_hess_cols_per", "_jac_rows_stacked",
                 "_jac_cols_stacked", "_hess_rows_stacked",
                 "_hess_cols_stacked")

    def __init__(self, tp: "TorchProblem", B: int):
        self._tp = tp
        self._B = B
        self._P = None
        self._jac_rows_per = np.asarray(tp._jac_rows, dtype=np.int64)
        self._jac_cols_per = np.asarray(tp._jac_cols, dtype=np.int64)
        self._hess_rows_per = np.asarray(tp._hess_rows, dtype=np.int64)
        self._hess_cols_per = np.asarray(tp._hess_cols, dtype=np.int64)
        n, m = tp._n, tp._m
        block_idx = np.arange(B, dtype=np.int64)[:, None]
        if self._jac_rows_per.size > 0 and m > 0:
            self._jac_rows_stacked = (
                self._jac_rows_per[None, :] + block_idx * m
            ).reshape(-1)
            self._jac_cols_stacked = (
                self._jac_cols_per[None, :] + block_idx * n
            ).reshape(-1)
        else:
            self._jac_rows_stacked = np.zeros(0, dtype=np.int64)
            self._jac_cols_stacked = np.zeros(0, dtype=np.int64)
        self._hess_rows_stacked = (
            self._hess_rows_per[None, :] + block_idx * n
        ).reshape(-1)
        self._hess_cols_stacked = (
            self._hess_cols_per[None, :] + block_idx * n
        ).reshape(-1)

    def objective(self, X):
        n = self._tp._n
        X_2d = _t(X).reshape(self._B, n)
        return float(vmap(self._tp._f)(X_2d, self._P).sum())

    def gradient(self, X):
        n = self._tp._n
        X_2d = _t(X).reshape(self._B, n)
        G_2d = vmap(self._tp._grad_f)(X_2d, self._P)
        return _to_np(G_2d).reshape(-1)

    def constraints(self, X):
        n, m = self._tp._n, self._tp._m
        if m == 0:
            return np.zeros(0, dtype=np.float64)
        X_2d = _t(X).reshape(self._B, n)
        G_2d = vmap(self._tp._g)(X_2d, self._P)
        return _to_np(G_2d).reshape(self._B * m)

    def jacobianstructure(self):
        return (self._jac_rows_stacked, self._jac_cols_stacked)

    def jacobian(self, X):
        if self._tp._m == 0:
            return np.zeros(0, dtype=np.float64)
        n = self._tp._n
        X_2d = _t(X).reshape(self._B, n)
        if self._tp._sparse:
            comp = vmap(self._tp._jac_compressed)(X_2d, self._P)
            vals = comp[:, self._tp._jac_seed_cols, self._jac_rows_per]
            return _to_np(vals).reshape(-1)
        J_3d = vmap(self._tp._jac_g_dense)(X_2d, self._P)
        vals = J_3d[:, self._jac_rows_per, self._jac_cols_per]
        return _to_np(vals).reshape(-1)

    def hessianstructure(self):
        return (self._hess_rows_stacked, self._hess_cols_stacked)

    def hessian(self, X, lam, obj_factor):
        n, m = self._tp._n, self._tp._m
        X_2d = _t(X).reshape(self._B, n)
        sigma = _t(float(obj_factor))
        if self._tp._sparse:
            if m > 0:
                Lam_2d = _t(lam).reshape(self._B, m)
                comp = vmap(
                    lambda x, lm, P: self._tp._hess_compressed(x, lm, sigma, P)
                )(X_2d, Lam_2d, self._P)
            else:
                comp = vmap(
                    lambda x, P: self._tp._hess_compressed(x, sigma, P)
                )(X_2d, self._P)
            vals = comp[:, self._tp._hess_seed_cols, self._hess_rows_per]
            return _to_np(vals).reshape(-1)
        if m > 0:
            Lam_2d = _t(lam).reshape(self._B, m)
            H_3d = vmap(
                lambda x, lm, P: self._tp._hess_lag(x, lm, sigma, P)
            )(X_2d, Lam_2d, self._P)
        else:
            H_3d = vmap(lambda x, P: self._tp._hess_lag(x, sigma, P))(X_2d, self._P)
        vals = H_3d[:, self._hess_rows_per, self._hess_cols_per]
        return _to_np(vals).reshape(-1)


# ----- held-factor (k_aug-style) back-solve helpers -----


def _factor_backsolve(solver: Solver, v_x: np.ndarray, n_x: int):
    """Multi-RHS back-solve against the held compound KKT factor.

    ``v_x`` is ``(N, n_x)``; the compound RHS embeds it in the x-block and
    zeros elsewhere (bounds / slacks don't depend on ``p`` here, and the
    factor already encodes active/inactive behaviour). Returns
    ``(u_x (N, n_x), u_y_c (N, n_y_c), u_y_d (N, n_y_d))`` — the primal
    block and the equality / inequality multiplier blocks of the solution
    in the solver's classified layout."""
    dims = solver.block_dims
    if dims is None:
        raise RuntimeError(
            "pounce.torch: factor-reuse backward requires a converged IPM "
            "factor, but the forward solve did not produce one (the IPM "
            "terminated without an acceptable factorisation). Tighten the "
            "solve (loosen `tol`, supply a better `x0`, rescale), or build "
            "the TorchProblem with `factor_reuse=False` to use the dense "
            "backward, which doesn't depend on the held factor."
        )
    n_x_, n_s_, n_y_c_, n_y_d_ = dims[0], dims[1], dims[2], dims[3]
    kkt_dim = solver.kkt_dim
    if v_x.shape[1] != n_x_:
        raise RuntimeError(
            f"pounce.torch: cotangent length {v_x.shape[1]} != Solver "
            f"n_x={n_x_} (fixed-variable treatment is not supported)."
        )
    N = v_x.shape[0]
    rhs = np.zeros((N, kkt_dim), dtype=np.float64)
    rhs[:, :n_x_] = v_x
    u = np.asarray(
        solver.kkt_solve_many(rhs.reshape(-1), N), dtype=np.float64,
    ).reshape(N, kkt_dim)
    y_c_off = n_x_ + n_s_
    y_d_off = y_c_off + n_y_c_
    u_x = u[:, :n_x_]
    u_y_c = u[:, y_c_off : y_c_off + n_y_c_]
    u_y_d = u[:, y_d_off : y_d_off + n_y_d_]
    return u_x, u_y_c, u_y_d


def _scatter_y_to_g(u_y_c, u_y_d, is_eq, m):
    """Scatter classified ``(y_c, y_d)`` multiplier blocks back to user
    g-order (``c_map``/``d_map`` preserve user order within each group)."""
    N = u_y_c.shape[0]
    out = np.zeros((N, m), dtype=np.float64)
    if m > 0:
        c_idx = np.flatnonzero(is_eq)
        d_idx = np.flatnonzero(~is_eq)
        out[:, c_idx] = u_y_c
        out[:, d_idx] = u_y_d
    return out


def _gather_g_to_y(rg, is_eq, n_y_c, n_y_d):
    """Inverse of :func:`_scatter_y_to_g` for the JVP RHS: user g-order
    ``(N, m)`` → classified ``(y_c (N, n_y_c), y_d (N, n_y_d))``."""
    N = rg.shape[0]
    c_idx = np.flatnonzero(is_eq)
    d_idx = np.flatnonzero(~is_eq)
    y_c = rg[:, c_idx] if n_y_c > 0 else np.zeros((N, 0))
    y_d = rg[:, d_idx] if n_y_d > 0 else np.zeros((N, 0))
    return y_c, y_d


def _bwd_factor_reuse_single(tp, solver, p, x_star, lam, v):
    """Single-solve k_aug VJP: reuse the converged factor for the KKT
    back-solve, autodiff the parameter sensitivities. Returns dL/dp."""
    f, g, n, m = tp._f, tp._g, tp._n, tp._m

    def lagrangian(x, p_):
        base = f(x, p_)
        if g is not None and m > 0:
            base = base + torch.dot(lam, g(x, p_))
        return base

    grad_L_of_p = lambda p_: grad(lagrangian, argnums=0)(x_star, p_)  # noqa: E731
    dgradL_dp = jacrev(grad_L_of_p)(p)
    if g is not None and m > 0:
        dg_dp = jacrev(lambda p_: g(x_star, p_))(p)
    else:
        dg_dp = torch.zeros((0,) + tuple(p.shape), dtype=_DT)

    v_np = _to_np(v)[None, :]
    u_x, u_y_c, u_y_d = _factor_backsolve(solver, v_np, n)
    u_x_t = torch.as_tensor(u_x[0], dtype=_DT)
    u_g_t = torch.as_tensor(
        _scatter_y_to_g(u_y_c, u_y_d, tp._is_eq, m)[0], dtype=_DT,
    )
    dL_dp = -torch.tensordot(u_x_t, dgradL_dp, dims=1)
    if m > 0:
        dL_dp = dL_dp - torch.tensordot(u_g_t, dg_dp, dims=1)
    return dL_dp


def _bwd_factor_reuse_batched(tp, solver, B, p_batch, x_star_batch, lam_batch, v_batch):
    """Stacked (A)+(B) VJP: one back-solve over the held stacked factor,
    per-block contraction with the autodiff parameter sensitivities.
    Returns ``(B,) + p_shape``."""
    f, g, n, m = tp._f, tp._g, tp._n, tp._m
    v_np = _to_np(v_batch).reshape(1, B * n)
    u_x, u_y_c, u_y_d = _factor_backsolve(solver, v_np, B * n)
    u_x_b = torch.as_tensor(u_x[0].reshape(B, n), dtype=_DT)
    # Stacked y_c / y_d are block-major; de-interleave then scatter.
    n_c_per = int(np.sum(tp._is_eq))
    n_d_per = m - n_c_per
    u_y_c_b = u_y_c[0].reshape(B, n_c_per) if n_c_per > 0 else np.zeros((B, 0))
    u_y_d_b = u_y_d[0].reshape(B, n_d_per) if n_d_per > 0 else np.zeros((B, 0))
    u_g_b = torch.as_tensor(
        _scatter_y_to_g(u_y_c_b, u_y_d_b, tp._is_eq, m), dtype=_DT,
    )

    grads = []
    for k in range(B):
        lam_k = lam_batch[k] if m > 0 else lam_batch
        def lagrangian(x, p_, lam_k=lam_k):
            base = f(x, p_)
            if g is not None and m > 0:
                base = base + torch.dot(lam_k, g(x, p_))
            return base
        dgradL_dp = jacrev(lambda p_: grad(lagrangian, argnums=0)(x_star_batch[k], p_))(p_batch[k])
        dL = -torch.tensordot(u_x_b[k], dgradL_dp, dims=1)
        if m > 0:
            dg_dp = jacrev(lambda p_: g(x_star_batch[k], p_))(p_batch[k])
            dL = dL - torch.tensordot(u_g_b[k], dg_dp, dims=1)
        grads.append(dL)
    return torch.stack(grads, dim=0)


def _jvp_factor_reuse_batched(tp, solver, B, p_batch, x_star_batch, lam_batch, dp_batch, cols):
    """Stacked forward (JVP) ``J @ dp`` against the held factor. Returns
    ``(delta_x (B, n), delta_lam (B, m))``."""
    f, g, n, m = tp._f, tp._g, tp._n, tp._m
    # Per-block parameter-side RHS: M_k · dp_k via autodiff.
    rx = np.zeros((B, n), dtype=np.float64)
    rg = np.zeros((B, m), dtype=np.float64)
    for k in range(B):
        lam_k = lam_batch[k] if m > 0 else lam_batch
        def lagrangian(x, p_, lam_k=lam_k):
            base = f(x, p_)
            if g is not None and m > 0:
                base = base + torch.dot(lam_k, g(x, p_))
            return base
        dgradL_dp = jacrev(lambda p_: grad(lagrangian, argnums=0)(x_star_batch[k], p_))(p_batch[k])
        if cols is not None:
            dgradL_dp = dgradL_dp[..., cols]
        rx[k] = _to_np(torch.tensordot(dgradL_dp, dp_batch[k], dims=dp_batch[k].ndim))
        if m > 0:
            dg_dp = jacrev(lambda p_: g(x_star_batch[k], p_))(p_batch[k])
            if cols is not None:
                dg_dp = dg_dp[..., cols]
            rg[k] = _to_np(torch.tensordot(dg_dp, dp_batch[k], dims=dp_batch[k].ndim))

    dims = solver.block_dims
    n_x_, n_s_, n_y_c_, n_y_d_ = dims[0], dims[1], dims[2], dims[3]
    kkt_dim = solver.kkt_dim
    rhs = np.zeros((1, kkt_dim), dtype=np.float64)
    rhs[0, :n_x_] = rx.reshape(-1)
    if m > 0:
        y_c, y_d = _gather_g_to_y(rg, tp._is_eq, n_y_c_, n_y_d_)
        y_c_off = n_x_ + n_s_
        y_d_off = y_c_off + n_y_c_
        if n_y_c_ > 0:
            rhs[0, y_c_off : y_c_off + n_y_c_] = y_c.reshape(-1)
        if n_y_d_ > 0:
            rhs[0, y_d_off : y_d_off + n_y_d_] = y_d.reshape(-1)
    w = np.asarray(solver.kkt_solve_many(rhs.reshape(-1), 1), dtype=np.float64).reshape(kkt_dim)
    w_x = w[:n_x_].reshape(B, n)
    delta_x = torch.as_tensor(-w_x, dtype=_DT)
    if m > 0:
        y_c_off = n_x_ + n_s_
        y_d_off = y_c_off + n_y_c_
        n_c_per = int(np.sum(tp._is_eq))
        n_d_per = m - n_c_per
        w_y_c = w[y_c_off : y_c_off + n_y_c_].reshape(B, n_c_per) if n_c_per > 0 else np.zeros((B, 0))
        w_y_d = w[y_d_off : y_d_off + n_y_d_].reshape(B, n_d_per) if n_d_per > 0 else np.zeros((B, 0))
        delta_lam = torch.as_tensor(
            -_scatter_y_to_g(w_y_c, w_y_d, tp._is_eq, m), dtype=_DT,
        )
    else:
        delta_lam = torch.zeros((B, 0), dtype=_DT)
    return delta_x, delta_lam


# ----- caller-owned held-factor handle (pounce#82) -----


class AnchorState:
    """Caller-owned handle to a held stacked KKT factor (pounce#82).

    Returned by :meth:`TorchProblem.anchor`,
    :meth:`TorchProblem.warm_anchor`, and (with ``return_state=True``)
    :meth:`TorchProblem.batched_solve_with_jacobian`. Holds the converged
    :class:`pounce.Solver` so several post-solve sensitivity calls reuse
    one factorisation without a re-solve.

    In eager PyTorch the factor lifetime is plain Python reference
    counting: the ``Solver`` is held as an attribute and released when the
    handle is :meth:`close`d (or garbage-collected). Prefer the context
    manager::

        with tp.anchor(p, x0) as state:
            J = tp.sensitivity(state)
    """

    __slots__ = ("_tp", "_solver", "_B", "_p_batch", "_x_star", "_lam",
                 "_duals", "_wrt_cols", "_closed")

    def __init__(self, tp, solver, B, p_batch, x_star, lam, duals, wrt_cols):
        self._tp = tp
        self._solver = solver
        self._B = B
        self._p_batch = p_batch
        self._x_star = x_star
        self._lam = lam
        self._duals = duals
        self._wrt_cols = wrt_cols
        self._closed = False

    @property
    def x_star(self):
        """Primal solution ``(B, n)`` captured at anchor time."""
        return self._x_star

    @property
    def duals(self):
        """``(lam, zL, zU)`` captured at anchor time."""
        return self._duals

    @property
    def closed(self) -> bool:
        return self._closed

    def _check_open(self):
        if self._closed:
            raise RuntimeError(
                "pounce.torch: AnchorState is closed; re-anchor with "
                "tp.anchor(...)."
            )

    def close(self) -> None:
        """Release the held factor (idempotent)."""
        self._solver = None
        self._closed = True

    def reanchor(self, p_batch, x0) -> "AnchorState":
        """Swap the held factor to a fresh solve in place."""
        self.close()
        x_star, duals, solver, (pb, xs, lam) = self._tp._anchor_forward(p_batch, x0)
        self._solver = solver
        self._p_batch, self._x_star, self._lam = pb, xs, lam
        self._duals = duals
        self._closed = False
        return self

    def __enter__(self) -> "AnchorState":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"AnchorState(B={self._B}, {state})"


class TorchProblem:
    """Reusable, differentiable parametric solve (PyTorch frontend).

    Construct once with ``f(x, p)`` and ``g(x, p)``; solve many times at
    different ``p`` without re-tracing the ``torch.func`` derivatives or
    re-running the random sparsity probe. Mirror of
    :class:`pounce.jax.JaxProblem`.

    Parameters
    ----------
    f : callable
        Objective ``f(x, p) -> scalar`` (tensors). ``torch.func``-traceable.
    g : callable or None
        Constraints ``g(x, p) -> (m,)``. Required when ``m > 0``.
    n, m : int
        Variable and constraint counts.
    p_example : array-like
        Example parameter (shape/dtype only) for the one-shot sparsity
        probe. Any later ``p`` must share this shape.
    lb, ub, cl, cu : array-like or None
        Variable and constraint bounds; same convention as
        :class:`pounce.Problem`.
    options : dict or None
        Pounce options applied via ``add_option`` at build time.
    seed : int
        Seed for the random sparsity probe(s).
    factor_reuse : bool
        When ``True`` (default), the differentiable backward reuses the
        IPM's converged compound KKT factor (k_aug-style; pounce#76). Set
        ``False`` for the dense in-framework backward (useful for
        higher-order differentiation).
    sparse : bool
        Colored/compressed forward Jacobian & Hessian (issue #83).
    n_probes : int or None
        Probes whose nonzero patterns are unioned (default 1 dense / 3
        sparse).
    """

    def __init__(
        self,
        *,
        f: Callable,
        g: Callable | None = None,
        n: int,
        m: int = 0,
        p_example,
        lb=None,
        ub=None,
        cl=None,
        cu=None,
        options: dict | None = None,
        seed: int = 0,
        factor_reuse: bool = True,
        sparse: bool = False,
        n_probes: int | None = None,
    ):
        if m > 0 and g is None:
            raise ValueError("g must be provided when m > 0")
        self._f = f
        self._g = g
        self._n = n
        self._m = m
        self._sparse = bool(sparse)
        self._n_probes = max(1, int(n_probes if n_probes is not None
                                    else (3 if sparse else 1)))
        self._lb = lb
        self._ub = ub
        self._cl = cl
        self._cu = cu
        self._options = dict(options or {})
        self._factor_reuse = bool(factor_reuse)

        if m > 0:
            self._cl_for_classify = np.asarray(cl, dtype=np.float64)
            self._cu_for_classify = np.asarray(cu, dtype=np.float64)
            self._is_eq = self._cl_for_classify == self._cu_for_classify
        else:
            self._cl_for_classify = np.zeros(0)
            self._cu_for_classify = np.zeros(0)
            self._is_eq = np.zeros(0, dtype=bool)

        self._build_closures()

        # Cached Problem instances (single-threaded reuse; the held
        # Solver captures the factor independently of the mutable _p).
        self._cached_pair = None
        self._cached_warm_pair = None
        self._cached_corrector_pair = None
        self._stacked_cache: dict[int, tuple] = {}
        self._stacked_warm_cache: dict[int, tuple] = {}

        # One-shot sparsity probe.
        p_arr = np.asarray(p_example, dtype=np.float64)
        self._p_shape = p_arr.shape
        rng = np.random.default_rng(seed)
        x_probes = [_t(rng.standard_normal(n)) for _ in range(self._n_probes)]
        p_probes = [_t(rng.standard_normal(p_arr.shape)) for _ in range(self._n_probes)]
        if m > 0:
            lam_probes = [_t(rng.standard_normal(m)) for _ in range(self._n_probes)]
            jac_denses = [
                _to_np(self._jac_g_dense(xp, pp))
                for xp, pp in zip(x_probes, p_probes)
            ]
            self._jac_rows, self._jac_cols = _detect_pattern_2d_multi(jac_denses)
            hess_denses = [
                _to_np(self._hess_lag(xp, lp, _t(1.0), pp))
                for xp, lp, pp in zip(x_probes, lam_probes, p_probes)
            ]
        else:
            self._jac_rows = np.zeros(0, dtype=np.int64)
            self._jac_cols = np.zeros(0, dtype=np.int64)
            hess_denses = [
                _to_np(self._hess_lag(xp, _t(1.0), pp))
                for xp, pp in zip(x_probes, p_probes)
            ]
        self._hess_rows, self._hess_cols = _detect_pattern_lower_multi(hess_denses)

        if self._sparse:
            self._build_compressed_closures()

    # ----- closure building -----

    def _build_closures(self) -> None:
        f, g, m = self._f, self._g, self._m
        self._grad_f = grad(f, argnums=0)
        if g is not None and m > 0:
            self._jac_g_dense = (
                jacfwd(g, argnums=0) if self._n < m else jacrev(g, argnums=0)
            )

            def lagrangian(x, lam, sigma, p):
                return sigma * f(x, p) + torch.dot(lam, g(x, p))

            self._hess_lag = hessian(lagrangian, argnums=0)
        else:
            self._jac_g_dense = None

            def lagrangian_unc(x, sigma, p):
                return sigma * f(x, p)

            self._hess_lag = hessian(lagrangian_unc, argnums=0)

    def _build_compressed_closures(self) -> None:
        n, m = self._n, self._m
        f, g = self._f, self._g
        if m > 0:
            jac_colors, k_jac = _color_columns(self._jac_rows, self._jac_cols, n)
            S_jac = _seed_matrix(jac_colors, k_jac, n)
            self._jac_seed_cols = jac_colors[self._jac_cols]

            def jac_compressed(x, p):
                return vmap(lambda s: jvp(lambda xx: g(xx, p), (x,), (s,))[1])(S_jac)

            self._jac_compressed = jac_compressed
        else:
            self._jac_seed_cols = np.zeros(0, dtype=np.int64)
            self._jac_compressed = None

        full_rows = np.concatenate([self._hess_rows, self._hess_cols])
        full_cols = np.concatenate([self._hess_cols, self._hess_rows])
        hess_colors, k_hess = _color_columns(full_rows, full_cols, n)
        S_hess = _seed_matrix(hess_colors, k_hess, n)
        self._hess_seed_cols = hess_colors[self._hess_cols]

        if m > 0:
            def lagrangian(x, lam, sigma, p):
                return sigma * f(x, p) + torch.dot(lam, g(x, p))
            grad_L = grad(lagrangian, argnums=0)

            def hess_compressed(x, lam, sigma, p):
                return vmap(
                    lambda s: jvp(lambda xx: grad_L(xx, lam, sigma, p), (x,), (s,))[1]
                )(S_hess)
        else:
            def lagrangian_unc(x, sigma, p):
                return sigma * f(x, p)
            grad_L = grad(lagrangian_unc, argnums=0)

            def hess_compressed(x, sigma, p):
                return vmap(
                    lambda s: jvp(lambda xx: grad_L(xx, sigma, p), (x,), (s,))[1]
                )(S_hess)

        self._hess_compressed = hess_compressed

    # ----- Problem builders (cached) -----

    def _build_problem(self):
        obj = _ReusableTorchNlp(self)
        prob = Problem(
            n=self._n, m=self._m, problem_obj=obj,
            lb=self._lb, ub=self._ub, cl=self._cl, cu=self._cu,
        )
        for k, v in self._options.items():
            prob.add_option(k, v)
        return obj, prob

    def _get_pair(self):
        if self._cached_pair is None:
            self._cached_pair = self._build_problem()
        return self._cached_pair

    def _get_warm_pair(self):
        if self._cached_warm_pair is None:
            obj, prob = self._build_problem()
            prob.add_option("warm_start_init_point", "yes")
            self._cached_warm_pair = (obj, prob)
        return self._cached_warm_pair

    def _get_corrector_pair(self):
        if self._cached_corrector_pair is None:
            obj, prob = self._build_problem()
            prob.add_option("warm_start_init_point", "yes")
            self._cached_corrector_pair = (obj, prob)
        return self._cached_corrector_pair

    def _build_stacked_problem(self, B: int):
        def tile(v, count):
            if v is None:
                return None
            arr = np.asarray(v, dtype=np.float64)
            return np.tile(arr, B) if count > 0 else np.zeros(0)
        n, m = self._n, self._m
        obj = _StackedTorchNlp(self, B)
        prob = Problem(
            n=B * n, m=B * m, problem_obj=obj,
            lb=tile(self._lb, n), ub=tile(self._ub, n),
            cl=tile(self._cl, m), cu=tile(self._cu, m),
        )
        for k, v in self._options.items():
            prob.add_option(k, v)
        return obj, prob

    def _get_stacked_pair(self, B: int):
        pair = self._stacked_cache.get(B)
        if pair is None:
            pair = self._build_stacked_problem(B)
            self._stacked_cache[B] = pair
        return pair

    def _get_stacked_warm_pair(self, B: int):
        pair = self._stacked_warm_cache.get(B)
        if pair is None:
            obj, prob = self._build_stacked_problem(B)
            prob.add_option("warm_start_init_point", "yes")
            pair = (obj, prob)
            self._stacked_warm_cache[B] = pair
        return pair

    # ----- host-side solves -----

    def _host_solve(self, p, x0):
        obj, prob = self._get_pair()
        obj._p = p
        solver = Solver(prob)
        x_np, info = solver.solve(x0=np.asarray(x0.detach().cpu(), dtype=np.float64))
        return np.asarray(x_np, dtype=np.float64), info, solver

    def _host_solve_warm(self, p, x0, lam, zL, zU):
        obj, prob = self._get_warm_pair()
        obj._p = p
        solver = Solver(prob)
        x_np, info = solver.solve(
            x0=np.asarray(x0.detach().cpu(), dtype=np.float64),
            lagrange=np.asarray(lam, dtype=np.float64),
            zl=np.asarray(zL, dtype=np.float64),
            zu=np.asarray(zU, dtype=np.float64),
        )
        return np.asarray(x_np, dtype=np.float64), info, solver

    def _host_batched_solve(self, p_batch, x0_batch):
        B = p_batch.shape[0]
        n, m = self._n, self._m
        obj, prob = self._get_stacked_pair(B)
        obj._P = p_batch
        X0 = np.asarray(x0_batch.detach().cpu(), dtype=np.float64).reshape(B * n)
        solver = Solver(prob)
        X_np, info = solver.solve(x0=X0)
        x_batch = np.asarray(X_np, dtype=np.float64).reshape(B, n)
        lam_batch = (
            np.asarray(info["mult_g"], dtype=np.float64).reshape(B, m)
            if m > 0 else np.zeros((B, 0))
        )
        mxL = np.asarray(info["mult_x_L"], dtype=np.float64).reshape(B, n)
        mxU = np.asarray(info["mult_x_U"], dtype=np.float64).reshape(B, n)
        return x_batch, lam_batch, mxL, mxU, solver, info

    def _host_batched_solve_warm(self, p_batch, x0_batch, lam_b, zL_b, zU_b):
        B = p_batch.shape[0]
        n, m = self._n, self._m
        obj, prob = self._get_stacked_warm_pair(B)
        obj._P = p_batch
        X0 = np.asarray(x0_batch.detach().cpu(), dtype=np.float64).reshape(B * n)
        lam_stack = np.asarray(lam_b, dtype=np.float64).reshape(B * m) if m > 0 else np.zeros(0)
        zL_stack = np.asarray(zL_b, dtype=np.float64).reshape(B * n)
        zU_stack = np.asarray(zU_b, dtype=np.float64).reshape(B * n)
        solver = Solver(prob)
        X_np, info = solver.solve(x0=X0, lagrange=lam_stack, zl=zL_stack, zu=zU_stack)
        x_batch = np.asarray(X_np, dtype=np.float64).reshape(B, n)
        lam_batch = (
            np.asarray(info["mult_g"], dtype=np.float64).reshape(B, m)
            if m > 0 else np.zeros((B, 0))
        )
        mxL = np.asarray(info["mult_x_L"], dtype=np.float64).reshape(B, n)
        mxU = np.asarray(info["mult_x_U"], dtype=np.float64).reshape(B, n)
        return x_batch, lam_batch, mxL, mxU, solver, info

    # ----- public differentiable solves -----

    def solve(self, p, x0):
        """Differentiable forward solve at parameter ``p``. Returns ``x*``."""
        p = _t(p)
        x0 = _t(x0)
        tp = self
        factor_reuse = self._factor_reuse

        class _SolveFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, p, x0):
                x_np, info, solver = tp._host_solve(p, x0)
                x_star = torch.as_tensor(x_np, dtype=_DT)
                lam = (
                    torch.as_tensor(np.asarray(info["mult_g"]), dtype=_DT)
                    if tp._m > 0 else torch.zeros(0, dtype=_DT)
                )
                mxL = torch.as_tensor(np.asarray(info["mult_x_L"]), dtype=_DT)
                mxU = torch.as_tensor(np.asarray(info["mult_x_U"]), dtype=_DT)
                ctx.save_for_backward(p, x_star, lam, mxL, mxU)
                ctx.solver = solver
                return x_star

            @staticmethod
            def backward(ctx, v):
                p, x_star, lam, mxL, mxU = ctx.saved_tensors
                if factor_reuse:
                    dL_dp = _bwd_factor_reuse_single(tp, ctx.solver, p, x_star, lam, v)
                else:
                    dL_dp = _kkt_backward(
                        tp._f, tp._g, tp._n, tp._m, tp._cl, tp._cu,
                        p, x_star, lam, mxL, mxU, v,
                    )
                return dL_dp, None

        return _SolveFn.apply(p, x0)

    def solve_with_warm(self, p, x0, warm_start: tuple | None = None):
        """Differentiable forward solve with dual warm-start.

        Returns ``(x*, (lam_out, zL_out, zU_out))``."""
        p = _t(p)
        x0 = _t(x0)
        n, m = self._n, self._m
        if warm_start is None:
            lam_w = np.zeros(m); zL_w = np.zeros(n); zU_w = np.zeros(n)
        else:
            lam_w, zL_w, zU_w = (np.asarray(_to_np(_t(a))) for a in warm_start)
        tp = self
        factor_reuse = self._factor_reuse

        class _WarmFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, p, x0):
                x_np, info, solver = tp._host_solve_warm(p, x0, lam_w, zL_w, zU_w)
                x_star = torch.as_tensor(x_np, dtype=_DT)
                lam = (
                    torch.as_tensor(np.asarray(info["mult_g"]), dtype=_DT)
                    if m > 0 else torch.zeros(0, dtype=_DT)
                )
                mxL = torch.as_tensor(np.asarray(info["mult_x_L"]), dtype=_DT)
                mxU = torch.as_tensor(np.asarray(info["mult_x_U"]), dtype=_DT)
                ctx.save_for_backward(p, x_star, lam, mxL, mxU)
                ctx.solver = solver
                ctx.extra = (lam, mxL, mxU)
                return x_star

            @staticmethod
            def backward(ctx, v):
                p, x_star, lam, mxL, mxU = ctx.saved_tensors
                if factor_reuse:
                    dL_dp = _bwd_factor_reuse_single(tp, ctx.solver, p, x_star, lam, v)
                else:
                    dL_dp = _kkt_backward(
                        tp._f, tp._g, n, m, tp._cl, tp._cu,
                        p, x_star, lam, mxL, mxU, v,
                    )
                return dL_dp, None

        # Run forward once to also recover the dual outputs to thread out.
        x_np, info, _solver = self._host_solve_warm(p, x0, lam_w, zL_w, zU_w)
        x_star = _WarmFn.apply(p, x0)
        lam_out = (
            torch.as_tensor(np.asarray(info["mult_g"]), dtype=_DT)
            if m > 0 else torch.zeros(0, dtype=_DT)
        )
        zL_out = torch.as_tensor(np.asarray(info["mult_x_L"]), dtype=_DT)
        zU_out = torch.as_tensor(np.asarray(info["mult_x_U"]), dtype=_DT)
        return x_star, (lam_out, zL_out, zU_out)

    def vmap_solve(self, p_batch, x0):
        """Sequential batched solve. Differentiable via per-element
        :meth:`solve`. ``x0`` may be ``(n,)`` (broadcast) or ``(B, n)``."""
        p_batch = _t(p_batch)
        B = p_batch.shape[0]
        x0_arr = _t(x0)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.expand(B, self._n)
        return torch.stack([self.solve(p_batch[i], x0_arr[i]) for i in range(B)], dim=0)

    def vmap_solve_parallel(self, p_batch, x0, workers: int | None = None):
        """Parallel batched solve via a threadpool. Differentiable via the
        per-element dense backward (no shared factor across threads)."""
        from ._diff import _solve_batch_threadpool

        p_batch = _t(p_batch)
        B = p_batch.shape[0]
        n, m = self._n, self._m
        x0_arr = _t(x0)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.expand(B, n).contiguous()
        tp = self

        # Wrap f/g to (x, p) signature already; reuse _diff threadpool with
        # the TorchProblem's own f/g and options.
        def f(x, p):
            return tp._f(x, p)
        g = (lambda x, p: tp._g(x, p)) if (tp._g is not None and m > 0) else None

        class _ParFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, p_batch, x0_batch):
                p_np = np.asarray(p_batch.detach().cpu(), dtype=np.float64)
                x0_np = np.asarray(x0_batch.detach().cpu(), dtype=np.float64)
                x_out, lam_out, zL_out, zU_out = _solve_batch_threadpool(
                    f, g, p_np, x0_np, n, m, tp._lb, tp._ub, tp._cl, tp._cu,
                    tp._options, workers,
                )
                x_star = torch.as_tensor(x_out, dtype=_DT)
                ctx.save_for_backward(
                    p_batch, x_star, torch.as_tensor(lam_out, dtype=_DT),
                    torch.as_tensor(zL_out, dtype=_DT),
                    torch.as_tensor(zU_out, dtype=_DT),
                )
                return x_star

            @staticmethod
            def backward(ctx, v_batch):
                p_b, x_star, lam, mxL, mxU = ctx.saved_tensors
                grads = [
                    _kkt_backward(
                        f, g, n, m, tp._cl, tp._cu, p_b[i], x_star[i],
                        lam[i] if m > 0 else lam, mxL[i], mxU[i], v_batch[i],
                    )
                    for i in range(p_b.shape[0])
                ]
                return torch.stack(grads, dim=0), None

        return _ParFn.apply(p_batch, x0_arr)

    def batched_solve(self, p_batch, x0):
        """Stacked block-diagonal batched solve (pounce#76 (A)). Returns
        ``x_batch`` of shape ``(B, n)``. Differentiable; the backward
        reuses the held stacked factor when ``factor_reuse=True``, else
        the per-block dense KKT solve."""
        p_batch = _t(p_batch)
        B = p_batch.shape[0]
        x0_arr = _t(x0)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.expand(B, self._n).contiguous()
        tp = self
        n, m = self._n, self._m
        factor_reuse = self._factor_reuse

        class _BatchedFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, p_batch, x0_batch):
                x_b, lam_b, mxL, mxU, solver, _info = tp._host_batched_solve(
                    p_batch, x0_batch,
                )
                x_star = torch.as_tensor(x_b, dtype=_DT)
                ctx.save_for_backward(
                    p_batch, x_star, torch.as_tensor(lam_b, dtype=_DT),
                    torch.as_tensor(mxL, dtype=_DT), torch.as_tensor(mxU, dtype=_DT),
                )
                ctx.solver = solver
                return x_star

            @staticmethod
            def backward(ctx, v_batch):
                p_b, x_star, lam_b, mxL, mxU = ctx.saved_tensors
                if factor_reuse:
                    dL = _bwd_factor_reuse_batched(
                        tp, ctx.solver, p_b.shape[0], p_b, x_star, lam_b, v_batch,
                    )
                else:
                    dL = torch.stack([
                        _kkt_backward(
                            tp._f, tp._g, n, m, tp._cl, tp._cu, p_b[i], x_star[i],
                            lam_b[i] if m > 0 else lam_b, mxL[i], mxU[i], v_batch[i],
                        )
                        for i in range(p_b.shape[0])
                    ], dim=0)
                return dL, None

        return _BatchedFn.apply(p_batch, x0_arr)

    def batched_solve_with_warm(self, p_batch, x0, warm_start: tuple | None = None):
        """Warm-started stacked batched solve (pounce#78). Returns
        ``(x_batch, (lam_batch, zL_batch, zU_batch))``."""
        p_batch = _t(p_batch)
        B = p_batch.shape[0]
        n, m = self._n, self._m
        x0_arr = _t(x0)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.expand(B, n).contiguous()
        if warm_start is None:
            lam_w = np.zeros((B, m)); zL_w = np.zeros((B, n)); zU_w = np.zeros((B, n))
        else:
            lam_w = _to_np(_t(warm_start[0])).reshape(B, m)
            zL_w = _to_np(_t(warm_start[1])).reshape(B, n)
            zU_w = _to_np(_t(warm_start[2])).reshape(B, n)
        tp = self
        factor_reuse = self._factor_reuse

        x_b, lam_b, mxL, mxU, _solver, info = self._host_batched_solve_warm(
            p_batch, x0_arr, lam_w, zL_w, zU_w,
        )

        class _BatchedWarmFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, p_batch, x0_batch):
                xb, lamb, mxl, mxu, solver, _i = tp._host_batched_solve_warm(
                    p_batch, x0_batch, lam_w, zL_w, zU_w,
                )
                x_star = torch.as_tensor(xb, dtype=_DT)
                ctx.save_for_backward(
                    p_batch, x_star, torch.as_tensor(lamb, dtype=_DT),
                    torch.as_tensor(mxl, dtype=_DT), torch.as_tensor(mxu, dtype=_DT),
                )
                ctx.solver = solver
                return x_star

            @staticmethod
            def backward(ctx, v_batch):
                p_b, x_star, lam_b, mxl, mxu = ctx.saved_tensors
                if factor_reuse:
                    dL = _bwd_factor_reuse_batched(
                        tp, ctx.solver, p_b.shape[0], p_b, x_star, lam_b, v_batch,
                    )
                else:
                    dL = torch.stack([
                        _kkt_backward(
                            tp._f, tp._g, n, m, tp._cl, tp._cu, p_b[i], x_star[i],
                            lam_b[i] if m > 0 else lam_b, mxl[i], mxu[i], v_batch[i],
                        )
                        for i in range(p_b.shape[0])
                    ], dim=0)
                return dL, None

        x_star = _BatchedWarmFn.apply(p_batch, x0_arr)
        lam_out = torch.as_tensor(lam_b, dtype=_DT)
        zL_out = torch.as_tensor(mxL, dtype=_DT)
        zU_out = torch.as_tensor(mxU, dtype=_DT)
        return x_star, (lam_out, zL_out, zU_out)

    # ----- post-solve sensitivity API (pounce#82) -----

    def _normalize_cols(self, wrt_cols):
        if wrt_cols is None:
            return None
        if len(self._p_shape) != 1:
            raise ValueError(
                "pounce.torch: wrt_cols is only supported for 1-D p "
                f"(got p_shape={self._p_shape})."
            )
        p_dim = self._p_shape[0]
        idx = np.arange(p_dim)[wrt_cols] if isinstance(wrt_cols, slice) \
            else np.asarray(wrt_cols, dtype=np.int64)
        return torch.as_tensor(np.atleast_1d(idx), dtype=torch.long)

    def _anchor_forward(self, p_batch, x0):
        p_arr = _t(p_batch)
        B = p_arr.shape[0]
        x0_arr = _t(x0)
        if x0_arr.ndim == 1:
            x0_arr = x0_arr.expand(B, self._n).contiguous()
        x_b, lam_b, mxL, mxU, solver, _info = self._host_batched_solve(p_arr, x0_arr)
        x_star = torch.as_tensor(x_b, dtype=_DT)
        lam = torch.as_tensor(lam_b, dtype=_DT)
        duals = (lam, torch.as_tensor(mxL, dtype=_DT), torch.as_tensor(mxU, dtype=_DT))
        return x_star, duals, solver, (p_arr, x_star, lam)

    def _slice_cols(self, arr, cols):
        if cols is None:
            return arr
        return torch.index_select(arr, -1, cols)

    def anchor(self, p_batch, x0, *, wrt_cols=None) -> AnchorState:
        """Solve once and hold the stacked KKT factor for reuse across
        post-solve sensitivity calls (pounce#82). Accepts a batched
        ``p_batch`` (``(B,) + p_shape``) or a single point (``p_shape`` →
        ``B=1``)."""
        cols = self._normalize_cols(wrt_cols)
        p_arr = _t(p_batch)
        if p_arr.ndim == len(self._p_shape):
            p_arr = p_arr[None]
        x_star, duals, solver, (pb, xs, lam) = self._anchor_forward(p_arr, x0)
        return AnchorState(self, solver, pb.shape[0], pb, xs, lam, duals, cols)

    def warm_anchor(self, p, x0, *, duals=None, mu=None, wrt_cols=None):
        """Warm-started, barrier-μ-seeded re-solve that also holds the
        converged factor as a ``B=1`` :class:`AnchorState` (pounce#90).
        Not differentiable (host-side corrector). Returns ``(state, info)``."""
        import math
        n, m = self._n, self._m
        cols = self._normalize_cols(wrt_cols)
        p_arr = _t(p)
        if p_arr.ndim != len(self._p_shape):
            raise ValueError(
                "pounce.torch: warm_anchor takes a single (un-batched) p."
            )
        x0_np = np.asarray(_to_np(_t(x0)), dtype=np.float64)
        if duals is None:
            lam_np = np.zeros(m); zL_np = np.zeros(n); zU_np = np.zeros(n)
        else:
            lam_np = _to_np(_t(duals[0])); zL_np = _to_np(_t(duals[1])); zU_np = _to_np(_t(duals[2]))
        mu_f = float("nan") if mu is None else float(mu)

        obj, prob = self._get_corrector_pair()
        obj._p = p_arr
        if math.isfinite(mu_f):
            prob.add_option("mu_init", mu_f)
            prob.add_option("warm_start_target_mu", mu_f)
        else:
            prob.add_option("mu_init", 0.1)
            prob.add_option("warm_start_target_mu", 0.0)
        solver = Solver(prob)
        x_np, info = solver.solve(
            x0=x0_np, lagrange=lam_np, zl=zL_np, zu=zU_np,
        )
        x_star = torch.as_tensor(np.asarray(x_np), dtype=_DT)[None]
        lam_out = torch.as_tensor(np.asarray(info["mult_g"]), dtype=_DT)[None]
        zL_out = torch.as_tensor(np.asarray(info["mult_x_L"]), dtype=_DT)[None]
        zU_out = torch.as_tensor(np.asarray(info["mult_x_U"]), dtype=_DT)[None]
        duals_out = (lam_out, zL_out, zU_out)
        state = AnchorState(self, solver, 1, p_arr[None], x_star, lam_out, duals_out, cols)
        return state, dict(info)

    def batched_solve_with_jacobian(self, p_batch, x0, *, wrt_cols=None, return_state=False):
        """Solve the stacked batch and return the full primal Jacobian
        ``J[k] = ∂x^(k)*/∂p^(k)`` from the held factor (pounce#82)."""
        n, m = self._n, self._m
        cols = self._normalize_cols(wrt_cols)
        x_star, duals, solver, (p_arr, xs, lam) = self._anchor_forward(p_batch, x0)
        B = p_arr.shape[0]
        basis = torch.eye(n, dtype=_DT)
        J_rows = []
        for i in range(n):
            v = basis[i].expand(B, n)
            J_rows.append(
                _bwd_factor_reuse_batched(self, solver, B, p_arr, xs, lam, v)
            )
        J = torch.stack(J_rows, dim=1)  # (B, n, p_dim)
        J = self._slice_cols(J, cols)
        if return_state:
            state = AnchorState(self, solver, B, p_arr, xs, lam, duals, cols)
            return x_star, duals, J, state
        return x_star, duals, J

    def batched_vjp_from_state(self, state: AnchorState, x_bar_batch):
        """``J^T @ x_bar`` against the held factor. Returns ``(B, p_dim)``."""
        if state._tp is not self:
            raise ValueError("AnchorState belongs to a different TorchProblem.")
        state._check_open()
        n = self._n
        x_bar = _t(x_bar_batch).reshape(state._B, n)
        dp = _bwd_factor_reuse_batched(
            self, state._solver, state._B, state._p_batch, state._x_star,
            state._lam, x_bar,
        )
        return self._slice_cols(dp, state._wrt_cols)

    def batched_jvp_from_state(self, state: AnchorState, dp_batch, *, with_duals=False):
        """``J @ dp`` against the held factor. Returns ``(B, n)`` (or
        ``(delta_x, delta_lam)`` with ``with_duals=True``)."""
        if state._tp is not self:
            raise ValueError("AnchorState belongs to a different TorchProblem.")
        state._check_open()
        dp = _t(dp_batch)
        delta_x, delta_lam = _jvp_factor_reuse_batched(
            self, state._solver, state._B, state._p_batch, state._x_star,
            state._lam, dp, state._wrt_cols,
        )
        if with_duals:
            return delta_x, delta_lam
        return delta_x

    # ----- single-problem ergonomic wrappers (pounce#88) -----

    def _require_single(self, state, who):
        if state._B != 1:
            raise ValueError(
                f"pounce.torch: {who} is the single-problem form (B=1); the "
                f"state was anchored with B={state._B}."
            )

    def solve_with_jacobian(self, p, x0, *, wrt_cols=None, return_state=False):
        """Single-problem :meth:`batched_solve_with_jacobian` (pounce#88)."""
        p1 = _t(p)[None]
        out = self.batched_solve_with_jacobian(
            p1, x0, wrt_cols=wrt_cols, return_state=return_state,
        )
        if return_state:
            x_star, (lam, zL, zU), J, state = out
            return x_star[0], (lam[0], zL[0], zU[0]), J[0], state
        x_star, (lam, zL, zU), J = out
        return x_star[0], (lam[0], zL[0], zU[0]), J[0]

    def sensitivity(self, state: AnchorState):
        """Full primal sensitivity ``∂x*/∂p`` from a held single-problem
        state (pounce#88)."""
        self._require_single(state, "sensitivity")
        n = self._n
        basis = torch.eye(n, dtype=_DT)
        J_rows = []
        for i in range(n):
            v = basis[i].expand(state._B, n)
            J_rows.append(
                _bwd_factor_reuse_batched(
                    self, state._solver, state._B, state._p_batch,
                    state._x_star, state._lam, v,
                )
            )
        J = torch.stack(J_rows, dim=1)  # (1, n, p_dim)
        return self._slice_cols(J, state._wrt_cols)[0]

    def jvp_from_state(self, state: AnchorState, dp, *, with_duals=False):
        """Single-problem :meth:`batched_jvp_from_state` (pounce#88)."""
        self._require_single(state, "jvp_from_state")
        dp1 = _t(dp)[None]
        if with_duals:
            dx, dlam = self.batched_jvp_from_state(state, dp1, with_duals=True)
            return dx[0], dlam[0]
        return self.batched_jvp_from_state(state, dp1)[0]

    def vjp_from_state(self, state: AnchorState, x_bar):
        """Single-problem :meth:`batched_vjp_from_state` (pounce#88)."""
        self._require_single(state, "vjp_from_state")
        xb1 = _t(x_bar)[None]
        return self.batched_vjp_from_state(state, xb1)[0]

    def sensitivity_at(self, x_star, theta, duals, *, wrt_cols=None):
        """Exact ``∂x*/∂θ`` at a supplied primal-dual point by re-assembling
        and solving the dense KKT there (pounce#87) — no IPM re-solve. Stays
        in-framework (differentiable)."""
        f, g, n, m = self._f, self._g, self._n, self._m
        cl, cu = self._cl, self._cu
        cols = self._normalize_cols(wrt_cols)
        x_star = _t(x_star)
        theta = _t(theta)
        lam, zL, zU = (_t(d) for d in duals)
        basis = torch.eye(n, dtype=_DT)
        J_rows = [
            _kkt_backward(f, g, n, m, cl, cu, theta, x_star, lam, zL, zU, basis[i])
            for i in range(n)
        ]
        J = torch.stack(J_rows, dim=0)  # (n,) + p_shape
        return self._slice_cols(J, cols)

    def active_set_margin(self, state: AnchorState, *, active_tol=_ACTIVE_TOL):
        """Distance to an active-set change at the anchor point (pounce#89)."""
        B = state._B
        lam_b, zL_b, zU_b = state.duals
        return self._margin_arrays(
            state._x_star, lam_b, zL_b, zU_b, state._p_batch, B, active_tol=active_tol,
        )

    def _margin_arrays(self, x_b, lam_b, zL_b, zU_b, p_b, B, *, active_tol=_ACTIVE_TOL):
        n, m = self._n, self._m
        INF = float("inf")
        x = _t(x_b).reshape(B, n)
        zL = _t(zL_b).reshape(B, n)
        zU = _t(zU_b).reshape(B, n)

        def _bound(arr, fill):
            if arr is None:
                return torch.full((B, n), fill, dtype=_DT)
            return _t(arr).expand(B, n)

        lb = _bound(self._lb, -INF)
        ub = _bound(self._ub, INF)
        act_L = zL > active_tol
        act_U = zU > active_tol
        inf_t = torch.full((), INF, dtype=_DT)
        mult_L = torch.where(act_L, zL, inf_t)
        mult_U = torch.where(act_U, zU, inf_t)
        slack_L = torch.where(act_L, inf_t, x - lb)
        slack_U = torch.where(act_U, inf_t, ub - x)
        mult_cols = [mult_L, mult_U]
        slack_cols = [slack_L, slack_U]
        if m > 0:
            lam = _t(lam_b).reshape(B, m)
            cl = torch.as_tensor(self._cl_for_classify, dtype=_DT)
            cu = torch.as_tensor(self._cu_for_classify, dtype=_DT)
            is_ineq = (cl != cu)[None, :]
            p_batch = _t(p_b).reshape((B,) + self._p_shape)
            g_val = vmap(self._g)(x, p_batch)
            ineq_slack = torch.minimum(g_val - cl[None, :], cu[None, :] - g_val)
            act_g = is_ineq & (torch.abs(lam) > active_tol)
            inact_g = is_ineq & ~act_g
            mult_cols.append(torch.where(act_g, torch.abs(lam), inf_t))
            slack_cols.append(torch.where(inact_g, ineq_slack, inf_t))
        min_mult = torch.cat(mult_cols, dim=1).min(dim=1).values
        min_slack = torch.cat(slack_cols, dim=1).min(dim=1).values
        margin = torch.minimum(min_mult, min_slack)
        return {"margin": margin, "min_mult": min_mult, "min_slack": min_slack}
