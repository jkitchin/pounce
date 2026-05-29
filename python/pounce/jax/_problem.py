"""Build-once, solve-many JAX problem (pounce#75).

The top-level :func:`pounce.jax.solve` / :func:`vmap_solve_parallel` /
:func:`solve_with_warm` rebuild a fresh ``_JaxProblem`` (re-JIT of
``jax.grad`` / ``jax.jacrev`` / ``jax.hessian`` plus the one-shot
random sparsity probe) and a fresh :class:`pounce.Problem` on every
call. For an iterative outer loop that solves the same-structure
problem many times — differentiable constrained layer in a training
loop, parametric sweep — that rebuild dominates wall-clock and makes
the JAX path 1–2 orders of magnitude slower than the underlying solver
(see pounce#75 for numbers).

:class:`JaxProblem` is the build-once handle: do the JIT and sparsity
probe in ``__init__``, then expose :meth:`solve`, :meth:`solve_with_warm`,
:meth:`vmap_solve`, :meth:`vmap_solve_parallel` as methods that
reuse the prebuilt state on every call.

Thread safety. ``vmap_solve_parallel`` dispatches solves across a
``ThreadPoolExecutor`` (pounce#74). The JIT-compiled JAX callables and
the sparsity pattern are immutable and thread-safe. The
:class:`pounce.Problem` instance and its bound ``problem_obj`` are
*not* — the obj closes over a mutable ``p`` that gets reset on each
solve. To avoid races, each worker thread gets its own
``(problem_obj, Problem)`` pair via a :class:`threading.local`
cache, so the per-thread build cost is paid at most once per worker
(typically ``min(B, 8)`` total) instead of ``B`` times per batch.

Factor-reuse backward (pounce#76 (B)). The classical IFT backward
through an NLP requires solving a KKT linear system at ``x*``. The
default pre-#76 path assembled ``[H J^T; J D]`` dense in JAX and
dispatched ``jnp.linalg.solve``, plus did explicit active-set masking
on bounds and slack inequality rows. The factor-reuse path now reuses
the IPM's converged compound KKT factor (the same one ``k_aug`` uses
for parametric sensitivity), which:

* avoids an O((n+m)^3) dense back-solve in JAX — the LDLᵀ factor is
  already sitting on the Rust side after the forward solve.
* drops the active-set masking entirely: the bound multiplier
  ``(z_l, z_u)`` rows in the compound block already encode active /
  inactive bound behaviour exactly. Same for slack inequalities via
  ``(v_l, v_u)``. The accuracy is O(μ) at the IPM barrier parameter,
  which for default ``tol=1e-8`` is well below typical training-loop
  noise.

Each fwd registers its ``pounce.Solver`` (which owns the factor) in a
bounded-LRU table on the JaxProblem keyed by an integer id that gets
stashed in the ``custom_vjp`` residual. The bwd reads it back via
``pure_callback``. See :meth:`JaxProblem._register_solver` for the
rationale on the bounded LRU.
"""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ._build import _detect_pattern_2d, _detect_pattern_lower, _to_np
from .._pounce import Problem, Solver

_ACTIVE_TOL = 1e-6


class _StackedJaxNlp:
    """Cyipopt-shaped problem object that wraps B replicas of a parent
    :class:`JaxProblem` into a single block-diagonal NLP (pounce#76 (A)).

    The stacked problem has variables ``X = [x^(1); ...; x^(B)]`` of
    size ``B*n``, constraints ``G(X, P) = concat(g(x^(k), p^(k)))`` of
    size ``B*m``, and objective ``F(X, P) = Σ_k f(x^(k), p^(k))``.
    The Jacobian and Hessian-of-Lagrangian are block-diagonal — every
    block-`k` row of ``G`` touches only the block-`k` slice of ``X``,
    and ``F`` is a pure sum so the Hessian has no cross-block coupling.

    Why a single stacked solve, vs ``vmap_solve_parallel``: the parallel
    path runs B independent IPM solves in worker threads; total cost is
    ``B / n_workers`` solves. The stacked path runs one IPM solve over
    a size-``B*n`` problem whose linear-system structure is exactly the
    same B blocks the parallel path would solve — but they share the
    barrier homotopy (a single ``μ`` schedule), the symbolic LDLᵀ
    factorisation, and one set of fill-reducing permutations. When
    convergence behaviour is similar across the batch this can be
    faster than the parallel path; when it isn't (one block needs many
    more iterations than the rest) the parallel path wins. Both stay
    available — they're complementary.

    Closes over a mutable ``_P`` (shape ``(B,) + p_shape``) that the
    parent updates between solves so the same Rust :class:`Problem` can
    serve a sequence of batched solves at different ``p_batch``.

    Not threadsafe on its own. Each :class:`JaxProblem` keeps a small
    LRU of these per worker thread via a ``threading.local``, keyed by
    batch size ``B`` (since ``B`` determines the problem dimensions and
    the block-tiled sparsity pattern — both fixed at Problem-build
    time).
    """

    __slots__ = ("_jp", "_B", "_P", "_jac_rows_per", "_jac_cols_per",
                 "_hess_rows_per", "_hess_cols_per", "_jac_rows_stacked",
                 "_jac_cols_stacked", "_hess_rows_stacked",
                 "_hess_cols_stacked")

    def __init__(self, jp: "JaxProblem", B: int):
        self._jp = jp
        self._B = B
        self._P = None  # set by JaxProblem before every solve
        # Per-block sparsity (cached on the parent JaxProblem from the
        # one-shot probe), promoted to int64 so the stacked-index arith
        # below doesn't overflow on huge B.
        self._jac_rows_per = np.asarray(jp._jac_rows, dtype=np.int64)
        self._jac_cols_per = np.asarray(jp._jac_cols, dtype=np.int64)
        self._hess_rows_per = np.asarray(jp._hess_rows, dtype=np.int64)
        self._hess_cols_per = np.asarray(jp._hess_cols, dtype=np.int64)
        # Lift to the stacked problem: block-`k` Jacobian nonzero at
        # per-block (i, j) lives at stacked (k*m + i, k*n + j). Same for
        # the Hessian with row/col offsets of k*n. Computed once at
        # construction so the per-solve `jacobianstructure` /
        # `hessianstructure` callbacks are O(1).
        n, m = jp._n, jp._m
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
        # Sum of per-block objectives — vmap then jnp.sum stays inside
        # JAX so the JIT-compiled per-block ``_f_jit`` is reused.
        n = self._jp._n
        X_2d = jnp.asarray(X).reshape(self._B, n)
        return float(
            jnp.sum(jax.vmap(self._jp._f_jit, in_axes=(0, 0))(X_2d, self._P))
        )

    def gradient(self, X):
        # Stacked gradient is just the per-block grads concatenated:
        # ∂F/∂x^(k) = ∂f(x^(k), p^(k))/∂x^(k) (no cross-block terms
        # because F is a sum).
        n = self._jp._n
        X_2d = jnp.asarray(X).reshape(self._B, n)
        G_2d = jax.vmap(self._jp._grad_f_jit, in_axes=(0, 0))(X_2d, self._P)
        return _to_np(G_2d).reshape(-1)

    def constraints(self, X):
        n, m = self._jp._n, self._jp._m
        X_2d = jnp.asarray(X).reshape(self._B, n)
        G_2d = jax.vmap(self._jp._g_jit, in_axes=(0, 0))(X_2d, self._P)
        return _to_np(G_2d).reshape(self._B * m)

    def jacobianstructure(self):
        return (self._jac_rows_stacked, self._jac_cols_stacked)

    def jacobian(self, X):
        # Compute per-block dense Jacobians via vmap, then gather the
        # nonzeros at the per-block sparsity pattern. We never assemble
        # the dense stacked Jacobian (would be (B*m, B*n)); the gather
        # below is order ``B * nnz_per_block`` work.
        n = self._jp._n
        X_2d = jnp.asarray(X).reshape(self._B, n)
        J_3d = jax.vmap(self._jp._jac_g_jit, in_axes=(0, 0))(X_2d, self._P)
        # J_3d: (B, m, n). Gather across the (m, n) axes via the cached
        # per-block index arrays, then flatten so the result is in
        # block-major order to match the stacked structure.
        vals = J_3d[:, self._jac_rows_per, self._jac_cols_per]  # (B, nnz_per_block)
        return _to_np(vals).reshape(-1)

    def hessianstructure(self):
        return (self._hess_rows_stacked, self._hess_cols_stacked)

    def hessian(self, X, lam, obj_factor):
        # Stacked Lagrangian factorises across blocks:
        #   L(X, λ, σ, P) = Σ_k [σ f(x^(k), p^(k))
        #                       + λ^(k)ᵀ g(x^(k), p^(k))]
        # so its Hessian is block-diagonal. Compute the per-block
        # Hessian via vmap and gather the lower-triangular pattern.
        n, m = self._jp._n, self._jp._m
        X_2d = jnp.asarray(X).reshape(self._B, n)
        if m > 0:
            Lam_2d = jnp.asarray(lam).reshape(self._B, m)
            H_3d = jax.vmap(
                self._jp._hess_lag_jit, in_axes=(0, 0, None, 0),
            )(X_2d, Lam_2d, obj_factor, self._P)
        else:
            H_3d = jax.vmap(
                self._jp._hess_lag_jit, in_axes=(0, None, 0),
            )(X_2d, obj_factor, self._P)
        vals = H_3d[:, self._hess_rows_per, self._hess_cols_per]
        return _to_np(vals).reshape(-1)


class _ReusableJaxNlp:
    """Cyipopt-shaped problem object whose JAX callables are owned by
    a parent :class:`JaxProblem`. Closes over a mutable ``_p`` that the
    parent updates between solves so the same Rust :class:`Problem`
    instance can serve a sequence of solves at different ``p``.

    Not threadsafe on its own. Each :class:`JaxProblem` keeps one of
    these per worker thread via a ``threading.local``.
    """

    __slots__ = ("_jp", "_p")

    def __init__(self, jp: "JaxProblem"):
        self._jp = jp
        self._p = None  # set by JaxProblem before every solve

    def objective(self, x):
        return float(self._jp._f_jit(jnp.asarray(x), self._p))

    def gradient(self, x):
        return _to_np(self._jp._grad_f_jit(jnp.asarray(x), self._p))

    def constraints(self, x):
        return _to_np(self._jp._g_jit(jnp.asarray(x), self._p))

    def jacobianstructure(self):
        return (self._jp._jac_rows, self._jp._jac_cols)

    def jacobian(self, x):
        J = _to_np(self._jp._jac_g_jit(jnp.asarray(x), self._p))
        return J[self._jp._jac_rows, self._jp._jac_cols]

    def hessianstructure(self):
        return (self._jp._hess_rows, self._jp._hess_cols)

    def hessian(self, x, lam, obj_factor):
        if self._jp._m > 0:
            H = _to_np(
                self._jp._hess_lag_jit(
                    jnp.asarray(x), jnp.asarray(lam), obj_factor, self._p,
                )
            )
        else:
            H = _to_np(
                self._jp._hess_lag_jit(jnp.asarray(x), obj_factor, self._p)
            )
        return H[self._jp._hess_rows, self._jp._hess_cols]


def _bwd_single_factor_reuse(
    f: Callable,
    g: Callable | None,
    n: int,
    m: int,
    cl,
    cu,
    jp: "JaxProblem",
    p,
    x_star,
    lam,
    solver_id,
    v,
):
    """k_aug-style VJP: reuse the converged compound IPM factor (pounce#76).

    Replaces the dense ``jnp.linalg.solve`` on a hand-assembled
    ``[H J^T; J D]`` block (the pre-#76 default) with a back-solve
    against the IPM compound KKT factor held by the
    :class:`pounce.Solver` registered in
    ``jp._solver_registry[solver_id]``.

    Why bother. Two reasons:

    1. **Perf.** The dense back-solve is O((n+m)^3) every bwd call;
       the LDLᵀ factor is already on the Rust side after the fwd, so
       the bwd back-solve is O(nnz(L)). For modest n this is small
       absolute savings; for n+m in the hundreds-to-thousands it
       dominates.
    2. **Correctness.** The compound block already encodes
       active-set behaviour via the barrier rows on the bound
       multipliers ``(z_l, z_u)``. At convergence active bounds
       have unbounded ``z`` (forces ``Δx_i = 0`` in the back-solve)
       and inactive bounds have ``z ≈ 0`` (leaves ``Δx_i`` free).
       Slack inequality rows in the user's ``g`` are handled the
       same way by the ``(v_l, v_u)`` rows. So this path drops the
       explicit ``_ACTIVE_TOL`` masking that the dense path does on
       ``H`` / ``J`` / ``v`` — that work falls out of the factor.
       Accuracy is O(μ) at the IPM barrier parameter, which is
       below ``tol`` after convergence.

    Why we still call JAX-AD. We still need ``∂²L/∂x∂p`` (``dgradL_dp``)
    and ``∂g/∂p`` (``dg_dp``) — those are the parameter sensitivities
    of the KKT residual that get contracted with ``u`` to form
    ``dL/dp``. They depend on how ``f`` and ``g`` were *written*, not
    on the solve, so the IPM can't produce them; autodiff over the
    user-supplied JAX callables is the right source.

    Why ``lam`` (not just the host-side Solver state). We rebuild the
    Lagrangian as a JAX-traced function to feed into ``jacrev``. The
    Lagrangian needs the multipliers as a constant inside the trace,
    so we close ``lam`` (already returned from the fwd in user g-order)
    into ``lagrangian(x, p_)``.
    """
    def lagrangian(x, p_):
        base = f(x, p_)
        if g is not None and m > 0:
            base = base + jnp.dot(lam, g(x, p_))
        return base

    grad_L_of_p = lambda p_: jax.grad(lagrangian, argnums=0)(x_star, p_)
    dgradL_dp = jax.jacrev(grad_L_of_p)(p)

    if g is not None and m > 0:
        dg_dp = jax.jacrev(lambda p_: g(x_star, p_))(p)
    else:
        dg_dp = jnp.zeros((0,) + jnp.shape(p))

    # Compound back-solve via pure_callback to host. Returns (u_x, u_g)
    # already permuted back to user g-order.
    u_x, u_g = _kkt_backsolve_pure_callback(jp, solver_id, v, n, m)

    dL_dp = -jnp.tensordot(u_x, dgradL_dp, axes=1)
    if m > 0:
        dL_dp = dL_dp - jnp.tensordot(u_g, dg_dp, axes=1)
    return dL_dp


def _kkt_backsolve_pure_callback(
    jp: "JaxProblem",
    solver_id: jnp.ndarray,
    v: jnp.ndarray,
    n: int,
    m: int,
):
    """Pure-callback: pack ``[v; 0; ...; 0]`` into the compound RHS,
    call ``Solver.kkt_solve``, scatter the y_c / y_d sub-blocks back
    to user g-order. Returns ``(u_x, u_g)``.

    Why a callback. ``Solver.kkt_solve`` is a Rust-side back-solve
    against the held LDLᵀ factor — no way to express it in JAX. We
    cross the host boundary via ``pure_callback``, which is opaque to
    JAX tracing (fine for the bwd, which doesn't itself need to be
    differentiable through the back-solve — second-order users on
    this code path should set ``factor_reuse=False``).

    Why the host scatter. Pounce's TNLP adapter classifies constraints
    into equalities (``c_map``) and inequalities (``d_map``) based on
    ``cl[i] == cu[i]``, preserving input order within each group
    (see crates/pounce-nlp/src/tnlp_adapter.rs:388-413). The compound
    KKT vector exposes them in that classified layout:
    ``y_c`` lives at offset ``n_x + n_s`` and ``y_d`` at
    ``n_x + n_s + n_y_c``. To make the bwd's contraction
    ``u_g · dg/dp`` align with the user's row ordering of ``g``, we
    scatter ``(u_y_c, u_y_d)`` back into a single length-``m`` vector
    keyed off the ``cl == cu`` mask cached on the JaxProblem.
    """
    result_shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
    )

    def host_call(sid_h, v_h):
        sid = int(np.asarray(sid_h))
        solver = jp._lookup_solver(sid)
        if solver is None:
            # The registered Solver was evicted from the bounded LRU
            # cache before the bwd ran. This is rare in normal use
            # (jacobian, grad, vmap_solve) but possible in
            # long-running training loops with very many distinct
            # forward solves whose grads come back out of order.
            # Easiest mitigation: increase
            # `jp._solver_registry_capacity` or run the grad sooner
            # after the fwd. The dense fallback path
            # (`factor_reuse=False`) is unaffected.
            raise RuntimeError(
                f"pounce.jax: missing Solver for backward (id={sid}). "
                "The factor was evicted from the LRU registry — bump "
                "`_solver_registry_capacity`, run grads closer to the "
                "fwd, or use `factor_reuse=False`."
            )
        dims = solver.block_dims  # [n_x, n_s, n_y_c, n_y_d, n_z_l, n_z_u, n_v_l, n_v_u]
        n_x = dims[0]
        n_s = dims[1]
        n_y_c = dims[2]
        n_y_d = dims[3]
        kkt = solver.kkt_dim
        rhs = np.zeros(kkt, dtype=np.float64)
        v_np = np.asarray(v_h, dtype=np.float64)
        # The JAX user-space n equals n_x exactly when no variables are
        # fixed (which is the JaxProblem's contract — fixed-variable
        # treatment isn't exposed). Assert and pack.
        if v_np.shape[0] != n_x:
            raise RuntimeError(
                f"pounce.jax: cotangent length {v_np.shape[0]} != "
                f"Solver n_x={n_x} (fixed-variable treatment is not "
                "supported on the JaxProblem factor-reuse path)."
            )
        # Embed the cotangent into the compound KKT RHS:
        # ``rhs = [v; 0_s; 0_yc; 0_yd; 0_zl; 0_zu; 0_vl; 0_vu]``.
        # We're computing ``u = K^{-T} · e_x v`` (K is symmetric here),
        # then contracting with ``∂R/∂p`` whose only nonzero blocks
        # are the x-row (``dgradL_dp``), y_c-row (``dg_c/dp``), and
        # y_d-row (``dg_d/dp``) — bounds and slacks don't depend on p
        # in the JAX path, so the corresponding RHS blocks are zero.
        rhs[:n_x] = v_np
        u = np.asarray(solver.kkt_solve(rhs), dtype=np.float64)
        u_x = u[:n_x].copy()
        y_c_off = n_x + n_s
        y_d_off = y_c_off + n_y_c
        u_y_c = u[y_c_off : y_c_off + n_y_c]
        u_y_d = u[y_d_off : y_d_off + n_y_d]

        # Scatter (u_y_c, u_y_d) back to user-g order via the cl == cu
        # mask. c_map and d_map preserve user order within each group,
        # which matches pounce-nlp's classification (tnlp_adapter.rs:388-413).
        u_g = np.zeros(m, dtype=np.float64)
        if m > 0:
            cl_arr = np.asarray(jp._cl_for_classify, dtype=np.float64)
            cu_arr = np.asarray(jp._cu_for_classify, dtype=np.float64)
            is_eq = cl_arr == cu_arr
            c_idx = np.flatnonzero(is_eq)
            d_idx = np.flatnonzero(~is_eq)
            u_g[c_idx] = u_y_c
            u_g[d_idx] = u_y_d
        return u_x, u_g

    # vmap_method="sequential" tells JAX to loop over the batch axis
    # rather than calling our impure host function on a batched RHS.
    # Needed for `jax.jacobian` (which vmaps the bwd over the n
    # cotangents) and for plain `jax.vmap` of the loss-gradient. The
    # host_call itself is single-direction: one v, one solver_id, one
    # back-solve. The Solver's underlying LDLᵀ factor *could* fan out
    # to multiple RHSes at once for true cost amortisation; doing
    # that needs a `kkt_solve_many(rhs_mat)` on the Rust side, which
    # is a worthwhile follow-up but out of scope for the initial
    # factor-reuse landing.
    return jax.pure_callback(
        host_call, result_shapes, solver_id, v, vmap_method="sequential",
    )


def _bwd_single_kkt(
    f: Callable,
    g: Callable | None,
    n: int,
    m: int,
    cl,
    cu,
    p,
    x_star,
    lam,
    mult_xL,
    mult_xU,
    v,
):
    """Implicit-function-theorem VJP at a single ``(p, x*, λ*)``.

    Same logic as the bwd in :func:`pounce.jax.solve` / the per-element
    bwd in :func:`vmap_solve_parallel` — factored out so the prebuilt
    paths share one source of truth for the active-set handling
    (pounce#73 fix).
    """
    active = (mult_xL > _ACTIVE_TOL) | (mult_xU > _ACTIVE_TOL)

    def lagrangian(x, p_):
        base = f(x, p_)
        if g is not None and m > 0:
            base = base + jnp.dot(lam, g(x, p_))
        return base

    H = jax.hessian(lagrangian, argnums=0)(x_star, p)
    grad_L_of_p = lambda p_: jax.grad(lagrangian, argnums=0)(x_star, p_)
    dgradL_dp = jax.jacrev(grad_L_of_p)(p)

    if g is not None and m > 0:
        J = jax.jacrev(g, argnums=0)(x_star, p)
        dg_dp = jax.jacrev(lambda p_: g(x_star, p_))(p)
        cl_arr = jnp.asarray(cl, dtype=H.dtype)
        cu_arr = jnp.asarray(cu, dtype=H.dtype)
        is_equality = cl_arr == cu_arr
        cons_active = is_equality | (jnp.abs(lam) > _ACTIVE_TOL)
        cons_inactive = ~cons_active
    else:
        J = jnp.zeros((0, n))
        dg_dp = jnp.zeros((0,) + jnp.shape(p))
        cons_inactive = jnp.zeros((0,), dtype=bool)

    active_mat = jnp.diag(active.astype(H.dtype))
    H_eff = jnp.where(active[:, None] | active[None, :], 0.0, H) + active_mat
    J_eff = jnp.where(
        cons_inactive[:, None] | active[None, :], 0.0, J
    )
    v_eff = jnp.where(active, 0.0, v)

    if m > 0:
        cons_inactive_diag = jnp.diag(cons_inactive.astype(H.dtype))
        top = jnp.concatenate([H_eff, J_eff.T], axis=1)
        bot = jnp.concatenate([J_eff, cons_inactive_diag], axis=1)
        K = jnp.concatenate([top, bot], axis=0)
        rhs = jnp.concatenate([v_eff, jnp.zeros(m, dtype=H.dtype)])
        u = jnp.linalg.solve(K, rhs)
        u_x, u_lam = u[:n], u[n:]
    else:
        u_x = jnp.linalg.solve(H_eff, v_eff)
        u_lam = jnp.zeros(0)

    dL_dp = -jnp.tensordot(u_x, dgradL_dp, axes=1)
    if m > 0:
        dL_dp = dL_dp - jnp.tensordot(u_lam, dg_dp, axes=1)
    return dL_dp


class JaxProblem:
    """Reusable, differentiable parametric solve (pounce#75).

    Construct once with ``f(x, p)`` and ``g(x, p)``; solve many times
    at different ``p`` without re-running the JAX JIT compilation or
    the random sparsity probe.

    Parameters
    ----------
    f : callable
        Objective ``f(x, p) -> scalar``. Must be JAX-traceable.
    g : callable or None
        Constraints ``g(x, p) -> (m,)``. Required when ``m > 0``.
    n, m : int
        Variable and constraint counts.
    p_example : array-like
        Example parameter vector. Used for the one-shot sparsity probe
        (shape and dtype only — the values are discarded). Any later
        ``p`` passed to a solve method must have the same shape.
    lb, ub, cl, cu : array-like or None
        Variable and constraint bounds; same convention as
        :class:`pounce.Problem`.
    options : dict or None
        Pounce options applied once via ``add_option`` at build time.
        Options that need to vary per-solve (e.g. ``warm_start_init_point``
        flipped from ``"no"`` to ``"yes"``) are toggled internally by
        :meth:`solve_with_warm`; otherwise the same dict is in force
        for every method call.
    seed : int
        Seed for the random sparsity probe.
    factor_reuse : bool
        When ``True`` (default), the differentiable backward reuses the
        IPM's converged compound KKT factor for the implicit-function
        back-solve (k_aug-style; pounce#76) — drops the
        ``jnp.linalg.solve`` on a freshly assembled
        ``(n+m) × (n+m)`` block and avoids the explicit active-set
        masking. Set ``False`` to fall back to the original dense path
        (useful for higher-order differentiation, since the dense path
        stays inside JAX and is itself differentiable).

    Notes
    -----
    Threadsafe: :meth:`vmap_solve_parallel` dispatches across worker
    threads, and each thread keeps its own per-thread
    :class:`pounce.Problem` (via :class:`threading.local`) to avoid
    racing on the mutable ``p`` slot. The JAX callables and the
    sparsity pattern itself are immutable.
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
    ):
        if m > 0 and g is None:
            raise ValueError("g must be provided when m > 0")
        self._f = f
        self._g = g
        self._n = n
        self._m = m
        self._lb = lb
        self._ub = ub
        self._cl = cl
        self._cu = cu
        self._options = dict(options or {})
        self._factor_reuse = bool(factor_reuse)
        # Cached arrays the bwd host-side closure reads (cl == cu mask)
        # to scatter the y_c / y_d sub-blocks back to user g-order.
        # Stored on the JaxProblem so the pure_callback host_call can
        # find them without re-marshalling per call.
        if m > 0:
            self._cl_for_classify = np.asarray(cl, dtype=np.float64)
            self._cu_for_classify = np.asarray(cu, dtype=np.float64)
        else:
            self._cl_for_classify = np.zeros(0)
            self._cu_for_classify = np.zeros(0)

        # Registry of converged Solvers keyed by the integer id stashed
        # in the custom_vjp residual. Bounded-LRU so we don't grow
        # without bound in a training loop with many solve calls.
        #
        # Why bounded instead of pop-on-bwd:
        #
        # `jax.jacobian` (and any other transform that produces
        # multiple cotangents per fwd) calls the bwd N times with the
        # same residual. With ``vmap_method="sequential"`` on the
        # back-solve pure_callback, JAX loops over the cotangents and
        # invokes our host_call once per direction. Each call carries
        # the same solver_id. If we popped on the first read we'd
        # crash on the rest. So we hold the Solver across all reads
        # and evict by LRU pressure instead.
        #
        # Why 128: ~all common patterns (single grad, jacobian up to
        # ~n=128, vmap_solve over a batch) reuse the cache before
        # evicting. Training loops walking past 128 distinct fwd calls
        # without re-reading old ones still bound memory at
        # 128 × sizeof(factor). Users with non-typical patterns can
        # call :meth:`clear_solver_cache` to drop everything early.
        self._solver_registry_capacity = 128
        self._solver_registry: "OrderedDict[int, Solver]" = OrderedDict()
        self._solver_id_counter = itertools.count(1)
        # Lock guards concurrent register/lookup against
        # vmap_solve_parallel worker threads.
        self._registry_lock = threading.Lock()

        # JIT-compiled derivatives over (x, p). These are stateless and
        # threadsafe — the JaxProblem is the canonical owner.
        self._f_jit = jax.jit(f)
        self._grad_f_jit = jax.jit(jax.grad(f, argnums=0))
        if g is not None and m > 0:
            self._g_jit = jax.jit(g)
            self._jac_g_jit = jax.jit(jax.jacrev(g, argnums=0))

            def lagrangian(x, lam, sigma, p):
                return sigma * f(x, p) + jnp.dot(lam, g(x, p))

            self._hess_lag_jit = jax.jit(jax.hessian(lagrangian, argnums=0))
        else:
            self._g_jit = None
            self._jac_g_jit = None

            def lagrangian_unc(x, sigma, p):
                return sigma * f(x, p)

            self._hess_lag_jit = jax.jit(jax.hessian(lagrangian_unc, argnums=0))

        # One-shot sparsity probe at random (x, p). The sparsity
        # *pattern* is assumed independent of p (true for smooth
        # pointwise compositions); see _JaxProblem for the same
        # assumption on the non-parametric path.
        p_arr = np.asarray(p_example, dtype=np.float64)
        self._p_shape = p_arr.shape
        self._p_dtype = jnp.float64
        rng = np.random.default_rng(seed)
        x_probe = jnp.asarray(rng.standard_normal(n))
        p_probe = jnp.asarray(rng.standard_normal(p_arr.shape))
        if m > 0:
            lam_probe = jnp.asarray(rng.standard_normal(m))
            jac_dense = _to_np(self._jac_g_jit(x_probe, p_probe))
            self._jac_rows, self._jac_cols = _detect_pattern_2d(jac_dense)
            hess_dense = _to_np(
                self._hess_lag_jit(x_probe, lam_probe, 1.0, p_probe)
            )
        else:
            self._jac_rows = np.zeros(0, dtype=np.int64)
            self._jac_cols = np.zeros(0, dtype=np.int64)
            hess_dense = _to_np(self._hess_lag_jit(x_probe, 1.0, p_probe))
        self._hess_rows, self._hess_cols = _detect_pattern_lower(hess_dense)

        # Per-thread (obj, Problem) cache. Built lazily on first solve
        # from the calling thread.
        self._tls = threading.local()

    # ----- internal: per-thread cached Problem -----

    def _build_problem(self) -> tuple[_ReusableJaxNlp, Problem]:
        obj = _ReusableJaxNlp(self)
        prob = Problem(
            n=self._n, m=self._m, problem_obj=obj,
            lb=self._lb, ub=self._ub, cl=self._cl, cu=self._cu,
        )
        for k, v in self._options.items():
            prob.add_option(k, v)
        return obj, prob

    def _thread_problem(self) -> tuple[_ReusableJaxNlp, Problem]:
        cached = getattr(self._tls, "pair", None)
        if cached is None:
            cached = self._build_problem()
            self._tls.pair = cached
        return cached

    def _build_stacked_problem(self, B: int) -> tuple[_StackedJaxNlp, Problem]:
        """Build the stacked Rust :class:`Problem` for batch size ``B``.

        Bounds are tiled B times. The per-block bounds (``lb``/``ub``/
        ``cl``/``cu``) are the same for every block — the parameter
        ``p`` is what varies across blocks, not the feasible region. If
        a future use-case wants per-block bounds, that's a separate
        public surface (``batched_solve_with_bounds`` etc.).
        """
        def tile(v, count):
            if v is None:
                return None
            arr = np.asarray(v, dtype=np.float64)
            return np.tile(arr, B) if count > 0 else np.zeros(0)

        n, m = self._n, self._m
        obj = _StackedJaxNlp(self, B)
        prob = Problem(
            n=B * n, m=B * m, problem_obj=obj,
            lb=tile(self._lb, n), ub=tile(self._ub, n),
            cl=tile(self._cl, m), cu=tile(self._cu, m),
        )
        for k, v in self._options.items():
            prob.add_option(k, v)
        return obj, prob

    def _thread_stacked_problem(self, B: int) -> tuple[_StackedJaxNlp, Problem]:
        """Per-thread LRU of stacked Problems keyed by batch size B.

        Cap is small (4) because in typical use a single training loop
        sticks to one batch size — the LRU is just a guard against
        cycling between two or three sizes (e.g. evaluation batch
        differs from training batch).
        """
        cache = getattr(self._tls, "stacked", None)
        if cache is None:
            cache = OrderedDict()
            self._tls.stacked = cache
        if B in cache:
            cache.move_to_end(B)
            return cache[B]
        pair = self._build_stacked_problem(B)
        cache[B] = pair
        cache.move_to_end(B)
        while len(cache) > 4:
            cache.popitem(last=False)
        return pair

    def _thread_problem_warm(self) -> tuple[_ReusableJaxNlp, Problem]:
        cached = getattr(self._tls, "pair_warm", None)
        if cached is None:
            obj, prob = self._build_problem()
            # Warm-start option must be wired at build time — toggling
            # mid-life isn't reliable across the C boundary.
            prob.add_option("warm_start_init_point", "yes")
            cached = (obj, prob)
            self._tls.pair_warm = cached
        return cached

    # ----- host-side solves (called from pure_callback host_call) -----

    def _register_solver(self, solver: Solver) -> int:
        """Register a converged Solver in the bwd registry. Returns a
        unique integer id the bwd uses to look it up.

        Why LRU instead of pop-on-read: ``jax.jacobian`` and friends
        call the bwd N times with the same residual id (one per output
        direction). We have to hold the factor across all of those
        calls. We bound the registry at ``_solver_registry_capacity``
        and evict the oldest entry on overflow so a long training
        loop doesn't grow without bound.
        """
        sid = next(self._solver_id_counter)
        with self._registry_lock:
            self._solver_registry[sid] = solver
            self._solver_registry.move_to_end(sid)
            while len(self._solver_registry) > self._solver_registry_capacity:
                # Pop the oldest entry. Its factor (held via Rc inside
                # the Rust Solver) is freed when this reference drops.
                self._solver_registry.popitem(last=False)
        return sid

    def _lookup_solver(self, sid: int) -> Solver | None:
        """LRU-touching lookup. Refreshes the entry's recency so an
        actively used factor doesn't get evicted while still in use."""
        with self._registry_lock:
            s = self._solver_registry.get(sid)
            if s is not None:
                self._solver_registry.move_to_end(sid)
            return s

    def clear_solver_cache(self) -> None:
        """Drop all cached IPM factors held for backward passes.

        The fwd path registers each converged factor for the bwd to
        consume (see :meth:`_register_solver`). Cached factors stay
        live until LRU eviction; if you want to free them earlier —
        e.g. you know no more grads are coming for in-flight forwards
        — call this between phases.
        """
        with self._registry_lock:
            self._solver_registry.clear()

    def _host_solve(self, p_np: np.ndarray, x0_np: np.ndarray, register: bool = True):
        """Forward solve. Returns ``(x, info_with_solver_id)`` — info
        carries a ``solver_id`` field that points into the per-JaxProblem
        Solver registry. Always allocates a fresh :class:`pounce.Solver`
        wrapping the per-thread cached :class:`pounce.Problem` so that
        interleaved fwd/bwd pairs (e.g. ``jax.grad(f)(p1)`` then
        ``jax.grad(f)(p2)``) don't clobber each other's factors.

        ``pounce.Solver(prob)`` is just a PyO3 wrapper around the
        Problem; the IPM build / factorization happens inside
        ``solver.solve()``, which costs the same as ``prob.solve()`` —
        so the only marginal cost of going through Solver is a tiny
        PyO3 allocation. The win is that the converged factor is now
        held and reusable in the bwd.

        ``register=False`` skips the registry hand-off — used by paths
        whose backward doesn't consume the factor (the parallel batched
        bwd is JAX-vmapped over the dense kernel), so we don't pin
        memory on factors nobody will read.
        """
        obj, prob = self._thread_problem()
        obj._p = jnp.asarray(p_np)
        solver = Solver(prob)
        x_np, info = solver.solve(x0=np.asarray(x0_np, dtype=np.float64))
        sid = (
            self._register_solver(solver)
            if (self._factor_reuse and register)
            else 0
        )
        info_out = dict(info)
        info_out["solver_id"] = sid
        return x_np, info_out

    def _host_solve_warm(
        self,
        p_np: np.ndarray,
        x0_np: np.ndarray,
        lam_np: np.ndarray,
        zL_np: np.ndarray,
        zU_np: np.ndarray,
    ):
        obj, prob = self._thread_problem_warm()
        obj._p = jnp.asarray(p_np)
        solver = Solver(prob)
        x_np, info = solver.solve(
            x0=np.asarray(x0_np, dtype=np.float64),
            lagrange=np.asarray(lam_np, dtype=np.float64),
            zl=np.asarray(zL_np, dtype=np.float64),
            zu=np.asarray(zU_np, dtype=np.float64),
        )
        sid = self._register_solver(solver) if self._factor_reuse else 0
        info_out = dict(info)
        info_out["solver_id"] = sid
        return x_np, info_out

    def _host_batched_solve(self, p_batch_np: np.ndarray, x0_batch_np: np.ndarray):
        """Forward solve for the stacked block-diagonal problem
        (pounce#76 (A)).

        Returns per-block residuals reshaped to leading-batch axis:
        ``x_batch: (B, n)``, ``lam_batch: (B, m)``, ``mult_xL_batch /
        mult_xU_batch: (B, n)``. The IPM under the hood factors a
        single ``(B*(n+m))^2`` block-diagonal KKT once and runs the
        barrier homotopy across all blocks together — i.e. one
        ``μ``-schedule for the whole batch.

        We don't register the stacked Solver in the factor-reuse
        registry: the per-element bwd path uses ``_bwd_single_kkt``
        vmapped over the batch, which doesn't consume a held compound
        factor. A future variant that back-solves the stacked LDLᵀ
        factor with a per-block-permuted RHS could land as ``(A)+(B)``
        composition, but it's out of scope for the initial landing
        (the wins overlap and isolating each track helps reasoning).
        """
        B = p_batch_np.shape[0]
        n, m = self._n, self._m
        obj, prob = self._thread_stacked_problem(B)
        obj._P = jnp.asarray(p_batch_np)
        # Initial X is the per-block ``x0`` tiled / concatenated.
        X0 = np.asarray(x0_batch_np, dtype=np.float64).reshape(B * n)
        X_np, info = prob.solve(x0=X0)
        x_batch = np.asarray(X_np, dtype=np.float64).reshape(B, n)
        lam_batch = (
            np.asarray(info["mult_g"], dtype=np.float64).reshape(B, m)
            if m > 0 else np.zeros((B, 0), dtype=np.float64)
        )
        mult_xL_batch = np.asarray(info["mult_x_L"], dtype=np.float64).reshape(B, n)
        mult_xU_batch = np.asarray(info["mult_x_U"], dtype=np.float64).reshape(B, n)
        return x_batch, lam_batch, mult_xL_batch, mult_xU_batch

    # ----- public: differentiable solve methods -----

    def solve(self, p, x0):
        """Differentiable forward solve at parameter ``p``. Returns ``x*``."""
        return self._solve_fn()(jnp.asarray(p), jnp.asarray(x0))

    def solve_with_warm(self, p, x0, warm_start: tuple | None = None):
        """Differentiable forward solve with dual warm-start (pounce#74).

        Returns ``(x*, (lam_out, zL_out, zU_out))``. Pass
        ``warm_start=None`` for an uninformed first call.
        """
        n, m = self._n, self._m
        if warm_start is None:
            lam_warm = jnp.zeros(m, dtype=jnp.float64)
            zL_warm = jnp.zeros(n, dtype=jnp.float64)
            zU_warm = jnp.zeros(n, dtype=jnp.float64)
        else:
            lam_warm, zL_warm, zU_warm = warm_start
            lam_warm = jnp.asarray(lam_warm, dtype=jnp.float64)
            zL_warm = jnp.asarray(zL_warm, dtype=jnp.float64)
            zU_warm = jnp.asarray(zU_warm, dtype=jnp.float64)
        fn = self._solve_with_warm_fn()
        x_star, lam_out, zL_out, zU_out = fn(
            jnp.asarray(p), jnp.asarray(x0), lam_warm, zL_warm, zU_warm,
        )
        return x_star, (lam_out, zL_out, zU_out)

    def vmap_solve(self, p_batch, x0):
        """Sequential batched solve over ``p_batch`` leading axis.

        Differentiable via per-element :meth:`solve`. ``x0`` may be a
        single ``(n,)`` vector (broadcast) or a ``(B, n)`` batch.
        """
        p_batch = jnp.asarray(p_batch)
        B = p_batch.shape[0]
        x0_arr = jnp.asarray(x0)
        if x0_arr.ndim == 1:
            x0_arr = jnp.broadcast_to(x0_arr, (B, self._n))

        def one(args):
            p_i, x0_i = args
            return self.solve(p_i, x0_i)

        return jax.lax.map(one, (p_batch, x0_arr))

    def vmap_solve_parallel(self, p_batch, x0, workers: int | None = None):
        """Parallel batched solve via :class:`ThreadPoolExecutor` (pounce#74).

        Each worker thread gets its own cached :class:`pounce.Problem`,
        so the build cost is paid once per worker (typically ``min(B, 8)``
        times total), not ``B`` times per batch.
        """
        p_batch = jnp.asarray(p_batch)
        B = p_batch.shape[0]
        x0_arr = jnp.asarray(x0)
        if x0_arr.ndim == 1:
            x0_arr = jnp.broadcast_to(x0_arr, (B, self._n))
        fn = self._vmap_solve_parallel_fn(workers)
        return fn(p_batch, x0_arr)

    def batched_solve(self, p_batch, x0):
        """Stacked block-diagonal batched solve (pounce#76 (A)).

        Build a single NLP with variables ``[x^(1); ...; x^(B)]``,
        constraints ``concat(g(x^(k), p^(k)))``, and objective
        ``Σ_k f(x^(k), p^(k))``, then solve once. The KKT matrix is
        block-diagonal, so the IPM gets all the per-block independence
        of :meth:`vmap_solve_parallel`, *plus* one shared barrier
        homotopy and one shared symbolic factorisation across the
        batch.

        Returns ``x_batch`` of shape ``(B, n)``. Differentiable via
        ``custom_vjp`` — the backward vmaps the dense per-element KKT
        back-solve, exploiting the block-diagonal coupling (each
        ``∂x^(k)*/∂p^(j)`` is zero for ``k ≠ j``).

        ``x0`` may be ``(n,)`` (broadcast over the batch) or ``(B, n)``.

        When to use which batched surface:

        * :meth:`vmap_solve` — sequential ``jax.lax.map``; safest for
          long batches where you want one solve per iterate without
          tying up host threads.
        * :meth:`vmap_solve_parallel` — ``ThreadPoolExecutor``; B
          independent IPM solves, GIL released per solve. Wins when
          batch elements have very different convergence behaviour
          (slow blocks don't drag fast ones).
        * :meth:`batched_solve` — one stacked IPM solve. Wins when
          blocks have similar convergence behaviour (shared
          homotopy and symbolic factorisation amortise) and when B is
          large enough that the per-call Python overhead of multiple
          fwd dispatches becomes visible (one Rust crossing instead
          of B).
        """
        p_batch = jnp.asarray(p_batch)
        B = p_batch.shape[0]
        x0_arr = jnp.asarray(x0)
        if x0_arr.ndim == 1:
            x0_arr = jnp.broadcast_to(x0_arr, (B, self._n))
        fn = self._batched_solve_fn(B)
        return fn(p_batch, x0_arr)

    # ----- custom_vjp factories -----

    def _solve_fn(self):
        f, g, n, m = self._f, self._g, self._n, self._m
        cl, cu = self._cl, self._cu
        jp = self
        factor_reuse = self._factor_reuse

        @jax.custom_vjp
        def solve_fn(p, x0):
            x_star, _ = _pure_callback_solve(jp, p, x0)
            return x_star

        def fwd(p, x0):
            x_star, info = _pure_callback_solve(jp, p, x0)
            lam = jnp.asarray(info["mult_g"]) if m > 0 else jnp.zeros(0)
            mult_xL = jnp.asarray(info["mult_x_L"])
            mult_xU = jnp.asarray(info["mult_x_U"])
            sid = jnp.asarray(info["solver_id"])
            return x_star, (p, x_star, lam, mult_xL, mult_xU, sid)

        def bwd(residuals, v):
            p, x_star, lam, mult_xL, mult_xU, sid = residuals
            if factor_reuse:
                dL_dp = _bwd_single_factor_reuse(
                    f, g, n, m, cl, cu, jp, p, x_star, lam, sid, v,
                )
            else:
                dL_dp = _bwd_single_kkt(
                    f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v,
                )
            return dL_dp, jnp.zeros((n,), dtype=jnp.float64)

        solve_fn.defvjp(fwd, bwd)
        return solve_fn

    def _solve_with_warm_fn(self):
        f, g, n, m = self._f, self._g, self._n, self._m
        cl, cu = self._cl, self._cu
        jp = self
        factor_reuse = self._factor_reuse

        @jax.custom_vjp
        def solve_fn(p, x0, lam_warm, zL_warm, zU_warm):
            x_star, lam_out, zL_out, zU_out, _sid = _pure_callback_warm_solve(
                jp, p, x0, lam_warm, zL_warm, zU_warm,
            )
            return x_star, lam_out, zL_out, zU_out

        def fwd(p, x0, lam_warm, zL_warm, zU_warm):
            x_star, lam_out, zL_out, zU_out, sid = _pure_callback_warm_solve(
                jp, p, x0, lam_warm, zL_warm, zU_warm,
            )
            return (
                (x_star, lam_out, zL_out, zU_out),
                (p, x_star, lam_out, zL_out, zU_out, sid),
            )

        def bwd(residuals, cotangents):
            p, x_star, lam, mult_xL, mult_xU, sid = residuals
            v = cotangents[0]
            if factor_reuse:
                dL_dp = _bwd_single_factor_reuse(
                    f, g, n, m, cl, cu, jp, p, x_star, lam, sid, v,
                )
            else:
                dL_dp = _bwd_single_kkt(
                    f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v,
                )
            return (
                dL_dp,
                jnp.zeros((n,), dtype=jnp.float64),
                jnp.zeros((m,), dtype=jnp.float64),
                jnp.zeros((n,), dtype=jnp.float64),
                jnp.zeros((n,), dtype=jnp.float64),
            )

        solve_fn.defvjp(fwd, bwd)
        return solve_fn

    def _vmap_solve_parallel_fn(self, workers: int | None):
        f, g, n, m = self._f, self._g, self._n, self._m
        cl, cu = self._cl, self._cu
        jp = self

        @jax.custom_vjp
        def solve_fn(p_batch, x0_batch):
            x_star, *_ = _pure_callback_parallel_solve(
                jp, p_batch, x0_batch, workers,
            )
            return x_star

        def fwd(p_batch, x0_batch):
            x_star, lam, mult_xL, mult_xU, sids = _pure_callback_parallel_solve(
                jp, p_batch, x0_batch, workers,
            )
            return x_star, (p_batch, x_star, lam, mult_xL, mult_xU, sids)

        def bwd_single(p, x_star, lam, mult_xL, mult_xU, v):
            # Dense path in JAX so the per-element bwd can vmap. The
            # factor-reuse path needs a per-element host callback,
            # which can't ride inside a JAX-traced vmap without giving
            # up the vectorisation. The batched compound back-solve is
            # the (A)-track follow-up.
            return _bwd_single_kkt(
                f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v,
            )

        def bwd(residuals, cot_x_batch):
            (
                p_batch, x_star_batch, lam_batch, mult_xL_batch,
                mult_xU_batch, _sids,
            ) = residuals
            # `_sids` are all zero — the parallel host_call uses
            # ``register=False`` (see _pure_callback_parallel_solve's
            # host_call below). They're kept in the residual only so
            # the residual-shape contract matches the single-solve fwd
            # for future code sharing.
            dL_dp_batch = jax.vmap(bwd_single)(
                p_batch, x_star_batch, lam_batch, mult_xL_batch, mult_xU_batch,
                cot_x_batch,
            )
            return dL_dp_batch, jnp.zeros_like(x_star_batch)

        solve_fn.defvjp(fwd, bwd)
        return solve_fn

    def _batched_solve_fn(self, B: int):
        """custom_vjp factory for :meth:`batched_solve` (pounce#76 (A)).

        The bwd is ``jax.vmap`` over the per-element dense KKT
        back-solve. Block-diagonal coupling in the stacked KKT means
        ``∂x^(k)*/∂p^(j) = 0`` for ``k ≠ j``, so vmapping the
        single-block bwd is exact — there's no cross-block correction
        to assemble.

        Why not reuse the stacked LDLᵀ factor (the ``(B)`` k_aug-style
        path) here? The stacked factor *would* work — back-solve once
        against the block-diagonal RHS ``diag(v^(1), ..., v^(B))`` —
        but plumbing per-block RHS packing through the Rust
        ``Solver.kkt_solve`` host_call is a separate surface, and the
        per-element dense bwd is already fast for the block sizes
        ``(A)`` is meant to win on (small per-block ``n``, large ``B``).
        We can compose ``(A)+(B)`` later without changing the public
        surface — the choice lives behind ``factor_reuse=``.
        """
        f, g, n, m = self._f, self._g, self._n, self._m
        cl, cu = self._cl, self._cu
        jp = self

        @jax.custom_vjp
        def solve_fn(p_batch, x0_batch):
            x_star, *_ = _pure_callback_batched_solve(jp, B, p_batch, x0_batch)
            return x_star

        def fwd(p_batch, x0_batch):
            x_star, lam, mult_xL, mult_xU = _pure_callback_batched_solve(
                jp, B, p_batch, x0_batch,
            )
            return x_star, (p_batch, x_star, lam, mult_xL, mult_xU)

        def bwd_single(p, x_star, lam, mult_xL, mult_xU, v):
            return _bwd_single_kkt(
                f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v,
            )

        def bwd(residuals, cot_x_batch):
            (
                p_batch, x_star_batch, lam_batch,
                mult_xL_batch, mult_xU_batch,
            ) = residuals
            dL_dp_batch = jax.vmap(bwd_single)(
                p_batch, x_star_batch, lam_batch,
                mult_xL_batch, mult_xU_batch, cot_x_batch,
            )
            return dL_dp_batch, jnp.zeros_like(x_star_batch)

        solve_fn.defvjp(fwd, bwd)
        return solve_fn


# ----- pure_callback wrappers (module-level, closed over a JaxProblem) -----


def _pure_callback_solve(jp: JaxProblem, p, x0):
    n, m = jp._n, jp._m
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
            "solver_id": jax.ShapeDtypeStruct((), jnp.int64),
        },
    )

    def host_call(p_h, x0_h):
        x_np, info = jp._host_solve(np.asarray(p_h), np.asarray(x0_h))
        info_out = {
            "obj_val": np.float64(info["obj_val"]),
            "status": np.int32(info["status"]),
            "iter_count": np.int32(info["iter_count"]),
            "g": np.asarray(info["g"], dtype=np.float64),
            "mult_g": np.asarray(info["mult_g"], dtype=np.float64),
            "mult_x_L": np.asarray(info["mult_x_L"], dtype=np.float64),
            "mult_x_U": np.asarray(info["mult_x_U"], dtype=np.float64),
            "solver_id": np.int64(info["solver_id"]),
        }
        return np.asarray(x_np, dtype=np.float64), info_out

    return jax.pure_callback(host_call, result_shapes, p, x0)


def _pure_callback_warm_solve(jp: JaxProblem, p, x0, lam_warm, zL_warm, zU_warm):
    n, m = jp._n, jp._m
    result_shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((m,), jnp.float64),
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.int64),
    )

    def host_call(p_h, x0_h, lam_h, zL_h, zU_h):
        x_np, info = jp._host_solve_warm(
            np.asarray(p_h), np.asarray(x0_h),
            np.asarray(lam_h), np.asarray(zL_h), np.asarray(zU_h),
        )
        return (
            np.asarray(x_np, dtype=np.float64),
            np.asarray(info["mult_g"], dtype=np.float64),
            np.asarray(info["mult_x_L"], dtype=np.float64),
            np.asarray(info["mult_x_U"], dtype=np.float64),
            np.int64(info["solver_id"]),
        )

    return jax.pure_callback(
        host_call, result_shapes, p, x0, lam_warm, zL_warm, zU_warm,
    )


def _pure_callback_batched_solve(jp: JaxProblem, B: int, p_batch, x0_batch):
    """Host-side dispatch for the stacked block-diagonal solve.

    The shapes in ``result_shapes`` are unconditionally ``(B, ·)`` — the
    custom_vjp factory bakes ``B`` in at construction time, so JAX
    tracing sees concrete shapes here even though ``B`` is a Python
    argument from the user's caller.
    """
    n, m = jp._n, jp._m
    result_shapes = (
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, m), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
    )

    def host_call(p_h, x0_h):
        return jp._host_batched_solve(np.asarray(p_h), np.asarray(x0_h))

    return jax.pure_callback(host_call, result_shapes, p_batch, x0_batch)


def _pure_callback_parallel_solve(jp: JaxProblem, p_batch, x0_batch, workers):
    n, m = jp._n, jp._m
    B = p_batch.shape[0]
    result_shapes = (
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, m), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B,), jnp.int64),
    )

    def host_call(p_h, x0_h):
        p_np = np.asarray(p_h)
        x0_np = np.asarray(x0_h)
        n_workers = workers or min(B, 8)
        x_out = np.empty((B, n), dtype=np.float64)
        lam_out = np.empty((B, m), dtype=np.float64)
        zL_out = np.empty((B, n), dtype=np.float64)
        zU_out = np.empty((B, n), dtype=np.float64)
        sid_out = np.empty((B,), dtype=np.int64)

        def one(i):
            # register=False: the parallel bwd is JAX-vmapped over the
            # dense kernel and never consults the registry; skip the
            # hand-off so we don't pin B factors in the registry for
            # no benefit. The follow-up batched compound bwd (#76 (A))
            # will need a different host-side surface anyway.
            x_np, info = jp._host_solve(p_np[i], x0_np[i], register=False)
            x_out[i] = x_np
            lam_out[i] = np.asarray(info["mult_g"], dtype=np.float64)
            zL_out[i] = np.asarray(info["mult_x_L"], dtype=np.float64)
            zU_out[i] = np.asarray(info["mult_x_U"], dtype=np.float64)
            sid_out[i] = np.int64(info["solver_id"])

        if n_workers <= 1 or B <= 1:
            for i in range(B):
                one(i)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                list(pool.map(one, range(B)))
        return x_out, lam_out, zL_out, zU_out, sid_out

    return jax.pure_callback(host_call, result_shapes, p_batch, x0_batch)
