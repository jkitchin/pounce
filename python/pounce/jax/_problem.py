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
import warnings
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
        # Under ``vmap_method="broadcast_all"`` (pounce#77 follow-up),
        # JAX prepends a leading batch axis to *every* input when this
        # callback is hit through a vmap — most importantly ``jax.jacrev``
        # over a JaxProblem solve, which fans out N cotangents and
        # otherwise paid one FFI hop per fan-out. Here we accept either:
        #   - unbatched: ``sid_h`` scalar, ``v_h`` shape ``(n,)``
        #   - batched:   ``sid_h`` shape ``(N,)``, ``v_h`` shape ``(N, n)``
        # The solver_id is the same for every fan-out cotangent (jacrev
        # holds the primal fixed and only varies cotangents), so we
        # extract a single int via ``np.ravel(...)[0]``.
        sid_arr = np.asarray(sid_h)
        sid = int(sid_arr.reshape(-1)[0])
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

        # All PySolver access must run on the same thread that
        # constructed it (pounce#77). The fwd pinned solver creation to
        # ``jp._factor_executor``; we pin every attribute read and the
        # ``kkt_solve`` call here to the same executor so JAX's
        # off-thread pure_callback dispatch (training loops, jit'd
        # outer wrappers, ...) doesn't trigger PyO3's unsendable panic.
        v_np = np.asarray(v_h, dtype=np.float64)
        # Unbatched: shape (n,). Batched (from jax.jacrev / jax.vmap of
        # the bwd): shape (N, n). Normalise to a 2D (N, n) and remember
        # whether to squeeze on return.
        squeeze = v_np.ndim == 1
        if squeeze:
            v_np = v_np[None, :]
        N = v_np.shape[0]

        def _do_kkt():
            dims = solver.block_dims  # [n_x, n_s, n_y_c, n_y_d, ...]
            if dims is None:
                # The Solver exists but its inner state holds no converged
                # factor — the IPM didn't converge to acceptable accuracy.
                # The factor-reuse bwd has no way forward here: kkt_solve
                # would crash on a missing factor, and silently returning
                # zeros (or NaNs) would mask the divergence in upstream
                # training loops. Raise loudly so the caller either
                # tightens the solve (loosen tol / better x0 / scale the
                # problem) or switches to `factor_reuse=False` — the dense
                # JAX backward assembles `(n+m) × (n+m)` from `f`, `g` at
                # `x*` without needing the held factor, so it still
                # produces *a* gradient (of whatever the IPM terminated
                # at, possibly poor quality) instead of crashing.
                raise RuntimeError(
                    "pounce.jax: factor-reuse backward requires a "
                    "converged IPM factor, but the forward solve did "
                    "not produce one (the IPM terminated without an "
                    "acceptable factorisation). Either tighten the "
                    "solve (check `info['status']` from `JaxProblem` "
                    "callbacks, loosen `tol`, supply a better `x0`, "
                    "or rescale the problem), or build the "
                    "`JaxProblem` with `factor_reuse=False` to fall "
                    "back to the dense JAX backward, which doesn't "
                    "depend on the held factor."
                )
            n_x_ = dims[0]
            n_s_ = dims[1]
            n_y_c_ = dims[2]
            n_y_d_ = dims[3]
            kkt_dim_ = solver.kkt_dim
            # The JAX user-space n equals n_x exactly when no variables
            # are fixed (which is the JaxProblem's contract —
            # fixed-variable treatment isn't exposed). Assert and pack.
            if v_np.shape[1] != n_x_:
                raise RuntimeError(
                    f"pounce.jax: cotangent length {v_np.shape[1]} != "
                    f"Solver n_x={n_x_} (fixed-variable treatment is "
                    "not supported on the JaxProblem factor-reuse path)."
                )
            # Embed the N cotangents into the compound KKT RHS rows:
            # ``rhs[k] = [v[k]; 0_s; 0_yc; 0_yd; 0_zl; 0_zu; 0_vl; 0_vu]``.
            # We're computing ``u[k] = K^{-T} · e_x v[k]`` (K symmetric),
            # then contracting with ``∂R/∂p`` whose only nonzero blocks
            # are the x-row (``dgradL_dp``), y_c-row (``dg_c/dp``), and
            # y_d-row (``dg_d/dp``) — bounds and slacks don't depend on
            # p in the JAX path, so the corresponding RHS blocks are
            # zero. One `kkt_solve_many` call per jacrev amortises the
            # FFI / executor-pin overhead across the N cotangents.
            rhs_flat = np.zeros((N, kkt_dim_), dtype=np.float64)
            rhs_flat[:, :n_x_] = v_np
            u_flat = np.asarray(
                solver.kkt_solve_many(rhs_flat.reshape(-1), N),
                dtype=np.float64,
            ).reshape(N, kkt_dim_)
            return n_x_, n_s_, n_y_c_, n_y_d_, u_flat

        n_x, n_s, n_y_c, n_y_d, u_mat = jp._run_pinned(_do_kkt)
        u_x_batch = u_mat[:, :n_x].copy()
        y_c_off = n_x + n_s
        y_d_off = y_c_off + n_y_c
        u_y_c_batch = u_mat[:, y_c_off : y_c_off + n_y_c]
        u_y_d_batch = u_mat[:, y_d_off : y_d_off + n_y_d]

        # Scatter (u_y_c, u_y_d) back to user-g order via the cl == cu
        # mask. c_map and d_map preserve user order within each group,
        # which matches pounce-nlp's classification (tnlp_adapter.rs:388-413).
        u_g_batch = np.zeros((N, m), dtype=np.float64)
        if m > 0:
            cl_arr = np.asarray(jp._cl_for_classify, dtype=np.float64)
            cu_arr = np.asarray(jp._cu_for_classify, dtype=np.float64)
            is_eq = cl_arr == cu_arr
            c_idx = np.flatnonzero(is_eq)
            d_idx = np.flatnonzero(~is_eq)
            u_g_batch[:, c_idx] = u_y_c_batch
            u_g_batch[:, d_idx] = u_y_d_batch
        if squeeze:
            return u_x_batch[0], u_g_batch[0]
        return u_x_batch, u_g_batch

    # vmap_method="broadcast_all" lets JAX hand us a batched RHS in a
    # single callback dispatch, which we fan out on the Rust side via
    # ``Solver.kkt_solve_many`` against the held LDLᵀ factor (pounce#77
    # follow-up). The host_call detects the leading batch axis on ``v``
    # and packs an ``(N, kkt_dim)`` RHS matrix. Critical for
    # ``jax.jacrev``, which vmaps the bwd over the N cotangents — under
    # the previous ``vmap_method="sequential"`` that was N separate
    # cross-thread ``pure_callback`` round-trips per jacrev call.
    return jax.pure_callback(
        host_call, result_shapes, solver_id, v, vmap_method="broadcast_all",
    )


def _bwd_batched_factor_reuse(
    f: Callable,
    g: Callable | None,
    n: int,
    m: int,
    jp: "JaxProblem",
    B: int,
    p_batch,
    x_star_batch,
    lam_batch,
    stacked_sid,
    cot_x_batch,
):
    """k_aug-style batched VJP composing pounce#76 (A)+(B): one stacked
    :func:`Solver.kkt_solve` over the held stacked LDLᵀ factor, then
    a per-block contraction with ``∂²L/∂x∂p`` and ``∂g/∂p``.

    Why this beats the per-element dense path. The stacked IPM held one
    block-diagonal compound KKT factor after the forward solve. The
    per-element dense bwd assembles a fresh ``(n+m) × (n+m)`` block per
    batch element and dispatches B independent ``jnp.linalg.solve``
    calls; the factor-reuse stacked bwd packs a single block-major RHS
    and calls ``Solver.kkt_solve`` *once* on the held factor — same
    flops scaling in B (one block-diagonal back-solve = B per-block
    back-solves modulo permutation overhead), but with one Rust
    crossing and no JAX reassembly. For modest per-block ``n`` and
    larger ``B`` this is the configuration the issue's "(A)+(B)
    together" point was after.

    The per-block ``dgradL_dp`` / ``dg_dp`` are still autodiff over the
    user-supplied ``f`` / ``g`` callables — they depend on how ``f``
    and ``g`` were *written*, not on the solve, so the IPM can't
    produce them. We vmap that work across the batch.
    """
    # One stacked back-solve. Returns (u_x_batch (B, n), u_g_batch (B, m))
    # already de-interleaved into per-block rows and scattered back into
    # user g-order per block.
    u_x_batch, u_g_batch = _kkt_backsolve_batched_pure_callback(
        jp, B, stacked_sid, cot_x_batch, n, m,
    )

    def per_block(p_k, x_star_k, lam_k, u_x_k, u_g_k):
        def lagrangian(x, p_):
            base = f(x, p_)
            if g is not None and m > 0:
                base = base + jnp.dot(lam_k, g(x, p_))
            return base

        grad_L_of_p = lambda p_: jax.grad(lagrangian, argnums=0)(x_star_k, p_)
        dgradL_dp = jax.jacrev(grad_L_of_p)(p_k)
        dL_dp_k = -jnp.tensordot(u_x_k, dgradL_dp, axes=1)
        if g is not None and m > 0:
            dg_dp = jax.jacrev(lambda p_: g(x_star_k, p_))(p_k)
            dL_dp_k = dL_dp_k - jnp.tensordot(u_g_k, dg_dp, axes=1)
        return dL_dp_k

    return jax.vmap(per_block)(
        p_batch, x_star_batch, lam_batch, u_x_batch, u_g_batch,
    )


def _kkt_backsolve_batched_pure_callback(
    jp: "JaxProblem",
    B: int,
    sid: jnp.ndarray,
    cot_x_batch: jnp.ndarray,
    n: int,
    m: int,
):
    """Pure-callback for the stacked (A)+(B) back-solve.

    Looks up the stacked Solver registered by :meth:`_host_batched_solve`,
    packs the (B, n) cotangent batch into the stacked compound RHS at
    the x-block (block-major flatten), back-solves once against the
    held stacked LDLᵀ factor, then de-interleaves the result:

    * ``u_x_batch`` from the stacked x-block, reshaped ``(B, n)``.
    * ``u_y_c_batch`` from the stacked y_c-block, reshaped ``(B, n_c)``.
    * ``u_y_d_batch`` from the stacked y_d-block, reshaped ``(B, n_d)``.

    The stacked y_c / y_d sub-blocks are block-major because the
    stacked problem's constraints are block-major (`[g^(1), g^(2),
    ..., g^(B)]`) and ``cl``/``cu`` are tiled per-block, so the
    classified ``c_map`` / ``d_map`` keeps block-k's equality
    multipliers contiguous before block-(k+1)'s.

    Then scatters per block back to user g-order via the per-block
    ``cl == cu`` mask cached on the parent JaxProblem.

    Returns ``(u_x_batch (B, n), u_g_batch (B, m))``.
    """
    result_shapes = (
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, m), jnp.float64),
    )

    def host_call(sid_h, cot_h):
        # Under ``vmap_method="broadcast_all"`` (pounce#77 follow-up),
        # both inputs gain a leading batch axis when this callback is
        # invoked through a vmap — most importantly ``jax.jacrev`` over
        # ``batched_solve``, which fans out N = B*n cotangents.
        # Accept either:
        #   - unbatched: ``sid_h`` scalar,   ``cot_h`` shape ``(B, n)``
        #   - batched:   ``sid_h`` (N,),     ``cot_h`` shape ``(N, B, n)``
        # solver_id is constant across cotangents in a jacrev (one
        # primal, N varying cotangents); pull the first.
        sid_arr = np.asarray(sid_h)
        sid_int = int(sid_arr.reshape(-1)[0])
        solver = jp._lookup_solver(sid_int)
        if solver is None:
            raise RuntimeError(
                f"pounce.jax: missing stacked Solver for batched backward "
                f"(id={sid_int}). The stacked factor was evicted from the "
                "LRU registry. Bump `_solver_registry_capacity`, run grads "
                "closer to the fwd, or use `factor_reuse=False`."
            )
        cot_np = np.asarray(cot_h, dtype=np.float64)
        squeeze = cot_np.ndim == 2  # (B, n) unbatched; (N, B, n) batched
        if squeeze:
            cot_np = cot_np[None, ...]
        N = cot_np.shape[0]

        # Pin all PySolver access to the executor thread that built
        # the stacked solver (pounce#77) so XLA-thread pure_callback
        # dispatch in training loops doesn't trip PyO3 unsendable.
        def _do_kkt():
            dims = solver.block_dims
            if dims is None:
                raise RuntimeError(
                    "pounce.jax: factor-reuse batched backward "
                    "requires a converged IPM factor on the stacked "
                    "solve, but the stacked IPM did not produce one. "
                    "Either tighten the stacked solve (loosen `tol`, "
                    "supply a better `x0`, rescale the problem), or "
                    "rebuild the `JaxProblem` with `factor_reuse="
                    "False` to fall back to the dense per-element JAX "
                    "backward."
                )
            n_x_ = dims[0]
            n_s_ = dims[1]
            n_y_c_ = dims[2]
            n_y_d_ = dims[3]
            kkt_dim_ = solver.kkt_dim
            # The stacked NLP has n_x = B * n_per_block by construction
            # (no fixed-variable treatment is exposed on the batched path).
            if n_x_ != B * n:
                raise RuntimeError(
                    f"pounce.jax: stacked solver n_x={n_x_} != "
                    f"B*n={B * n} (B={B}, n={n}). Internal invariant "
                    "violated — please file an issue."
                )
            # Block-major flatten per cotangent: each row k of
            # ``rhs_flat`` is ``[cot^(1)_k; cot^(2)_k; ...; cot^(B)_k;
            # 0_s; 0_yc; 0_yd; 0_zl; 0_zu; 0_vl; 0_vu]``. One
            # ``kkt_solve_many`` against the held stacked LDLᵀ factor
            # amortises FFI / executor-pin overhead across the N
            # cotangents from ``jax.jacrev``.
            rhs_flat = np.zeros((N, kkt_dim_), dtype=np.float64)
            rhs_flat[:, :n_x_] = cot_np.reshape(N, B * n)
            u_flat = np.asarray(
                solver.kkt_solve_many(rhs_flat.reshape(-1), N),
                dtype=np.float64,
            ).reshape(N, kkt_dim_)
            return n_x_, n_s_, n_y_c_, n_y_d_, u_flat

        n_x, n_s, n_y_c, n_y_d, u_mat = jp._run_pinned(_do_kkt)

        u_x_batch = u_mat[:, :n_x].reshape(N, B, n).copy()

        u_g_batch = np.zeros((N, B, m), dtype=np.float64)
        if m > 0:
            cl_arr = jp._cl_for_classify
            cu_arr = jp._cu_for_classify
            is_eq = cl_arr == cu_arr
            n_c_per = int(np.sum(is_eq))
            n_d_per = m - n_c_per
            y_c_off = n_x + n_s
            y_d_off = y_c_off + n_y_c
            if n_c_per > 0:
                u_y_c_batch = u_mat[:, y_c_off : y_c_off + n_y_c].reshape(
                    N, B, n_c_per
                )
                c_idx = np.flatnonzero(is_eq)
                u_g_batch[:, :, c_idx] = u_y_c_batch
            if n_d_per > 0:
                u_y_d_batch = u_mat[:, y_d_off : y_d_off + n_y_d].reshape(
                    N, B, n_d_per
                )
                d_idx = np.flatnonzero(~is_eq)
                u_g_batch[:, :, d_idx] = u_y_d_batch
        if squeeze:
            return u_x_batch[0], u_g_batch[0]
        return u_x_batch, u_g_batch

    return jax.pure_callback(
        host_call, result_shapes, sid, cot_x_batch, vmap_method="broadcast_all",
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
        masking. Set ``False`` to fall back to the dense JAX path
        (useful for higher-order differentiation, since the dense path
        stays inside JAX and is itself differentiable).

        **When to pick which (pounce#77 follow-up).** The right
        setting depends on two independent axes: the AD shape (how
        many cotangents the backward sees per forward) and the
        per-block KKT size ``n+m``.

        ===============================  ==========================  ==========================
        ``n+m``                          single cotangent            many cotangents
                                         (``value_and_grad``, grad,  (``jacrev``, ``jacfwd``)
                                          single ``jvp``/``vjp``)
        ===============================  ==========================  ==========================
        ≲ 100                            either (≈ tie)              ``factor_reuse=False``
        ~100 – ~5000                     ``factor_reuse=True``       ``factor_reuse=False``
        ≳ 10000                          ``factor_reuse=True``       *no good option today*
                                                                     (MINRES/GMRES bwd not
                                                                     implemented — file an
                                                                     issue if you need it)
        ===============================  ==========================  ==========================

        Mechanism. The dense path (``factor_reuse=False``) assembles
        ``(n+m) × (n+m)`` per block at pounce's converged
        ``(x*, λ*, μ_l*, μ_u*)`` (saved in the custom_vjp residual)
        and lets XLA JIT fuse the per-block ``jnp.linalg.solve``.
        Crucially, XLA shares the LAPACK factorization across every
        cotangent in a ``jacrev`` / ``jacfwd``, so the per-cotangent
        cost is one vectorized BLAS back-substitution — that's why
        the dense path wins at many-cotangent shapes until the dense
        matrix runs out of RAM (at f64, ``(n+m)² × 8`` bytes per
        block; ≳ n+m ~ 10000 on typical hardware).

        The reuse path (``factor_reuse=True``) skips the dense
        factorization entirely and back-solves through pounce's
        sparse LDLᵀ via ``Solver.kkt_solve_many``. With no factor to
        pay for, it wins outright on single-cotangent shapes as soon
        as the dense factor costs anything (``n+m`` above a few
        hundred). On many-cotangent shapes the sparse back-subs are
        scalar / cache-unfriendly and the vectorized dense path
        beats them even at large ``n+m``, until the dense matrix
        won't fit.

        Above ``n+m`` ≳ 10000 with a ``jacrev`` / ``jacfwd`` shape,
        neither path is good: dense exhausts memory, sparse LDLᵀ on
        N cotangents runs into ``O(N · kkt_dim)`` sequential
        back-subs. The matrix-free MINRES/GMRES bwd that would cover
        this regime isn't implemented yet — JAX has no sparse direct
        solver (``jax.scipy.sparse.linalg`` exposes only ``cg`` /
        ``bicgstab`` / ``gmres``; ``jnp.linalg.solve`` does not
        accept ``BCOO``; XLA does not detect numerical zeros and
        switch algorithms). A ``UserWarning`` is emitted at
        construction when ``n + m > 10000`` to flag this gap.

        Always benchmark wrapped in ``jax.jit`` — eager-mode numbers
        run 5–6× slower than the same code under ``jit`` and don't
        reflect a real training-loop shape.

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
        # Above this per-block KKT size, neither bwd path is great:
        # factor_reuse=False materializes a dense (n+m)x(n+m) KKT per
        # block (O((n+m)^3) and O((n+m)^2) memory), and factor_reuse=
        # True still hops FFI per cotangent so jacrev/jacfwd loses the
        # XLA factor-sharing the dense path gets. The right pick
        # depends on AD shape — see the JaxProblem.factor_reuse
        # docstring for the regime table. The matrix-free MINRES/
        # GMRES bwd that would close the gap for many-cotangent AD at
        # this scale is not implemented (pounce#77).
        if (n + m) > 10000:
            warnings.warn(
                f"JaxProblem with n+m={n + m} > 10000: both backward "
                "paths have known scaling limits at this size. "
                "factor_reuse=False scales O((n+m)^3) per block in time "
                "and O((n+m)^2) in memory; factor_reuse=True is FFI-"
                "bound per cotangent so jax.jacrev / jax.jacfwd lose "
                "the LAPACK factor sharing that helps the dense path. "
                "Pick based on AD shape: single cotangent (value_and_"
                "grad) -> factor_reuse=True; many cotangents (jacrev/"
                "jacfwd) -> factor_reuse=False as the lesser evil. "
                "See the JaxProblem.factor_reuse docstring for the "
                "full regime table; file pounce#77 if you need the "
                "matrix-free MINRES/GMRES bwd.",
                UserWarning,
                stacklevel=2,
            )
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
        # JIT closures + per-process runtime state (lock / TLS / executor
        # / registry / id counter) live behind helpers so pickle
        # round-trips can drop and rebuild them — see
        # :meth:`__getstate__` / :meth:`__setstate__`.
        self._build_jit_closures()
        self._init_runtime_state()

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

    # ----- internal: rebuildable per-process state -----

    def _build_jit_closures(self) -> None:
        """(Re)build the JIT'd JAX callables from ``self._f`` /
        ``self._g``. Called from ``__init__`` once and from
        ``__setstate__`` after a pickle round-trip — JAX JIT closures
        don't survive pickle, but ``self._f`` / ``self._g`` do (as long
        as the user's callables are picklable, which is on them)."""
        f = self._f
        g = self._g
        m = self._m
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

    def _init_runtime_state(self) -> None:
        """(Re)create per-process mutable state: bwd-registry id
        counter and storage, registry lock, per-thread Problem cache,
        and the dedicated single-thread executor that pins every
        ``pure_callback`` host_call to one thread (pounce#77).

        Called from ``__init__`` and again from ``__setstate__`` — both
        arrive at a fresh process/thread context where every one of
        these has to start empty. Pickling drops them on the sending
        side (none are picklable: ``threading.Lock``,
        ``threading.local``, ``ThreadPoolExecutor``'s internal
        ``SimpleQueue``, and any held :class:`pounce.Solver` which
        carries unsendable Rust state).

        Dedicated single-thread executor for every host-side solve and
        bwd dispatch (pounce#77). PySolver is
        ``#[pyclass(unsendable)]`` because the inner ``RustSolver``
        holds ``Rc<RefCell<dyn TNLP>>``; PyO3 panics if a PySolver is
        touched from any thread other than the one that constructed
        it. JAX's ``pure_callback`` dispatches ``host_call`` on XLA
        worker threads whose identity is unstable across calls — so
        the per-instance ``threading.local`` cache of
        ``(JaxNlp, Problem)`` would miss almost every dispatch under
        ``jit`` (observed: 8 unique thread ids over 10 dispatches).
        Routing every host_call body through this single-worker pool
        gives a stable thread, which means the TLS cache hits by
        construction and we pay the build cost once per problem (per
        batch size B for the stacked path) for the whole process
        lifetime. Single code path for ``factor_reuse`` ∈ {True, False}.

        The one documented exception is :meth:`vmap_solve_parallel` /
        :func:`_pure_callback_parallel_solve`: that API's whole point
        is host-side B-way concurrency, so it spins up its own
        ``ThreadPoolExecutor`` inside the host_call and intentionally
        builds one Problem per worker thread. That path calls
        :meth:`_host_solve` with ``register=False`` to opt out of
        pinning.
        """
        self._solver_registry: "OrderedDict[int, Solver]" = OrderedDict()
        self._solver_id_counter = itertools.count(1)
        # Lock guards concurrent register/lookup against
        # vmap_solve_parallel worker threads.
        self._registry_lock = threading.Lock()
        # Per-thread (obj, Problem) cache. Built lazily on first solve
        # from the calling thread.
        self._tls = threading.local()
        self._factor_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"pounce-jp-factor-{id(self)}",
        )

    # Attributes that don't survive a process boundary: drop on
    # __getstate__, rebuild on __setstate__. Kept as a single source of
    # truth so the two hooks can't drift.
    _PICKLE_DROP = (
        "_f_jit", "_grad_f_jit", "_g_jit", "_jac_g_jit", "_hess_lag_jit",
        "_registry_lock", "_tls", "_factor_executor",
        "_solver_registry", "_solver_id_counter",
    )

    def __getstate__(self):
        """Pickle support (pounce#77 follow-up): drop per-process
        runtime state — JAX JIT closures (not picklable), the
        ``threading.Lock`` / ``threading.local`` / ``ThreadPoolExecutor``
        from the bwd factor-reuse path, the held LDLᵀ factor registry
        (held :class:`pounce.Solver` instances are unsendable Rust
        state), and the id counter (``itertools.count`` pickles with a
        deprecation warning on 3.12+).

        :meth:`__setstate__` rebuilds all of them. The JIT closures are
        rebuilt from ``self._f`` / ``self._g``; the user's callables
        themselves need to be picklable (module-level / cloudpickle-
        compatible) for the round-trip to work at all.

        The sparsity-pattern arrays (``_jac_rows`` etc.) *are* pickled,
        so the receiving side does not redo the one-shot probe.
        """
        state = self.__dict__.copy()
        for k in self._PICKLE_DROP:
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_jit_closures()
        self._init_runtime_state()

    # ----- internal: per-thread cached Problem -----

    def _run_pinned(self, fn):
        """Run ``fn()`` on the dedicated single-thread executor (pounce#77).

        The default for every ``pure_callback`` host_call body, both
        ``factor_reuse=True`` and ``factor_reuse=False``: it pins the
        TLS Problem cache to a stable thread (so it never misses
        under XLA's worker-thread fanout) and keeps any held
        :class:`pounce.Solver` on the thread that constructed it
        (PySolver is ``#[pyclass(unsendable)]``). The one exception
        is :meth:`vmap_solve_parallel`, which intentionally calls
        ``_host_solve(..., register=False)`` inline on its own
        worker pool so its B-way concurrency isn't serialized
        through this single thread.
        """
        return self._factor_executor.submit(fn).result()

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

    def _thread_stacked_problem_warm(self, B: int) -> tuple[_StackedJaxNlp, Problem]:
        """Per-thread LRU of *warm-started* stacked Problems keyed by
        batch size B (pounce#78). Mirrors :meth:`_thread_stacked_problem`
        but the cached Problem has ``warm_start_init_point="yes"`` set
        once at build time, so :meth:`_host_batched_solve_warm` can pass
        per-block ``(lam, zL, zU)`` through to the underlying Solver in
        the same single-Rust-crossing shape as the cold batched path.

        Warm- and cold-cached pairs live in separate caches so a training
        loop that interleaves warm and cold batched solves doesn't flip
        the warm-start option mid-life (which isn't reliable across the
        C boundary, same constraint that motivates :meth:`_thread_problem_warm`).
        """
        cache = getattr(self._tls, "stacked_warm", None)
        if cache is None:
            cache = OrderedDict()
            self._tls.stacked_warm = cache
        if B in cache:
            cache.move_to_end(B)
            return cache[B]
        obj, prob = self._build_stacked_problem(B)
        prob.add_option("warm_start_init_point", "yes")
        cache[B] = (obj, prob)
        cache.move_to_end(B)
        while len(cache) > 4:
            cache.popitem(last=False)
        return cache[B]

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

        ``register=False`` skips the registry hand-off **and** the
        single-thread pin — used by :meth:`vmap_solve_parallel` whose
        backward is JAX-vmapped over the dense kernel (so no factor
        is ever read), and which spins up its own B-way executor
        inside the host_call. Pinning that path through the
        single-thread executor would defeat the API's whole purpose.

        ``register=True`` (the default for every differentiable
        surface) pins the solve onto :attr:`_factor_executor` so the
        TLS ``(JaxNlp, Problem)`` cache lives on a stable thread and
        never misses under XLA's worker-thread fanout (pounce#77).
        When ``factor_reuse=True`` the converged Solver is then
        registered for the bwd to back-solve against; otherwise the
        Solver is dropped inside the pinned closure so its
        unsendable Rust state never crosses thread boundaries.
        """
        do_register = register and self._factor_reuse

        def _do():
            obj, prob = self._thread_problem()
            obj._p = jnp.asarray(p_np)
            solver = Solver(prob)
            x_np, info = solver.solve(x0=np.asarray(x0_np, dtype=np.float64))
            # Register or drop the unsendable Solver while we still
            # hold the pinned thread; never let it escape.
            sid = self._register_solver(solver) if do_register else 0
            return x_np, info, sid

        x_np, info, sid = self._run_pinned(_do) if register else _do()
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
        do_register = self._factor_reuse

        def _do():
            obj, prob = self._thread_problem_warm()
            obj._p = jnp.asarray(p_np)
            solver = Solver(prob)
            x_np, info = solver.solve(
                x0=np.asarray(x0_np, dtype=np.float64),
                lagrange=np.asarray(lam_np, dtype=np.float64),
                zl=np.asarray(zL_np, dtype=np.float64),
                zu=np.asarray(zU_np, dtype=np.float64),
            )
            sid = self._register_solver(solver) if do_register else 0
            return x_np, info, sid

        # Warm-start path is always a differentiable surface entry
        # point, so pin every call (pounce#77) — TLS cache stability
        # at minimum, and the bwd's kkt_solve when factor_reuse=True.
        # Register/drop happens inside the pinned closure so the
        # unsendable Solver never crosses thread boundaries.
        x_np, info, sid = self._run_pinned(_do)
        info_out = dict(info)
        info_out["solver_id"] = sid
        return x_np, info_out

    def _host_batched_solve(self, p_batch_np: np.ndarray, x0_batch_np: np.ndarray):
        """Forward solve for the stacked block-diagonal problem
        (pounce#76 (A), composes with (B) when ``factor_reuse=True``).

        Returns per-block residuals reshaped to leading-batch axis:
        ``x_batch: (B, n)``, ``lam_batch: (B, m)``, ``mult_xL_batch /
        mult_xU_batch: (B, n)``, plus a ``stacked_sid`` int that points
        at the held stacked LDLᵀ factor in the bwd registry. The IPM
        under the hood factors a single block-diagonal compound KKT
        once and runs the barrier homotopy across all blocks together —
        i.e. one ``μ``-schedule for the whole batch.

        When ``factor_reuse=True`` the stacked Solver is registered so
        the bwd can do *one* :func:`Solver.kkt_solve` against the held
        stacked factor instead of B per-element dense JAX KKT solves.
        That's the (A)+(B) composition: one factorisation reused for
        both the forward and the per-block sensitivities. When
        ``factor_reuse=False`` we don't register (the bwd uses the
        dense per-element vmap path that doesn't consult the held
        factor).
        """
        B = p_batch_np.shape[0]
        n, m = self._n, self._m
        # Initial X is the per-block ``x0`` tiled / concatenated.
        X0 = np.asarray(x0_batch_np, dtype=np.float64).reshape(B * n)

        do_register = self._factor_reuse

        def _do():
            obj, prob = self._thread_stacked_problem(B)
            obj._P = jnp.asarray(p_batch_np)
            solver = Solver(prob)
            X_np, info = solver.solve(x0=X0)
            sid = self._register_solver(solver) if do_register else 0
            return X_np, info, sid

        # Pin every batched_solve dispatch (pounce#77) — TLS cache
        # stability at minimum, and the bwd's stacked kkt_solve when
        # factor_reuse=True. Register/drop happens inside the pinned
        # closure so the unsendable Solver never crosses thread
        # boundaries.
        X_np, info, sid = self._run_pinned(_do)
        x_batch = np.asarray(X_np, dtype=np.float64).reshape(B, n)
        lam_batch = (
            np.asarray(info["mult_g"], dtype=np.float64).reshape(B, m)
            if m > 0 else np.zeros((B, 0), dtype=np.float64)
        )
        mult_xL_batch = np.asarray(info["mult_x_L"], dtype=np.float64).reshape(B, n)
        mult_xU_batch = np.asarray(info["mult_x_U"], dtype=np.float64).reshape(B, n)
        return x_batch, lam_batch, mult_xL_batch, mult_xU_batch, sid

    def _host_batched_solve_warm(
        self,
        p_batch_np: np.ndarray,
        x0_batch_np: np.ndarray,
        lam_batch_np: np.ndarray,
        zL_batch_np: np.ndarray,
        zU_batch_np: np.ndarray,
    ):
        """Warm-started stacked block-diagonal solve (pounce#78).

        The stacked NLP packs variables block-major
        (``[x^(1); ...; x^(B)]``) and constraints block-major
        (``[g^(1); ...; g^(B)]``), so per-block warm vectors flatten
        block-major via a simple ``reshape(-1)``:
        ``lam_stack[k*m:(k+1)*m] = lam_batch[k]``, same for
        ``zL_stack`` / ``zU_stack`` (size ``B*n`` each). One stacked
        Solver call covers the whole batch in a single barrier-homotopy
        run, mirroring the cold :meth:`_host_batched_solve` shape.

        Returns ``(x_batch, lam_batch, mult_xL_batch, mult_xU_batch, sid)``
        in the same shape as the cold path, so the warm and cold
        callback wrappers share the rest of the bwd plumbing.
        """
        B = p_batch_np.shape[0]
        n, m = self._n, self._m
        X0 = np.asarray(x0_batch_np, dtype=np.float64).reshape(B * n)
        lam_stack = (
            np.asarray(lam_batch_np, dtype=np.float64).reshape(B * m)
            if m > 0 else np.zeros(0, dtype=np.float64)
        )
        zL_stack = np.asarray(zL_batch_np, dtype=np.float64).reshape(B * n)
        zU_stack = np.asarray(zU_batch_np, dtype=np.float64).reshape(B * n)

        do_register = self._factor_reuse

        def _do():
            obj, prob = self._thread_stacked_problem_warm(B)
            obj._P = jnp.asarray(p_batch_np)
            solver = Solver(prob)
            X_np, info = solver.solve(
                x0=X0,
                lagrange=lam_stack,
                zl=zL_stack,
                zu=zU_stack,
            )
            sid = self._register_solver(solver) if do_register else 0
            return X_np, info, sid

        X_np, info, sid = self._run_pinned(_do)
        x_batch = np.asarray(X_np, dtype=np.float64).reshape(B, n)
        lam_batch = (
            np.asarray(info["mult_g"], dtype=np.float64).reshape(B, m)
            if m > 0 else np.zeros((B, 0), dtype=np.float64)
        )
        mult_xL_batch = np.asarray(info["mult_x_L"], dtype=np.float64).reshape(B, n)
        mult_xU_batch = np.asarray(info["mult_x_U"], dtype=np.float64).reshape(B, n)
        return x_batch, lam_batch, mult_xL_batch, mult_xU_batch, sid

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

    def batched_solve_with_warm(
        self,
        p_batch,
        x0,
        warm_start: tuple | None = None,
    ):
        """Warm-started stacked block-diagonal batched solve (pounce#78).

        Combines :meth:`batched_solve`'s single-Rust-crossing /
        shared-symbolic-factorisation / factor-reuse-backward shape with
        :meth:`solve_with_warm`'s per-block ``(lam, zL, zU)`` warm
        threading. Intended for differentiable constrained projection
        layers inside a training loop, where the parameter ``p`` drifts
        only slightly per step and the previous step's converged primal +
        dual state is a near-optimal seed for the whole batch.

        Parameters
        ----------
        p_batch : ``(B,) + p_shape``
            Per-block parameters.
        x0 : ``(n,)`` or ``(B, n)``
            Primal initial point; ``(n,)`` is broadcast across the batch
            to match :meth:`batched_solve`.
        warm_start : tuple ``(lam_batch, zL_batch, zU_batch)`` or None
            Per-block warm vectors of shapes ``(B, m)``, ``(B, n)``,
            ``(B, n)``. Pass ``None`` for an uninformed first call (warm
            buffers default to zeros, matching the cold-batch behaviour
            of :meth:`batched_solve` modulo the warm-start option flip).

        Returns
        -------
        ``(x_batch, (lam_batch_out, zL_batch_out, zU_batch_out))``
            Stacked primal + dual outputs reshaped per-block.
            ``x_batch`` is ``(B, n)``; the dual outputs are ``(B, m)``,
            ``(B, n)``, ``(B, n)`` so the caller can thread them straight
            into the next step's ``warm_start``.

        The custom-VJP treats the warm inputs and ``x0`` as stop-gradient
        — only ``p_batch`` gets a gradient — matching the convention
        :meth:`solve_with_warm` already uses on the single-sample path.
        """
        p_batch = jnp.asarray(p_batch)
        B = p_batch.shape[0]
        n, m = self._n, self._m
        x0_arr = jnp.asarray(x0)
        if x0_arr.ndim == 1:
            x0_arr = jnp.broadcast_to(x0_arr, (B, n))
        if warm_start is None:
            lam_warm = jnp.zeros((B, m), dtype=jnp.float64)
            zL_warm = jnp.zeros((B, n), dtype=jnp.float64)
            zU_warm = jnp.zeros((B, n), dtype=jnp.float64)
        else:
            lam_warm, zL_warm, zU_warm = warm_start
            lam_warm = jnp.asarray(lam_warm, dtype=jnp.float64).reshape(B, m)
            zL_warm = jnp.asarray(zL_warm, dtype=jnp.float64).reshape(B, n)
            zU_warm = jnp.asarray(zU_warm, dtype=jnp.float64).reshape(B, n)
        fn = self._batched_solve_with_warm_fn(B)
        x_star, lam_out, zL_out, zU_out = fn(
            p_batch, x0_arr, lam_warm, zL_warm, zU_warm,
        )
        return x_star, (lam_out, zL_out, zU_out)

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
        """custom_vjp factory for :meth:`batched_solve` (pounce#76 (A),
        composes with (B) when ``factor_reuse=True``).

        Two bwd paths, selected by ``self._factor_reuse``:

        * ``factor_reuse=True`` ((A)+(B) composition): one
          ``Solver.kkt_solve`` against the held stacked LDLᵀ factor
          (see :func:`_bwd_batched_factor_reuse`). Per-block
          ``∂²L/∂x∂p`` and ``∂g/∂p`` are still autodiff over the user's
          ``f`` / ``g`` — those depend on how the functions were
          written, not on the solve. One Rust crossing total for the
          whole batched back-solve; the JAX work is just the per-block
          contraction.
        * ``factor_reuse=False`` ((A) only): ``jax.vmap`` of the
          per-element dense KKT back-solve. Block-diagonal coupling in
          the stacked KKT means ``∂x^(k)*/∂p^(j) = 0`` for ``k ≠ j``,
          so vmapping the single-block bwd is exact — there's no
          cross-block correction to assemble.
        """
        f, g, n, m = self._f, self._g, self._n, self._m
        cl, cu = self._cl, self._cu
        jp = self
        factor_reuse = self._factor_reuse

        @jax.custom_vjp
        def solve_fn(p_batch, x0_batch):
            x_star, *_ = _pure_callback_batched_solve(jp, B, p_batch, x0_batch)
            return x_star

        def fwd(p_batch, x0_batch):
            x_star, lam, mult_xL, mult_xU, sid = _pure_callback_batched_solve(
                jp, B, p_batch, x0_batch,
            )
            return x_star, (p_batch, x_star, lam, mult_xL, mult_xU, sid)

        def bwd_single(p, x_star, lam, mult_xL, mult_xU, v):
            return _bwd_single_kkt(
                f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v,
            )

        def bwd(residuals, cot_x_batch):
            (
                p_batch, x_star_batch, lam_batch,
                mult_xL_batch, mult_xU_batch, sid,
            ) = residuals
            if factor_reuse:
                dL_dp_batch = _bwd_batched_factor_reuse(
                    f, g, n, m, jp, B,
                    p_batch, x_star_batch, lam_batch, sid, cot_x_batch,
                )
            else:
                dL_dp_batch = jax.vmap(bwd_single)(
                    p_batch, x_star_batch, lam_batch,
                    mult_xL_batch, mult_xU_batch, cot_x_batch,
                )
            return dL_dp_batch, jnp.zeros_like(x_star_batch)

        solve_fn.defvjp(fwd, bwd)
        return solve_fn

    def _batched_solve_with_warm_fn(self, B: int):
        """custom_vjp factory for :meth:`batched_solve_with_warm` (pounce#78).

        Shares both bwd paths with :meth:`_batched_solve_fn`:

        * ``factor_reuse=True`` — :func:`_bwd_batched_factor_reuse` over
          the held stacked LDLᵀ factor. The warm-start path doesn't
          change the converged KKT structure (same primal stationarity
          + active-set encoding), so the same one-stacked-back-solve
          machinery applies unmodified.
        * ``factor_reuse=False`` — ``jax.vmap`` of the per-element dense
          KKT bwd; block-diagonal coupling makes vmapping exact.

        Warm inputs and ``x0`` are treated as stop-gradient — only
        ``p_batch`` gets a gradient — matching :meth:`solve_with_warm`'s
        convention on the single-sample warm path.
        """
        f, g, n, m = self._f, self._g, self._n, self._m
        cl, cu = self._cl, self._cu
        jp = self
        factor_reuse = self._factor_reuse

        @jax.custom_vjp
        def solve_fn(p_batch, x0_batch, lam_warm, zL_warm, zU_warm):
            x_star, lam_out, zL_out, zU_out, _sid = (
                _pure_callback_batched_warm_solve(
                    jp, B, p_batch, x0_batch, lam_warm, zL_warm, zU_warm,
                )
            )
            return x_star, lam_out, zL_out, zU_out

        def fwd(p_batch, x0_batch, lam_warm, zL_warm, zU_warm):
            x_star, lam_out, zL_out, zU_out, sid = (
                _pure_callback_batched_warm_solve(
                    jp, B, p_batch, x0_batch, lam_warm, zL_warm, zU_warm,
                )
            )
            return (
                (x_star, lam_out, zL_out, zU_out),
                (p_batch, x_star, lam_out, zL_out, zU_out, sid),
            )

        def bwd_single(p, x_star, lam, mult_xL, mult_xU, v):
            return _bwd_single_kkt(
                f, g, n, m, cl, cu, p, x_star, lam, mult_xL, mult_xU, v,
            )

        def bwd(residuals, cotangents):
            (
                p_batch, x_star_batch, lam_batch,
                mult_xL_batch, mult_xU_batch, sid,
            ) = residuals
            # Only the x* cotangent is used — dual cotangents are
            # dropped, same as :meth:`solve_with_warm`'s single-sample
            # bwd does (the user almost always pulls grad only through
            # ``x_star``, and the dual outputs are there so the next
            # step can warm-start from them).
            cot_x_batch = cotangents[0]
            if factor_reuse:
                dL_dp_batch = _bwd_batched_factor_reuse(
                    f, g, n, m, jp, B,
                    p_batch, x_star_batch, lam_batch, sid, cot_x_batch,
                )
            else:
                dL_dp_batch = jax.vmap(bwd_single)(
                    p_batch, x_star_batch, lam_batch,
                    mult_xL_batch, mult_xU_batch, cot_x_batch,
                )
            return (
                dL_dp_batch,
                jnp.zeros_like(x_star_batch),
                jnp.zeros((B, m), dtype=jnp.float64),
                jnp.zeros((B, n), dtype=jnp.float64),
                jnp.zeros((B, n), dtype=jnp.float64),
            )

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

    ``stacked_sid`` is a scalar int64 — the id of the held stacked
    LDLᵀ factor in the JaxProblem's bwd registry (or 0 when
    ``factor_reuse=False``). The bwd reads it back via the residual
    pytree to do one stacked ``kkt_solve`` against the held factor
    instead of B per-element dense KKT solves.
    """
    n, m = jp._n, jp._m
    result_shapes = (
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, m), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.int64),
    )

    def host_call(p_h, x0_h):
        return jp._host_batched_solve(np.asarray(p_h), np.asarray(x0_h))

    return jax.pure_callback(host_call, result_shapes, p_batch, x0_batch)


def _pure_callback_batched_warm_solve(
    jp: JaxProblem,
    B: int,
    p_batch,
    x0_batch,
    lam_warm,
    zL_warm,
    zU_warm,
):
    """Host-side dispatch for the warm-started stacked solve (pounce#78).

    Wraps :meth:`JaxProblem._host_batched_solve_warm`. Output shapes
    match :func:`_pure_callback_batched_solve` so the bwd plumbing is
    identical — the only difference upstream is that this surface
    threads per-block ``(lam, zL, zU)`` into the stacked solver.
    """
    n, m = jp._n, jp._m
    result_shapes = (
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, m), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((B, n), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.int64),
    )

    def host_call(p_h, x0_h, lam_h, zL_h, zU_h):
        return jp._host_batched_solve_warm(
            np.asarray(p_h), np.asarray(x0_h),
            np.asarray(lam_h), np.asarray(zL_h), np.asarray(zU_h),
        )

    return jax.pure_callback(
        host_call, result_shapes,
        p_batch, x0_batch, lam_warm, zL_warm, zU_warm,
    )


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
