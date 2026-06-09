"""Build a :class:`pounce.Problem` from PyTorch-traced ``f(x)`` and
``g(x)`` (pounce#109).

PyTorch frontend mirror of :mod:`pounce.jax._build`. The user supplies
the problem in mathematical form and we derive:

* ``gradient`` = ``torch.func.grad(f)``
* ``jacobian`` = derivative of ``g`` projected onto the precomputed
  sparsity pattern of nonzero entries
* ``hessian`` = Hessian of the Lagrangian
  ``L(x, λ, σ) = σ f(x) + λᵀ g(x)``

``torch.func`` (functorch, merged into core) mirrors JAX's functional AD
API, so the port is near-mechanical: ``jax.grad`` → ``torch.func.grad``,
``jax.jacrev`` / ``jax.jacfwd`` → ``torch.func.jacrev`` / ``jacfwd``,
``jax.hessian`` → ``torch.func.hessian``, ``jax.jvp`` →
``torch.func.jvp``, ``jax.vmap`` → ``torch.func.vmap``.

There are two ways the Jacobian / Hessian get computed per evaluation,
selected by the ``sparse`` flag on :func:`from_torch`:

* **dense (default).** ``jacrev`` / ``jacfwd`` / ``hessian`` produce the
  full dense matrix, which we then slice to the nonzeros. The *direction*
  (``jacrev`` vs ``jacfwd``) is chosen to minimise the number of AD
  passes: ``jacrev`` costs ~``m`` passes, ``jacfwd`` ~``n``, so we pick
  ``jacfwd`` when ``n < m`` (issue #83, option A).

* **compressed (``sparse=True``).** CPR-style colored AD (issue #83,
  option B). Structurally-orthogonal columns are colored, one JVP/HVP is
  taken per color (``k ≪ n`` colors), ``vmap``'d over the seeds, and the
  result scattered back to the known nonzeros.

dtype. pounce is a double-precision solver, and the Newton / KKT solves
need float64 throughout (convergence stalls in float32). Every traced
evaluation runs on ``torch.float64`` tensors; :func:`from_torch`
validates that any user-supplied bounds are finite-or-inf float64 and the
cyipopt callbacks coerce the incoming NumPy buffers to float64. There is
no global x64 flag as in JAX — float64 is requested per tensor.

All eval methods return ``numpy.ndarray`` (via ``.numpy()``) so the Rust
bridge receives a contiguous CPU buffer; PyTorch CPU tensors are
zero-copy to/from NumPy.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np
import torch
from torch.func import grad, hessian, jacfwd, jacrev, jvp, vmap

from .._pounce import Problem
# Framework-neutral sparsity helpers shared with the JAX frontend.
from .._ad_common import (
    _color_columns,
    _detect_pattern_2d_multi,
    _detect_pattern_lower_multi,
)

# Real (CPU, float64) tensor dtype used throughout the frontend.
_DT = torch.float64

# ``torch.func`` (functorch) transforms push/pop a *process-global*
# dynamic-layer stack, which is NOT thread-safe: concurrent
# grad/jacrev/jacfwd/jvp/hessian evaluations from different threads
# corrupt the nesting ("Trying to access a forward AD level with an
# invalid index"). The parallel batched path (vmap_solve_parallel)
# dispatches solves across a ThreadPoolExecutor whose per-iteration
# cyipopt callbacks evaluate these transforms, so we serialize the AD
# evaluations with this lock. The genuine parallelism — the Rust IPM
# linear algebra — runs in ``py.allow_threads`` (GIL released, no Python,
# no lock held), so the lock only orders the (already GIL-bound) Python
# derivative callbacks. In the single-threaded path the lock is
# uncontended and costs only an uncontended acquire.
_FUNC_LOCK = threading.RLock()


def _t(a) -> torch.Tensor:
    """Coerce array-like to a contiguous float64 CPU tensor."""
    if isinstance(a, torch.Tensor):
        return a.to(dtype=_DT)
    return torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=_DT)


def _to_np(a) -> np.ndarray:
    """Detach a tensor (or array-like) to a float64 NumPy array."""
    if isinstance(a, torch.Tensor):
        return a.detach().to(dtype=_DT).cpu().numpy().astype(np.float64)
    return np.asarray(a, dtype=np.float64)


def _seed_matrix(colors: np.ndarray, num_colors: int, n: int) -> torch.Tensor:
    """``(num_colors, n)`` 0/1 seed matrix: row ``c`` selects every
    column assigned color ``c``. Seeding a JVP/HVP with row ``c`` sums
    the structurally-orthogonal columns of that color."""
    S = np.zeros((num_colors, int(n)), dtype=np.float64)
    S[colors, np.arange(int(n))] = 1.0
    return torch.as_tensor(S, dtype=_DT)


class _TorchProblem:
    """Cyipopt-shaped problem object backed by ``torch.func``-AD callables."""

    def __init__(
        self,
        f: Callable,
        g: Callable | None,
        n: int,
        m: int,
        seed: int = 0,
        sparse: bool = False,
        n_probes: int = 1,
    ):
        self._f = f
        self._grad_f = grad(f)
        self._n = n
        self._m = m
        self._sparse = sparse

        # --- dense AD callables (always built; used for the probe and,
        # when sparse=False, for per-eval Jacobian/Hessian) ---
        if g is not None and m > 0:
            self._g = g
            # Option A: jacrev costs ~m passes, jacfwd ~n. Pick the
            # cheaper direction for the dense path.
            self._jac_g_dense = jacfwd(g) if n < m else jacrev(g)

            def lagrangian(x, lam, sigma):
                return sigma * f(x) + torch.dot(lam, g(x))

            self._lagrangian = lagrangian
            self._hess_lag_dense = hessian(lagrangian, argnums=0)
        else:
            self._g = None
            self._jac_g_dense = None

            def lagrangian_unc(x, sigma):
                return sigma * f(x)

            self._lagrangian = lagrangian_unc
            self._hess_lag_dense = hessian(lagrangian_unc, argnums=0)

        # --- detect sparsity from a union of random probes ---
        # The probe evaluates torch.func transforms, which share a
        # process-global layer stack — guard with _FUNC_LOCK so workers
        # building _TorchProblems concurrently (vmap_solve_parallel) don't
        # corrupt each other's AD level nesting.
        rng = np.random.default_rng(seed)
        n_probes = max(1, int(n_probes))
        x_probes = [_t(rng.standard_normal(n)) for _ in range(n_probes)]
        with _FUNC_LOCK:
            if m > 0:
                lam_probes = [_t(rng.standard_normal(m)) for _ in range(n_probes)]
                jac_denses = [_to_np(self._jac_g_dense(xp)) for xp in x_probes]
                self._jac_rows, self._jac_cols = _detect_pattern_2d_multi(jac_denses)
                hess_denses = [
                    _to_np(self._hess_lag_dense(xp, lp, _t(1.0)))
                    for xp, lp in zip(x_probes, lam_probes)
                ]
            else:
                self._jac_rows = np.zeros(0, dtype=np.int64)
                self._jac_cols = np.zeros(0, dtype=np.int64)
                hess_denses = [
                    _to_np(self._hess_lag_dense(xp, _t(1.0))) for xp in x_probes
                ]
        self._hess_rows, self._hess_cols = _detect_pattern_lower_multi(hess_denses)

        # --- compressed (colored) AD callables, when requested ---
        if sparse:
            self._build_compressed()

    def _build_compressed(self) -> None:
        """Build the colored JVP (Jacobian) and HVP (Hessian) callables
        and the scatter indices that map compressed columns back to the
        stored ``(rows, cols)`` nonzeros (issue #83, option B)."""
        n = self._n

        # Jacobian: color columns of the (m, n) pattern, one JVP per
        # color. For nonzero (i, j) the value lives at
        # compressed[color[j], i].
        if self._m > 0:
            jac_colors, k_jac = _color_columns(self._jac_rows, self._jac_cols, n)
            S_jac = _seed_matrix(jac_colors, k_jac, n)
            self._jac_seed_cols = jac_colors[self._jac_cols]
            g = self._g

            def jac_compressed(x):
                # vmap one forward-mode JVP per seed column → (k, m).
                return vmap(lambda s: jvp(g, (x,), (s,))[1])(S_jac)

            self._jac_compressed = jac_compressed
        else:
            self._jac_seed_cols = np.zeros(0, dtype=np.int64)
            self._jac_compressed = None

        # Hessian: symmetric, so color the *full* pattern (both
        # triangles) — a column's compressed value sums contributions
        # from every row, including the upper triangle we don't store.
        full_rows = np.concatenate([self._hess_rows, self._hess_cols])
        full_cols = np.concatenate([self._hess_cols, self._hess_rows])
        hess_colors, k_hess = _color_columns(full_rows, full_cols, n)
        S_hess = _seed_matrix(hess_colors, k_hess, n)
        self._hess_seed_cols = hess_colors[self._hess_cols]

        # HVP via jvp of the Lagrangian gradient: H @ s. (k, n).
        lag = self._lagrangian
        grad_L = grad(lag, argnums=0)
        if self._m > 0:
            def hess_compressed(x, lam, sigma):
                return vmap(
                    lambda s: jvp(
                        lambda xx: grad_L(xx, lam, sigma), (x,), (s,)
                    )[1]
                )(S_hess)
        else:
            def hess_compressed(x, sigma):
                return vmap(
                    lambda s: jvp(lambda xx: grad_L(xx, sigma), (x,), (s,))[1]
                )(S_hess)

        self._hess_compressed = hess_compressed

    # --- cyipopt-shaped methods ---

    def objective(self, x):
        return float(self._f(_t(x)))

    def gradient(self, x):
        with _FUNC_LOCK:
            return _to_np(self._grad_f(_t(x)))

    def constraints(self, x):
        return _to_np(self._g(_t(x)))

    def jacobianstructure(self):
        return (self._jac_rows, self._jac_cols)

    def jacobian(self, x):
        with _FUNC_LOCK:
            if self._sparse:
                # Compressed (k, m); scatter back: J[rows, cols] lives at
                # comp[color(cols), rows].
                comp = _to_np(self._jac_compressed(_t(x)))
                return comp[self._jac_seed_cols, self._jac_rows]
            J = _to_np(self._jac_g_dense(_t(x)))
            return J[self._jac_rows, self._jac_cols]

    def hessianstructure(self):
        return (self._hess_rows, self._hess_cols)

    def hessian(self, x, lam, obj_factor):
        sigma = _t(float(obj_factor))
        with _FUNC_LOCK:
            if self._sparse:
                # Compressed (k, n); H[r, c] lives at comp[color(c), r].
                if self._m > 0:
                    comp = _to_np(self._hess_compressed(_t(x), _t(lam), sigma))
                else:
                    comp = _to_np(self._hess_compressed(_t(x), sigma))
                return comp[self._hess_seed_cols, self._hess_rows]
            if self._m > 0:
                H = _to_np(self._hess_lag_dense(_t(x), _t(lam), sigma))
            else:
                H = _to_np(self._hess_lag_dense(_t(x), sigma))
            return H[self._hess_rows, self._hess_cols]


def from_torch(
    f: Callable,
    g: Callable | None = None,
    *,
    n: int,
    m: int = 0,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    seed: int = 0,
    sparse: bool = False,
    n_probes: int | None = None,
) -> Problem:
    """Build a pounce :class:`Problem` from PyTorch-traced functions.

    PyTorch mirror of :func:`pounce.jax.from_jax`.

    Parameters
    ----------
    f : callable
        Objective. Takes a 1-D float64 tensor of length ``n`` and returns
        a scalar tensor. Must be traceable by ``torch.func``.
    g : callable or None
        Constraint function. Takes ``x`` of length ``n`` and returns a
        1-D tensor of length ``m``. Required when ``m > 0``.
    n, m : int
        Variable / constraint counts.
    lb, ub, cl, cu : array-like or None
        Variable bounds and constraint bounds; same convention as
        :class:`Problem`.
    seed : int
        Seed for the random probe(s) used to detect sparsity patterns.
    sparse : bool
        When ``True``, compute the Jacobian and Lagrangian Hessian with
        CPR-style colored AD (one JVP/HVP per color) instead of forming
        the dense matrix and slicing (issue #83). The reported structure
        is identical either way. Defaults to ``False``.
    n_probes : int or None
        Number of random probes whose nonzero patterns are unioned to
        detect sparsity. ``None`` (default) uses 1 probe for the dense
        path and 3 for ``sparse=True``.

    Returns
    -------
    Problem
        With the user object built from ``torch.func`` AD. Pass options
        via ``problem.add_option(...)`` and call ``problem.solve(x0)``.
    """
    if m > 0 and g is None:
        raise ValueError("g must be provided when m > 0")
    if n_probes is None:
        n_probes = 3 if sparse else 1
    obj = _TorchProblem(
        f=f, g=g, n=n, m=m, seed=seed, sparse=sparse, n_probes=n_probes,
    )
    return Problem(
        n=n, m=m, problem_obj=obj, lb=lb, ub=ub, cl=cl, cu=cu,
    )
