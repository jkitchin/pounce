"""Build a :class:`pounce.Problem` from JAX-traced ``f(x)`` and ``g(x)``.

The user supplies the problem in mathematical form and we derive:

* ``gradient`` = ``jax.grad(f)``
* ``jacobian`` = derivative of ``g`` projected onto the precomputed
  sparsity pattern of nonzero entries
* ``hessian`` = Hessian of the Lagrangian
  ``L(x, λ, σ) = σ f(x) + λᵀ g(x)``

There are two ways the Jacobian / Hessian get computed per evaluation,
selected by the ``sparse`` flag on :func:`from_jax`:

* **dense (default).** ``jax.jacrev`` / ``jax.jacfwd`` / ``jax.hessian``
  produce the full dense matrix, which we then slice to the nonzeros.
  The *direction* (``jacrev`` vs ``jacfwd``) is chosen to minimise the
  number of AD passes: ``jacrev`` costs ~``m`` passes, ``jacfwd`` ~``n``,
  so we pick ``jacfwd`` when ``n < m`` (issue #83, option A). The work
  and memory are still ``O(m·n)`` / ``O(n²)`` regardless of sparsity.

* **compressed (``sparse=True``).** CPR-style colored AD (issue #83,
  option B). Structurally-orthogonal columns are colored, one
  JVP/HVP is taken per color (``k ≪ n`` colors), ``vmap``'d over the
  seeds, and the result scattered back to the known nonzeros. The cost
  drops from ``O(n)`` to ``O(k)`` AD passes. This mirrors the colored
  Hessian path the Rust ``.nl`` tape already uses.

Sparsity is detected by random probes: evaluate the dense Jacobian /
Hessian at ``n_probes`` random ``x`` (and ``λ`` for the Hessian) and
record the *union* of indices where the magnitude exceeds a threshold.
A single probe is fine for the common case where the *structure*
doesn't depend on the value (any composition of smooth pointwise
operations); the union of several probes hardens against branchy /
value-dependent structure (``where`` / ``abs``), which matters more
under ``sparse=True`` because a mis-probe there corrupts the
compression seed, not just a reported nonzero. Users with truly
value-dependent sparsity should hand-roll the pattern via the
:class:`Problem` API.

All eval methods are JIT-compiled with ``jax.jit`` and return
``numpy.ndarray`` (via ``np.asarray(jnp_result)``) so the Rust bridge
receives a contiguous CPU buffer.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .._pounce import Problem
# Framework-neutral sparsity helpers shared with the PyTorch frontend
# (pounce#109). Re-exported from this module so existing
# ``from pounce.jax._build import _color_columns`` style imports (and the
# test-suite's direct use of them) keep working unchanged.
from .._ad_common import (  # noqa: F401
    _SPARSITY_EPS,
    _color_columns,
    _detect_pattern_2d,
    _detect_pattern_2d_multi,
    _detect_pattern_lower,
    _detect_pattern_lower_multi,
    _to_np,
    _union_mask,
)


def _seed_matrix(colors: np.ndarray, num_colors: int, n: int) -> jnp.ndarray:
    """``(num_colors, n)`` 0/1 seed matrix: row ``c`` selects every
    column assigned color ``c``. Seeding a JVP/HVP with row ``c`` sums
    the structurally-orthogonal columns of that color."""
    S = np.zeros((num_colors, int(n)), dtype=np.float64)
    S[colors, np.arange(int(n))] = 1.0
    return jnp.asarray(S)


class _JaxProblem:
    """Cyipopt-shaped problem object backed by JAX-AD callables."""

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
        self._f = jax.jit(f)
        self._grad_f = jax.jit(jax.grad(f))
        self._n = n
        self._m = m
        self._sparse = sparse

        # --- dense AD callables (always built; used for the probe and,
        # when sparse=False, for per-eval Jacobian/Hessian) ---
        if g is not None and m > 0:
            self._g = jax.jit(g)
            # Option A: jacrev costs ~m passes, jacfwd ~n. Pick the
            # cheaper direction for the dense path.
            jac_g = jax.jacfwd(g) if n < m else jax.jacrev(g)
            self._jac_g_dense = jax.jit(jac_g)

            def lagrangian(x, lam, sigma):
                return sigma * f(x) + jnp.dot(lam, g(x))

            self._lagrangian = lagrangian
            self._hess_lag_dense = jax.jit(jax.hessian(lagrangian, argnums=0))
        else:
            self._g = None
            self._jac_g_dense = None

            def lagrangian_unc(x, sigma):
                return sigma * f(x)

            self._lagrangian = lagrangian_unc
            self._hess_lag_dense = jax.jit(jax.hessian(lagrangian_unc, argnums=0))

        # --- detect sparsity from a union of random probes ---
        rng = np.random.default_rng(seed)
        n_probes = max(1, int(n_probes))
        x_probes = [jnp.asarray(rng.standard_normal(n)) for _ in range(n_probes)]
        if m > 0:
            lam_probes = [
                jnp.asarray(rng.standard_normal(m)) for _ in range(n_probes)
            ]
            jac_denses = [_to_np(self._jac_g_dense(xp)) for xp in x_probes]
            self._jac_rows, self._jac_cols = _detect_pattern_2d_multi(jac_denses)
            hess_denses = [
                _to_np(self._hess_lag_dense(xp, lp, 1.0))
                for xp, lp in zip(x_probes, lam_probes)
            ]
        else:
            self._jac_rows = np.zeros(0, dtype=np.int64)
            self._jac_cols = np.zeros(0, dtype=np.int64)
            hess_denses = [_to_np(self._hess_lag_dense(xp, 1.0)) for xp in x_probes]
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
        # color. J @ S has shape (m, k); for nonzero (i, j) the value
        # lives at compressed[i, color[j]].
        if self._m > 0:
            jac_colors, k_jac = _color_columns(self._jac_rows, self._jac_cols, n)
            S_jac = _seed_matrix(jac_colors, k_jac, n)
            # color of each stored nonzero's column → its compressed col.
            self._jac_seed_cols = jac_colors[self._jac_cols]
            g = self._g

            def jac_compressed(x):
                # vmap one forward-mode JVP per seed column → (k, m).
                return jax.vmap(lambda s: jax.jvp(g, (x,), (s,))[1])(S_jac)

            self._jac_compressed = jax.jit(jac_compressed)
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
        grad_L = jax.grad(lag, argnums=0)
        if self._m > 0:
            def hess_compressed(x, lam, sigma):
                return jax.vmap(
                    lambda s: jax.jvp(
                        lambda xx: grad_L(xx, lam, sigma), (x,), (s,)
                    )[1]
                )(S_hess)
        else:
            def hess_compressed(x, sigma):
                return jax.vmap(
                    lambda s: jax.jvp(lambda xx: grad_L(xx, sigma), (x,), (s,))[1]
                )(S_hess)

        self._hess_compressed = jax.jit(hess_compressed)

    # --- cyipopt-shaped methods ---

    def objective(self, x):
        return float(self._f(jnp.asarray(x)))

    def gradient(self, x):
        return _to_np(self._grad_f(jnp.asarray(x)))

    def constraints(self, x):
        return _to_np(self._g(jnp.asarray(x)))

    def jacobianstructure(self):
        return (self._jac_rows, self._jac_cols)

    def jacobian(self, x):
        if self._sparse:
            # Compressed (k, m); scatter back: J[rows, cols] lives at
            # comp[color(cols), rows].
            comp = _to_np(self._jac_compressed(jnp.asarray(x)))
            return comp[self._jac_seed_cols, self._jac_rows]
        J = _to_np(self._jac_g_dense(jnp.asarray(x)))
        return J[self._jac_rows, self._jac_cols]

    def hessianstructure(self):
        return (self._hess_rows, self._hess_cols)

    def hessian(self, x, lam, obj_factor):
        if self._sparse:
            # Compressed (k, n); H[r, c] lives at comp[color(c), r].
            if self._m > 0:
                comp = _to_np(
                    self._hess_compressed(jnp.asarray(x), jnp.asarray(lam), obj_factor)
                )
            else:
                comp = _to_np(self._hess_compressed(jnp.asarray(x), obj_factor))
            return comp[self._hess_seed_cols, self._hess_rows]
        if self._m > 0:
            H = _to_np(self._hess_lag_dense(jnp.asarray(x), jnp.asarray(lam), obj_factor))
        else:
            H = _to_np(self._hess_lag_dense(jnp.asarray(x), obj_factor))
        return H[self._hess_rows, self._hess_cols]


def from_jax(
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
    """Build a pounce :class:`Problem` from JAX-traced functions.

    Parameters
    ----------
    f : callable
        Objective. Must be ``jax.jit``-able; takes a 1-D array of
        length ``n`` and returns a scalar.
    g : callable or None
        Constraint function. Takes ``x`` of length ``n`` and returns
        an array of length ``m``. Required when ``m > 0``.
    n, m : int
        Variable / constraint counts.
    lb, ub, cl, cu : array-like or None
        Variable bounds and constraint bounds; same convention as
        :class:`Problem`.
    seed : int
        Seed for the random probe(s) used to detect sparsity patterns.
        Change only if your problem has structural zeros that happen to
        align with the default probe.
    sparse : bool
        When ``True``, compute the Jacobian and Lagrangian Hessian with
        CPR-style colored AD (one JVP/HVP per color) instead of forming
        the dense matrix and slicing (issue #83). This drops the per-eval
        cost from ``O(n)`` to ``O(k)`` AD passes, where ``k`` is the
        number of colors — a large win on genuinely sparse problems and a
        small loss on dense ones (extra coloring + scatter bookkeeping).
        The reported structure is identical either way. Defaults to
        ``False`` (dense, with forward/reverse mode chosen by shape).
    n_probes : int or None
        Number of random probes whose nonzero patterns are unioned to
        detect sparsity. ``None`` (default) uses 1 probe for the dense
        path and 3 for ``sparse=True`` (a mis-probe under compression
        corrupts the seed structure, not just a reported nonzero, so
        hardening detection matters more there). Pass an explicit integer
        to override.

    Returns
    -------
    Problem
        With the user object built from JAX AD. Pass options via
        ``problem.add_option(...)`` and call ``problem.solve(x0)``.
    """
    if m > 0 and g is None:
        raise ValueError("g must be provided when m > 0")
    if n_probes is None:
        n_probes = 3 if sparse else 1
    obj = _JaxProblem(
        f=f, g=g, n=n, m=m, seed=seed, sparse=sparse, n_probes=n_probes,
    )
    return Problem(
        n=n, m=m, problem_obj=obj, lb=lb, ub=ub, cl=cl, cu=cu,
    )
