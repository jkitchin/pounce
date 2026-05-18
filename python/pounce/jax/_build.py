"""Build a :class:`pounce.Problem` from JAX-traced ``f(x)`` and ``g(x)``.

The user supplies the problem in mathematical form and we derive:

* ``gradient`` = ``jax.grad(f)``
* ``jacobian`` = ``jax.jacrev(g)`` (returns a dense ``(m, n)`` matrix
  which we project onto the precomputed sparsity pattern of nonzero
  entries)
* ``hessian`` = ``jax.hessian`` of the Lagrangian
  ``L(x, λ, σ) = σ f(x) + λᵀ g(x)``

Sparsity is detected by a single random probe: evaluate the dense
Jacobian / Hessian at a random ``x`` (and ``λ`` for the Hessian) and
record the indices where the magnitude exceeds a threshold. This is
fast and works for the common case where the *structure* doesn't depend
on the value, which is true for any composition of smooth pointwise
operations. Users with value-dependent sparsity should hand-roll the
pattern via the :class:`Problem` API.

All five eval methods are JIT-compiled with ``jax.jit`` and return
``numpy.ndarray`` (via ``np.asarray(jnp_result)``) so the Rust bridge
receives a contiguous CPU buffer.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .._pounce import Problem

# Threshold below which a Jacobian/Hessian entry is treated as
# structurally zero during the one-shot pattern probe. Tight enough to
# reject genuine zeros from constant terms, loose enough that random
# probe values don't accidentally hit a numerical cancellation that
# would drop a real entry.
_SPARSITY_EPS = 1e-12


def _detect_pattern_2d(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(np.abs(dense) > _SPARSITY_EPS)
    return rows.astype(np.int64), cols.astype(np.int64)


def _detect_pattern_lower(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower-triangle sparsity pattern of a symmetric matrix."""
    n = dense.shape[0]
    mask = np.abs(dense) > _SPARSITY_EPS
    rows, cols = np.tril_indices(n)
    keep = mask[rows, cols]
    return rows[keep].astype(np.int64), cols[keep].astype(np.int64)


def _to_np(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


class _JaxProblem:
    """Cyipopt-shaped problem object backed by JAX-AD callables."""

    def __init__(
        self,
        f: Callable,
        g: Callable | None,
        n: int,
        m: int,
        seed: int = 0,
    ):
        self._f = jax.jit(f)
        self._grad_f = jax.jit(jax.grad(f))
        if g is not None and m > 0:
            self._g = jax.jit(g)
            jac_g_dense = jax.jit(jax.jacrev(g))
            self._jac_g_dense = jac_g_dense

            # Lagrangian Hessian: σ f(x) + λᵀ g(x).
            def lagrangian(x, lam, sigma):
                return sigma * f(x) + jnp.dot(lam, g(x))

            self._hess_lag_dense = jax.jit(jax.hessian(lagrangian, argnums=0))
        else:
            self._g = None
            self._jac_g_dense = None

            def lagrangian_unc(x, sigma):
                return sigma * f(x)

            self._hess_lag_dense = jax.jit(jax.hessian(lagrangian_unc, argnums=0))

        self._n = n
        self._m = m

        # Detect sparsity once. Use a random probe so we don't sit on
        # structural zeros that happen to be valid for x=0 / λ=0.
        rng = np.random.default_rng(seed)
        x_probe = jnp.asarray(rng.standard_normal(n))
        if m > 0:
            lam_probe = jnp.asarray(rng.standard_normal(m))
            jac_dense = _to_np(self._jac_g_dense(x_probe))
            self._jac_rows, self._jac_cols = _detect_pattern_2d(jac_dense)
            hess_dense = _to_np(self._hess_lag_dense(x_probe, lam_probe, 1.0))
        else:
            self._jac_rows = np.zeros(0, dtype=np.int64)
            self._jac_cols = np.zeros(0, dtype=np.int64)
            hess_dense = _to_np(self._hess_lag_dense(x_probe, 1.0))
        self._hess_rows, self._hess_cols = _detect_pattern_lower(hess_dense)

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
        J = _to_np(self._jac_g_dense(jnp.asarray(x)))
        return J[self._jac_rows, self._jac_cols]

    def hessianstructure(self):
        return (self._hess_rows, self._hess_cols)

    def hessian(self, x, lam, obj_factor):
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
        Seed for the random probe used to detect sparsity patterns.
        Change only if your problem has structural zeros that happen to
        align with the default probe.

    Returns
    -------
    Problem
        With the user object built from JAX AD. Pass options via
        ``problem.add_option(...)`` and call ``problem.solve(x0)``.
    """
    if m > 0 and g is None:
        raise ValueError("g must be provided when m > 0")
    obj = _JaxProblem(f=f, g=g, n=n, m=m, seed=seed)
    return Problem(
        n=n, m=m, problem_obj=obj, lb=lb, ub=ub, cl=cl, cu=cu,
    )
