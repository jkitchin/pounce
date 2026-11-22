"""Sparse-matrix support for the ``pounce.ode`` integrators (gh #1228).

:func:`scipy.integrate.solve_ivp` accepts a ``jac`` that is a sparse matrix,
or a callable returning one, and the method-of-lines / forward-sensitivity
problems that motivate it have Jacobians that are sparse in the normal case.
When the Jacobian (or the mass matrix) is sparse, the integrators switch to
a *sparse mode*: ``J`` and ``M`` are held as CSC matrices, the shifted stage
operators are assembled sparse, and they are factored with SuperLU
(:func:`scipy.sparse.linalg.splu`) instead of the dense faer LU. Nothing is
densified on the per-step path.

The objects here wrap SuperLU in the same ``factor`` / ``solve`` interface
the dense :class:`pounce._pounce.DenseLU` exposes, so the step loops do not
branch on the mode beyond choosing which factorization object to build.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

issparse = sp.issparse
__all__ = ["sp", "issparse", "to_dense", "to_csc", "abs_max", "solve",
           "SparseLU", "SparseComplexLU"]


def to_dense(A):
    """``A`` as a float ndarray (densifying a sparse matrix)."""
    if sp.issparse(A):
        return np.asarray(A.toarray(), dtype=float)
    return np.asarray(A, dtype=float)


def to_csc(A):
    """``A`` as a float CSC sparse matrix."""
    if sp.issparse(A):
        return sp.csc_matrix(A, dtype=float)
    return sp.csc_matrix(np.asarray(A, dtype=float))


def abs_max(A, axis):
    """Dense ``(n,)`` max of ``|A|`` along ``axis``, for dense or sparse ``A``."""
    if sp.issparse(A):
        return np.asarray(abs(sp.csc_matrix(A)).max(axis=axis).toarray(),
                          dtype=float).ravel()
    return np.max(np.abs(np.asarray(A, dtype=float)), axis=axis)


def solve(A, b):
    """Solve ``A x = b`` for dense or sparse ``A``.

    Raises :class:`numpy.linalg.LinAlgError` on a singular matrix in both
    cases (``spsolve`` would instead warn and return NaN)."""
    if sp.issparse(A):
        try:
            return spla.splu(sp.csc_matrix(A, dtype=float)).solve(
                np.asarray(b, dtype=float))
        except RuntimeError as e:
            raise np.linalg.LinAlgError(str(e)) from e
    return np.linalg.solve(A, b)


def _splu(A):
    lu = spla.splu(sp.csc_matrix(A))
    # SuperLU reports an exactly-zero pivot as RuntimeError; a pivot that is
    # merely tiny is accepted, but an operator with inf/nan entries produces a
    # non-finite factor — surface that as the same failure so the caller's
    # shrink-h / refresh-J response engages, as it does for the dense LU.
    if not (np.all(np.isfinite(lu.U.data)) and np.all(np.isfinite(lu.L.data))):
        raise RuntimeError("sparse LU: non-finite factor")
    return lu


class SparseLU:
    """Real sparse LU with the ``factor(Mat)`` / ``solve(b)`` interface."""

    def __init__(self):
        self._lu = None

    def factor(self, Mat):
        self._lu = _splu(Mat)

    def solve(self, b):
        return self._lu.solve(np.asarray(b, dtype=float))


class SparseComplexLU:
    """Complex sparse LU presented through the real ``2n`` embedding.

    The dense path factors the complex operator ``C = Re + i·Im`` as the real
    block matrix ``[[Re, −Im], [Im, Re]]`` and solves against the stacked
    ``[b_re; b_im]``. Sparse mode factors ``C`` directly as a complex matrix
    (half the dimension, a quarter of the fill) and translates the stacked
    right-hand side and solution at the boundary, so callers are unchanged.
    """

    def __init__(self, n):
        self.n = n
        self._lu = None

    def factor(self, C):
        self._lu = _splu(sp.csc_matrix(C, dtype=complex))

    def solve(self, b):
        n = self.n
        x = self._lu.solve(np.asarray(b[:n], dtype=complex) + 1j * b[n:])
        return np.concatenate([x.real, x.imag])
