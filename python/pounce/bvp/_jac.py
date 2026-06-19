"""Exact sparse Jacobian of the Hermite--Simpson collocation residual.

The dense finite-difference Jacobian of the *whole* residual costs ``N+1``
residual evaluations (each ``O(N)``) per interior-point iteration and forces
a dense ``N x N`` factorisation — ``O(N^2)`` to build, ``O(N^3)`` to factor,
with ``N = n*m + k``. That is what made the NumPy ``solve_bvp`` scale badly
in the mesh size ``m``.

This module assembles the analytic Jacobian instead. Each collocation block
couples only the two endpoints of its interval (and the unknown parameters);
the boundary block couples only the two domain ends (and the parameters). So
the global Jacobian is **sparse** (``O(m)`` nonzeros), and the per-node
blocks ``df/dy`` / ``df/dp`` are obtained either from a user ``fun_jac`` /
``bc_jac`` or by finite differences that perturb each *state component across
the whole mesh at once* — ``O(n)`` vectorised ``fun`` calls, not ``O(n*m)``.

Index convention matches :mod:`pounce.bvp._core` (state-major flatten):

* variable column for ``Y[a, i]`` is ``a*m + i``; for ``p[l]`` it is
  ``n*m + l``;
* residual row for ``col_res[a, i]`` is ``a*(m-1) + i``; for boundary
  residual ``t`` it is ``n*(m-1) + t``.

The collocation derivative formulas are the standard bvp4c ones
(cf. Kierzenka & Shampine; identical to SciPy's ``construct_global_jac``):

    y_mid_i      = (y_i + y_{i+1})/2 - h_i/8 (f_{i+1} - f_i)
    d r_i/d y_i      = -I - h_i/6 (J_i  + 4 J_mid (I/2 + h_i/8 J_i))
    d r_i/d y_{i+1}  =  I - h_i/6 (J_{i+1} + 4 J_mid (I/2 - h_i/8 J_{i+1}))
    d r_i/d p        = -h_i/6 (K_i + K_{i+1} + 4 K_mid
                               - (h_i/2) J_mid (K_{i+1} - K_i))

with ``J_* = df/dy`` and ``K_* = df/dp`` at the node / midpoint.
"""

from __future__ import annotations

import numpy as np

_FD_EPS = np.sqrt(np.finfo(np.float64).eps)


def _fd_df_dy(nfun, xq, Yq, p, fq):
    """Finite-difference ``df/dy`` at the given points.

    ``xq`` ``(mq,)``, ``Yq`` ``(n, mq)``, ``fq = nfun(xq, Yq, p)`` ``(n, mq)``.
    Returns ``(mq, n, n)`` with ``J[i, a, b] = df_a/dy_b`` at point ``i``.
    """
    n, mq = Yq.shape
    J = np.empty((mq, n, n), dtype=np.float64)
    for b in range(n):
        step = _FD_EPS * np.maximum(1.0, np.abs(Yq[b]))
        Yb = Yq.copy()
        Yb[b] += step
        fb = np.asarray(nfun(xq, Yb, p), dtype=np.float64)
        J[:, :, b] = ((fb - fq) / step).T
    return J


def _fd_df_dp(nfun, xq, Yq, p, fq):
    """Finite-difference ``df/dp`` at the given points -> ``(mq, n, k)``."""
    n, mq = Yq.shape
    k = p.shape[0]
    K = np.empty((mq, n, k), dtype=np.float64)
    for l in range(k):
        step = _FD_EPS * max(1.0, abs(p[l]))
        pl = p.copy()
        pl[l] += step
        fl = np.asarray(nfun(xq, Yq, pl), dtype=np.float64)
        K[:, :, l] = ((fl - fq) / step).T
    return K


def _fd_dbc(nbc, ya, yb, p, b0):
    """Finite-difference boundary Jacobians.

    Returns ``(dbc_dya, dbc_dyb, dbc_dp)`` of shapes ``(n+k, n)``,
    ``(n+k, n)``, ``(n+k, k)``.
    """
    n = ya.shape[0]
    k = p.shape[0]
    nb = b0.shape[0]
    dya = np.empty((nb, n), dtype=np.float64)
    dyb = np.empty((nb, n), dtype=np.float64)
    dp = np.empty((nb, k), dtype=np.float64)
    for b in range(n):
        s = _FD_EPS * max(1.0, abs(ya[b]))
        yav = ya.copy(); yav[b] += s
        dya[:, b] = (np.asarray(nbc(yav, yb, p), dtype=np.float64) - b0) / s
        s = _FD_EPS * max(1.0, abs(yb[b]))
        ybv = yb.copy(); ybv[b] += s
        dyb[:, b] = (np.asarray(nbc(ya, ybv, p), dtype=np.float64) - b0) / s
    for l in range(k):
        s = _FD_EPS * max(1.0, abs(p[l]))
        pl = p.copy(); pl[l] += s
        dp[:, l] = (np.asarray(nbc(ya, yb, pl), dtype=np.float64) - b0) / s
    return dya, dyb, dp


class CollocationJacobian:
    """Sparse Jacobian of the collocation residual w.r.t. ``z = [vec(Y); p]``.

    The sparsity pattern (``rows``, ``cols``) is fixed and built once;
    :meth:`values` recomputes only the nonzero values at a given ``z``.
    """

    def __init__(self, nfun, nbc, x, n, m, k, df_blocks=None, dbc_block=None,
                 bc_len=None):
        self._nfun = nfun
        self._nbc = nbc
        self._x = np.asarray(x, dtype=np.float64)
        self._n = n
        self._m = m
        self._k = k
        # Number of boundary residuals. Defaults to n + k (a fully
        # determined BVP); the constrained solver may pass fewer (an
        # under-determined system whose remaining freedom an objective
        # resolves — an optimal-control collocation).
        self._bc_len = (n + k) if bc_len is None else bc_len
        self._df_blocks = df_blocks   # callable(xq, Yq, p, fq) -> (J (mq,n,n), K (mq,n,k)) or None
        self._dbc_block = dbc_block   # callable(ya, yb, p, b0) -> (dya, dyb, dp) or None
        self._I = np.eye(n, dtype=np.float64)
        self._build_structure()

    # -- fixed sparsity pattern ------------------------------------------
    def _build_structure(self):
        n, m, k = self._n, self._m, self._k
        rows = []
        cols = []

        # Collocation blocks: for each interval i and (row a, col b),
        # left -> col y_i = b*m + i ; right -> col y_{i+1} = b*m + i + 1.
        I, A, B = np.meshgrid(
            np.arange(m - 1), np.arange(n), np.arange(n), indexing="ij"
        )
        col_row = (A * (m - 1) + I).ravel()
        rows.append(col_row)                          # left
        cols.append((B * m + I).ravel())
        rows.append(col_row)                          # right
        cols.append((B * m + I + 1).ravel())

        if k > 0:
            Ip, Ap, L = np.meshgrid(
                np.arange(m - 1), np.arange(n), np.arange(k), indexing="ij"
            )
            rows.append((Ap * (m - 1) + Ip).ravel())  # d r_i / d p
            cols.append((n * m + L).ravel())

        # Boundary block: rows n*(m-1)+t, cols y_0 / y_{m-1} / p.
        nb = self._bc_len
        T, Bb = np.meshgrid(np.arange(nb), np.arange(n), indexing="ij")
        bc_row = (n * (m - 1) + T).ravel()
        rows.append(bc_row)                           # d bc / d ya
        cols.append((Bb * m + 0).ravel())
        rows.append(bc_row)                           # d bc / d yb
        cols.append((Bb * m + (m - 1)).ravel())
        if k > 0:
            Tp, Lp = np.meshgrid(np.arange(nb), np.arange(k), indexing="ij")
            rows.append((n * (m - 1) + Tp).ravel())   # d bc / d p
            cols.append((n * m + Lp).ravel())

        self._rows = np.concatenate(rows).astype(np.int64)
        self._cols = np.concatenate(cols).astype(np.int64)

    def structure(self):
        return self._rows, self._cols

    # -- nonzero values at z ---------------------------------------------
    def values(self, Y, p):
        n, m, k = self._n, self._m, self._k
        x = self._x
        Im = self._I

        f_nodes = np.asarray(self._nfun(x, Y, p), dtype=np.float64)   # (n, m)
        h = x[1:] - x[:-1]                                            # (m-1,)
        y_mid = 0.5 * (Y[:, 1:] + Y[:, :-1]) - 0.125 * h * (
            f_nodes[:, 1:] - f_nodes[:, :-1]
        )
        x_mid = x[:-1] + 0.5 * h
        f_mid = np.asarray(self._nfun(x_mid, y_mid, p), dtype=np.float64)

        if self._df_blocks is not None:
            J_nodes, K_nodes = self._df_blocks(x, Y, p, f_nodes)
            J_mid, K_mid = self._df_blocks(x_mid, y_mid, p, f_mid)
        else:
            J_nodes = _fd_df_dy(self._nfun, x, Y, p, f_nodes)         # (m, n, n)
            J_mid = _fd_df_dy(self._nfun, x_mid, y_mid, p, f_mid)     # (m-1, n, n)
            if k > 0:
                K_nodes = _fd_df_dp(self._nfun, x, Y, p, f_nodes)     # (m, n, k)
                K_mid = _fd_df_dp(self._nfun, x_mid, y_mid, p, f_mid)
            else:
                K_nodes = K_mid = None

        h3 = (h / 6.0)[:, None, None]
        h8 = (h / 8.0)[:, None, None]
        Ji = J_nodes[:-1]                                            # (m-1, n, n)
        Jip1 = J_nodes[1:]
        Jmid = J_mid

        # d r_i / d y_i  and  d r_i / d y_{i+1}
        D_left = -Im[None] - h3 * Ji - 4.0 * h3 * (Jmid @ (0.5 * Im[None] + h8 * Ji))
        D_right = Im[None] - h3 * Jip1 - 4.0 * h3 * (Jmid @ (0.5 * Im[None] - h8 * Jip1))

        vals = [D_left.ravel(), D_right.ravel()]

        if k > 0:
            Ki = K_nodes[:-1]
            Kip1 = K_nodes[1:]
            h2 = (h / 2.0)[:, None, None]
            D_p = -h3 * (Ki + Kip1 + 4.0 * K_mid - h2 * (Jmid @ (Kip1 - Ki)))
            vals.append(D_p.ravel())

        # Boundary block.
        ya = Y[:, 0]
        yb = Y[:, -1]
        b0 = np.asarray(self._nbc(ya, yb, p), dtype=np.float64)
        if self._dbc_block is not None:
            dbc_dya, dbc_dyb, dbc_dp = self._dbc_block(ya, yb, p, b0)
        else:
            dbc_dya, dbc_dyb, dbc_dp = _fd_dbc(self._nbc, ya, yb, p, b0)
        vals.append(np.asarray(dbc_dya, dtype=np.float64).ravel())
        vals.append(np.asarray(dbc_dyb, dtype=np.float64).ravel())
        if k > 0:
            vals.append(np.asarray(dbc_dp, dtype=np.float64).ravel())

        return np.concatenate(vals)
