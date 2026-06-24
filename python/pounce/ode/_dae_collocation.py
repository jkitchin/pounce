"""Fixed-mesh backward-Euler collocation for a fully-implicit DAE, with the
numpy pieces the differentiable frontends need: a host forward solve and the
transpose-Jacobian solve for the implicit-function-theorem backward.

On a *fixed* mesh ``t_0 < ... < t_{m-1}`` the node states ``Y = (y_1, ..., y_{m-1})``
(``y_0`` given) solve the backward-Euler residual

    r_k = F(t_{k+1}, y_{k+1}, (y_{k+1} - y_k) / h_k) = 0,   k = 0 .. m-2,

an ``n(m-1)``-dimensional root-find ``R(Y; theta, y0) = 0``. Backward Euler is
L-stable, so it integrates stiff / index-1 DAEs without spurious oscillation;
order 1, so accuracy is controlled by the mesh (as with ``jax.odeint``'s fixed
mesh). ``R_Y`` is block lower-bidiagonal: ``r_k`` couples ``y_k`` and ``y_{k+1}``.

The differentiable map ``(theta, y0) -> Y`` is defined implicitly by ``R = 0``;
its VJP needs ``R_Y^T u = v`` (here, via FERAL's sparse LU) plus ``(dR/dp)^T u``
(supplied by the framework autodiff of a traced residual at ``Y*``). Forward and
transpose-solve are numpy; the traced residual lives in the jax/torch frontends.
"""

from __future__ import annotations

import numpy as np

from .._pounce import SparseLU

_FD = np.sqrt(np.finfo(float).eps)


def _node_jacs(Ffun, t1, w, wp, h, jac):
    """``(dF/dy, dF/dy')`` at a node, analytic if ``jac`` else forward-diff."""
    F0 = np.asarray(Ffun(t1, w, wp), float)
    if jac is not None:
        Fy, Fyp = jac(t1, w, wp)
        return np.asarray(Fy, float), np.asarray(Fyp, float), F0
    n = w.size
    Fy = np.empty((n, n)); Fyp = np.empty((n, n))
    for j in range(n):
        dy = _FD * max(1.0, abs(w[j]))
        wj = w.copy(); wj[j] += dy
        Fy[:, j] = (np.asarray(Ffun(t1, wj, wp), float) - F0) / dy
        dp = _FD * max(1.0, abs(wp[j]))
        pj = wp.copy(); pj[j] += dp
        Fyp[:, j] = (np.asarray(Ffun(t1, w, pj), float) - F0) / dp
    return Fy, Fyp, F0


def be_forward(Ffun, t, y0, *, jac=None, tol=1e-10, maxiter=50):
    """Sequential backward-Euler march; returns ``Y`` of shape ``(n, m)``."""
    t = np.asarray(t, float)
    y0 = np.asarray(y0, float)
    n = y0.size
    m = t.size
    Y = np.empty((n, m))
    Y[:, 0] = y0
    yk = y0.copy()
    for k in range(m - 1):
        h = t[k + 1] - t[k]
        w = yk.copy()                         # warm start from previous node
        for _ in range(maxiter):
            wp = (w - yk) / h
            Fy, Fyp, F0 = _node_jacs(Ffun, t[k + 1], w, wp, h, jac)
            if np.linalg.norm(F0) <= tol * (1.0 + np.linalg.norm(w)):
                break
            J = Fy + Fyp / h
            w = w + np.linalg.solve(J, -F0)
        Y[:, k + 1] = w
        yk = w
    return Y


def _coo_pattern(n, M):
    """COO (rows, cols) for the block lower-bidiagonal ``R_Y`` (``N = n*M``)."""
    rows = []; cols = []
    for k in range(M):
        r0 = k * n
        # diagonal block (k,k): d r_k / d y_{k+1}
        for a in range(n):
            for b in range(n):
                rows.append(r0 + a); cols.append(k * n + b)
        # subdiagonal block (k,k-1): d r_k / d y_k  (k>=1; y_0 is fixed)
        if k >= 1:
            for a in range(n):
                for b in range(n):
                    rows.append(r0 + a); cols.append((k - 1) * n + b)
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64)


def _jac_values(Ffun, t, Y, y0, n, M, jac, rows_per):
    """Values aligned to :func:`_coo_pattern`, evaluated at ``Y``."""
    t = np.asarray(t, float)
    vals = []
    for k in range(M):
        h = t[k + 1] - t[k]
        w = Y[:, k + 1]
        yk = y0 if k == 0 else Y[:, k]
        wp = (w - yk) / h
        Fy, Fyp, _ = _node_jacs(Ffun, t[k + 1], w, wp, h, jac)
        diag = Fy + Fyp / h               # d r_k / d y_{k+1}
        vals.append(diag.reshape(-1))
        if k >= 1:
            sub = -Fyp / h                # d r_k / d y_k
            vals.append(sub.reshape(-1))
    return np.concatenate(vals)


def be_transpose_solve(Ffun, t, Y, y0, v, *, jac=None):
    """Solve ``R_Y^T u = v`` at the converged nodes ``Y`` (``(n, m)``).

    ``Y`` includes the fixed ``y_0`` column; the unknown block is ``y_1..y_{m-1}``.
    Returns ``u`` of shape ``(n*(m-1),)``.
    """
    n = Y.shape[0]
    M = Y.shape[1] - 1
    N = n * M
    rows, cols = _coo_pattern(n, M)
    lu = SparseLU(N, rows, cols)
    lu.factor(_jac_values(Ffun, t, Y, y0, n, M, jac, None))
    return np.asarray(lu.solve_transpose(np.asarray(v, float).reshape(-1)))
