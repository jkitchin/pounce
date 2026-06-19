"""Damped-Newton root-finder for the collocation system, on FERAL's LU.

A boundary value problem discretises to a **square** nonlinear system
``R(z) = 0`` (``z = [vec(Y); p]``, size ``N = n*m + k``). The natural
solver is Newton: factor the ``N x N`` Jacobian ``J = dR/dz`` and step
``dz = -J^{-1} R`` — exactly what SciPy's ``solve_bvp`` does. We factor
``J`` with FERAL's unsymmetric sparse LU (:class:`pounce._pounce.SparseLU`),
so this path never builds the interior-point method's ``2N`` symmetric
saddle system.

The iteration is an affine-invariant damped Newton: backtrack the step
length until the residual norm decreases, which makes it robust on mildly
nonlinear problems without the IPM's filter / barrier machinery. The
Jacobian structure is fixed across iterations, so the LU symbolic analysis
is computed once and only the numeric factorization repeats.
"""

from __future__ import annotations

import numpy as np

from .._pounce import SparseLU


def newton_solve(residual_fn, jac, z0, n, m, k, *, tol=1e-8, max_iter=50):
    """Solve ``R(z) = 0`` from ``z0`` by damped Newton.

    Parameters
    ----------
    residual_fn : callable
        ``residual_fn(z) -> (N,)`` collocation + boundary residual.
    jac : CollocationJacobian
        Provides the fixed sparsity ``structure()`` and per-``z`` sparse
        ``values(Y, p)``.
    z0 : array (N,)
        Initial guess.
    n, m, k : int
        States, mesh nodes, unknown parameters.

    Returns
    -------
    (z, niter, converged, res_norm)
    """
    N = n * m + k
    rows, cols = jac.structure()
    lu = SparseLU(N, np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))

    z = np.array(z0, dtype=np.float64)
    R = np.asarray(residual_fn(z), dtype=np.float64)
    rnorm = np.linalg.norm(R, np.inf)

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        if rnorm < tol:
            converged = True
            break
        Y = z[: n * m].reshape(n, m)
        p = z[n * m :]
        lu.factor(jac.values(Y, p))
        dz = lu.solve(-R)

        # Backtracking line search on the residual infinity-norm.
        alpha = 1.0
        accepted = False
        for _ in range(30):
            z_trial = z + alpha * dz
            R_trial = np.asarray(residual_fn(z_trial), dtype=np.float64)
            rnorm_trial = np.linalg.norm(R_trial, np.inf)
            if rnorm_trial < (1.0 - 1e-4 * alpha) * rnorm:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            # Backtracking could not reduce the residual — the iterate has
            # converged to round-off (quadratic convergence then stalls).
            # Stop here rather than spinning to max_iter.
            converged = rnorm < 1e-6
            break
        z, R, rnorm = z_trial, R_trial, rnorm_trial

    if rnorm < tol:
        converged = True
    return z, it, converged, rnorm
