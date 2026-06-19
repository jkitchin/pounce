"""Damped-Newton root-finder for the collocation system, on FERAL's LU.

A boundary value problem discretises to a **square** nonlinear system
``R(z) = 0`` (``z = [vec(Y); p]``, size ``N = n*m + k``). The natural
solver is Newton: factor the ``N x N`` Jacobian ``J = dR/dz`` and step
``dz = -J^{-1} R`` — exactly what SciPy's ``solve_bvp`` does. We factor
``J`` with FERAL's unsymmetric sparse LU (:class:`pounce._pounce.SparseLU`),
so this path never builds the interior-point method's ``2N`` symmetric
saddle system.

The iteration is a **modified (frozen-Jacobian) damped Newton**: the LU is
reused across steps and only refactored when progress stalls (the line
search fails, or a step's residual reduction is poor). Because the
factorization dominates the per-iteration cost — and the collocation
Jacobian changes slowly between steps — reusing it typically cuts the
number of factorizations from one-per-iteration to one or two for the whole
solve, which is what makes this competitive with (and often faster than)
SciPy, whose ``solve_newton`` freezes the Jacobian the same way. The LU
symbolic analysis is computed once (fixed sparsity); refactoring only
re-runs the numeric factorization.
"""

from __future__ import annotations

import numpy as np

from .._pounce import SparseLU

# Refactor the frozen Jacobian when an accepted step reduces the residual
# norm by less than this factor — convergence has slowed enough that a fresh
# factor (restoring quadratic convergence) is worth its cost.
_REFACTOR_RATIO = 0.5

# Outcome codes returned by :func:`newton_solve` (mapped to BVPResult.status
# by the caller). Distinct from SciPy's mesh codes (1 = max nodes,
# 2 = singular Jacobian) so the two can't be confused.
STATUS_CONVERGED = 0
STATUS_SINGULAR = 2
STATUS_NOT_CONVERGED = 4


def newton_solve(residual_fn, jac, z0, n, m, k, *, tol=1e-8, max_iter=50):
    """Solve ``R(z) = 0`` from ``z0`` by modified (frozen-Jacobian) Newton.

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
    (z, niter, status, res_norm)
        ``status`` is one of :data:`STATUS_CONVERGED` (0),
        :data:`STATUS_SINGULAR` (2), :data:`STATUS_NOT_CONVERGED` (4).
    """
    N = n * m + k
    rows, cols = jac.structure()
    lu = SparseLU(N, np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))

    z = np.array(z0, dtype=np.float64)
    R = np.asarray(residual_fn(z), dtype=np.float64)
    rnorm = np.linalg.norm(R, np.inf)

    need_factor = True   # (re)factor the held LU at the top of the next step
    fresh = False        # True iff the held factor was built at the current z
    status = STATUS_NOT_CONVERGED
    it = 0
    for it in range(1, max_iter + 1):
        if rnorm < tol:
            status = STATUS_CONVERGED
            break
        if need_factor:
            Y = z[: n * m].reshape(n, m)
            p = z[n * m :]
            try:
                lu.factor(jac.values(Y, p))
            except RuntimeError:
                # FERAL raises on a singular factorisation (e.g. a poor
                # initial guess). Mirror SciPy, which returns a result with
                # status 2 rather than propagating an exception.
                status = STATUS_SINGULAR
                break
            need_factor = False
            fresh = True
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
            if fresh:
                # A fresh factor still can't reduce the residual: the iterate
                # has stalled (typically at round-off, but possibly short of
                # `tol`). Judge success against the requested `tol` rather
                # than a hardcoded threshold so a loose stall is reported as
                # non-convergence.
                status = STATUS_CONVERGED if rnorm < tol else STATUS_NOT_CONVERGED
                break
            # The frozen factor gave a poor direction; refresh it at the
            # current z and retry this step (no move taken).
            need_factor = True
            continue

        ratio = rnorm_trial / rnorm
        z, R, rnorm = z_trial, R_trial, rnorm_trial
        fresh = False
        # Refactor next step if the frozen factor is going stale — slow
        # reduction or a heavily damped step both signal the linearisation
        # no longer matches; a fresh factor restores fast convergence.
        if ratio > _REFACTOR_RATIO or alpha < 1.0:
            need_factor = True
    else:
        # Loop ran to max_iter without the top-of-loop tol test firing.
        status = STATUS_CONVERGED if rnorm < tol else STATUS_NOT_CONVERGED

    return z, it, status, rnorm
