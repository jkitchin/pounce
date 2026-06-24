"""Fully-implicit index-1 DAE integration ``F(t, y, y') = 0`` via Radau IIA(5).

This generalises the mass-matrix path in ``_radau.py``. The Radau stage system
there is ``M K_i = f(t + c_i h, Y_i)``; here the unknown stage derivatives ``K``
(the ``y'`` values at the stages) solve the residual form

    F(t + c_i h, Y_i, K_i) = 0,   Y_i = y + h (A K)_i.

The simplified-Newton matrix and the embedded error operator generalise the
mass form by the substitution  ``M -> F_y' = dF/dy'``  and  ``-J -> F_y =
dF/dy``  (the mass form is ``F = M y' - f``, so ``F_y' = M`` and ``F_y =
-df/dy``):

    stage matrix    I3 (x) F_y'  +  h (A (x) F_y)
    error operator  (MU_REAL/h) F_y'  +  F_y

and the error right-hand side ``M(Z.E/h) + f0`` (``f0 = f = M y'_start``)
becomes ``F_y' (Z.E/h + y'_start)``, with ``y'_start`` the consistent
derivative at the step start (the user's / projected ``yp0`` at ``t0``, then
the last stage ``K`` of each accepted step, since Radau IIA is stiffly
accurate). Everything else — sparse LU + pattern reuse, the stage predictor,
the adaptive controller, the h-hold band, the dense output — is reused from
``_radau``.

The differential / algebraic variable split is detected from ``F_y'``: a
variable ``j`` is *algebraic* when ``F`` does not depend on ``y'_j`` (column
``j`` of ``F_y'`` is structurally zero). That split drives the consistent
initial-condition projection (:func:`consistent_initial_conditions`).
"""

from __future__ import annotations

import numpy as np

from . import _radau as R

_FD = np.sqrt(np.finfo(float).eps)
_ALG_TOL = 1e-9            # column of F_y' below this (relative) => algebraic


def _fd_dae_jacs(Ffun, t, y, yp, F0):
    """Forward-difference ``(F_y, F_y')`` at ``(t, y, yp)``; ``2n`` evals."""
    n = y.size
    Fy = np.empty((n, n))
    Fyp = np.empty((n, n))
    for j in range(n):
        dy = _FD * max(1.0, abs(y[j]))
        yj = y.copy(); yj[j] += dy
        Fy[:, j] = (Ffun(t, yj, yp) - F0) / dy
        dp = _FD * max(1.0, abs(yp[j]))
        pj = yp.copy(); pj[j] += dp
        Fyp[:, j] = (Ffun(t, y, pj) - F0) / dp
    return Fy, Fyp


class _DaeProblem:
    """Holds the residual ``F``, optional analytic Jacobians, and counters."""

    def __init__(self, Ffun, n, jac=None):
        self.Ffun = Ffun
        self.n = n
        self._user_jac = jac          # optional (t,y,yp) -> (F_y, F_y')
        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    def F(self, t, y, yp):
        self.nfev += 1
        return np.asarray(self.Ffun(t, y, yp), dtype=float)

    def jacs(self, t, y, yp, F0):
        self.njev += 1
        if self._user_jac is not None:
            Fy, Fyp = self._user_jac(t, y, yp)
            return np.asarray(Fy, float), np.asarray(Fyp, float)
        return _fd_dae_jacs(self.F, t, y, yp, F0)


def _algebraic_mask(Fyp):
    """Boolean ``(n,)``: variable ``j`` is algebraic iff ``F`` is independent of
    ``y'_j`` (column ``j`` of ``F_y'`` is ~0)."""
    col = np.max(np.abs(Fyp), axis=0)
    return col <= _ALG_TOL * max(1.0, float(col.max()))


def consistent_initial_conditions(prob, t0, y0, yp0=None, *, tol=1e-10,
                                  max_iter=25):
    """Project ``(y0, yp0)`` onto ``F(t0, y0, y'0) = 0`` for an index-1 DAE.

    Differential vs. algebraic variables are detected from ``F_y'`` (see
    :func:`_algebraic_mask`). Holding the *differential* ``y`` components and
    the *algebraic* ``y'`` components fixed, Newton-solves ``F = 0`` for the
    differential ``y'`` and algebraic ``y`` components — the IDA
    ``IDA_YA_YDP_INIT`` computation. Returns ``(y0, yp0)`` consistent to ``tol``
    (raises ``RuntimeError`` if Newton fails to converge).
    """
    n = np.asarray(y0).size
    y = np.asarray(y0, float).copy()
    yp = np.zeros(n) if yp0 is None else np.asarray(yp0, float).copy()

    F0 = prob.F(t0, y, yp)
    _, Fyp = prob.jacs(t0, y, yp, F0)
    alg = _algebraic_mask(Fyp)
    diff = ~alg
    for _ in range(max_iter):
        F0 = prob.F(t0, y, yp)
        if np.linalg.norm(F0) <= tol * (1.0 + np.linalg.norm(y) + np.linalg.norm(yp)):
            return y, yp
        Fy, Fyp = prob.jacs(t0, y, yp, F0)
        # Unknown j is y'_j (differential) or y_j (algebraic): pick that column.
        Ju = np.where(diff[None, :], Fyp, Fy)
        try:
            du = np.linalg.solve(Ju, -F0)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "consistent_initial_conditions: singular Jacobian — the DAE may "
                "be higher than index 1, or the differential/algebraic split is "
                "ambiguous."
            ) from e
        yp[diff] += du[diff]
        y[alg] += du[alg]
    raise RuntimeError(
        "consistent_initial_conditions: Newton did not converge; pass a better "
        f"yp0 guess (residual ||F||={np.linalg.norm(prob.F(t0, y, yp)):.3e})."
    )


def _solve_stages_dae(prob, t, y, h, yp_base, lu3, scale, newton_tol, K0=None):
    """Simplified-Newton solve of ``F(t+c_i h, Y_i, K_i) = 0`` for stage K."""
    n = prob.n
    K = np.broadcast_to(yp_base, (3, n)).copy() if K0 is None else K0.copy()
    dnorm_prev = None
    rate = None
    converged = False
    sc = scale * np.sqrt(3 * n)
    for _ in range(R._NEWTON_MAXITER):
        G = np.empty((3, n))
        for i in range(3):
            Yi = y + h * (R.RADAU_A[i] @ K)
            G[i] = prob.F(t + R.RADAU_C[i] * h, Yi, K[i])
        dK = lu3.solve(-G.reshape(-1)).reshape(3, n)
        K = K + dK
        dnorm = np.linalg.norm(dK / sc)
        if dnorm_prev is not None and dnorm_prev > 0:
            rate = dnorm / dnorm_prev
            if rate >= 1:
                break
            if rate / (1 - rate) * dnorm < newton_tol:
                converged = True
                break
        elif dnorm < newton_tol:
            converged = True
            break
        dnorm_prev = dnorm
    return K, converged, rate


def _error_estimate_dae(Fyp, h, K, yp_base, lu_real):
    """Embedded order-3 error estimate, generalising the mass-matrix form."""
    Z = h * (R.RADAU_A @ K)
    rhs = Fyp @ (Z.T @ (R.RADAU_E / h) + yp_base)
    return lu_real.solve(rhs)


def integrate_dae(Ffun, t0, t1, y0, yp0, *, rtol=1e-3, atol=1e-6,
                  first_step=None, max_step=np.inf, jac=None, t_eval=None,
                  dense_output=False, max_steps=10**6):
    """Adaptive Radau IIA(5) integration of ``F(t, y, y') = 0`` from t0 to t1.

    ``yp0`` must be consistent (``F(t0, y0, yp0) == 0``); use
    :func:`consistent_initial_conditions` first if it is not. Returns the same
    dict shape as ``_radau.integrate``.
    """
    y0 = np.asarray(y0, float)
    yp = np.asarray(yp0, float).copy()
    n = y0.size
    prob = _DaeProblem(Ffun, n, jac=jac)
    forward = t1 >= t0
    s = 1.0 if forward else -1.0

    eps = np.finfo(float).eps
    newton_tol = max(10 * eps / rtol, min(0.03, rtol ** 0.5))

    t = t0
    y = y0.copy()
    F0 = prob.F(t, y, yp)
    Fy, Fyp = prob.jacs(t, y, yp, F0)

    if first_step is not None:
        h = abs(first_step)
    else:                                       # Hairer h0 from y0, yp0
        scale = atol + np.abs(y0) * rtol
        d0 = np.linalg.norm(y0 / scale) / np.sqrt(n)
        d1 = np.linalg.norm(yp / scale) / np.sqrt(n)
        h = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    h = min(h, abs(t1 - t0), max_step)

    records = [] if (dense_output or t_eval is not None) else None
    ts = [t]; ys = [y.copy()]
    nstep = nrej = 0
    jac_current = True
    rejected = False

    lu3 = R._dense_lu_pattern(3 * n)
    lu_real = R._dense_lu_pattern(n)
    h_lu = None
    need_factor = True
    K_prev = None
    h_prev = None
    status, message = 0, R._ODE_MESSAGES[0]

    while (t - t1) * s < -1e-12 and nstep < max_steps:
        h = min(h, abs(t1 - t))
        hs = s * h
        if need_factor or h != h_lu:
            big = np.kron(np.eye(3), Fyp) + hs * np.kron(R.RADAU_A, Fy)
            R._refactor(lu3, big)
            R._refactor(lu_real, R.MU_REAL / hs * Fyp + Fy)
            prob.nlu += 2
            h_lu = h
            need_factor = False

        scale = atol + rtol * np.abs(y)
        K0 = (R._predict_stages(K_prev, h / h_prev)
              if K_prev is not None and h_prev else None)
        K, converged, rate = _solve_stages_dae(
            prob, t, y, hs, yp, lu3, scale, newton_tol, K0=K0)

        if not converged:
            if not jac_current:
                F0 = prob.F(t, y, yp)
                Fy, Fyp = prob.jacs(t, y, yp, F0); jac_current = True
            else:
                h *= 0.5
                rejected = True
            need_factor = True
            if h < 1e-13 * max(1.0, abs(t)):
                status, message = -1, R._ODE_MESSAGES[-1].format(t=t)
                break
            continue

        y_new = y + hs * (R.RADAU_B @ K)
        err = _error_estimate_dae(Fyp, hs, K, yp, lu_real)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        enorm = np.sqrt(np.mean((err / scale) ** 2))

        if enorm > 1:
            h *= max(R._MIN_FACTOR, R._SAFETY * enorm ** R._ERR_EXP)
            need_factor = True
            nrej += 1
            rejected = True
            if not jac_current:
                F0 = prob.F(t, y, yp)
                Fy, Fyp = prob.jacs(t, y, yp, F0); jac_current = True
            continue

        # Accept.
        if records is not None:
            records.append((t, hs, y.copy(), K.copy()))
        K_prev = K.copy(); h_prev = h
        t = t + hs
        y = y_new
        yp = K[2].copy()                # stiffly accurate: consistent y' at t
        ts.append(t); ys.append(y.copy())
        nstep += 1
        fac = R._MAX_FACTOR if enorm == 0 else R._SAFETY * enorm ** R._ERR_EXP
        fac = min(R._MAX_FACTOR, max(R._MIN_FACTOR, fac))
        if rejected:
            fac = min(fac, 1.0); rejected = False
        if 1.0 <= fac < R._HOLD_HI:
            fac = 1.0
        h_new = min(h * fac, max_step)
        if h_new != h:
            need_factor = True
        h = h_new
        if rate is not None and rate > 1e-3:
            F0 = prob.F(t, y, yp)
            Fy, Fyp = prob.jacs(t, y, yp, F0); jac_current = True
            need_factor = True
        else:
            jac_current = False

    if status == 0 and (t - t1) * s < -1e-12:
        status, message = -1, R._ODE_MESSAGES[-2]

    ts = np.array(ts)
    ys = np.array(ys).T
    out = dict(t=ts, y=ys, nstep=nstep, nrej=nrej, nfev=prob.nfev,
               njev=prob.njev, nlu=prob.nlu, status=status, message=message,
               success=status == 0)
    if records is not None and len(records) > 0:
        sol = R._make_dense(records, n)      # collocation poly of K — reused
        out["sol"] = sol
        if t_eval is not None:
            te = np.asarray(t_eval, dtype=float)
            out["t"] = te
            out["y"] = sol(te)
    elif t_eval is not None:
        te = np.asarray(t_eval, dtype=float)
        out["t"] = te
        out["y"] = np.full((n, te.size), np.nan)
    return out
