"""Adaptive Radau IIA (order 5) integrator — the engine behind ``solve_ivp``.

A 3-stage Radau IIA collocation method (Hairer & Wanner, *Solving Ordinary
Differential Equations II*, §IV.8 — the ``RADAU5`` scheme, the same method
SciPy's ``Radau`` implements). It is L-stable and 5th-order, so it handles
**stiff** systems that explicit methods choke on, and — written in
mass-matrix form ``M y' = f(t, y)`` — it also integrates **index-1
differential-algebraic equations** when ``M`` is singular.

Each step solves the coupled 3-stage system by a simplified Newton
iteration; the ``(3n × 3n)`` stage Jacobian (and the ``n × n`` error-
estimate operator) are factored with FERAL's sparse LU
(:class:`pounce._pounce.SparseLU`). The Jacobian and its factorisation are
reused across steps and only refreshed when Newton convergence degrades —
the standard Hairer-Wanner efficiency trick.

Adaptivity uses the embedded order-3 error estimate (the ``E`` weights with
the ``(γ/h)M − J`` smoothing) and an elementary step-size controller.
"""

from __future__ import annotations

import numpy as np

from .._pounce import SparseLU

# --- Radau IIA (5) tableau (Hairer-Wanner) --------------------------------
_S6 = 6 ** 0.5
RADAU_C = np.array([(4 - _S6) / 10, (4 + _S6) / 10, 1.0])
RADAU_A = np.array([
    [(88 - 7 * _S6) / 360, (296 - 169 * _S6) / 1800, (-2 + 3 * _S6) / 225],
    [(296 + 169 * _S6) / 1800, (88 + 7 * _S6) / 360, (-2 - 3 * _S6) / 225],
    [(16 - _S6) / 36, (16 + _S6) / 36, 1 / 9],
])
RADAU_B = RADAU_A[-1]                       # stiffly accurate: y_{n+1} = Y_s
# Embedded order-3 error-estimate weights and the real eigenvalue of A^{-1}.
RADAU_E = np.array([-13 - 7 * _S6, -13 + 7 * _S6, -1.0]) / 3
MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
# Stage predictor: inverse Vandermonde on the collocation nodes. Used to warm-
# start the simplified-Newton stage solve by extrapolating the *previous*
# step's collocation polynomial (its derivative interpolates K at the nodes) to
# the new stage points, instead of cold-starting from K = 0 every step.
_PRED_VINV = np.linalg.inv(np.vander(RADAU_C, 3, increasing=True))
# Hold h (and so reuse the cached LU) when the controller's growth factor is
# below this. 1.2 matches RADAU5's reference QUOT2; 2.0 reuses the factor more
# aggressively, a net win for pounce because its (3n×3n) dense factor is
# relatively expensive — no accuracy cost, a few more steps, and the largest
# effect on big stiff systems where the LU dominates the per-step cost.
_HOLD_HI = 2.0

# Dense-output: the collocation polynomial through (y_n, stage values). We
# build per-step interpolation coefficients from the stages.
_NEWTON_MAXITER = 8
_MIN_FACTOR, _MAX_FACTOR, _SAFETY = 0.2, 10.0, 0.9
_ERR_EXP = -1 / 4                           # 1 / (error-estimator order + 1)

# Status codes / messages, following SciPy's solve_ivp convention (0 = reached
# the end of the interval; negative = the integration step failed). The solver
# never raises on a numerical failure — it returns the partial trajectory with
# a failure status, so `if res.success:` works as a drop-in.
_ODE_MESSAGES = {
    0: "The solver successfully reached the end of the integration interval.",
    -1: "Required step size is less than the smallest allowed (step underflow"
        " at t={t}); the system may be too stiff or ill-posed for this tol.",
    -2: "The maximum number of internal steps (max_steps) was reached before"
        " the end of the integration interval.",
}


def _dense_lu_pattern(N):
    """Build a reusable FERAL ``SparseLU`` over the full dense ``N × N`` pattern.

    The sparsity pattern is fixed across a whole solve — only the matrix values
    change as ``h`` and ``J`` vary — so the pattern object (and FERAL's symbolic
    analysis, which it caches internally) is built **once** and the matrix is
    refactored in place with :func:`_refactor` each step. Re-creating it per
    refactor (re-bucketing the ``N²`` COO entries and re-analysing) dominated
    the per-step cost on large systems.
    """
    idx = np.arange(N, dtype=np.int64)
    return SparseLU(N, np.repeat(idx, N), np.tile(idx, N))


def _refactor(lu, Mat):
    """Numerically refactor ``lu`` (a fixed-pattern dense ``SparseLU``) from the
    row-major values of ``Mat`` — reusing the cached symbolic analysis."""
    lu.factor(np.ascontiguousarray(Mat, dtype=np.float64).reshape(-1))


def _fd_jac(f, t, y, f0):
    """Forward-difference Jacobian ``df/dy``."""
    n = y.size
    J = np.empty((n, n))
    for j in range(n):
        d = (np.sqrt(np.finfo(float).eps)) * max(1.0, abs(y[j]))
        yj = y.copy()
        yj[j] += d
        J[:, j] = (f(t, yj) - f0) / d
    return J


class _RadauProblem:
    """Holds the RHS, mass matrix, Jacobian, and counters for one solve."""

    def __init__(self, fun, n, mass=None, jac=None):
        self.fun = fun
        self.n = n
        self.M = np.eye(n) if mass is None else np.asarray(mass, dtype=float)
        self._user_jac = jac
        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    def f(self, t, y):
        self.nfev += 1
        return np.asarray(self.fun(t, y), dtype=float)

    def jac(self, t, y, f0):
        self.njev += 1
        if self._user_jac is not None:
            return np.asarray(self._user_jac(t, y), dtype=float)
        # FD costs n extra fun evals (counted).
        J = _fd_jac(self.f, t, y, f0)
        return J


def _predict_stages(K_prev, ratio):
    """Warm-start guess for the stage derivatives from the previous step.

    The previous step's collocation polynomial has derivative ``u'(τ) = Σ aⱼτʲ``
    with ``a = Vinv @ K_prev``; its value at the new stage points (in the
    previous step's normalised coordinate ``τ = 1 + cᵢ·ratio``, where
    ``ratio = h_new / h_prev``) is the natural prediction of the new stage
    derivatives. Returns ``(3, n)``. This is the standard RADAU5 predictor and
    typically halves the Newton iterations versus a cold ``K = 0`` start.
    """
    tau = 1.0 + RADAU_C * ratio
    Vpred = np.vander(tau, 3, increasing=True)      # (3,3)
    return (Vpred @ _PRED_VINV) @ K_prev            # (3,n)


def _solve_stages(prob, t, y, h, lu3, scale, newton_tol, K0=None):
    """Simplified-Newton solve of the 3-stage system for stage derivatives K.

    ``M K_i = f(t + c_i h, y + h Σ_j A_ij K_j)``. Returns ``(K, converged,
    rate)``; ``lu3`` factors ``I_3 ⊗ M − h (A ⊗ J)`` (the Jacobian is already
    baked into this factor, reused across the Newton iterations of this step).

    Convergence uses the Hairer-Wanner criterion: the increments must shrink
    (rate ``Θ = ‖ΔK‖/‖ΔK_prev‖ < 1``) and the predicted remaining error
    ``Θ/(1−Θ)·‖ΔK‖`` must drop below ``newton_tol`` in the scaled RMS norm.
    Norms are taken in the same ``scale = atol + rtol·|y|`` used for the step
    error, so the stage solve is only as accurate as the tolerance needs.
    """
    n = prob.n
    K = np.zeros((3, n)) if K0 is None else K0.copy()
    dnorm_prev = None
    rate = None
    converged = False
    sc = scale * np.sqrt(3 * n)              # RMS denominator for (3n,) stack
    for it in range(_NEWTON_MAXITER):
        G = np.empty((3, n))
        for i in range(3):
            stage_y = y + h * (RADAU_A[i] @ K)
            G[i] = prob.M @ K[i] - prob.f(t + RADAU_C[i] * h, stage_y)
        dK = lu3.solve(-G.reshape(-1)).reshape(3, n)
        K = K + dK
        dnorm = np.linalg.norm(dK / sc)
        if dnorm_prev is not None and dnorm_prev > 0:
            rate = dnorm / dnorm_prev
            if rate >= 1:
                break          # diverging — bail, caller shrinks h / refreshes J
            # Predicted error of the converged iterate; stop once it's tiny.
            if rate / (1 - rate) * dnorm < newton_tol:
                converged = True
                break
        elif dnorm < newton_tol:
            converged = True   # already at tolerance on the first increment
            break
        dnorm_prev = dnorm
    return K, converged, rate


def _error_estimate(prob, t, y, h, K, lu_real):
    """Embedded order-3 error estimate, smoothed by ``(MU_REAL/h)M − J``.

    The ``M @`` factor multiplies the stage-increment combination for *any*
    mass matrix (it reduces to the plain ``y' = f`` form when ``M = I``), so
    it is applied unconditionally — keying it on singularity would mis-scale
    the estimate for a full-rank non-identity ``M``.
    """
    Z = h * (RADAU_A @ K)                       # stage increments Y_i − y
    f0 = prob.f(t, y)
    rhs = prob.M @ (Z.T @ (RADAU_E / h)) + f0
    return lu_real.solve(rhs)


def _select_initial_step(prob, t0, y0, f0, s, rtol, atol):
    """Hairer-Wanner / SciPy initial step heuristic (error-estimator order 3)."""
    n = y0.size
    scale = atol + np.abs(y0) * rtol
    d0 = np.linalg.norm(y0 / scale) / np.sqrt(n)
    d1 = np.linalg.norm(f0 / scale) / np.sqrt(n)
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    y1 = y0 + s * h0 * f0
    f1 = prob.f(t0 + s * h0, y1)
    d2 = np.linalg.norm((f1 - f0) / scale) / np.sqrt(n) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.25      # 1/(order+1), order=3
    return min(100 * h0, h1)


def integrate(fun, t0, t1, y0, *, rtol=1e-3, atol=1e-6, first_step=None,
              max_step=np.inf, mass=None, jac=None, t_eval=None,
              dense_output=False, max_steps=10**6):
    """Adaptive Radau IIA(5) integration of ``M y' = f`` from ``t0`` to ``t1``.

    Returns a dict with ``t``, ``y`` (shape ``(n, n_points)``), plus
    ``nfev`` / ``njev`` / ``nlu`` / ``nstep`` / ``nrej`` and (when requested)
    a dense-output callable ``sol`` and the value at ``t_eval``.
    """
    y0 = np.asarray(y0, dtype=float)
    n = y0.size
    prob = _RadauProblem(fun, n, mass=mass, jac=jac)
    forward = t1 >= t0
    s = 1.0 if forward else -1.0

    # Newton stage-solve tolerance, tied to the integration tolerance
    # (Hairer-Wanner): no point solving stages tighter than the step error.
    eps = np.finfo(float).eps
    newton_tol = max(10 * eps / rtol, min(0.03, rtol ** 0.5))

    t = t0
    y = y0.copy()
    f0 = prob.f(t, y)
    J = prob.jac(t, y, f0)

    if first_step is not None:
        h = abs(first_step)
    else:
        h = _select_initial_step(prob, t, y, f0, s, rtol, atol)
    h = min(h, abs(t1 - t0), max_step)

    # Step records for dense output: (t_start, h, y_start, K).
    records = [] if (dense_output or t_eval is not None) else None
    ts = [t]
    ys = [y.copy()]
    nstep = nrej = 0
    jac_current = True
    rejected = False

    # Cached LU factors of the (h, J)-dependent stage / error operators. The
    # RADAU5 efficiency trick is to refactor only when J is refreshed or h
    # changes — so we hold the factor across steps (see the step-size band
    # below, which freezes h on mild growth to keep reusing it).
    # Reusable LU factors over the FIXED dense patterns: the (3n×3n) stage
    # operator and the (n×n) real error operator. Only the values change across
    # steps, so the pattern objects (and FERAL's symbolic analysis) are built
    # once here and refactored in place inside the loop.
    lu3 = _dense_lu_pattern(3 * prob.n)
    lu_real = _dense_lu_pattern(prob.n)
    h_lu = None
    need_factor = True

    # Stage predictor carry: the previous accepted step's converged stage
    # derivatives and (unsigned) step, used to warm-start the next Newton solve.
    K_prev = None
    h_prev = None

    status, message = 0, _ODE_MESSAGES[0]

    while (t - t1) * s < -1e-12 and nstep < max_steps:
        h = min(h, abs(t1 - t))
        hs = s * h
        if need_factor or h != h_lu:
            big = np.kron(np.eye(3), prob.M) - hs * np.kron(RADAU_A, J)
            _refactor(lu3, big)
            _refactor(lu_real, MU_REAL / hs * prob.M - J)
            prob.nlu += 2
            h_lu = h
            need_factor = False

        scale = atol + rtol * np.abs(y)
        # Warm-start Newton from the previous step's collocation polynomial
        # (cold K=0 on the first step). Re-predicts with the current h after a
        # rejection/refactor, so a shrunk step gets a matching guess.
        K0 = (_predict_stages(K_prev, h / h_prev)
              if K_prev is not None and h_prev else None)
        K, converged, rate = _solve_stages(prob, t, y, hs, lu3, scale,
                                           newton_tol, K0=K0)
        if not converged:
            # A stale Jacobian is the usual reason simplified Newton stalls at
            # a larger step; refresh it at the current point and retry the
            # same h before giving up and shrinking (Hairer-Wanner / SciPy).
            if not jac_current:
                J = prob.jac(t, y, prob.f(t, y)); jac_current = True
            else:
                h *= 0.5            # J is fresh; the step really is too big
                rejected = True
            need_factor = True      # J or h changed -> refactor next pass
            if h < 1e-13 * max(1.0, abs(t)):
                status, message = -1, _ODE_MESSAGES[-1].format(t=t)
                break
            continue

        y_new = y + hs * (RADAU_B @ K)
        err = _error_estimate(prob, t, y, hs, K, lu_real)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        enorm = np.sqrt(np.mean((err / scale) ** 2))

        if enorm > 1:
            h *= max(_MIN_FACTOR, _SAFETY * enorm ** _ERR_EXP)
            need_factor = True      # h shrank -> refactor
            nrej += 1
            rejected = True
            if not jac_current:
                J = prob.jac(t, y, prob.f(t, y)); jac_current = True
            continue

        # Accept.
        if records is not None:
            records.append((t, hs, y.copy(), K.copy()))
        K_prev = K.copy()           # warm-start seed for the next step
        h_prev = h
        t = t + hs
        y = y_new
        ts.append(t)
        ys.append(y.copy())
        nstep += 1
        # Step growth: don't grow right after a rejection (avoids oscillation).
        fac = _MAX_FACTOR if enorm == 0 else _SAFETY * enorm ** _ERR_EXP
        fac = min(_MAX_FACTOR, max(_MIN_FACTOR, fac))
        if rejected:
            fac = min(fac, 1.0)
            rejected = False
        # Hold h on growth up to 2x so the cached LU is reused across steps
        # (Hairer-Wanner: only change h, and pay the refactor, when it buys
        # enough). A shrink always changes h and forces a refactor. The 2x band
        # matters most for large stiff systems, where the (3n×3n) refactor is
        # the dominant per-step cost.
        if 1.0 <= fac < _HOLD_HI:
            fac = 1.0
        h_new = min(h * fac, max_step)
        if h_new != h:
            need_factor = True
        h = h_new
        # Jacobian for the next step: if Newton convergence was anything but
        # excellent, refresh J now at the new point; otherwise mark it stale
        # so a later convergence failure triggers an on-demand refresh. This
        # keeps a frozen Jacobian only while it is genuinely working.
        if rate is not None and rate > 1e-3:
            f_new = prob.f(t, y)
            J = prob.jac(t, y, f_new); jac_current = True
            need_factor = True
        else:
            jac_current = False

    # Reached the cap without arriving at t1 -> a (partial) failure, not
    # success. SciPy likewise returns status<0 rather than raising.
    if status == 0 and (t - t1) * s < -1e-12:
        status, message = -1, _ODE_MESSAGES[-2]

    ts = np.array(ts)
    ys = np.array(ys).T                          # (n, n_points), SciPy layout

    out = dict(t=ts, y=ys, nstep=nstep, nrej=nrej,
               nfev=prob.nfev, njev=prob.njev, nlu=prob.nlu,
               status=status, message=message, success=status == 0)

    if records is not None and len(records) > 0:
        sol = _make_dense(records, n)
        out["sol"] = sol
        if t_eval is not None:
            te = np.asarray(t_eval, dtype=float)
            # Only evaluate t_eval points covered by the (possibly partial)
            # trajectory; clip keeps a failed solve from extrapolating wildly.
            out["t"] = te
            out["y"] = sol(te)
    elif t_eval is not None:
        # No steps were recorded (e.g. the first step underflowed) yet output
        # points were requested: return NaNs, not uninitialized memory, so a
        # caller that ignores success=False sees an obvious failure.
        te = np.asarray(t_eval, dtype=float)
        out["t"] = te
        out["y"] = np.full((n, te.size), np.nan)
    return out


def _make_dense(records, n):
    """Continuous extension via the per-step collocation polynomial.

    On ``[t_k, t_k + h]`` the collocation solution is the degree-3 polynomial
    with ``u(t_k) = y_k`` and ``u'(t_k + c_i h) = K_i``; we evaluate it from
    the Lagrange form of the stage derivatives.
    """
    starts = np.array([r[0] for r in records])
    hs = np.array([r[1] for r in records])
    ends = starts + hs
    # Locate query points regardless of integration direction. For backward
    # integration (hs < 0) the step times descend, so search on a sorted copy
    # of each step's lower time bound and map back through the permutation.
    lo = np.minimum(starts, ends)
    order = np.argsort(lo)                        # ascending search order
    lo_sorted = lo[order]

    # Precompute, per step, the monomial coefficients of u(τ), τ∈[0,1]:
    # u(τ) = y_k + h Σ_m P[m] τ^{m+1}, where P maps stage derivatives K to the
    # polynomial whose derivative interpolates K at the nodes c_i.
    # Fit u'(τ) = Σ_i K_i ℓ_i(τ) (Lagrange on nodes c), integrate.
    C = RADAU_C
    # Vandermonde for u'(τ)=Σ a_j τ^j (degree 2) matching u'(c_i)=K_i.
    V = np.vander(C, 3, increasing=True)         # (3,3)
    Vinv = np.linalg.inv(V)

    def sol(tq):
        tq = np.atleast_1d(np.asarray(tq, dtype=float))
        # locate step index for each query point (ascending search key)
        j = np.clip(np.searchsorted(lo_sorted, tq, side="right") - 1,
                    0, len(records) - 1)
        idx = order[j]
        out = np.empty((n, tq.size))
        for q in range(tq.size):
            k = idx[q]
            t_k, h, y_k, K = records[k]
            tau = (tq[q] - t_k) / h
            a = Vinv @ K                          # (3,n): coeffs of u'(τ)
            # u(τ) = y_k + h ∫_0^τ Σ a_j τ'^j dτ' = y_k + h Σ a_j τ^{j+1}/(j+1)
            powers = np.array([tau ** (j + 1) / (j + 1) for j in range(3)])
            out[:, q] = y_k + h * (powers @ a)
        return out

    return sol
