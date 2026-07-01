"""Adaptive Radau IIA (order 5) integrator — the engine behind ``solve_ivp``.

A 3-stage Radau IIA collocation method (Hairer & Wanner, *Solving Ordinary
Differential Equations II*, §IV.8 — the ``RADAU5`` scheme, the same method
SciPy's ``Radau`` implements). It is L-stable and 5th-order, so it handles
**stiff** systems that explicit methods choke on, and — written in
mass-matrix form ``M y' = f(t, y)`` — it also integrates **index-1
differential-algebraic equations** when ``M`` is singular.

Each step solves the coupled 3-stage system by a simplified Newton
iteration; the ``(3n × 3n)`` stage Jacobian (and the ``n × n`` error-
estimate operator) are dense and tiny, so they are factored with a dense
partial-pivoting LU (:class:`pounce._pounce.DenseLU`, faer-backed) — the
same dense factorisation SciPy's ``Radau`` uses. The Jacobian and its
factorisation are reused across steps and only refreshed when Newton
convergence degrades — the standard Hairer-Wanner efficiency trick.

Adaptivity uses the embedded order-3 error estimate (the ``E`` weights with
the ``(γ/h)M − J`` smoothing) and an elementary step-size controller.
"""

from __future__ import annotations

import numpy as np

from .._pounce import DenseLU
from ._jacobian import make_ode_jac

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
# RADAU5 eigendecomposition of the Butcher matrix (A = T Λ T⁻¹). The simplified
# Newton stage solve runs in the eigenbasis W = T⁻¹ Z, where it decouples into
# one real and one complex n×n system against the *shifted* operators
# (μ/h·M − J) — which stay nonsingular even when J is singular (e.g. Robertson
# at equilibrium), unlike the full I₃⊗M − h(A⊗J) operator (pounce#175).
# MU_REAL / MU_COMPLEX are the eigenvalues of A⁻¹; T / TI = T⁻¹ are the standard
# Radau IIA(5) transform matrices (Hairer-Wanner, matching SciPy's Radau).
MU_COMPLEX = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
              - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
_T = np.array([
    [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
    [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
    [1.0, 1.0, 0.0]])
_TI = np.array([
    [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
    [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
    [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])
_TI_REAL = _TI[0]
_TI_COMPLEX = _TI[1] + 1j * _TI[2]
# Stage values ↔ derivatives: Z_i = h Σ_j A_ij K_j, so K = (1/h) A⁻¹ Z.
RADAU_AINV = np.linalg.inv(RADAU_A)
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
    1: "A termination event occurred.",
}


def _normalize_events(events):
    """SciPy-style events -> (funcs, directions, terminal-counts). ``terminal``
    may be ``bool`` or a positive ``int`` (stop after that many occurrences);
    ``direction`` filters rising (>0) / falling (<0) / either (0) crossings."""
    if events is None:
        return None, None, None
    if callable(events):
        events = [events]
    funcs = list(events)
    dirs = np.array([float(getattr(e, "direction", 0.0) or 0.0) for e in funcs])
    terms = [int(getattr(e, "terminal", False) or 0) for e in funcs]
    return funcs, dirs, terms


def _bisect_root(g, p, q, gp, gq, xtol=1e-12, maxiter=100):
    """Root of ``g`` bracketed by ``[p, q]`` with ``g(p)=gp``, ``g(q)=gq`` of
    opposite sign (order-agnostic in ``p``/``q``)."""
    for _ in range(maxiter):
        c = 0.5 * (p + q)
        gc = g(c)
        if gc == 0.0 or abs(q - p) <= xtol * (1.0 + abs(c)):
            return c
        if (gp < 0.0) != (gc < 0.0):
            q, gq = c, gc
        else:
            p, gp = c, gc
    return 0.5 * (p + q)


def _detect_events(events, dirs, t0, g0, t1, g1, step_data, n):
    """Crossings of each event on the just-accepted step ``[t0, t1]``.

    ``step_data = (t_k, h_signed, y_k, K)`` builds the step's collocation
    polynomial for the root-find. Returns ``[(idx, t_root, y_root), ...]``,
    direction-filtered, ordered by event index.
    """
    sol = _make_dense([step_data], n)
    found = []
    for i, ev in enumerate(events):
        a, b = g0[i], g1[i]
        cross = (a < 0.0 < b) or (a > 0.0 > b)
        if not cross:
            continue
        d = dirs[i]
        if d != 0.0 and not ((d > 0.0 and a < 0.0) or (d < 0.0 and a > 0.0)):
            continue
        gfun = lambda tt: float(ev(tt, sol(tt)[:, 0]))
        tr = _bisect_root(gfun, t0, t1, a, b)
        found.append((i, tr, sol(tr)[:, 0]))
    return found


def _dense_lu_pattern(N):
    """Build a reusable dense ``N × N`` LU (faer partial-pivoting, via
    :class:`pounce._pounce.DenseLU`).

    The stage / error operators are dense and tiny; a dense LU is both faster
    than a sparse factorisation on these blocks and, like LAPACK, stays
    permissive on the ill-conditioned (large-``h``) operators a stiff/DAE
    problem reaches on its slow manifold — where the old sparse-LU path
    hard-failed as ``SingularBasis`` (pounce#175). Only the values change
    across steps, so the object is built once and re-:func:`_refactor`-ed in
    place each step.
    """
    return DenseLU(N)


def _refactor(lu, Mat):
    """Numerically refactor ``lu`` from the row-major values of ``Mat``."""
    lu.factor(np.ascontiguousarray(Mat, dtype=np.float64).reshape(-1))


class _RadauProblem:
    """Holds the RHS, mass matrix, Jacobian, and counters for one solve."""

    def __init__(self, fun, n, mass=None, jac=None):
        self.fun = fun
        self.n = n
        self.M = np.eye(n) if mass is None else np.asarray(mass, dtype=float)
        self._user_jac = jac
        self._auto_jac = None      # lazily built JAX-or-central-diff strategy
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
        # No analytic Jacobian: use exact JAX autodiff when the RHS is
        # traceable, else an accurate central difference (see _jacobian.py).
        if self._auto_jac is None:
            self._auto_jac = make_ode_jac(self.fun, self.f)
        return self._auto_jac(t, y, f0)


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


def _solve_stages(prob, t, y, h, lu_real, lu_cplx, scale, newton_tol, K0=None):
    """Simplified-Newton solve of the 3-stage system via the RADAU5
    eigendecomposition.

    Solves ``M K_i = f(t + c_i h, y + h Σ_j A_ij K_j)``. Rather than factoring
    the full ``I₃⊗M − h(A⊗J)`` operator (which inherits any singularity of
    ``J`` — Robertson at equilibrium, pounce#175), the iteration runs in the
    eigenbasis ``W = T⁻¹ Z`` of the Butcher matrix, where it decouples into one
    real and one complex ``n×n`` solve against the shifted operators
    ``μ_real/h·M − J`` (``lu_real``) and ``μ_cplx/h·M − J`` (``lu_cplx``, held as
    its real ``2n×2n`` embedding ``[[Re,−Im],[Im,Re]]``). Those stay
    well-conditioned even at a singular-``J`` equilibrium, so the warm start is
    corrected properly instead of leaving an uncorrectable near-null component.

    Iterates on stage *values* ``Z`` (``Z_i = Y_i − y``) internally and returns
    the stage *derivatives* ``K = (1/h) A⁻¹ Z`` for the caller, so the rest of
    the integrator (step update, error estimate, dense output) is unchanged.
    Convergence is the Hairer-Wanner criterion (SciPy's Radau): increments must
    contract (rate ``Θ < 1``) with predicted remaining error
    ``Θ/(1−Θ)·‖ΔW‖`` below ``newton_tol`` in the scaled RMS norm.
    """
    n = prob.n
    m_real = MU_REAL / h
    m_cplx = MU_COMPLEX / h
    Z = np.zeros((3, n)) if K0 is None else h * (RADAU_A @ K0)
    W = _TI @ Z
    sc = scale * np.sqrt(3 * n)              # RMS denominator for (3n,) stack
    dnorm_prev = None
    rate = None
    converged = False
    for it in range(_NEWTON_MAXITER):
        F = np.empty((3, n))
        for i in range(3):
            F[i] = prob.f(t + RADAU_C[i] * h, y + Z[i])
        if not np.all(np.isfinite(F)):
            break              # blew up — bail, caller shrinks h / refreshes J
        f_real = F.T @ _TI_REAL - m_real * (prob.M @ W[0])
        f_cplx = F.T @ _TI_COMPLEX - m_cplx * (prob.M @ (W[1] + 1j * W[2]))
        dW_real = lu_real.solve(f_real)
        sol = lu_cplx.solve(np.concatenate([f_cplx.real, f_cplx.imag]))
        dW = np.array([dW_real, sol[:n], sol[n:]])
        dnorm = np.linalg.norm(dW / sc)
        if dnorm_prev is not None and dnorm_prev > 0:
            rate = dnorm / dnorm_prev
            # Bail if not contracting (Θ ≥ 1) or the predicted remaining error
            # won't reach tol in the iterations left — caller shrinks h.
            if rate >= 1 or rate ** (_NEWTON_MAXITER - it) / (1 - rate) * dnorm > newton_tol:
                break
        W = W + dW
        Z = _T @ W
        if dnorm == 0.0 or (rate is not None and rate / (1 - rate) * dnorm < newton_tol):
            converged = True
            break
        dnorm_prev = dnorm
    K = (RADAU_AINV @ Z) / h
    return K, converged, rate, it + 1


def _error_estimate(prob, t, y, h, K, lu_real, refine_from=None):
    """Embedded order-3 error estimate, smoothed by ``(MU_REAL/h)M − J``.

    The ``M @`` factor multiplies the stage-increment combination for *any*
    mass matrix (it reduces to the plain ``y' = f`` form when ``M = I``), so
    it is applied unconditionally — keying it on singularity would mis-scale
    the estimate for a full-rank non-identity ``M``.

    ``refine_from`` performs the Hairer-Wanner / SciPy nonlinear refinement:
    re-evaluate ``f`` at ``y + refine_from`` (a previous error estimate) instead
    of at ``y``, so a linear estimate that is over-pessimistic at large ``h`` on
    a stiff problem doesn't force a needless step rejection.
    """
    Z = h * (RADAU_A @ K)                       # stage increments Y_i − y
    fpt = prob.f(t, y if refine_from is None else y + refine_from)
    rhs = prob.M @ (Z.T @ (RADAU_E / h)) + fpt
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
              dense_output=False, max_steps=10**6, events=None):
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

    # Cached LU factors of the (h, J)-dependent operators. The RADAU5 efficiency
    # trick is to refactor only when J is refreshed or h changes — so we hold
    # the factors across steps (see the step-size band below, which freezes h on
    # mild growth to keep reusing them). The eigendecomposed stage solve uses two
    # shifted operators: the real (n×n) ``μ_real/h·M − J`` (which doubles as the
    # error-estimate operator) and the complex ``μ_cplx/h·M − J``, held as its
    # real (2n×2n) embedding — both far better conditioned than the full
    # (3n×3n) ``I₃⊗M − h(A⊗J)`` and cheaper to factor.
    lu_real = _dense_lu_pattern(prob.n)
    lu_cplx = _dense_lu_pattern(2 * prob.n)
    h_lu = None
    need_factor = True

    # Stage predictor carry: the previous accepted step's converged stage
    # derivatives and (unsigned) step, used to warm-start the next Newton solve.
    K_prev = None
    h_prev = None

    status, message = 0, _ODE_MESSAGES[0]

    # Event detection (SciPy-style): track each event function's value at the
    # last accepted point and root-find a crossing on each step's dense poly.
    ev_funcs, ev_dirs, ev_terms = _normalize_events(events)
    if ev_funcs is not None:
        g_prev = np.array([float(e(t, y)) for e in ev_funcs])
        t_events = [[] for _ in ev_funcs]
        y_events = [[] for _ in ev_funcs]
        ev_count = [0 for _ in ev_funcs]

    while (t - t1) * s < -1e-12 and nstep < max_steps:
        h = min(h, abs(t1 - t))
        hs = s * h
        if need_factor or h != h_lu:
            # Shifted RADAU5 operators (eigenbasis of the Butcher matrix): the
            # real ``μ_real/h·M − J`` and the complex ``μ_cplx/h·M − J``, the
            # latter held as its real 2n×2n embedding ``[[Re,−Im],[Im,Re]]``.
            mc = MU_COMPLEX / hs
            c_re = mc.real * prob.M - J
            c_im = mc.imag * prob.M
            try:
                _refactor(lu_real, MU_REAL / hs * prob.M - J)
                _refactor(lu_cplx, np.block([[c_re, -c_im], [c_im, c_re]]))
            except RuntimeError:
                # A stage-operator factorization failure is treated like a
                # rejected step (shrink h, refresh a stale J, retry) rather than
                # killing the integration — the Hairer-Wanner RADAU5 response to
                # a singular decomposition, mirroring the Newton-non-convergence
                # branch below. The dense-LU backend factors near-singular
                # operators like LAPACK, so this is a safety net for a genuinely
                # unrecoverable (inf/nan) operator rather than the common path.
                if not jac_current:
                    J = prob.jac(t, y, prob.f(t, y)); jac_current = True
                else:
                    h *= 0.5
                    rejected = True
                need_factor = True
                if h < 1e-13 * max(1.0, abs(t)):
                    status, message = -1, _ODE_MESSAGES[-1].format(t=t)
                    break
                continue
            prob.nlu += 2
            h_lu = h
            need_factor = False

        scale = atol + rtol * np.abs(y)
        # Warm-start Newton from the previous step's collocation polynomial
        # (cold K=0 on the first step). Re-predicts with the current h after a
        # rejection/refactor, so a shrunk step gets a matching guess.
        K0 = (_predict_stages(K_prev, h / h_prev)
              if K_prev is not None and h_prev else None)
        K, converged, rate, n_iter = _solve_stages(prob, t, y, hs, lu_real,
                                                   lu_cplx, scale, newton_tol,
                                                   K0=K0)
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
        if rejected and enorm > 1:
            # Nonlinear refinement of the stiff error estimate (so an
            # over-pessimistic linear estimate at large h doesn't needlessly
            # reject step growth on the slow manifold — Hairer-Wanner / SciPy).
            err = _error_estimate(prob, t, y, hs, K, lu_real, refine_from=err)
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

        if ev_funcs is not None:
            t_new = t + hs
            g_new = np.array([float(e(t_new, y_new)) for e in ev_funcs])
            found = _detect_events(ev_funcs, ev_dirs, t, g_prev, t_new, g_new,
                                   (t, hs, y, K), n)
            g_prev = g_new
            term_t = term_y = None
            for (i, tr, yr) in found:
                t_events[i].append(tr)
                y_events[i].append(yr)
                ev_count[i] += 1
                if ev_terms[i] and ev_count[i] >= ev_terms[i]:
                    if term_t is None or abs(tr - t) < abs(term_t - t):
                        term_t, term_y = tr, yr
            if term_t is not None:          # stop at the earliest terminal event
                t, y = term_t, term_y
                ts.append(t); ys.append(y.copy()); nstep += 1
                status, message = 1, _ODE_MESSAGES[1]
                break

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
        # Jacobian for the next step (SciPy Radau policy): recompute J only when
        # the just-completed Newton was *both* slow (took more than 2 iterations)
        # and contracting poorly (rate > 1e-3). Otherwise keep J frozen and skip
        # the refactor. Near a singular-Jacobian steady state (e.g. the Robertson
        # DAE on its slow manifold) the dynamics change slowly, so a frozen J
        # stays valid for many steps and the simplified Newton converges in ≤ 2
        # iterations — refreshing it every step instead just burns evaluations
        # and factorisations. A stale J that later stalls Newton is refreshed
        # on-demand by the non-convergence branch above.
        if rate is not None and n_iter > 2 and rate > 1e-3:
            J = prob.jac(t, y, prob.f(t, y)); jac_current = True
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
               status=status, message=message, success=status >= 0)
    if ev_funcs is not None:
        out["t_events"] = [np.array(te) for te in t_events]
        out["y_events"] = [np.array(ye).reshape(-1, n) for ye in y_events]

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
