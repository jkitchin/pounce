"""Numerical-continuation / predictor–corrector path following on the
held KKT factor (pounce#90).

Traces a solution path of a parametric NLP

    min_x  f(x, θ)   s.t.  g(x, θ) = 0,  lb ≤ x ≤ ub

by *composing* the post-solve sensitivity primitives rather than
re-solving the NLP at every step:

* **predict** — extrapolate ``x`` along the held-factor sensitivity
  ``∂x*/∂θ`` (:meth:`JaxProblem.jvp_from_state`, pounce#82/#88);
* **monitor** (no solve) — the KKT residual at the predicted point plus
  the active-set margin (:meth:`JaxProblem.active_set_margin`, pounce#89);
* **correct** — only when the monitor trips: a warm-started, barrier-μ
  seeded re-solve that also re-anchors the factor in one solve
  (:meth:`JaxProblem.warm_anchor`, pounce#86).

Two entry points:

* :meth:`PathFollower.follow` — *parameter continuation* along a
  prescribed ``θ(s)``, ``s ∈ [s0, s1]``. This is the inverse /
  uncertainty-mapping and operability-tracing case: the path parameter
  ``s`` is prescribed and monotone, so there is no fold in ``s``.
* :meth:`PathFollower.trace_arclength` — *pseudo-arclength continuation*
  of the solution curve of a **scalar-parameter**, equality /
  unconstrained family, which traces **past turning points (folds)** in
  ``θ`` where ``∂x*/∂θ`` is singular and parameter continuation cannot
  proceed.

Scope (v1). Bifurcation / branch switching, Hopf detection, and general
DAE continuation are out of scope. The arclength mode handles folds for
equality / unconstrained scalar-θ systems (no inequality active-set
changes along the traced branch); inequality-active folds are not
covered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

_OK_STATUS = ("Solve_Succeeded", "Solved_To_Acceptable_Level")


@dataclass
class PathTrace:
    """Result of a path-following run.

    Attributes
    ----------
    s : (K,)
        Path parameter at each recorded point (arclength in the
        arclength mode).
    theta : (K,) + p_shape
        Parameter value at each point.
    x : (K, n)
        Primal solution at each point.
    lam : (K, m)
        Equality/inequality multipliers at each point.
    n_steps, n_correctors, n_accepts : int
        Total steps taken, of which corrected (a solve) vs. accepted on
        the predictor alone (no solve). ``n_correctors`` is the solve
        count beyond the initial anchor.
    active_set_changes : list[float]
        Path-parameter values at which the active set changed (parameter
        mode).
    turning_points : list[float]
        θ values at detected folds (arclength mode).
    status : str
        ``"ok"`` on a full traverse, or a reason string on early stop.
    """

    s: np.ndarray
    theta: np.ndarray
    x: np.ndarray
    lam: np.ndarray
    n_steps: int
    n_correctors: int
    n_accepts: int
    active_set_changes: list = field(default_factory=list)
    turning_points: list = field(default_factory=list)
    status: str = "ok"


class PathFollower:
    """Predictor–corrector path follower over a :class:`JaxProblem`
    (pounce#90).

    Parameters
    ----------
    jp : JaxProblem
        The parametric NLP to trace.
    monitor_tol : float
        KKT-residual threshold at the predicted point; above it the step
        is corrected (a warm re-solve), below it accepted on the
        predictor alone.
    active_margin_tol : float
        Active-set margin (pounce#89) threshold; a predicted point closer
        than this to a critical-region boundary forces a correction so
        the predictor never extrapolates across a discontinuity.
    ds0, ds_min, ds_max : float
        Initial / minimum / maximum step in the path parameter.
    grow, shrink : float
        Step-size adaptation factors (on accept / easy correction vs.
        hard or failed correction).
    max_steps : int
        Safety cap on the number of steps.
    """

    def __init__(
        self,
        jp,
        *,
        monitor_tol: float = 1e-6,
        active_margin_tol: float = 1e-4,
        ds0: float = 0.05,
        ds_min: float = 1e-4,
        ds_max: float = 0.25,
        grow: float = 1.5,
        shrink: float = 0.5,
        max_steps: int = 100_000,
    ):
        self._jp = jp
        self._monitor_tol = float(monitor_tol)
        self._active_margin_tol = float(active_margin_tol)
        self._ds0 = float(ds0)
        self._ds_min = float(ds_min)
        self._ds_max = float(ds_max)
        self._grow = float(grow)
        self._shrink = float(shrink)
        self._max_steps = int(max_steps)

    # ----- monitors -----

    def _kkt_residual(self, x, theta, lam, zL, zU) -> float:
        """First-order optimality residual at ``(x, θ)`` with the given
        multipliers — the smooth-drift monitor. No solve."""
        jp = self._jp
        m = jp._m
        gradf = jax.grad(lambda xx: jp._f(xx, theta))(x)
        stat = gradf - zL + zU
        cviol = 0.0
        if m > 0:
            gx = jp._g(x, theta)
            Jg = jax.jacobian(lambda xx: jp._g(xx, theta))(x)
            stat = stat + Jg.T @ lam
            cviol = float(jnp.max(jnp.abs(gx)))
        r = float(jnp.max(jnp.abs(stat))) + cviol
        if jp._lb is not None:
            r += float(jnp.max(jnp.maximum(0.0, jnp.asarray(jp._lb) - x)))
        if jp._ub is not None:
            r += float(jnp.max(jnp.maximum(0.0, x - jnp.asarray(jp._ub))))
        return r

    def _margin_at(self, x, lam, zL, zU, theta) -> float:
        """Active-set margin at an explicit predicted point (pounce#89)."""
        r = self._jp._margin_arrays(
            x[None], lam[None], zL[None], zU[None],
            jnp.asarray(theta)[None], 1,
        )
        return float(r["margin"][0])

    def _active_signature(self, lam, zL, zU, active_tol=1e-6):
        """Boolean active-set fingerprint: which bounds / inequalities are
        active. Used to detect active-set changes across corrections."""
        jp = self._jp
        sig = (
            tuple(bool(v) for v in np.asarray(zL > active_tol))
            + tuple(bool(v) for v in np.asarray(zU > active_tol))
        )
        if jp._m > 0:
            cl = np.asarray(jp._cl_for_classify)
            cu = np.asarray(jp._cu_for_classify)
            is_ineq = cl != cu
            lam_np = np.asarray(lam)
            sig = sig + tuple(
                bool(is_ineq[i] and abs(lam_np[i]) > active_tol)
                for i in range(jp._m)
            )
        return sig

    # ----- parameter continuation -----

    def follow(self, theta_of_s, s_span, x0) -> PathTrace:
        """Trace ``x*(θ(s))`` for a prescribed path ``θ(s)``,
        ``s ∈ [s0, s1]`` (pounce#90).

        Anchors once with a cold solve at ``θ(s0)``, then steps: predict
        with the held-factor sensitivity, monitor (KKT residual +
        active-set margin) with no solve, and correct (warm-μ re-solve +
        re-anchor) only when the monitor trips. The step size adapts to
        the corrector effort and backs off near active-set boundaries.

        Parameters
        ----------
        theta_of_s : callable
            ``s (float) -> θ`` (shape ``p_shape``).
        s_span : (float, float)
            ``(s0, s1)`` with ``s1 > s0``.
        x0 : (n,)
            Primal initial guess for the anchor solve.

        Returns
        -------
        PathTrace
        """
        jp = self._jp
        s0, s1 = float(s_span[0]), float(s_span[1])
        if not (s1 > s0):
            raise ValueError("s_span must have s1 > s0")

        def th(s):
            return jnp.asarray(theta_of_s(s), dtype=jnp.float64)

        # Anchor: cold solve at s0 (no dual / μ seed).
        state, info = jp.warm_anchor(th(s0), jnp.asarray(x0, dtype=jnp.float64))
        if info["status_msg"] not in _OK_STATUS:
            state.close()
            raise RuntimeError(
                f"pounce.jax: anchor solve failed at s0 "
                f"({info['status_msg']})."
            )
        x = jnp.asarray(state.x_star[0])
        lam, zL, zU = (jnp.asarray(d[0]) for d in state.duals)
        mu = float(info["mu"])
        sig = self._active_signature(lam, zL, zU)

        S = [s0]
        TH = [np.asarray(th(s0))]
        X = [np.asarray(x)]
        LAM = [np.asarray(lam)]
        n_corr = 0
        n_acc = 0
        n_steps = 0
        as_changes: list = []
        status = "ok"

        s = s0
        ds = self._ds0
        while s < s1 - 1e-12 and n_steps < self._max_steps:
            n_steps += 1
            ds = min(ds, s1 - s)
            s_new = s + ds
            th_cur = th(s)
            th_new = th(s_new)
            dth = th_new - th_cur

            # PREDICT: step both primal and duals along the sensitivity
            # (∂x*/∂θ, ∂λ*/∂θ) · dθ — a held-factor back-solve. Predicting
            # the multipliers too is what lets the KKT-residual monitor
            # recognise an accurate predictor (a stale λ would otherwise
            # inflate the residual even at an exact x_pred).
            if jp._m > 0:
                dx, dlam = jp.jvp_from_state(state, dth, with_duals=True)
                lam_pred = lam + dlam
            else:
                dx = jp.jvp_from_state(state, dth)
                lam_pred = lam
            x_pred = x + dx

            # MONITOR (no solve): residual + active-set margin at the
            # predicted point.
            r = self._kkt_residual(x_pred, th_new, lam_pred, zL, zU)
            margin = self._margin_at(x_pred, lam_pred, zL, zU, th_new)

            if r <= self._monitor_tol and margin > self._active_margin_tol:
                # Accept on the predictor — no solve. Carry the predicted
                # primal and duals forward; the next predictor keeps
                # extrapolating from the same held factor.
                x = x_pred
                lam = lam_pred
                s = s_new
                n_acc += 1
                S.append(s)
                TH.append(np.asarray(th_new))
                X.append(np.asarray(x))
                LAM.append(np.asarray(lam))
                ds = min(ds * self._grow, self._ds_max)
                continue

            # CORRECT: warm-μ re-solve from the predicted point, which
            # also re-anchors the held factor for the next predictor.
            new_state, cinfo = jp.warm_anchor(
                th_new, x_pred, duals=(lam, zL, zU), mu=mu,
            )
            if cinfo["status_msg"] not in _OK_STATUS:
                # Back off and retry a shorter step from the same anchor.
                new_state.close()
                ds *= self._shrink
                if ds < self._ds_min:
                    status = "corrector_failed"
                    break
                continue

            state.close()
            state = new_state
            x = jnp.asarray(state.x_star[0])
            lam, zL, zU = (jnp.asarray(d[0]) for d in state.duals)
            mu = float(cinfo["mu"])
            n_corr += 1
            s = s_new

            new_sig = self._active_signature(lam, zL, zU)
            if new_sig != sig:
                as_changes.append(s)
                sig = new_sig
                # Resolve the region near the change finely.
                ds = min(self._ds0, max(ds * self._shrink, self._ds_min))
            else:
                iters = int(cinfo["iter_count"])
                if iters <= 3:
                    ds = min(ds * self._grow, self._ds_max)
                elif iters >= 10:
                    ds = max(ds * self._shrink, self._ds_min)

            S.append(s)
            TH.append(np.asarray(th_new))
            X.append(np.asarray(x))
            LAM.append(np.asarray(lam))

        state.close()
        if n_steps >= self._max_steps and s < s1 - 1e-12:
            status = "max_steps"
        return PathTrace(
            s=np.asarray(S),
            theta=np.asarray(TH),
            x=np.asarray(X),
            lam=np.asarray(LAM),
            n_steps=n_steps,
            n_correctors=n_corr,
            n_accepts=n_acc,
            active_set_changes=as_changes,
            turning_points=[],
            status=status,
        )

    # ----- pseudo-arclength continuation (folds) -----

    def trace_arclength(
        self,
        x0,
        theta0,
        *,
        ds: float = 0.05,
        n_steps: int = 200,
        direction: float = 1.0,
        newton_tol: float = 1e-9,
        newton_max: int = 40,
    ) -> PathTrace:
        """Pseudo-arclength continuation of the solution curve for a
        **scalar** parameter ``θ``, tracing *past folds* (pounce#90).

        Solves the stationarity / feasibility system ``R(x, λ, θ) = 0``
        — ``R = [∇_x f + J_gᵀλ ; g]`` — along the arclength of its
        solution curve in ``(x, λ, θ)`` space, with a tangent predictor
        and a Newton corrector on the augmented system ``[R ; arclength]``.
        Because the curve is parametrised by arclength rather than ``θ``,
        it passes through turning points where ``∂x*/∂θ`` is singular
        (the fold), which parameter continuation cannot.

        Restricted to equality / unconstrained problems (the active set is
        fixed along the traced branch). Bounds / inequality-active folds
        are out of scope for v1.

        Parameters
        ----------
        x0 : (n,)
            Primal guess near a point on the curve; projected onto
            ``R = 0`` at ``θ0`` before tracing.
        theta0 : float
            Starting (scalar) parameter value.
        ds : float
            Arclength step.
        n_steps : int
            Number of arclength steps.
        direction : float
            Sign of the initial step in ``θ`` (+1 increasing).
        newton_tol, newton_max : float, int
            Corrector Newton tolerance and iteration cap.

        Returns
        -------
        PathTrace  (``turning_points`` lists the θ values at detected folds)
        """
        jp = self._jp
        n, m = jp._n, jp._m
        if len(jp._p_shape) != 0 and jp._p_shape != (1,):
            raise ValueError(
                "trace_arclength supports a scalar parameter only "
                f"(p_shape={jp._p_shape}); use follow(...) for a path."
            )
        d = n + m  # number of equations in R

        def _theta_arg(theta_scalar):
            # Present θ to the user callables in their declared shape.
            return theta_scalar if jp._p_shape == () else theta_scalar[None]

        def R(u):
            x = u[:n]
            theta = _theta_arg(u[d])
            gradf = jax.grad(lambda xx: jp._f(xx, theta))(x)
            if m > 0:
                lam = u[n:d]
                Jg = jax.jacobian(lambda xx: jp._g(xx, theta))(x)
                return jnp.concatenate([gradf + Jg.T @ lam, jp._g(x, theta)])
            return gradf

        R_jit = jax.jit(R)
        dR_jit = jax.jit(jax.jacobian(R))  # (d, d+1)

        f64 = jnp.float64
        u = jnp.concatenate([
            jnp.asarray(x0, dtype=f64),
            jnp.zeros(m, dtype=f64),
            jnp.asarray([float(theta0)], dtype=f64),
        ])

        # Project the initial point onto R = 0 at fixed θ0 (Newton on the
        # first d coordinates).
        for _ in range(newton_max):
            res = R_jit(u)
            if float(jnp.max(jnp.abs(res))) < newton_tol:
                break
            A = dR_jit(u)[:, :d]
            u = u.at[:d].add(-jnp.linalg.solve(A, res))

        def tangent(u_at, t_prev):
            A = dR_jit(u_at)                 # (d, d+1)
            _, _, Vt = jnp.linalg.svd(A)
            t = Vt[-1]
            t = t / jnp.linalg.norm(t)
            if t_prev is None:
                if float(t[d]) * direction < 0:
                    t = -t
            elif float(jnp.dot(t, t_prev)) < 0:
                t = -t
            return t

        t = tangent(u, None)
        X = [np.asarray(u[:n])]
        TH = [float(u[d])]
        LAM = [np.asarray(u[n:d])]
        turning: list = []
        status = "ok"

        for _ in range(n_steps):
            u_pred = u + ds * t
            uc = u_pred
            ok = False
            for _ in range(newton_max):
                res = R_jit(uc)
                arc = jnp.dot(t, uc - u_pred)
                F = jnp.concatenate([res, jnp.asarray([arc])])
                if float(jnp.max(jnp.abs(F))) < newton_tol:
                    ok = True
                    break
                A = dR_jit(uc)                       # (d, d+1)
                Aaug = jnp.concatenate([A, t[None, :]], axis=0)  # (d+1, d+1)
                uc = uc - jnp.linalg.solve(Aaug, F)
            if not ok:
                status = "corrector_failed"
                break
            u = uc
            t_new = tangent(u, t)
            # Fold = the θ-component of the tangent changes sign.
            if float(t[d]) * float(t_new[d]) < 0:
                turning.append(float(u[d]))
            t = t_new
            X.append(np.asarray(u[:n]))
            TH.append(float(u[d]))
            LAM.append(np.asarray(u[n:d]))

        K = len(X)
        return PathTrace(
            s=np.arange(K) * ds,
            theta=np.asarray(TH),
            x=np.asarray(X),
            lam=np.asarray(LAM),
            n_steps=K - 1,
            n_correctors=K - 1,
            n_accepts=0,
            active_set_changes=[],
            turning_points=turning,
            status=status,
        )
