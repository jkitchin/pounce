"""POUNCE adapter — the only file in the suite that imports pounce.

Maps the four arms onto pounce's two algorithm paths:

===========  ===================================  =========================
arm          ``algorithm``                        warm-start payload
===========  ===================================  =========================
cold-ipm     ``interior-point`` (default)         none
cold-sqp     ``active-set-sqp``                   none
warm-ipm     ``interior-point``                   ``WarmStart`` (x, λ, z, μ)
warm-sqp     ``active-set-sqp``                   working set + previous x
pred-ipm     ``interior-point``                   previous state, primal seed
                                                  stepped by the held-factor
                                                  tangent
predcorr-ipm ``interior-point``                   as above, multipliers
                                                  stepped too, reanchored on
                                                  an active-set event
cold-qp-ipm  ``pounce.solve_qp`` (pounce-convex)  none
warm-qp-ipm  ``pounce.solve_qp``                  previous ``QpResult``
===========  ===================================  =========================

Two deliberate choices worth knowing when reading the numbers:

* **The ``Problem`` is rebuilt at every step.** Several families move
  their variable or constraint bounds with θ, and bounds are fixed at
  construction. Rebuilding everywhere keeps the arms symmetric, and
  costs nothing in the measurement because only the ``solve()`` call
  is timed. It also sidesteps the fact that ``WarmStart`` applies its
  enabling options through ``add_option``, which would otherwise
  persist on a reused handle and quietly contaminate a later solve.

* **The QP arms do not go through the callback interface at all.** The
  convex solver takes matrix data, so each step assembles ``(P, c, A,
  b, G, h, lb, ub)`` from the family (see :mod:`..qpform`) and hands it
  over. The assembly is inside the timed region and routed through the
  harness's counters, because it is work a caller genuinely has to do —
  but it happens *once per step*, where the callback-driven arms
  re-evaluate every iteration. That is a real advantage of the QP path
  on a QP, not a measurement artifact, and it is why the report keeps
  these arms in their own section rather than in the headline table.
  ``check_psd`` is off: the suite's self-test already proves ``P`` is
  positive semidefinite for every family that claims to be a QP, so
  leaving the guard on would time an O(n³) eigenvalue decomposition
  that a caller who knew their problem would not pay.

* **Tolerances are pinned on both paths** rather than left at their
  defaults, since the two paths have separate convergence-test knobs
  and an iteration-count comparison across differently-converged
  solves means nothing. The achieved KKT error is recorded per step
  so any residual asymmetry is visible in the data instead of being
  taken on trust.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

import pounce

from .. import qpform
from ..spec import ParametricFamily, StepResult, WarmState
from ..sparsity import SparseCallbacks
from .base import (
    ARMS,
    QP_ARMS,
    SolverAdapter,
    is_sqp,
    is_warm,
    predicts_duals,
    uses_homotopy,
    uses_predictor,
)

# ApplicationReturnStatus values that count as a solved step.
_OK_STATUS = (0, 1)  # SolveSucceeded, SolvedToAcceptableLevel

# `solve_qp` status strings that count as a solved step.
_OK_QP_STATUS = ("optimal", "optimal_inaccurate")


class PounceAdapter(SolverAdapter):
    name = "pounce"

    def __init__(self, max_iter: int = 500, recentering: str = "residual"):
        self.max_iter = max_iter
        # `warm_start_recentering` (pounce#606 / #620). The warm-start
        # baseline this corpus reports moves with it, so it is a swept
        # axis rather than a default: see docs/src/continuation.md.
        self.recentering = recentering

    def supports(self, arm: str) -> bool:
        return arm in ARMS

    # -- problem construction --------------------------------------

    def _build(self, family: ParametricFamily, callbacks, arm: str, tol: float):
        b = family.bounds()
        prob = pounce.Problem(
            n=family.n,
            m=family.m,
            problem_obj=callbacks,
            lb=b.lb,
            ub=b.ub,
            cl=b.cl,
            cu=b.cu,
        )
        prob.add_option("print_level", 0)
        prob.add_option("sb", "yes")
        if is_sqp(arm):
            prob.add_option("algorithm", "active-set-sqp")
            prob.add_option("sqp_print_level", 0)
            prob.add_option("sqp_tol", tol)
            prob.add_option("sqp_constr_viol_tol", 1e-6)
            # The only difference between an arm and its `-hom` twin.
            prob.add_option(
                "sqp_qp_use_homotopy", "yes" if uses_homotopy(arm) else "no"
            )
            prob.add_option("sqp_max_iter", self.max_iter)
        else:
            prob.add_option("tol", tol)
            prob.add_option("constr_viol_tol", 1e-6)
            prob.add_option("max_iter", self.max_iter)
        return prob

    # -- the convex-QP path ----------------------------------------

    def _solve_qp_ipm(
        self,
        family: ParametricFamily,
        callbacks: SparseCallbacks,
        arm: str,
        warm: Optional[WarmState],
        step: int,
        tol: float,
    ) -> Tuple[StepResult, Optional[WarmState]]:
        callbacks.reset_counts()
        seed = warm.extra.get("qp") if (warm is not None and warm.extra) else None

        t0 = time.perf_counter()
        qp = qpform.extract(family, callbacks)
        res = pounce.solve_qp(
            P=qp.P,
            c=qp.c,
            A=qp.A,
            b=qp.b,
            G=qp.G,
            h=qp.h,
            lb=qp.lb,
            ub=qp.ub,
            tol=tol,
            max_iter=self.max_iter,
            warm_start=seed if is_warm(arm) else None,
            check_psd=False,
        )
        elapsed = time.perf_counter() - t0

        resid = res.residuals or {}
        result = StepResult(
            step=step,
            theta=[],
            success=res.status in _OK_QP_STATUS,
            status=0 if res.status in _OK_QP_STATUS else -1,
            status_msg=str(res.status),
            iters=int(res.iters),
            solve_time=elapsed,
            # The QP form drops the objective's constant term; add it
            # back or every objective comparison against the other arms
            # is off by a per-step offset.
            obj=float(res.obj) + qp.f0,
            kkt_error=float(resid.get("kkt_error", np.nan)),
            constr_viol=float(resid.get("primal_infeasibility", np.nan)),
            # No working set: this is an interior-point method.
            n_active=None,
            n_qp_solves=None,
            n_qp_ws_changes=None,
            **callbacks.counts(),
        )
        next_warm = WarmState(
            x=np.asarray(res.x, dtype=float).copy(),
            extra={"qp": res},
        )
        return result, next_warm

    # -- the tangent predictor (pounce#608) ------------------------

    def _predict(self, family, warm, arm, x, lam, zl, zu):
        """Step the previous state along ``∂·/∂θ`` for this step's Δθ.

        The sensitivity is a back-solve against the previous solve's
        held factor, carried in `warm.extra`. Everything here degrades
        to the plain warm start rather than failing: a factor that
        cannot answer, a missing Δθ, or an active-set event on the
        previous step all return the seed unchanged, which is the
        zero-order transfer.
        """
        extra = warm.extra or {}
        session = extra.get("session")
        theta_prev = extra.get("theta")
        theta_now = family.current_theta()
        if session is None or theta_prev is None or theta_now is None:
            return x, lam, zl, zu
        dtheta = np.asarray(theta_now, float) - np.asarray(theta_prev, float)
        if not dtheta.size:
            return x, lam, zl, zu
        if predicts_duals(arm) and extra.get("active_set_event"):
            return x, lam, zl, zu          # reanchor: drop the stale tangent

        pins = list(family.pin_rows)
        deltas = [float(v) for v in np.asarray(dtheta, dtype=float).ravel()]
        if len(deltas) != len(pins):
            return x, lam, zl, zu
        n = family.n
        b = family.bounds()
        try:
            if not predicts_duals(arm):
                dx = np.asarray(session.parametric_step(pins, deltas),
                                dtype=float)[:n]
                return np.clip(x + dx, b.lb, b.ub), lam, zl, zu
            full = np.asarray(session.parametric_step_full(pins, deltas),
                              dtype=float)
        except Exception:
            return x, lam, zl, zu

        dims = list(session.block_dims)
        x = np.clip(x + full[:n], b.lb, b.ub)
        if lam is not None:
            dlam = np.zeros(family.m)
            rows = session.multiplier_rows(list(range(family.m)))
            for i, r in enumerate(rows):
                if r is not None and 0 <= r < full.size:
                    dlam[i] = full[r]
            lam = lam + dlam
        off = dims[0] + dims[1] + dims[2] + dims[3]
        # Bound multipliers are nonnegative; a linear step can drive one
        # through zero, which is the active-set event the solve resolves.
        if zl is not None and off + dims[4] <= full.size:
            k = min(n, dims[4])
            zl = zl.copy()
            zl[:k] = np.maximum(zl[:k] + full[off:off + k], 0.0)
        if zu is not None and off + dims[4] + dims[5] <= full.size:
            k = min(n, dims[5])
            zu = zu.copy()
            zu[:k] = np.maximum(
                zu[:k] + full[off + dims[4]:off + dims[4] + k], 0.0)
        return x, lam, zl, zu

    # -- one step --------------------------------------------------

    def solve(
        self,
        family: ParametricFamily,
        callbacks: SparseCallbacks,
        arm: str,
        x0: np.ndarray,
        warm: Optional[WarmState],
        step: int,
        tol: float,
    ) -> Tuple[StepResult, Optional[WarmState]]:
        if arm in QP_ARMS:
            return self._solve_qp_ipm(family, callbacks, arm, warm, step, tol)

        prob = self._build(family, callbacks, arm, tol)
        callbacks.reset_counts()
        # The predictor arms need the session handle, because the tangent
        # is a back-solve against the factor this step leaves behind. The
        # session is created for every IPM arm so that arm-to-arm timing
        # is not confounded by the wrapper itself.
        session = pounce.Solver(prob) if not is_sqp(arm) else None

        # Step 0 of a warm arm has nothing to warm from: it is a cold
        # solve, and is reported as one.
        use_warm = is_warm(arm) and warm is not None

        kwargs = {}
        if use_warm:
            if is_sqp(arm):
                kwargs["x0"] = warm.x
                if warm.working_set is not None:
                    kwargs["working_set"] = warm.working_set
            else:
                seed_x = warm.x
                lam, zl, zu = warm.mult_g, warm.mult_x_L, warm.mult_x_U
                if uses_predictor(arm):
                    seed_x, lam, zl, zu = self._predict(
                        family, warm, arm, seed_x, lam, zl, zu
                    )
                kwargs["warm_start"] = pounce.WarmStart(
                    x=seed_x, lagrange=lam, zl=zl, zu=zu, mu=warm.mu,
                    recentering=self.recentering,
                )
        else:
            kwargs["x0"] = np.asarray(x0, dtype=float)

        t0 = time.perf_counter()
        if session is not None:
            # `Solver.solve` takes the seeds directly; the WarmStart's
            # enabling options still have to be installed on the Problem.
            ws = kwargs.pop("warm_start", None)
            if ws is not None:
                for key, val in ws.options().items():
                    prob.add_option(key, val)
                sk = ws.solve_kwargs()
                sk.pop("working_set", None)
                x, info = session.solve(x0=ws.x, **sk)
            else:
                x, info = session.solve(x0=kwargs["x0"])
        else:
            x, info = prob.solve(**kwargs)
        elapsed = time.perf_counter() - t0

        status = int(info.get("status", -99))
        ws = info.get("working_set")
        n_active = None
        if ws is not None:
            n_active = int(np.count_nonzero(ws[0]) + np.count_nonzero(ws[1]))

        result = StepResult(
            step=step,
            theta=[],  # filled by the runner, which owns the path
            success=status in _OK_STATUS,
            status=status,
            status_msg=str(info.get("status_msg", "")),
            iters=int(info.get("iter_count", -1)),
            solve_time=elapsed,
            obj=float(info.get("obj_val", np.nan)),
            kkt_error=float(
                info.get("final_unscaled_kkt_error",
                         info.get("final_kkt_error", np.nan))
            ),
            constr_viol=float(
                info.get("final_unscaled_constr_viol",
                         info.get("final_constr_viol", np.nan))
            ),
            n_active=n_active,
            # Recorded as counts (possibly 0) on the SQP path and as
            # None on the IPM path, which has no QP subproblems at all.
            # Keyed off the arm rather than off the value, so a warm
            # solve that converged without solving a single QP records
            # an honest 0 instead of looking like "not measured".
            n_qp_solves=int(info.get("n_qp_solves", 0)) if is_sqp(arm) else None,
            n_qp_ws_changes=(
                int(info.get("n_qp_ws_changes", 0)) if is_sqp(arm) else None
            ),
            **callbacks.counts(),
        )

        next_warm = WarmState(
            x=np.asarray(x, dtype=float).copy(),
            mult_g=np.asarray(info.get("mult_g"), dtype=float).copy()
            if info.get("mult_g") is not None
            else None,
            mult_x_L=np.asarray(info.get("mult_x_L"), dtype=float).copy()
            if info.get("mult_x_L") is not None
            else None,
            mult_x_U=np.asarray(info.get("mult_x_U"), dtype=float).copy()
            if info.get("mult_x_U") is not None
            else None,
            mu=float(info["mu"]) if info.get("mu") else None,
            working_set=(
                (np.asarray(ws[0]).copy(), np.asarray(ws[1]).copy())
                if ws is not None
                else None
            ),
        )
        if uses_predictor(arm):
            # What the next step's tangent needs: the live session (it
            # owns the converged factor), the theta this solve was run
            # at (the runner fills `theta` after the fact, so record it
            # here from the family), and whether the bound activity just
            # moved -- which invalidates the tangent for `predcorr-ipm`.
            act = None
            if info.get("mult_x_L") is not None and info.get("mult_x_U") is not None:
                act = (np.asarray(info["mult_x_L"]) > 1e-6,
                       np.asarray(info["mult_x_U"]) > 1e-6)
            prev_act = (warm.extra or {}).get("active") if warm else None
            next_warm.extra = {
                "session": session if status in _OK_STATUS else None,
                # Delta theta for the next step is formed against this,
                # not handed down: an adapter never sees the runner's path.
                "theta": family.current_theta(),
                "active": act,
                "active_set_event": bool(
                    prev_act is not None and act is not None
                    and (np.any(act[0] != prev_act[0])
                         or np.any(act[1] != prev_act[1]))
                ),
            }
        return result, next_warm
