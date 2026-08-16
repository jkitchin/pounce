"""Frontend-neutral predictor--corrector continuation over repeated NLPs
(pounce#608).

pounce#90 delivered a predictor--corrector path follower, but only for
the differentiable frontend: :class:`pounce.jax.PathFollower` is written
against ``JaxProblem`` and reaches for ``jax.grad`` / ``jax.jacobian``
and the AD-side ``jvp_from_state`` / ``warm_anchor`` primitives. Nothing
about the *algorithm* needs autodiff — the tangent predictor is a
back-solve against the held KKT factor, which the generic frontend
already exposes as :meth:`pounce.Solver.parametric_step` — so this
module is the same loop driven through the public ``Problem`` /
``Solver`` API instead.

The two drivers share their step-size policy verbatim: both use
:class:`StepController` below, so "how far to step next" has exactly one
implementation in the tree.

What it does per step
---------------------

1. **predict** — extrapolate the previous iterate to the new parameter.
   With `pins` declared the predictor is the tangent
   ``Δx ≈ ∂x*/∂θ · Δθ`` read off the previous solve's held factor
   (:meth:`pounce.Solver.parametric_step`; the multipliers come from
   :meth:`~pounce.Solver.parametric_step_full`). Without them it falls
   back to a zero-order transfer — the previous iterate carried over
   unchanged.
2. **transfer** — when the problem's *shape* changes between steps (a
   horizon shift, a remesh), an explicit user map moves the state onto
   the new layout. The mapper protocol is
   :meth:`pounce.WarmStart.transfer`'s: ``mapper(ctx) -> dict`` of
   replacement arrays.
3. **monitor** — the KKT residual of the predicted point, evaluated
   through the problem's own callbacks. No solve.
4. **correct** — a warm-started re-solve seeded with the predicted
   primal, duals, and the previous barrier ``μ``.
5. **adapt** — :class:`StepController` grows or shrinks the step from
   the corrector's iteration count, and re-anchors on an active-set
   event.

Read the docs page (``docs/src/continuation.md``) before reaching for
this on an interior-point problem: measured on pounce's own warm-start
corpus, a warm-started IPM solve along a continuation path already
converges in **one iteration per step**, so a tangent predictor has no
iterations left to remove. Continuation pays here by *skipping solves*
(:meth:`Continuation.follow`), not by making them cheaper
(:meth:`Continuation.run`).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, List, Optional, Sequence

import numpy as np

from ._warm_start import WarmStart

__all__ = [
    "Continuation",
    "ContinuationStep",
    "ContinuationTrace",
    "StepController",
    "kkt_residual_monitor",
]

#: Solve statuses that count as a converged step.
_OK_STATUS = (0, 1)

#: Multiplier magnitude above which a bound counts as active, for the
#: active-set fingerprint. Matches ``pounce.jax._ad_common.ACTIVE_TOL``.
ACTIVE_TOL = 1e-6


# ---------------------------------------------------------------------------
# Step-size policy — shared with pounce.jax.PathFollower (pounce#90).
# ---------------------------------------------------------------------------


class StepController:
    """Adaptive continuation step size.

    Extracted from :meth:`pounce.jax.PathFollower.follow` (pounce#90) so
    the AD and non-AD drivers cannot drift apart. Pure arithmetic — it
    holds no problem state and imports nothing.

    The policy: grow after a step accepted on the predictor alone or
    corrected cheaply (``iters <= easy``), shrink after an expensive
    correction (``iters >= hard``) or a rejected one, and drop back to
    ``ds0`` after an active-set event so the region boundary is resolved
    finely rather than jumped over.
    """

    def __init__(
        self,
        *,
        ds0: float = 0.05,
        ds_min: float = 1e-4,
        ds_max: float = 0.25,
        grow: float = 1.5,
        shrink: float = 0.5,
        easy: int = 3,
        hard: int = 10,
    ):
        self.ds0 = float(ds0)
        self.ds_min = float(ds_min)
        self.ds_max = float(ds_max)
        self.grow = float(grow)
        self.shrink = float(shrink)
        self.easy = int(easy)
        self.hard = int(hard)
        self.ds = float(ds0)

    def accepted(self) -> float:
        """Step taken on the predictor alone — no solve. Grow."""
        self.ds = min(self.ds * self.grow, self.ds_max)
        return self.ds

    def corrected(self, iters: int, active_set_event: bool = False) -> float:
        """Step corrected by a solve that converged."""
        if active_set_event:
            self.ds = min(self.ds0, max(self.ds * self.shrink, self.ds_min))
        elif iters <= self.easy:
            self.ds = min(self.ds * self.grow, self.ds_max)
        elif iters >= self.hard:
            self.ds = max(self.ds * self.shrink, self.ds_min)
        return self.ds

    def rejected(self) -> Optional[float]:
        """Corrector failed. Returns the shorter step to retry with, or
        ``None`` when the step floor is reached and the trace must stop."""
        self.ds *= self.shrink
        if self.ds < self.ds_min:
            return None
        return self.ds


# ---------------------------------------------------------------------------
# Transfer (prolongation)
# ---------------------------------------------------------------------------


def _apply_transfer(ws: WarmStart, problem, mapper) -> WarmStart:
    """Map `ws` onto `problem` through `mapper`.

    Delegates to :meth:`pounce.WarmStart.transfer` (#607), which owns the
    length checks, the signing, and the ``replay="mapped"`` bookkeeping.
    The mapper is handed a :class:`pounce.TransferContext` and returns a
    dict of replacement arrays.
    """
    return ws.transfer(problem, mapper)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ContinuationStep:
    """One point of a continuation trace.

    Attributes:
        index: Position in the trace.
        s: Path parameter (``follow``) or step index (``run``).
        theta: Parameter value at this point.
        predictor: How the seed was built — ``"cold"`` (the anchor),
            ``"zero"`` (previous iterate carried over), ``"tangent"``
            (held-factor sensitivity step), or ``"transfer"`` (a user
            prolongation map was applied on top).
        predictor_residual: Max-norm KKT residual at the predicted
            point, before any correction. ``None`` when no monitor ran.
        corrected: Whether a solve ran at this point.
        rejections: Corrector failures absorbed before this point was
            accepted.
        active_set_event: Whether the active-set fingerprint changed
            here relative to the previous accepted point.
        iters, solve_time, obj, kkt_error, status, status_msg: The
            corrector solve's outcome (zeros / ``None`` on an accepted
            predictor step, which runs no solve).
        n_obj, n_grad, n_cons, n_jac, n_hess: Callback counts for this
            step, monitor evaluations included.
    """

    index: int
    s: float
    theta: np.ndarray
    predictor: str
    predictor_residual: Optional[float] = None
    corrected: bool = False
    rejections: int = 0
    active_set_event: bool = False
    iters: int = 0
    solve_time: float = 0.0
    obj: float = float("nan")
    kkt_error: float = float("nan")
    status: Optional[int] = None
    status_msg: str = ""
    n_obj: int = 0
    n_grad: int = 0
    n_cons: int = 0
    n_jac: int = 0
    n_hess: int = 0

    @property
    def evals(self) -> int:
        """Total callback evaluations charged to this step."""
        return self.n_obj + self.n_grad + self.n_cons + self.n_jac + self.n_hess


@dataclasses.dataclass
class ContinuationTrace:
    """Result of a continuation run.

    The counters are the ones pounce#608 asks a driver to report:
    predictor residual (per step, in :attr:`steps`), corrections, step
    rejections, active-set events, and total evaluations.
    """

    steps: List[ContinuationStep] = dataclasses.field(default_factory=list)
    x: List[np.ndarray] = dataclasses.field(default_factory=list)
    status: str = "ok"

    # -- aggregate counters ------------------------------------------

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def n_corrections(self) -> int:
        """Solves run, the initial anchor included."""
        return sum(1 for st in self.steps if st.corrected)

    @property
    def n_predictor_accepts(self) -> int:
        """Points reached without a solve."""
        return sum(1 for st in self.steps if not st.corrected)

    @property
    def n_rejections(self) -> int:
        return sum(st.rejections for st in self.steps)

    @property
    def n_active_set_events(self) -> int:
        return sum(1 for st in self.steps if st.active_set_event)

    @property
    def total_evals(self) -> int:
        return sum(st.evals for st in self.steps)

    @property
    def total_iters(self) -> int:
        return sum(st.iters for st in self.steps)

    @property
    def total_time(self) -> float:
        return sum(st.solve_time for st in self.steps)

    @property
    def theta(self) -> np.ndarray:
        return np.asarray([st.theta for st in self.steps])

    def report(self) -> str:
        """One-line-per-counter summary, for a log or a notebook."""
        res = [st.predictor_residual for st in self.steps
               if st.predictor_residual is not None]
        worst = max(res) if res else float("nan")
        return "\n".join([
            f"continuation: {self.status}",
            f"  points            {self.n_steps}",
            f"  corrections       {self.n_corrections}",
            f"  predictor accepts {self.n_predictor_accepts}",
            f"  step rejections   {self.n_rejections}",
            f"  active-set events {self.n_active_set_events}",
            f"  worst predictor residual {worst:.3e}",
            f"  solver iterations {self.total_iters}",
            f"  total evaluations {self.total_evals}",
            f"  solve time        {self.total_time * 1e3:.1f} ms",
        ])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class Continuation:
    """Trace a parametric NLP sequence through the generic ``Problem`` API.

    Args:
        update: ``update(theta) -> Problem``. The frontend-neutral model
            update: build (or mutate and return) the problem at ``theta``.
            It is called once per attempted point. Anything the frontend
            needs to do to install ``theta`` — rewriting ``cl``/``cu``,
            setting a Pyomo ``Param``, re-emitting an ``.nl`` file — goes
            here, which is what makes the driver frontend-neutral.
        pins: 0-based indices into ``g(x)`` of the parameter-pin equality
            rows ``g_i(x) = theta_i`` (the sIPOPT convention;
            ``cl[i] == cu[i]``). Supplying them enables the **tangent**
            predictor. Omit them and the driver falls back to a zero-order
            warm transfer, which pounce#608 asks for explicitly.
        transfer: Optional prolongation map for steps that change the
            problem's shape (horizon shift, remesh). Same protocol as
            :meth:`pounce.WarmStart.transfer`: ``mapper(ctx) -> dict`` of
            replacement arrays.
        monitor: ``monitor(theta, x, lam, zl, zu) -> float | None``, the
            KKT residual at a *predicted* point, with no solve. Required
            for :meth:`follow` to accept a step on the predictor alone;
            without it every step is corrected. :func:`kkt_residual_monitor`
            builds one from cyipopt-shaped callbacks.
        bounds: ``(lb, ub, cl, cu)``, or ``bounds(theta)`` returning them.
            Used to keep a predicted seed inside the box and to place the
            default anchor ``x0``. The predictor points where the linear
            model says, which can be outside.
        predict_duals: Also step the multipliers along the sensitivity
            (:meth:`pounce.Solver.parametric_step_full`). Off by default:
            measured on pounce's warm-start corpus it is a wash at small
            steps and can cost iterations at large ones (see
            ``docs/src/continuation.md``).
        monitor_tol: KKT-residual threshold at the predicted point. In
            :meth:`follow`, a point under it is accepted with no solve.
        bound_push: Forwarded to :class:`~pounce.WarmStart`.
        controller: A configured :class:`StepController`, or ``None`` for
            the default policy.
        max_steps: Safety cap on :meth:`follow`.
    """

    def __init__(
        self,
        update: Callable[[np.ndarray], object],
        *,
        pins: Optional[Sequence[int]] = None,
        transfer: Optional[Callable[[object], dict]] = None,
        monitor: Optional[Callable] = None,
        bounds: Optional[Callable] = None,
        predict_duals: bool = False,
        monitor_tol: float = 1e-6,
        bound_push: float = 1e-9,
        controller: Optional[StepController] = None,
        max_steps: int = 100_000,
    ):
        if not callable(update):
            raise TypeError("Continuation: `update` must be callable")
        self._update = update
        self._pins = None if pins is None else [int(i) for i in pins]
        self._transfer = transfer
        self._monitor = monitor
        self._bounds = bounds
        self._predict_duals = bool(predict_duals)
        self._monitor_tol = float(monitor_tol)
        self._bound_push = float(bound_push)
        self._controller = controller
        self._max_steps = int(max_steps)

    # -- predictor ----------------------------------------------------

    @property
    def has_tangent(self) -> bool:
        """Whether a sensitivity-based predictor is available.

        ``False`` means the driver runs the zero-order fallback — the
        supported degraded mode, not an error.
        """
        return bool(self._pins)

    def _tangent(self, solver, problem, dtheta):
        """``(dx, dlam, dzl, dzu)`` from the held factor, or all-``None``.

        Never raises: a factor that cannot answer (a solve that stopped
        at an acceptable level, an inertia-corrected factor the
        back-solve rejects) degrades to the zero-order transfer rather
        than ending the trace.
        """
        none4 = (None, None, None, None)
        if not self._pins or solver is None or not solver.converged:
            return none4
        deltas = [float(v) for v in np.asarray(dtheta, float).ravel()]
        if len(deltas) != len(self._pins):
            raise ValueError(
                f"Continuation: theta has {len(deltas)} components but "
                f"{len(self._pins)} pin constraints were declared; the pin "
                "rows must correspond one-to-one with theta"
            )
        n = int(problem.n)
        try:
            if not self._predict_duals:
                dx = np.asarray(solver.parametric_step(self._pins, deltas), float)
                return dx[:n], None, None, None
            full = np.asarray(
                solver.parametric_step_full(self._pins, deltas), float
            )
        except Exception:
            return none4

        dims = list(solver.block_dims)
        dx = full[:n]
        m = int(problem.m)
        dlam = np.zeros(m)
        for i, row in enumerate(solver.multiplier_rows(list(range(m)))):
            if row is not None and 0 <= row < full.size:
                dlam[i] = full[row]
        # z_l / z_u sit after (x, s, y_c, y_d) in the compound vector.
        off = dims[0] + dims[1] + dims[2] + dims[3]
        dzl = dzu = None
        if off + dims[4] <= full.size:
            dzl = np.zeros(n)
            k = min(n, dims[4])
            dzl[:k] = full[off:off + k]
        if off + dims[4] + dims[5] <= full.size:
            dzu = np.zeros(n)
            k = min(n, dims[5])
            dzu[:k] = full[off + dims[4]:off + dims[4] + k]
        return dx, dlam, dzl, dzu

    # -- monitor ------------------------------------------------------

    def _monitor_at(self, theta, ws) -> Optional[float]:
        """Predicted point's KKT residual, or ``None`` when unavailable."""
        if self._monitor is None:
            return None
        r = self._monitor(np.asarray(theta, float), ws.x, ws.lagrange,
                          ws.zl, ws.zu)
        return None if r is None else float(r)

    @staticmethod
    def _active_signature(zl, zu, tol=ACTIVE_TOL):
        """Boolean bound-activity fingerprint, as in pounce#90's
        ``PathFollower._active_signature``."""
        if zl is None or zu is None:
            return None
        return (np.asarray(zl, float) > tol, np.asarray(zu, float) > tol)

    @staticmethod
    def _signature_changed(a, b) -> bool:
        if a is None or b is None:
            return False
        return bool(np.any(a[0] != b[0]) or np.any(a[1] != b[1]))

    # -- the solve ----------------------------------------------------

    def _solve_at(self, theta, ws, counter):  # noqa: C901
        """Build the problem at `theta` and solve it, warm if `ws`.

        Returns ``(problem, solver, x, info, elapsed)``.
        """
        problem = self._update(np.asarray(theta, float))
        if problem is None:
            raise TypeError(
                "Continuation: `update(theta)` must return the Problem at "
                "theta; it returned None"
            )
        if ws is not None and self._transfer is not None:
            ws = _apply_transfer(ws, problem, self._transfer)
        if counter is not None:
            counter.reset_counts()

        solver = _Solver(problem)
        kwargs = {}
        if ws is not None:
            kwargs = ws.solve_kwargs()
            kwargs.pop("working_set", None)  # not a Solver.solve keyword
            x0 = ws.x
        else:
            x0 = (self._anchor_x0 if self._anchor_x0 is not None
                  else self._x0_for(problem, theta))
        # The warm-start overlay belongs to *this step*, not to the
        # Problem: `add_option` is append-only, and a continuation run
        # applies the overlay once per step, so without the snapshot the
        # seven `warm_start_*` / `mu_init` options outlive the run and a
        # later solve that looks cold is not (pounce#607). The restore
        # has to survive an exception, or a step that fails poisons the
        # rest of the path.
        snapshot = problem.options_snapshot() if ws is not None else None
        try:
            if ws is not None:
                for key, val in ws.options().items():
                    problem.add_option(key, val)
            t0 = time.perf_counter()
            x, info = solver.solve(x0=x0, **kwargs)
        finally:
            if snapshot is not None:
                problem.restore_options(snapshot)
        return problem, solver, x, info, time.perf_counter() - t0

    def _box(self, theta):
        """``(lb, ub)`` at `theta`, or ``(None, None)`` when not supplied."""
        if self._bounds is None:
            return None, None
        b = self._bounds(np.asarray(theta, float)) if callable(self._bounds) \
            else self._bounds
        return np.asarray(b[0], float), np.asarray(b[1], float)

    def _x0_for(self, problem, theta):
        lb, ub = self._box(theta)
        x0 = np.zeros(int(problem.n))
        return x0 if lb is None else np.clip(x0, lb, ub)

    # -- entry points -------------------------------------------------

    def run(self, thetas, *, x0=None, counter=None) -> ContinuationTrace:
        """Trace a **prescribed** parameter sequence, solving every point.

        This is the repeated-NLP case pounce#608's first acceptance
        criterion names: the caller has a list of parameter values and
        wants each one solved, without rebuilding the transfer,
        warm-start, and predictor plumbing in user code.

        Every point is corrected, because every point is an answer the
        caller asked for. The predictor's job here is to make each solve
        cheaper — which, on an interior-point method, it largely cannot;
        see the module docstring and ``docs/src/continuation.md``. Use
        :meth:`follow` when the intermediate points are a means rather
        than an end.

        Args:
            thetas: Iterable of parameter values.
            x0: Primal guess for the anchor solve. Defaults to zero
                clipped into the box.
            counter: Optional object with ``reset_counts()`` and
                ``counts()`` returning ``{"n_obj": ..., ...}``, used to
                fill the per-step evaluation counts.

        Returns:
            ContinuationTrace
        """
        trace = ContinuationTrace()
        thetas = [np.asarray(t, float) for t in thetas]
        if not thetas:
            return trace

        ws = None
        solver = None
        prev_theta = None
        prev_sig = None
        self._anchor_x0 = None if x0 is None else np.asarray(x0, float)

        for k, theta in enumerate(thetas):
            kind = "cold"
            dx = dlam = dzl = dzu = None
            if ws is not None:
                kind = "transfer" if self._transfer is not None else "zero"
                dtheta = theta - prev_theta
                dx, dlam, dzl, dzu = self._tangent(solver, self._last_problem,
                                                   dtheta)
                if dx is not None:
                    kind = "tangent"
                    ws = self._advance(ws, dx, dlam, dzl, dzu, theta)

            problem, solver, x, info, elapsed = self._solve_at(
                theta, ws, counter
            )
            self._last_problem = problem
            step = self._record(k, float(k), theta, kind, info, elapsed,
                                counter, corrected=True)

            sig = self._active_signature(info.get("mult_x_L"),
                                         info.get("mult_x_U"))
            step.active_set_event = self._signature_changed(prev_sig, sig)
            prev_sig = sig
            trace.steps.append(step)
            trace.x.append(np.asarray(x, float).copy())

            if step.status not in _OK_STATUS:
                ws = None
                solver = None
                trace.status = f"solve_failed at step {k}: {step.status_msg}"
                continue
            ws = WarmStart.from_info(x, info, bound_push=self._bound_push)
            prev_theta = theta

        return trace

    def follow(self, theta_of_s, s_span, *, x0=None, counter=None
               ) -> ContinuationTrace:
        """Trace ``x*(θ(s))`` over ``s ∈ [s0, s1]`` with an adaptive step.

        The continuation case, and the one where a predictor earns its
        keep: a point whose predicted KKT residual is under
        `monitor_tol` is **accepted with no solve at all**. The path
        parameter is the driver's to choose, so the step size adapts to
        the corrector's work and backs off across active-set events.

        This is :meth:`pounce.jax.PathFollower.follow` (pounce#90) driven
        through the generic ``Problem`` API, sharing its step policy via
        :class:`StepController`.

        Args:
            theta_of_s: ``s (float) -> theta``.
            s_span: ``(s0, s1)`` with ``s1 > s0``.
            x0: Primal guess for the anchor solve at ``s0``.
            counter: As in :meth:`run`.

        Returns:
            ContinuationTrace
        """
        s0, s1 = float(s_span[0]), float(s_span[1])
        if not s1 > s0:
            raise ValueError("Continuation.follow: s_span must have s1 > s0")

        ctl = self._controller or StepController()
        trace = ContinuationTrace()
        self._anchor_x0 = None if x0 is None else np.asarray(x0, float)

        theta0 = np.asarray(theta_of_s(s0), float)
        problem, solver, x, info, elapsed = self._solve_at(theta0, None, counter)
        self._last_problem = problem
        anchor = self._record(0, s0, theta0, "cold", info, elapsed, counter,
                              corrected=True)
        trace.steps.append(anchor)
        trace.x.append(np.asarray(x, float).copy())
        if anchor.status not in _OK_STATUS:
            trace.status = f"anchor_failed: {anchor.status_msg}"
            return trace

        ws = WarmStart.from_info(x, info, bound_push=self._bound_push)
        sig = self._active_signature(info.get("mult_x_L"), info.get("mult_x_U"))
        s = s0
        theta = theta0
        index = 1
        rejections = 0

        while s < s1 - 1e-12 and index <= self._max_steps:
            ds = min(ctl.ds, s1 - s)
            s_new = s + ds
            theta_new = np.asarray(theta_of_s(s_new), float)
            dx, dlam, dzl, dzu = self._tangent(solver, self._last_problem,
                                               theta_new - theta)
            kind = "tangent" if dx is not None else "zero"
            ws_pred = self._advance(ws, dx, dlam, dzl, dzu, theta_new)

            # MONITOR (no solve): residual of the predicted point at the
            # new parameter, from the caller-supplied monitor. Without one
            # the driver has no way to know a predicted point is good, so
            # it corrects every step -- the safe degradation.
            resid = self._monitor_at(theta_new, ws_pred)

            if resid is not None and resid <= self._monitor_tol:
                step = ContinuationStep(
                    index=index, s=s_new, theta=theta_new, predictor=kind,
                    predictor_residual=resid, corrected=False,
                    rejections=rejections,
                )
                if counter is not None:
                    step = dataclasses.replace(step, **counter.counts())
                trace.steps.append(step)
                trace.x.append(np.asarray(ws_pred.x, float).copy())
                ws = ws_pred
                theta, s, index, rejections = theta_new, s_new, index + 1, 0
                ctl.accepted()
                continue

            # CORRECT.
            problem, solver, x, info, elapsed = self._solve_at(
                theta_new, ws_pred, counter
            )
            self._last_problem = problem
            status = int(info.get("status", -99))
            if status not in _OK_STATUS:
                rejections += 1
                if ctl.rejected() is None:
                    trace.status = "corrector_failed"
                    break
                continue

            step = self._record(index, s_new, theta_new, kind, info, elapsed,
                                counter, corrected=True)
            step.predictor_residual = resid
            step.rejections = rejections
            new_sig = self._active_signature(info.get("mult_x_L"),
                                             info.get("mult_x_U"))
            step.active_set_event = self._signature_changed(sig, new_sig)
            sig = new_sig
            trace.steps.append(step)
            trace.x.append(np.asarray(x, float).copy())
            ws = WarmStart.from_info(x, info, bound_push=self._bound_push)
            theta, s, index, rejections = theta_new, s_new, index + 1, 0
            ctl.corrected(step.iters, step.active_set_event)

        if index > self._max_steps and s < s1 - 1e-12:
            trace.status = "max_steps"
        return trace

    # -- helpers ------------------------------------------------------

    def _advance(self, ws, dx, dlam, dzl, dzu, theta=None) -> WarmStart:
        """Apply a predictor step to a warm start, clipped into the box."""
        if dx is None:
            return ws
        x = np.asarray(ws.x, float) + dx
        if theta is not None:
            lb, ub = self._box(theta)
            if lb is not None:
                x = np.clip(x, lb, ub)
        lam = ws.lagrange
        if lam is not None and dlam is not None:
            lam = np.asarray(lam, float) + dlam
        zl, zu = ws.zl, ws.zu
        # Bound multipliers are nonnegative; a linear step can drive one
        # through zero, which is the active-set event the corrector is
        # there to resolve. Clip rather than seed an infeasible dual.
        if zl is not None and dzl is not None:
            zl = np.maximum(np.asarray(zl, float) + dzl, 0.0)
        if zu is not None and dzu is not None:
            zu = np.maximum(np.asarray(zu, float) + dzu, 0.0)
        return dataclasses.replace(ws, x=x, lagrange=lam, zl=zl, zu=zu)

    @staticmethod
    def _record(index, s, theta, kind, info, elapsed, counter, *, corrected):
        step = ContinuationStep(
            index=index, s=s, theta=np.asarray(theta, float),
            predictor=kind, corrected=corrected,
            iters=int(info.get("iter_count", -1)),
            solve_time=elapsed,
            obj=float(info.get("obj_val", float("nan"))),
            kkt_error=float(info.get(
                "final_unscaled_kkt_error",
                info.get("final_kkt_error", float("nan")))),
            status=int(info.get("status", -99)),
            status_msg=str(info.get("status_msg", "")),
        )
        if counter is not None:
            for key, val in counter.counts().items():
                setattr(step, key, int(val))
        return step


def kkt_residual_monitor(problem_obj, bounds):
    """Build a :class:`Continuation` `monitor` from cyipopt-shaped callbacks.

    Returns ``monitor(theta, x, lam, zl, zu) -> float``: the max-norm
    first-order optimality residual at an explicit point, plus the
    constraint and bound violations. No solve, and no autodiff — this is
    the non-AD counterpart of :meth:`pounce.jax.PathFollower._kkt_residual`
    (pounce#90), assembled from ``gradient`` / ``constraints`` /
    ``jacobian`` instead of from ``jax.grad`` / ``jax.jacobian``.

    Args:
        problem_obj: The object handed to ``Problem(problem_obj=...)``.
            Needs ``gradient``; also ``constraints``, ``jacobian`` and
            ``jacobianstructure`` when the problem has constraints.
        bounds: ``(lb, ub, cl, cu)``, or ``bounds(theta)`` returning them.
            Bounds that move with ``theta`` need the callable form, or the
            monitor reads the residual against the wrong box.
    """
    def monitor(theta, x, lam, zl, zu):
        b = bounds(np.asarray(theta, float)) if callable(bounds) else bounds
        lb, ub, cl, cu = (np.asarray(v, float) for v in b)
        x = np.asarray(x, float).ravel()

        stat = np.asarray(problem_obj.gradient(x), float).ravel().copy()
        if zl is not None:
            stat -= np.asarray(zl, float).ravel()
        if zu is not None:
            stat += np.asarray(zu, float).ravel()

        viol = 0.0
        if cl.size:
            c = np.asarray(problem_obj.constraints(x), float).ravel()
            rows, cols = problem_obj.jacobianstructure()
            vals = np.asarray(problem_obj.jacobian(x), float).ravel()
            if lam is not None:
                lam = np.asarray(lam, float).ravel()
                np.add.at(stat, np.asarray(cols, np.int64),
                          vals * lam[np.asarray(rows, np.int64)])
            viol = float(np.max(np.maximum(np.maximum(cl - c, c - cu), 0.0)))

        box = float(max(np.max(np.maximum(0.0, lb - x)),
                        np.max(np.maximum(0.0, x - ub))))
        return float(np.max(np.abs(stat))) + viol + box

    return monitor


def _Solver(problem):
    """`pounce.Solver` bound late, so importing this module does not
    depend on the extension being importable at import time."""
    from . import _pounce

    return _pounce.Solver(problem)
