"""The trust-region filter algorithm.

Solves the glass box / black box problem

    min  f(x)
    s.t. h(x) = 0,  g(x) <= 0        # glass box: cheap, exact derivatives
         y = d(w)                    # black box: expensive, procedural

where ``w`` and ``y`` are subvectors of ``x``. The black box ``d`` is typically
a simulation -- a CFD solve, a converged unit model, a trained network -- that
cannot be handed to an NLP solver as equations.

The algorithm never gives the black box to the solver. Instead it solves a
sequence of ordinary NLPs in which ``d`` is replaced by a surrogate ``r_k``
that is trusted only inside a shrinking box around the current point, and it
uses a filter on ``(theta, f)`` -- where ``theta = ||y - d(w)||`` -- to decide
which steps to keep. Under the standard assumptions this converges to a
first-order KKT point of the *truth* model.

Two design points are worth knowing, because they are where naive
implementations go wrong:

**The trust region is applied as variable bounds, not as a constraint.**
Following Biegler (2024) Sec. 3.1-II it is imposed only on the decision
variables ``u``, as ``max(lb, u_k - Delta) <= u <= min(ub, u_k + Delta)``.
Writing it as a norm constraint over all of ``x`` makes the subproblem harder
and, when the glass-box variables are poorly scaled, badly conditioned.

**The sampling radius is decoupled from the trust radius.** ``sigma_k <= Delta_k``:
``sigma`` governs how accurate the surrogate is, ``Delta`` governs how far the
step may go. Eason & Biegler (2018) introduced this separation and reported it
more than doubled the number of problems their test set could solve -- with a
single radius doing both jobs, the algorithm is forced to take tiny steps near
the solution just to keep the model accurate.

References
----------
Eason, J.P. & Biegler, L.T. AIChE J. 62, 3124-3136 (2016).
Eason, J.P. & Biegler, L.T. AIChE J. 64, 3934-3943 (2018).  [Algorithm 2]
Yoshio, N. & Biegler, L.T. AIChE J. 67, e17054 (2021).
Biegler, L.T. Digital Chemical Engineering 13, 100197 (2024).
Pedrozo et al. Comput. Chem. Eng. 200, 109199 (2025).  [Figs. 2-3, pseudocode]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .._minimize import minimize
from ._filter import FilterPoint, TRFilter
from ._surrogate import (
    Basis,
    CorrectedSurrogate,
    QuadraticBasis,
    ZeroBasis,
    central_difference_design,
    finite_difference_jacobian,
    forward_difference_design,
    quadratic_design,
)

__all__ = ["trf_minimize", "TRFResult", "TRFConfig"]

# No "linear" entry: an affine basis cancels identically under ZOC/FOC. See the
# note in _surrogate.py.
_BASES = {"zero": ZeroBasis, "quadratic": QuadraticBasis}

_INFEASIBLE_STATUS = 2  # Infeasible_Problem_Detected


def _subproblem_status(res) -> str:
    """Classify a subproblem outcome as ``ok`` / ``incompatible`` / ``failed``.

    The extra distinction beyond ``success`` is that an *incompatible*
    subproblem -- the glass-box constraints having no solution inside the
    current trust region -- is not a numerical breakdown. It calls for
    expanding the trust region rather than contracting it, so the loop has to
    tell it apart from every other failure.

    ``res.success`` can be trusted for the rest. That is worth stating because
    it was briefly untrue: ``pounce.minimize`` used to upgrade a refused solve
    to ``success=True`` when its (never-computed, defaulted-to-zero) KKT error
    looked small, so an over-determined subproblem came back successful with an
    ``x`` outside its own bounds. Feeding that to the filter as a trial step
    corrupted the run. Residuals now default to NaN, which fails that test
    closed.
    """
    if res.success:
        return "ok"
    status = res.info.get("status") if hasattr(res, "info") else None
    if status is not None and int(status) == _INFEASIBLE_STATUS:
        return "incompatible"
    return "failed"


@dataclass
class TRFConfig:
    """Tuning parameters.

    Defaults follow the initialization block of Pedrozo et al. (2025) Fig. 6,
    which in turn follows Eason & Biegler (2018) and Yoshio & Biegler (2021).
    """

    trust_radius: float = 0.1
    trust_radius_max: float = 1e3
    trust_radius_min: float = 1e-8
    sampling_radius: float | None = None  # default: 0.1 * trust_radius
    sampling_radius_min: float = 1e-10

    gamma_contract: float = 0.5  # gamma_c
    gamma_expand: float = 2.0  # gamma_e

    gamma_theta: float = 0.01  # filter envelope
    gamma_f: float = 0.01
    theta_min: float = 0.1  # f-steps only below this infeasibility
    theta_max: float = 1e6

    kappa_theta: float = 0.01  # switching condition
    gamma_s: float = 0.9

    eta1: float = 0.4  # ratio-test thresholds
    eta2: float = 0.8

    feasibility_tol: float = 1e-8
    criticality_tol: float = 1e-6
    step_tol: float = 1e-10
    max_iterations: int = 50
    stall_window: int = 12

    central_differences: bool = False
    verbose: bool = False


@dataclass
class TRFIterate:
    """One row of the iteration history."""

    k: int
    objective: float
    theta: float
    trust_radius: float
    sampling_radius: float
    step_norm: float
    kind: str  # f-step | theta-step | rejected | incompatible | subproblem-failed


@dataclass
class TRFResult:
    """Outcome of :func:`trf_minimize`."""

    x: np.ndarray
    fun: float
    success: bool
    message: str
    nit: int
    n_truth_evals: int
    theta: float
    trust_radius: float
    history: list[TRFIterate] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"TRFResult(success={self.success}, fun={self.fun:.10g}, "
            f"theta={self.theta:.3g}, nit={self.nit}, "
            f"n_truth_evals={self.n_truth_evals})"
        )


class _TruthModel:
    """Wraps the user's black box with call counting and memoization.

    Truth-model evaluations dominate the cost of this method -- 82% of total CPU
    in the CFD case study of Pedrozo et al. (2024) -- and the algorithm revisits
    the same ``w`` more than once per iteration (the base point is both the
    surrogate's anchor and a finite-difference stencil point). Caching is not an
    optimization here so much as a correctness-of-accounting measure: without
    it, reported call counts are not comparable to the literature.
    """

    def __init__(self, fun: Callable, jac: Callable | None = None) -> None:
        self._fun = fun
        self._jac = jac
        self._cache: dict[bytes, np.ndarray] = {}
        self.n_evals = 0

    def __call__(self, w: np.ndarray) -> np.ndarray:
        w = np.ascontiguousarray(np.asarray(w, dtype=float).ravel())
        key = w.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        val = np.atleast_1d(np.asarray(self._fun(w), dtype=float))
        self._cache[key] = val
        self.n_evals += 1
        return val

    def jacobian(self, w: np.ndarray, sigma: float, central: bool):
        """Analytic Jacobian if available, else finite differences."""
        if self._jac is not None:
            return np.atleast_2d(np.asarray(self._jac(w), dtype=float))
        J, _ = finite_difference_jacobian(
            self, w, sigma, f0=self(w), central=central
        )
        return J


def _as_bounds_array(bounds, n: int) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    if bounds is None:
        return lo, hi
    for i, b in enumerate(bounds):
        if b is None:
            continue
        bl, bu = b
        if bl is not None:
            lo[i] = bl
        if bu is not None:
            hi[i] = bu
    return lo, hi


def trf_minimize(
    fun: Callable[[np.ndarray], float],
    x0: Sequence[float] | np.ndarray,
    truth_model: Callable[[np.ndarray], np.ndarray],
    w_index: Sequence[int],
    y_index: Sequence[int],
    *,
    jac: Callable | None = None,
    truth_jac: Callable | None = None,
    bounds: Sequence | None = None,
    constraints: Sequence | dict | None = None,
    basis: Basis | str = "zero",
    refit_basis: bool | None = None,
    dof_index: Sequence[int] | None = None,
    config: TRFConfig | None = None,
    solver_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> TRFResult:
    """Minimize a glass box / black box problem by the trust-region filter method.

    Parameters
    ----------
    fun, jac:
        Objective ``f(x)`` and its gradient, over the *full* variable vector.
    x0:
        Starting point. Must include entries for both ``w`` and ``y``.
    truth_model:
        The black box ``d(w) -> y``. Called with a vector of length
        ``len(w_index)``, must return one of length ``len(y_index)``.
    w_index, y_index:
        Positions within ``x`` of the black-box inputs and outputs. The
        algorithm adds the constraint ``x[y_index] == r_k(x[w_index])`` to each
        subproblem; you should *not* include that relationship in
        ``constraints`` yourself.
    truth_jac:
        Optional analytic ``J_d(w)`` of shape ``(n_y, n_w)``. Supplying it is
        the single highest-value optimization available: it removes ``n_w``
        truth-model calls per iteration and makes the corrected surrogate
        exactly first-order accurate rather than accurate to the
        finite-difference step. Many simulators can provide it (COMSOL's
        sensitivity module, Aspen EO), and the ZOC/FOC variant is built around
        the assumption that it is available.
    bounds, constraints:
        Passed through to :func:`pounce.minimize`. ``constraints`` uses the
        scipy dict form ``{'type': 'eq'|'ineq', 'fun': ..., 'jac': ...}``; note
        that ``scipy.optimize.NonlinearConstraint`` is *not* accepted by
        ``pounce.minimize``.
    basis:
        ``"zero"`` (default), ``"quadratic"``, or any object implementing the
        :class:`~pounce.trf.Basis` protocol. ``"zero"`` makes the surrogate a
        plain linearization of the truth model, which is what
        ``pyomo.contrib.trustregion`` does by default.

        Note there is no ``"linear"``: under ZOC/FOC an affine basis cancels
        identically, so it would cost ``n_w + 1`` truth-model calls per
        iteration to reproduce ``"zero"`` exactly. Only curvature survives the
        correction.
    refit_basis:
        Whether to re-fit the basis from fresh truth-model samples at every
        iteration. Default (``None``) means **refit string bases, freeze
        user-supplied objects** -- if you fitted a model yourself, it is not
        clobbered.

        Freezing is the pattern that pays on an expensive truth model. A basis
        fitted once from a designed dataset costs *nothing* per iteration: each
        iteration then needs only the base-point evaluation (plus a gradient),
        because ZOC/FOC re-anchors the fixed basis to the truth model at the
        new point. This is how the literature uses ALAMO-style surrogates
        (Pedrozo et al. 2025). Refitting locally instead spends ``n_w + 1`` or
        ``(n_w+1)(n_w+2)/2`` calls per iteration and is only worth it when
        truth-model calls are cheap.
    dof_index:
        Variables the trust region constrains. Defaults to ``w_index``, which is
        almost always what you want -- the surrogate is only inaccurate as a
        function of ``w``, so that is what needs confining.
    config:
        A :class:`TRFConfig`. Individual fields may also be passed as keyword
        arguments.

    Returns
    -------
    TRFResult

    Notes
    -----
    This method assumes the truth model is **deterministic and smooth**. Eason &
    Biegler are explicit that noise handling is outside the theory ("we will
    assume in this study that noise is negligible"), so it is not appropriate
    for optimization against physical measurements.
    """
    cfg = config or TRFConfig()
    for key, val in kwargs.items():
        if not hasattr(cfg, key):
            raise TypeError(f"unknown option {key!r}")
        setattr(cfg, key, val)

    x = np.asarray(x0, dtype=float).ravel().copy()
    n = x.size
    w_index = np.asarray(w_index, dtype=int)
    y_index = np.asarray(y_index, dtype=int)
    dof = np.asarray(w_index if dof_index is None else dof_index, dtype=int)
    n_w, n_y = w_index.size, y_index.size

    if isinstance(basis, str):
        try:
            cls = _BASES[basis]
        except KeyError:
            extra = (
                " (there is no 'linear': an affine basis cancels identically "
                "under ZOC/FOC, so it is exactly 'zero' at n_w+1 extra "
                "truth-model calls per iteration)"
                if basis == "linear"
                else ""
            )
            raise ValueError(
                f"unknown basis {basis!r}; expected one of {sorted(_BASES)} "
                f"or an object implementing the Basis protocol{extra}"
            ) from None
        basis_obj = cls(n_y, n_w) if cls is ZeroBasis else cls()
        refit = True if refit_basis is None else bool(refit_basis)
    else:
        basis_obj = basis
        # A user-supplied object is assumed already fitted unless told otherwise.
        refit = False if refit_basis is None else bool(refit_basis)

    lo, hi = _as_bounds_array(bounds, n)
    truth = _TruthModel(truth_model, truth_jac)
    filt = TRFilter(cfg.gamma_theta, cfg.gamma_f, cfg.theta_max)

    delta = float(cfg.trust_radius)
    sigma = float(
        cfg.sampling_radius if cfg.sampling_radius is not None else 0.1 * delta
    )
    user_cons = list(constraints) if constraints is not None else []

    def theta_of(xv: np.ndarray) -> float:
        return float(np.linalg.norm(xv[y_index] - truth(xv[w_index])))

    f_cur = float(fun(x))
    theta_cur = theta_of(x)
    history: list[TRFIterate] = []
    message = "Maximum iterations reached"
    success = False

    for k in range(int(cfg.max_iterations)):
        w_k = x[w_index]

        # ---- 1. build the surrogate -------------------------------------
        needs_samples = refit and not isinstance(basis_obj, ZeroBasis)
        if needs_samples:
            if isinstance(basis_obj, QuadraticBasis):
                design = quadratic_design(w_k, sigma)
            elif cfg.central_differences:
                design = central_difference_design(w_k, sigma)
            else:
                design = forward_difference_design(w_k, sigma)
            Y = np.vstack([truth(wi) for wi in design])
            basis_obj.fit(design, Y)

        d_k = truth(w_k)
        Jd_k = truth.jacobian(w_k, sigma, cfg.central_differences)
        r_k = CorrectedSurrogate(basis_obj, w_k, d_k, Jd_k)

        # ---- 2. solve the trust-region subproblem ------------------------
        def surrogate_con(xv, _r=r_k):
            return xv[y_index] - _r.predict(xv[w_index])

        def surrogate_jac(xv, _r=r_k):
            J = np.zeros((n_y, n))
            J[np.arange(n_y), y_index] = 1.0
            J[:, w_index] -= _r.jacobian(xv[w_index])
            return J

        sub_lo = lo.copy()
        sub_hi = hi.copy()
        sub_lo[dof] = np.maximum(lo[dof], x[dof] - delta)
        sub_hi[dof] = np.minimum(hi[dof], x[dof] + delta)

        res = minimize(
            fun,
            x,
            jac=jac,
            bounds=list(zip(sub_lo, sub_hi)),
            constraints=user_cons
            + [{"type": "eq", "fun": surrogate_con, "jac": surrogate_jac}],
            **(solver_options or {}),
        )
        x_trial = np.asarray(res.x, dtype=float).ravel()
        step = x_trial - x
        # The trust radius bounds |dw| only (the degree-of-freedom block), so
        # the step length that feeds the contraction / expansion rules must be
        # measured over that block: the y block is slaved to w through the
        # surrogate and can be orders of magnitude larger (gh#989).
        step_norm = float(np.linalg.norm(step[dof]))

        # A failed subproblem returns whatever iterate the solver stopped on,
        # which is not a point we may trust or evaluate the filter against.
        # Treat it the way the published algorithm treats an incompatible
        # subproblem: shrink the trust region and try again. (The real fix is a
        # restoration phase, which this implementation does not have -- if the
        # region keeps shrinking, the trust_radius_min exit below reports it.)
        outcome = _subproblem_status(res)
        if outcome != "ok":
            # An *incompatible* subproblem is the case the published algorithm
            # handles with a restoration phase: the glass-box constraints have
            # no solution inside the current trust region. Contracting is
            # exactly backwards -- a tighter box is even less likely to contain
            # a feasible point -- so expand instead, up to trust_radius_max.
            # This is a poor substitute for real restoration (it cannot help
            # when the constraints are infeasible for reasons unrelated to the
            # trust region) but it resolves the common case where the initial
            # radius is simply too small, and it degrades to a clear error
            # rather than a wrong answer when it cannot.
            #
            # Any *other* failure is a genuine numerical breakdown, where
            # shrinking is the right response.
            if outcome == "incompatible":
                if delta >= cfg.trust_radius_max:
                    message = (
                        "Trust-region subproblem is incompatible at the maximum "
                        "trust radius: the glass-box constraints appear "
                        "infeasible independently of the trust region. A "
                        "restoration phase would be needed here, and is not "
                        "implemented."
                    )
                    break
                delta = min(delta / cfg.gamma_contract, cfg.trust_radius_max)
                label = "incompatible"
            else:
                delta = cfg.gamma_contract * delta
                sigma = min(sigma, delta)
                label = "subproblem-failed"

            history.append(
                TRFIterate(k, f_cur, theta_cur, delta, sigma, step_norm, label)
            )
            if cfg.verbose:
                print(
                    f"  {k:4d}  {label} ({res.message}); Delta -> {delta:.3e}"
                )
            if delta < cfg.trust_radius_min:
                message = (
                    f"Trust region collapsed after repeated subproblem failures "
                    f"(last: {res.message})"
                )
                break
            continue

        # ---- 3. evaluate the truth model at the trial point --------------
        f_trial = float(fun(x_trial))
        theta_trial = theta_of(x_trial)
        trial_pt = FilterPoint(theta_trial, f_trial)

        # ---- 4. filter acceptance ----------------------------------------
        if not filt.is_acceptable(trial_pt, current=FilterPoint(theta_cur, f_cur)):
            kind = "rejected"
            delta = cfg.gamma_contract * (step_norm if step_norm > 0 else delta)
            sigma = min(sigma, delta)
        else:
            switching = (
                f_cur - f_trial >= cfg.kappa_theta * theta_cur**cfg.gamma_s
                and theta_cur <= cfg.theta_min
            )
            if switching:
                kind = "f-step"
                delta = min(
                    max(cfg.gamma_expand * step_norm, delta), cfg.trust_radius_max
                )
            else:
                kind = "theta-step"
                filt.add(FilterPoint(theta_cur, f_cur))
                rho = (
                    1.0 - theta_trial / theta_cur
                    if theta_cur > cfg.feasibility_tol
                    else 1.0
                )
                if rho < cfg.eta1:
                    delta = cfg.gamma_contract * delta
                elif rho > cfg.eta2:
                    delta = min(cfg.gamma_expand * delta, cfg.trust_radius_max)
                sigma = min(sigma, delta)

            x, f_cur, theta_cur = x_trial, f_trial, theta_trial

            # Track the sampling radius down to the step scale. The trust
            # radius alone cannot do this: under ZOC/FOC it is free to grow
            # (that is the point of the variant), so without a separate rule
            # sigma would stay pinned at its initial value forever. That caps
            # the accuracy of a finite-differenced Jacobian at O(sigma) and
            # the iterates stall a few digits short of the true solution.
            if step_norm > 0.0:
                sigma = float(
                    np.clip(
                        min(sigma, max(step_norm, cfg.sampling_radius_min)),
                        cfg.sampling_radius_min,
                        delta,
                    )
                )

        history.append(
            TRFIterate(k, f_cur, theta_cur, delta, sigma, step_norm, kind)
        )
        if cfg.verbose:
            print(
                f"  {k:4d}  f={f_cur: .10g}  theta={theta_cur:.3e}  "
                f"Delta={delta:.3e}  |s|={step_norm:.3e}  {kind}"
            )

        # ---- 5. convergence ----------------------------------------------
        #
        # Two conditions, and both are load-bearing.
        #
        # (a) theta <= feasibility_tol -- the surrogate agrees with the truth
        #     model at the accepted point, so the point is feasible for the
        #     real problem.
        #
        # (b) the step in w is small. This one is easy to leave out and the
        #     result is a plausible-looking wrong answer. ZOC/FOC forces
        #     grad r_k == grad d only *at the base point* w_k. The subproblem's
        #     KKT conditions are written with grad r_k, so if the accepted step
        #     lands far from w_k they are the KKT conditions of the wrong
        #     problem: the constraint Jacobian is stale by
        #     grad d(w_k) - grad d(w*), which is O(||w* - w_k||). Testing (a)
        #     alone will happily report convergence at a feasible point whose
        #     true gradient is nowhere near zero.
        #
        # Note that testing `step_norm <= step_tol` as the *only* second
        # condition does not work either: without a restoration phase the
        # algorithm can settle into a stable f-step / theta-step / rejected
        # limit cycle in which theta creeps down but the step length never
        # shrinks, so the test never fires and the run burns its whole
        # iteration budget. The stall detector below catches that case and
        # reports it rather than silently spinning.
        w_step = float(np.linalg.norm(x_trial[w_index] - w_k))
        w_scale = max(1.0, float(np.linalg.norm(w_k)))
        if (
            theta_cur <= cfg.feasibility_tol
            and kind != "rejected"
            and w_step <= cfg.criticality_tol * w_scale
        ):
            success = True
            message = "Converged: feasible to tolerance at a first-order critical point"
            break

        if delta < cfg.trust_radius_min:
            message = "Trust region collapsed below trust_radius_min"
            break

        if len(history) >= cfg.stall_window:
            window = history[-cfg.stall_window :]
            thetas = [h.theta for h in window]
            objs = [h.objective for h in window]
            theta_gain = (max(thetas) - min(thetas)) / max(max(thetas), 1e-300)
            obj_gain = max(objs) - min(objs)
            if theta_gain < 1e-3 and abs(obj_gain) < cfg.feasibility_tol:
                message = (
                    f"Stalled: no progress in infeasibility or objective over "
                    f"{cfg.stall_window} iterations. This is the failure mode a "
                    f"restoration phase exists to fix; restoration is not "
                    f"implemented. Try a larger trust_radius, a richer basis, or "
                    f"a looser feasibility_tol."
                )
                break

    return TRFResult(
        x=x,
        fun=f_cur,
        success=success,
        message=message,
        nit=len(history),
        n_truth_evals=truth.n_evals,
        theta=theta_cur,
        trust_radius=delta,
        history=history,
    )
