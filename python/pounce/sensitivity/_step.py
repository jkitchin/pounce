"""Parametric steps, and what they do about the bounds.

The step is `dx ~ (dx*/dp) dp` against the held factor: how the solution
moves when the parameters do. :func:`solution` returns the moved point,
:func:`solution_report` says what the step did about the bounds on the
way, and :func:`active_set_changes` lists the active-set events a path
step applies.

Everything here takes a :class:`~pounce.sensitivity.SensSession` and
addresses parameters by the full-g row of the defining equality each is
pinned by. `who` names the caller in diagnostics, so a wrapper can keep
its own function name in the messages its users see.

The steps and their measurements are `pounce_sens_core::step`, written
against the engine traits so they run over any interior-point engine,
and the moved point is measured with the session's own `nl` functions.
This module checks the arguments, hands them to that code, and turns
what it reports into the results and the warnings below.
"""
import warnings
from collections import namedtuple

import numpy as np

from .. import _pounce
from ._session import user_row_names

def weakly_active(session):
    """The degenerate bounds at the held solve, cached per session.

    `(variable name, "lower" or "upper")` for every bound the
    classifier could call neither active nor inactive. Empty when the
    solve relaxed its bounds, since the classifier cannot read shifted
    slacks. Cached because the factor is fixed per solve and
    `sens_jacobian()` runs in tight loops.
    """
    cached = getattr(session, "_weakly_active_cache", None)
    if cached is None:
        if session.solver.bound_relax_factor:
            cached = []
        else:
            full_of = {row: full
                       for full, row in enumerate(session._primal_row_map())
                       if row is not None}
            cached = [
                (session.var_names[full_of[vr]],
                 "lower" if low else "upper")
                for vr, low in session.solver.weakly_active_bounds()
                if vr in full_of
            ]
        session._weakly_active_cache = cached
    return cached


def check_margins(bound_eps, max_pdpert, who):
    """Reject the values the CLI's option surface rejects.

    `sens_bound_eps` and `sens_max_pdpert` are both registered with
    `add_lower_bounded_number_option(..., 0.0, strict)`, so a value the
    CLI turns down is turned down here too. A `bound_eps` of zero
    reinstates the roundoff pinning the floor exists to prevent, a NaN
    makes every "outside a bound" comparison false so the refinement
    pins nothing and still reports settled, and a negative `max_pdpert`
    always refuses with a message that reads as though the factor were
    bad.
    """
    for name, value in (("bound_eps", bound_eps), ("max_pdpert", max_pdpert)):
        if value is None:
            continue
        if not float(value) > 0.0:
            raise ValueError(
                f"{who}: {name} must be a positive number, got {value!r}")


def pdpert_error(who, worst, perturbations, max_pdpert):
    """The refusal for a factor carrying more inertia correction than
    `max_pdpert`."""
    pert = [abs(float(v)) for v in perturbations]
    return ValueError(
        f"{who}: the converged KKT factor carries a perturbation of "
        f"{worst:.3e} (dx={pert[0]:.3e}, ds={pert[1]:.3e}, "
        f"dc={pert[2]:.3e}, dd={pert[3]:.3e}), above the requested "
        f"max_pdpert={max_pdpert:.3e}. The factor the step would "
        f"invert is not this problem's KKT matrix. Raise max_pdpert "
        f"to accept it anyway.")


def refuse_on_pdpert(session, max_pdpert, who):
    """Refuse when the converged factor carries more inertia correction
    than the caller will accept.

    A non-zero entry means the factorization every sensitivity output
    inverts is not this problem's KKT matrix but a nearby one's, and how
    nearby is what the entries measure. `SolutionReport.perturbations`
    reports them either way, and this is the caller choosing to stop
    rather than to read them.

    The comparison comes from `pounce_sens_core::step::pdpert_verdict`,
    which `sens_max_pdpert` reads too, so the two surfaces cannot drift
    apart on what counts as too perturbed. The message is written here
    rather than taken from there, since that one names a CLI option and
    says the sensitivity was skipped, and neither is true of a call
    that raises.
    """
    if max_pdpert is None:
        return
    refuse, worst = session.solver.pdpert_verdict(float(max_pdpert))
    if refuse:
        raise pdpert_error(who, worst, session.solver.kkt_perturbations,
                           max_pdpert)


def _degeneracy_iter(degeneracy_iter, degeneracy, who):
    """Warn when the budget is inert: only the "directional" decision
    spends back-solves, so a budget passed under another option changes
    nothing, the same shape as bound_eps under a mode that runs no
    refinement."""
    if degeneracy_iter is not None and degeneracy != "directional":
        warnings.warn(
            f"{who}: degeneracy_iter budgets the directional decision's "
            f"back-solves and degeneracy={degeneracy!r} makes no "
            "decision, so it changes nothing here.")


#: Findings the core reports that this module has already raised as
#: warnings itself, in the order it always raised them, from the
#: arguments alone.
_ARGUMENT_FINDINGS = ("degeneracy_iter_inert", "bound_eps_inert")


def _raise(err, who, max_pdpert):
    kind = err[0]
    if kind == "raise":
        raise err[1]
    if kind == "pdpert":
        raise pdpert_error(who, err[1], err[2], max_pdpert)
    if kind == "not_positive":
        raise ValueError(
            f"{who}: {err[1]} must be a positive number, got {err[2]!r}")
    raise AssertionError(f"unknown error {err!r}")


def _warn_step_findings(session, findings, who, mode, predictor_iter,
                        record=False):
    """Each finding of the core, as the warning this package has always
    raised for it."""
    for f in findings:
        kind = f[0]
        if kind in _ARGUMENT_FINDINGS:
            continue
        if kind == "directional_fallback":
            what = "record" if record else "step"
            warnings.warn(
                f"{who}: {f[1]}. Falling back to the one-sided "
                f"{what}, the degeneracy='one_sided' behavior.")
        elif kind == "corrector_weak":
            c = f[1]
            warnings.warn(
                f"{who}: the corrector spent "
                f"{c['iterations']} back-solve(s) and moved the "
                f"residual from {c['initial_residual']:.2e} to "
                f"{c['residual']:.2e}, measured from the point the "
                "iterations start at rather than from the step handed in, "
                "so the estimate is close to the uncorrected step. One "
                "cause is a bound that must leave the active set with "
                "nothing else for the iterations to reduce; "
                "mode=\"fix_relax\" and mode=\"path\" decide such releases "
                "where the step shows them.")
        elif kind == "still_outside":
            _, vars_, pinned, changes, clamped = f
            names = [session.var_names[i] for i in vars_]
            if mode == "fix_relax":
                n_pinned, stop = pinned
                did = f"pinned {n_pinned} variable(s)"
                # The refinement says why it stopped rather than the
                # count being read as a proxy for it: a pass pins every
                # crossing it sees, so the number of pins says nothing
                # about whether predictor_iter bound the work (gh#732).
                why = {
                    "iteration_limit":
                        "the safety limit of %d pass(es) was reached, so "
                        "raising predictor_iter may finish it" % predictor_iter,
                    "degrees_of_freedom":
                        "holding them all would need more pins than the "
                        "problem has degrees of freedom, so no step does",
                    "worse_than_plain":
                        "the refinement ended further outside the bounds "
                        "than the plain step, which was returned instead",
                }.get(stop, "the refinement settled here")
            else:
                n_changes = changes
                did = f"applied {n_changes} active-set change(s)"
                why = ("the limit of %d was reached, so raising "
                       "predictor_iter may finish it" % predictor_iter
                       if n_changes >= predictor_iter else
                       "the path settled the active set here")
            warnings.warn(
                f"{who}: {mode} {did} and "
                f"still leaves the bounds for {names}, because {why}."
                + (" The values were clamped, which breaks the constraints "
                   "the pins were solved against." if clamped else
                   " The values are returned unclamped."))
        elif kind == "linear_clamped":
            names = [session.var_names[i] for i in f[1]]
            warnings.warn(
                f"{who}: linear step leaves the variable bounds for "
                f"{names}; values were clamped and the active set likely "
                "changed, so the estimate is unreliable there. mode="
                "'fix_relax' pins them and re-solves instead.")
        else:
            raise AssertionError(f"unknown finding {f!r}")


def _check_step_args(mode, degeneracy, degeneracy_iter, bound_eps,
                     max_pdpert, who):
    if mode not in ("linear", "fix_relax", "path"):
        raise ValueError(
            f"{who}: mode must be 'linear', 'fix_relax' or 'path', got "
            f"{mode!r}")
    if degeneracy not in ("directional", "one_sided", "release_all"):
        raise ValueError(
            f"{who}: degeneracy must be 'directional', 'one_sided' or "
            f"'release_all', got {degeneracy!r}")
    _degeneracy_iter(degeneracy_iter, degeneracy, who)
    check_margins(bound_eps, max_pdpert, who)
    if bound_eps is not None and mode != "fix_relax":
        warnings.warn(
            f"{who}: bound_eps is the margin the fix_relax refinement "
            f"pins against and mode={mode!r} runs no refinement, so it "
            "changes nothing here.")


def _opt(v):
    return None if v is None else float(v)


def solution(session, pin_rows, deltas, clamp=True, mode="linear",
             predictor_iter=16, degeneracy="directional", degeneracy_iter=None,
             corrector_iter=0, bound_eps=None, max_pdpert=None,
             who="solution"):
    """First-order estimate of the solution at perturbed parameter values.

    pin_rows: the full-g rows of the defining equalities the parameters
    are pinned by. deltas: how far each one's right-hand side moves.
    Returns the estimated point as a full-x array, ordered like
    `session.var_names`. Values are clamped to variable bounds (with a
    warning) unless clamp=False.

    mode selects what happens when the step leaves a variable bound.
    "linear" (the default) takes the step and clamps, which truncates the
    crossing variable and leaves every other one at its predictor value.
    "fix_relax" instead pins the crossing variable at its bound and
    re-solves, so the others move to stay consistent under the pin, which
    is what `solution_report`'s step fraction says the step needs. It
    costs a dense solve and a backsolve per crossing and tracks a full
    re-solve far more closely. Where nothing crosses the two agree
    exactly.

    "path" applies the perturbation a little at a time, stopping at
    each fraction where the active set changes and continuing under
    the changed set, so a variable can reach a bound partway through
    the change and leave it again further along. Where the two settle
    the same active set it agrees with "fix_relax". At a change large
    enough that they disagree, "fix_relax" decides every change from
    a single step taken at the base point while "path" applies each
    change at the fraction where it happens. `active_set_changes()`
    returns the record of those changes.

    predictor_iter bounds that work. Under "fix_relax" it caps the
    passes, each of which pins every crossing it can see and costs a
    dense solve whose size grows with the number of pins. It is a
    safety limit there rather than a budget, since the loop ends when
    nothing is left outside a bound. Under "path" it caps the number of
    active-set changes applied, and past the cap the rest of the
    perturbation is taken in one step under the active set reached.

    degeneracy selects what happens when the base point itself sits at
    an active-set kink: a bound the classifier can call neither active
    nor inactive, on the bound with a multiplier of the same order as
    the slack. The solution has two one-sided derivatives there.
    "directional" (the default) decides the weakly active bounds by the
    directional-derivative QP for the perturbation's own direction, in
    every mode, at the cost of a few extra backsolves paid only at a
    kink. degeneracy_iter budgets those backsolves: the all-released
    solve and one basis column per engaged bound count against it, and
    a budget the engagement cannot fit falls back to the one-sided
    step with a warning naming the counts. Only
    degeneracy="directional" reads it, and passing it under another
    option warns and changes nothing. "one_sided" takes the
    single-sided value today's thresholds produce, bit-identical to
    the release before this option existed. "release_all" releases
    every weakly active bound undecided, at one back-solve and no QP:
    the step is the all-released direction, and the bounds this
    perturbation actually holds come back as bound crossings for the
    mode or a correction to handle. It trades the decision's cost for
    downstream repair: under mode="fix_relax" or "path" the crossings
    are pinned or walked by the mode's own machinery, while
    mode="linear" clamps each crossing coordinate and leaves its
    neighbors carrying the released coupling.

    corrector_iter runs Newton iterations on the barrier system after
    the step, against an operator assembled at the predicted point:
    one derivative evaluation and one factorization there, every
    block including the barrier diagonal, then a back-solve per
    iteration. It aims at the barrier solution at the mu the solve
    finished on rather than at a re-solve, so the accuracy it reaches
    is bounded by that offset, and it stops as soon as an iteration
    fails to improve the residual, warning when the whole correction
    failed to improve it. Where the perturbation needs a bound to
    leave the active set and the step's endpoint does not show the
    release, no released row is applied: the iterations can move the
    variable partway off the bound on the weak diagonal entry the
    step's clamped multiplier builds, and the estimate is not the
    re-solve. mode="fix_relax" and mode="path" decide such releases
    themselves, and the corrector applies the rows they decided. It
    applies under every mode, refining whatever step that mode
    produced.

    bound_eps sets how far outside a variable bound a step has to end
    to count as having left it, which decides what mode="fix_relax"
    pins, what `solution_report()` reports in `crossed`, and what the
    clamp below acts on. It is absolute, as the refinement's own test
    is. Unset, it is how far outside the solve itself was willing to
    settle, floored so an unrelaxed solve does not pin on roundoff. A
    constraint row keeps its own floor, and a bound is released when the
    step drives its multiplier negative past the solve's own margin,
    whatever bound_eps is. Only mode="fix_relax" reads it, and passing
    it under another mode warns and changes nothing. It must be
    positive, as the CLI's sens_bound_eps is.

    max_pdpert refuses rather than answering when the converged KKT
    factor carries an inertia correction larger than the value given.
    Every sensitivity output inverts that factor, so a perturbed one
    answers for a nearby problem rather than this one.
    `solution_report().perturbations` reports the same numbers for a
    caller who would rather read them than stop. It must be positive,
    as the CLI's sens_max_pdpert is, and the same argument is on
    sens_jacobian(), solution(), solution_report(),
    active_set_changes(),
    covariance() and information().

    clamp keeps its meaning in both modes: it clamps whatever is still
    outside a bound at the end. Under "fix_relax" the pins usually
    leave nothing to clamp, and when they do not, the warning says
    which of the refinement's stopping conditions was reached.

    `deltas` are measured from the SOLVE point -- each is how far that
    pin row's right-hand side moves from the value it carried when the
    factor was taken -- so a caller that has already written the new
    parameter values somewhere else gets the same answer. This is what
    makes the receding-horizon pattern (solve at a prediction, record
    the measurement, then ask) come out right.

    A limit the caller expressed as a CONSTRAINT rather than a variable
    bound moves with the perturbation, so it is not clamped against
    here and raises no clamp warning; the step already respects it to
    first order.
    """
    _check_step_args(mode, degeneracy, degeneracy_iter, bound_eps,
                     max_pdpert, who)
    res = _pounce._sens_solution(
        session.solver, session.nl, session.base_x,
        [int(p) for p in pin_rows], [float(d) for d in deltas], mode,
        int(predictor_iter), degeneracy,
        None if degeneracy_iter is None else int(degeneracy_iter),
        int(corrector_iter), _opt(bound_eps), _opt(max_pdpert), bool(clamp))
    if res["error"] is not None and res["error"][0] == "pdpert":
        _raise(res["error"], who, max_pdpert)
    _warn_step_findings(session, res["findings"], who, mode, predictor_iter)
    if res["error"] is not None:
        _raise(res["error"], who, max_pdpert)
    return np.asarray(res["x"])


class SolutionReport:
    """What the linear step behind `solution()` does about the bounds.

    Attributes
    ----------
    alpha : float
        The fraction of the requested perturbation that can be taken
        before the first bound is reached, from the ratio test along
        the step. On a solve that kept its bounds, at least 1.0 means
        the full step stays inside every one of them. That does not
        follow when `bounds_relaxed` is true, since the solve can leave
        a coordinate outside a bound before the step starts. Infinity
        means no bound lies in the step direction.
    first : str or None
        Name of the variable or constraint reaching its bound at
        `alpha`, None when `alpha` is infinite.
    first_kind : str or None
        Either "variable" or "constraint", naming what `first` is.
    crossed : mapping
        Variable data to the distance by which the full step leaves
        that variable's bound. These are the variables `solution()`
        clamps. Measured at the predicted point against both bounds, so
        on a relaxed solve an entry can be a coordinate the SOLVE left
        outside its bound rather than one the step carried past. The
        step fraction looks only along the step direction, so the two
        answer different questions there and `alpha` can be at or above
        one while this is non-empty.
    crossed_rows : mapping
        Constraint data to the same distance, for inequality
        constraints.

    Those two are keyed by model components, because every coordinate
    that crosses has one. The two classifications below are keyed by
    solve-space name instead, because they cover every coordinate of
    the solve, including the ones the declared-parameter surgery
    created, which have no counterpart on the model.
    violation : float
        Largest constraint violation at the predicted point, measured
        against the perturbed right-hand sides. This is the primal
        half of the residual. The dual half needs the multipliers at
        the perturbed point, so it is computed by the corrector, which
        holds them, rather than assembled here.
    mu : float
        Barrier parameter of the held factorization.
    activity, row_activity : dict
        Name to activity classification, over variables and over
        constraints. Both are empty when `bounds_relaxed` is true.

        Entries the cheap classifier could not call are **refined**
        before they get here -- unless `refine_activity=False`, which
        leaves the cheap class standing and `refined` empty. Either
        way `refined` is what says which reading produced a verdict. The
        classifier normalizes a variable's barrier diagonal by the
        Hessian's diagonal and a row's by the curvature along the row's
        own gradient, while the multiplier that produced it is
        generated by the *reduced* curvature -- the ratio is
        `reduced/diagonal`, so a genuine kink whose coordinate is
        coupled lands in `"ambiguous"` and stays there however tightly
        the problem is re-solved. Reading that class as "probably not a
        kink" is the gh#763 defect; the reduced normalizer is what
        answers it, and it costs one back-solve per entry, so it is
        spent only on the entries that need it.
    refined : dict
        Solve-space name to `(before, after)` for every entry the
        reduced normalizer re-classified, over variables and rows
        together. Empty when nothing was ambiguous, or when
        `refine_activity=False`. A caller who wants to know whether a
        verdict is the cheap one or the refined one reads this rather
        than guessing from the class.
    refined_entries : list
        Solve-space names of every entry the reduced normalizer
        classified, variables first: the entries the cheap classifier
        called "ambiguous", at one back-solve each. `refined` holds the
        ones whose class changed. Empty when `refine_activity=False`.
    perturbations : list or None
        The held factor's inertia-correction perturbations. Any
        non-zero entry means the factor is regularized, so the step is
        taken against a modified matrix and differs from the exact
        active-set answer by that much. None when the engine does not
        report them, which is not the same as all zero.
    corrector : dict or None
        What `corrector_iter` iterations did, None when none were run.
        Holds the back-solves spent under `iterations`, the residual
        before and after under `initial_residual` and `residual`
        (`initial_residual` is measured at the point the iterations
        start from, after the active-set decision and the clamp, not at
        the step handed in), and
        that residual split into `stationarity`, `feasibility` and
        `complementarity`. `released` counts the bounds the step took
        out of the active set and `pinned` the ones it brought in, with
        `active_set_changes` their total. `converged` says the loop
        stopped because an iteration failed to improve rather than
        because it ran out of budget. This is the dual half `violation`
        refers to: it needs the multipliers at the perturbed point,
        which the corrector holds.
    refine_stop : str or None
        Why the `mode="fix_relax"` refinement stopped, one of
        "settled", "iteration_limit", "degrees_of_freedom" or
        "worse_than_plain". None under the other two modes, which run
        no refinement. A pass pins every crossing it sees, so the
        number of pins says nothing about which limit was reached and
        this is the only thing that does. "worse_than_plain" means the
        step reported here is the unrefined one.
    bounds_relaxed : bool
        True when the solve ran with a non-zero `bound_relax_factor`,
        which lets a variable settle outside the bound the model
        declares. The classifier raises for such a solve, since relaxed
        bounds shift the slacks it reads, and the ratio test measures
        distances to bounds the solve did not hold to.

    The last three, with `mu`, are what separate this predictor from
    the exact value at the perturbed active set. A caller comparing the
    estimate against a re-solve reads them to tell which one explains
    the difference.
    """

    def __init__(self, alpha, first, first_kind, crossed, crossed_rows,
                 violation, mu, activity, row_activity, perturbations,
                 bounds_relaxed, corrector=None, refine_stop=None,
                 refined=None, refined_entries=None):
        self.alpha = alpha
        self.first = first
        self.first_kind = first_kind
        self.crossed = crossed
        self.crossed_rows = crossed_rows
        self.violation = violation
        self.mu = mu
        self.activity = activity
        self.row_activity = row_activity
        self.perturbations = perturbations
        self.bounds_relaxed = bounds_relaxed
        self.corrector = corrector
        self.refine_stop = refine_stop
        self.refined = {} if refined is None else refined
        self.refined_entries = [] if refined_entries is None else refined_entries

    def __repr__(self):
        n = len(self.crossed) + len(self.crossed_rows)
        where = f", first {self.first_kind} {self.first}" if self.first else ""
        flags = "".join([
            ", regularized" if self.perturbations and any(self.perturbations)
            else ", perturbations unreported" if self.perturbations is None
            else "",
            ", bounds relaxed" if self.bounds_relaxed else "",
            f", refined {len(self.refined)}" if self.refined else "",
        ])
        return (f"SolutionReport(alpha={self.alpha:.6g}{where}, "
                f"crossed={n}, violation={self.violation:.3e}, "
                f"mu={self.mu:.3e}{flags})")


#: No bound at all reaches here as the reader's sentinel rather than an
#: infinity: `read_nl` seeds every bound vector with +-1e19, so an
#: `isfinite` test passes on it and scores a crossing at 1e18 times the
#: perturbation. The magnitude test is the convention the rest of the
#: package already uses (`preflight`, `block_init`, gh #401 / #403).
_NO_BOUND = 1e19


def _ratio_test(base, step, lo, hi, names, live=None, on_bound=None,
                mu=float("nan"), tol=None):
    """Smallest fraction of `step` that reaches a bound, and where.

    Only coordinates strictly inside their bounds take part, because a
    coordinate already on a bound is not one the step can cross. That
    exclusion is what makes the answer mean anything: at an active
    bound the remaining gap is the small slack the barrier leaves, the
    step component is the same size, and their quotient can take any
    value. It would then become the minimum on any model carrying an
    active bound. Activity is reported through the classification
    instead.

    A coordinate the full step carries past a bound is always scored,
    whatever the exclusion would otherwise say. That case IS a crossing,
    at a fraction below one by construction, and the caller reads it in
    `crossed` off the same predicate and the same tolerance.

    `on_bound` carries the classification, which is exact where the
    classifier commits, applied per SIDE rather than per coordinate,
    since a coordinate held at one bound can still be carried across
    its other one. A distance test scaled by sqrt(mu) rules on the
    coordinates the classifier declines to, capped because it is
    applied relative to the coordinate's own magnitude, and without a
    classification it falls back to a fixed threshold. Coordinates
    whose step is below the roundoff of their own magnitude take no
    part. `pounce_sens_core::step::ratio_test`.
    """
    base = np.asarray(base, dtype=float)
    if tol is not None:
        tol = np.broadcast_to(np.asarray(tol, dtype=float), base.shape)
    frac, i = _pounce._ratio_test(
        base, np.asarray(step, dtype=float), np.asarray(lo, dtype=float),
        np.asarray(hi, dtype=float),
        None if live is None else [bool(v) for v in np.asarray(live)],
        None if on_bound is None else [bool(v) for v in np.asarray(on_bound)],
        float(mu), tol)
    return float(frac), (None if i is None else names[i])


#: Classifications that put a coordinate ON its bound. Both have the
#: slack going to zero, so neither can be crossed by a step. Strongly
#: active leaves an O(mu) gap and weakly active an O(sqrt(mu)) one, and
#: neither gap is room the step can move through. `ambiguous` and
#: `unidentified` are absent on purpose: the first spans coordinates on
#: their bound AND coordinates near it with room left, and the second
#: says only that the curvature is below scale, so neither label
#: answers the question. `_ratio_test`'s distance test rules on those.
_AT_BOUND = ("strongly_active", "weakly_active")


def solution_report(session, pin_rows, deltas, max_iter=None,
                    degeneracy="directional", degeneracy_iter=None,
                    corrector_iter=0, mode="linear", predictor_iter=16,
                    bound_eps=None, max_pdpert=None, refine_activity=True,
                    who="solution_report"):
    """Report what `solution()`'s step does about the bounds.

    degeneracy and degeneracy_iter match `solution()`'s arguments of
    the same names, so the step measured here is the step `solution()`
    takes for the same arguments, including the directional-derivative
    correction at a degenerate base point. degeneracy_iter budgets that
    correction's back-solves.

    refine_activity re-classifies the entries the cheap classifier
    reported as "ambiguous" using the reduced curvature, and records
    what moved under `SolutionReport.refined`. On by default: a coupled
    kink is ambiguous to the cheap rule at every tolerance, so leaving
    it unrefined reports "undetermined" for a question that has an
    answer -- the gh#763 misreading, in the report a caller actually
    looks at.

    It costs one back-solve per ambiguous entry, so the price is set by
    the **ambiguous population**, not by the model size, and the two are
    not the same thing. Measured in review of #889 on a 62k-variable
    Radau collocation column: 675 ambiguous entries, ~29 ms each, and
    the call goes 0.67 s -> 20.2 s against the same call with the
    refinement off. Nothing when none are ambiguous. Pass False to skip
    it, at the price the next paragraph names.

    What skipping costs is not speed-for-nothing: a coupled kink is
    ambiguous to the cheap rule at every tolerance, so the entries that
    come back "ambiguous" unrefined are a *mixture* of genuine kinks and
    genuine non-kinks, and no amount of re-solving separates them. A
    caller who reads that class as "probably not a kink" has made the
    gh#763 inference. `python/pounce/examples/asnmpc_cstr.py` is the
    worked case: it calls this in a latency-measured control loop, and
    its online guard is narrower *because* the refinement runs, so
    switching it off there widens the guard rather than only speeding it
    up. Decide it that way round -- what the class is for -- rather than
    on the timing alone.

    max_iter is accepted for positional compatibility and does nothing.
    It used to budget the directional decision here, so passing it, in
    particular `max_iter=0` to force the one-sided fallback, changed the
    reported step. `degeneracy_iter` is that knob now. Passing it raises
    a DeprecationWarning rather than being ignored in silence, since the
    two readings differ and nothing else would say so.

    mode and predictor_iter match `solution()`'s arguments of the same
    names, so the step measured here is the step `solution()` takes for
    the same arguments. `violation` and `corrector` are properties of
    that step and move with the mode. Reporting the linear step's
    violation for a `fix_relax` estimate would describe a step the
    caller did not take.

    `alpha`, `first`, `crossed` and `crossed_rows` measure where the
    step leaves a bound. Under "fix_relax" and "path" it does not,
    since both stop at the bound by construction, so `alpha` is 1.0 and
    `crossed` is empty for every model. That is the correct answer for
    such a step rather than a missing one. A `bound_eps` wide enough to
    cover the crossing is the exception: the refinement then pins
    nothing, so the step reaches a bound at a fraction below one. It is
    not quite the linear step either, since the release test keeps the
    solve's own margin whatever `bound_eps` is, so a bound the step
    drives negative is released and the step moves by that multiplier's
    size. The reason to run those modes is what "linear" reports at the
    same perturbation.

    `activity`, `row_activity` and `mu` come from the converged base
    point and `perturbations` and `bounds_relaxed` from the solve, so
    none of the five depends on the mode.

    bound_eps sets how far outside a variable bound a step has to end
    to count as having left it, which decides what mode="fix_relax"
    pins and what `crossed` and `alpha` measure against. It is absolute,
    as the refinement's own test is. Unset, it is how far outside the
    solve itself was willing to settle, floored so an unrelaxed solve
    does not pin on roundoff, and `crossed` keeps that floor so a
    coordinate the solve itself left outside a relaxed bound is still
    named. A constraint row keeps its own floor, and a bound is released
    when the step drives its multiplier negative past the solve's own
    margin, whatever bound_eps is. Only mode="fix_relax" reads it, and
    passing it under another mode warns. It must be positive, as the
    CLI's sens_bound_eps is.

    max_pdpert refuses rather than answering when the converged KKT
    factor carries an inertia correction larger than the value given.
    Every sensitivity output inverts that factor, so a perturbed one
    answers for a nearby problem rather than this one.
    `solution_report().perturbations` reports the same numbers for a
    caller who would rather read them than stop. It must be positive,
    as the CLI's sens_max_pdpert is, and the same argument is on
    sens_jacobian(), solution(), solution_report(),
    active_set_changes(),
    covariance() and information().

    corrector_iter runs the same Newton iterations `solution()` runs and
    reports what they did on the `corrector` attribute, without changing
    anything the rest of the report measures. Those describe the step
    handed to the corrector, which is what a caller comparing the two
    wants.

    Takes the same perturbation argument `solution()` takes and returns
    a SolutionReport. Nothing about the estimate changes: this runs
    the same step and measures it.

    The ratio test divides each bounded coordinate's distance to the
    bound it moves toward by the size of its step component and keeps
    the smallest value, over variables and over inequality constraint
    constraints. Equality constraints do not take part, since the step
    holds them to first order and they admit no activity change. The
    constraints pinning the declared Params are excluded on the same
    grounds, because the perturbation moves their right-hand sides by
    construction.
    """
    _check_step_args(mode, degeneracy, degeneracy_iter, bound_eps,
                     max_pdpert, who)
    res = _pounce._sens_solution_report(
        session.solver, session.nl, session.base_x,
        [int(p) for p in pin_rows], [float(d) for d in deltas], mode,
        int(predictor_iter), degeneracy,
        None if degeneracy_iter is None else int(degeneracy_iter),
        int(corrector_iter), _opt(bound_eps), _opt(max_pdpert),
        bool(refine_activity))
    if res["error"] is not None and res["error"][0] == "pdpert":
        _raise(res["error"], who, max_pdpert)
    if max_iter is not None:
        warnings.warn(
            f"{who}: max_iter no longer does anything "
            "here and is "
            "ignored; it used to budget the directional decision, which "
            "degeneracy_iter budgets now. Pass degeneracy_iter instead.",
            DeprecationWarning, stacklevel=2)
    _warn_step_findings(session, res["findings"], who, mode, predictor_iter)
    if res["error"] is not None:
        _raise(res["error"], who, max_pdpert)

    row_names = user_row_names(session)

    def name_of(target):
        kind, idx = target
        return session.var_names[idx] if kind == "variable" else row_names[idx]

    first = None if res["first"] is None else name_of(res["first"])
    first_kind = None if res["first"] is None else res["first"][0]
    crossed = session.new_keymap()
    for i, dist in res["crossed"]:
        ov = session.var_key(i)
        if ov is not None:
            crossed[ov] = float(dist)
    crossed_rows = session.new_keymap()
    # resolved only when a row actually crossed: the resolution is once
    # per session, but the common case has nothing to resolve for
    row_data = session.user_row_data() if res["crossed_rows"] else ()
    for j, dist in res["crossed_rows"]:
        oc = row_data[j]
        if oc is not None:
            crossed_rows[oc] = float(dist)
    bounds_relaxed = bool(res["bounds_relaxed"])
    if bounds_relaxed:
        activity, row_status = {}, {}
    else:
        activity = dict(zip(session.var_names, res["var_status"]))
        row_status = dict(zip(row_names, res["row_status"]))
    refined = {name_of(t): (before, after)
               for t, before, after in res["refined"]}
    refined_entries = [name_of(t) for t in res["refined_entries"]]
    return SolutionReport(
        alpha=float(res["alpha"]), first=first, first_kind=first_kind,
        crossed=crossed, crossed_rows=crossed_rows,
        violation=float(res["violation"]), mu=float(res["mu"]),
        activity=activity, row_activity=row_status,
        perturbations=(None if res["perturbations"] is None
                       else [float(v) for v in res["perturbations"]]),
        bounds_relaxed=bounds_relaxed, corrector=res["corrector"],
        refine_stop=res["refine_stop"], refined=refined,
        refined_entries=refined_entries,
    )


#: One active-set change along `solution(mode="path")`'s path.
#: `fraction` is how far along the perturbation the change happens,
#: `var` is the variable (its solve-space name when the solve created
#: it without a model counterpart), `bound` is "lower" or "upper", and
#: `action` is "reaches" when the variable arrives at the bound and is
#: held there, "leaves" when it comes off it.
#:
#: A weakly active bound can be recorded as "reaches" at a fraction of
#: essentially zero, which does not contradict the variable having been
#: on that bound at the base point: what the working set gained there
#: is the *hold*. Undecided, such a bound sits in the factorization as
#: an order-one penalty that bends the step without enforcing anything,
#: so a perturbation pressing into it is a breakpoint like any other
#: (gh#852).
#:
#: `var` names the quantity the bound is on, and it is not always a
#: variable: a limit written as a constraint row, `g(x) <= cap`, bounds
#: that row's SLACK, and its entry names the constraint instead
#: (gh#928). The two are told apart by what the object is -- a
#: `Constraint` rather than a `Var` under `pyomo_pounce`, a row name
#: rather than a column name in the base session -- and both kinds of
#: limit produce the same `bound` and `action` values. The field keeps
#: its name so that unpacking a record written before row limits were
#: reachable still works.
ActiveSetChange = namedtuple(
    "ActiveSetChange", ["fraction", "var", "bound", "action"])


def active_set_changes(session, pin_rows, deltas, predictor_iter=16,
                       degeneracy="directional", degeneracy_iter=None,
                       max_pdpert=None, who="active_set_changes"):
    """The active-set changes `solution(mode="path")` applies, in order.

    Takes the same perturbation argument `solution()` takes and returns
    a list of `ActiveSetChange` entries, one per change, in the order
    the path applies them. Nothing about the estimate changes: this
    runs the same path and returns its record.

    The record is what `mode="path"` produces that no other mode does.
    The first entry's fraction is how much of the measurement
    discrepancy the held solve's active set survives unchanged, and the
    list as a whole says which bounds the re-optimized solution enters
    and leaves between the predicted state and the measured one.

    A list of length `predictor_iter` means the cap stopped the path before
    the target, the same condition `solution()` warns about.

    degeneracy and degeneracy_iter match `solution()`'s arguments of the
    same names. Under "directional" (the default), a weakly active bound
    the perturbation releases appears in the record as a departure at
    the fraction where its multiplier reaches zero: essentially zero at
    an exact kink, and partway along the step when the held solve sits
    inside the ambiguous band, where the bound is genuinely active for
    the first stretch. degeneracy_iter budgets that decision's
    back-solves, and a budget it cannot fit falls back to the one-sided
    record with a warning, and predictor_iter above still caps the path
    itself. Under "release_all" every weakly active bound starts the
    path released undecided, so a bound the perturbation holds appears
    as a return to its bound along the path rather than a decision at
    the base point.

    max_pdpert refuses rather than answering when the converged KKT
    factor carries an inertia correction larger than the value given.
    This runs the same predictor `solution(mode="path")` runs and
    inverts the same factor, so it takes the same cap. There is no
    bound_eps here, since the path decides its changes from where a
    multiplier reaches zero and reads no margin.
    """
    if degeneracy not in ("directional", "one_sided", "release_all"):
        raise ValueError(
            f"{who}: degeneracy must be 'directional', "
            f"'one_sided' or 'release_all', got {degeneracy!r}")
    _degeneracy_iter(degeneracy_iter, degeneracy, who)
    check_margins(None, max_pdpert, who)
    res = _pounce._sens_active_set_changes(
        session.solver, [int(p) for p in pin_rows],
        [float(d) for d in deltas], int(predictor_iter), degeneracy,
        None if degeneracy_iter is None else int(degeneracy_iter),
        _opt(max_pdpert))
    if res["error"] is not None and res["error"][0] == "pdpert":
        _raise(res["error"], who, max_pdpert)
    _warn_step_findings(session, res["findings"], who, "path",
                        predictor_iter, record=True)
    if res["error"] is not None:
        _raise(res["error"], who, max_pdpert)
    out = []
    for frac, (kind, idx), lower, reaches in res["changes"]:
        if kind == "variable":
            key = session.var_key(idx)
            if key is None:
                key = session.var_names[idx]
        else:
            key = session.row_key(idx)
            if key is None:
                key = session.con_names[idx]
        out.append(ActiveSetChange(
            fraction=float(frac),
            var=key,
            bound="lower" if lower else "upper",
            action="reaches" if reaches else "leaves",
        ))
    return out
