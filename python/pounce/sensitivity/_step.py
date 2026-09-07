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
"""
import warnings
from collections import namedtuple

import numpy as np

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


def _bound_margin(session, bound_eps):
    """How far outside a bound a coordinate has to end to count as
    having left it.

The refinement pins against this margin and `solution()` clamps
    against it. `solution_report()` measures `crossed` against the same
    number whenever the caller sets one, since deciding it twice is
    what let `refine_stop == "settled"` come back beside a coordinate
    reported 2.0 outside its bound.

    Unset it is how far outside the solve itself was willing to settle,
    floored so an unrelaxed solve does not pin on roundoff. The report
    keeps the floor there instead, because a coordinate the SOLVE left
    outside its bound is one of the things `crossed` exists to name,
    and the relaxation is exactly how far out it is.
    """
    if bound_eps is not None:
        return float(bound_eps)
    return max(abs(session.solver.bound_relax_factor or 0.0), 1e-9)


def refuse_on_pdpert(session, max_pdpert, who):
    """Refuse when the converged factor carries more inertia correction
    than the caller will accept.

    A non-zero entry means the factorization every sensitivity output
    inverts is not this problem's KKT matrix but a nearby one's, and how
    nearby is what the entries measure. `SolutionReport.perturbations`
    reports them either way, and this is the caller choosing to stop
    rather than to read them.

    The comparison comes from `pounce_sensitivity::pdpert_verdict`,
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
        pert = [abs(float(v)) for v in session.solver.kkt_perturbations]
        raise ValueError(
            f"{who}: the converged KKT factor carries a perturbation of "
            f"{worst:.3e} (dx={pert[0]:.3e}, ds={pert[1]:.3e}, "
            f"dc={pert[2]:.3e}, dd={pert[3]:.3e}), above the requested "
            f"max_pdpert={max_pdpert:.3e}. The factor the step would "
            f"invert is not this problem's KKT matrix. Raise max_pdpert "
            f"to accept it anyway.")


def _degeneracy_iter(degeneracy_iter, degeneracy, who):
    """Resolve the budget's default and warn when it is inert: only
    the "directional" decision spends back-solves, so a budget passed
    under another option changes nothing, the same shape as bound_eps
    under a mode that runs no refinement."""
    if degeneracy_iter is None:
        return 16
    if degeneracy != "directional":
        warnings.warn(
            f"{who}: degeneracy_iter budgets the directional decision's "
            f"back-solves and degeneracy={degeneracy!r} makes no "
            "decision, so it changes nothing here.")
    return degeneracy_iter


def _correct(session, pin_idx, deltas, step, corrector_iter):
    """Refine a step by Newton iterations on the barrier system.

    Returns the refined primal step and what the iterations did.

    The corrector starts from a point, so it needs the multipliers as
    well as the primal step, and only the plain parametric step reports
    both. Every mode refines the same underlying step, so the start is
    the mode's own primal block carrying the plain step's multipliers.
    The iterations correct the multipliers from there, which is what
    they are for.

    Two consequences worth naming. This spends a second solve even under
    mode="linear" with degeneracy="one_sided", where the primal block it
    then overwrites is the one it just computed. And under
    degeneracy="directional" the multipliers come from the one-sided
    step rather than the directional one that produced `step`, since
    there is no directional full step to ask. Under
    degeneracy="release_all" the mismatch is sharper still: the held
    factorization supplies bound multipliers at exactly the bounds the
    step just released to zero, over the maximal weak set rather than
    a decided subset.
    """
    full = np.asarray(session.solver.parametric_step_full(pin_idx, deltas))
    n_x = len(step)
    full[:n_x] = np.asarray(step)
    out, info = session.solver.correct_step(
        pin_idx, deltas, list(full), corrector_iter)
    return np.asarray(out)[:n_x], dict(info)


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
    if mode not in ("linear", "fix_relax", "path"):
        raise ValueError(
            f"{who}: mode must be 'linear', 'fix_relax' or 'path', got "
            f"{mode!r}")
    if degeneracy not in ("directional", "one_sided", "release_all"):
        raise ValueError(
            f"{who}: degeneracy must be 'directional', 'one_sided' or "
            f"'release_all', got {degeneracy!r}")
    degeneracy_iter = _degeneracy_iter(
        degeneracy_iter, degeneracy, who)
    check_margins(bound_eps, max_pdpert, who)
    if bound_eps is not None and mode != "fix_relax":
        warnings.warn(
            f"{who}: bound_eps is the margin the fix_relax refinement "
            f"pins against and mode={mode!r} runs no refinement, so it "
            "changes nothing here.")
    refuse_on_pdpert(session, max_pdpert, who)

    pin_idx, deltas = list(pin_rows), list(deltas)

    # parametric_step returns the factor's x block (var-x); base_x and
    # everything below it (nl.x_l/x_u, var_names) are full-x
    segments = []
    fell_back = False
    if degeneracy == "directional":
        # At a degenerate base point the weakly active bounds are
        # decided by the directional-derivative QP for this
        # perturbation's own direction; at a clean base point these are
        # the plain calls at no extra solve cost. A failed decision
        # (budget exhausted, no sign-consistent working set) falls back
        # to the one-sided step and says so.
        try:
            step, held_rows, _ = (
                session.solver.parametric_step_directional(
                    pin_idx, deltas, degeneracy_iter))
            if mode == "fix_relax":
                step, pinned, stop = (
                    session.solver.parametric_step_bounded_decided(
                        pin_idx, deltas, held_rows, predictor_iter,
                        bound_eps))
            elif mode == "path":
                step, segments = (
                    session.solver.parametric_step_path_decided(
                        pin_idx, deltas, held_rows, predictor_iter))
        except RuntimeError as e:
            if "directional derivative" not in str(e):
                raise
            warnings.warn(
                f"{who}: {e}. Falling back to the one-sided step, "
                "the degeneracy='one_sided' behavior.")
            fell_back = True
    if degeneracy == "release_all":
        # Every weakly active bound is released undecided, at one
        # back-solve and no QP: the bounds this perturbation actually
        # holds come back as crossings for the mode or a correction to
        # handle. fix_relax and path run their own machinery under the
        # all-released treatment, the decided variants' empty held set.
        if mode == "fix_relax":
            step, pinned, stop = (
                session.solver.parametric_step_bounded_decided(
                    pin_idx, deltas, [], predictor_iter, bound_eps))
        elif mode == "path":
            step, segments = (
                session.solver.parametric_step_path_decided(
                    pin_idx, deltas, [], predictor_iter))
        else:
            step, _released = session.solver.parametric_step_release_all(
                pin_idx, deltas)
    if degeneracy == "one_sided" or fell_back:
        if mode == "fix_relax":
            step, pinned, stop = session.solver.parametric_step_bounded(
                pin_idx, deltas, predictor_iter, bound_eps)
        elif mode == "path":
            step, segments = session.solver.parametric_step_path(
                pin_idx, deltas, predictor_iter)
        else:
            step = session.solver.parametric_step(pin_idx, deltas)
    corrector = None
    if corrector_iter:
        step, corrector = _correct(
            session, pin_idx, deltas, step, corrector_iter)
        # A correction that works drives the residual down by orders;
        # one whose Newton direction finds nothing to reduce shaves a
        # few percent off and leaves the estimate where it was.
        # Halving is a low bar that separates them cleanly, and saying
        # nothing would let the second case pass for the first.
        # Written as `not (<=)` rather than `>`, so a non-finite
        # residual warns instead of comparing false and passing in
        # silence. gh#845 was exactly that: the corrector normed an
        # all-NaN residual to 0.0, `0.0 > 0.5 * 6.09` was false, and
        # `solution()` returned {x: nan} with nothing said. The Rust
        # side no longer produces that number, and this side no longer
        # depends on it not doing so.
        if corrector is not None and not (
                corrector["residual"] <= 0.5 * corrector[
                    "initial_residual"]):
            warnings.warn(
                f"{who}: the corrector spent "
                f"{corrector['iterations']} back-solve(s) and moved the "
                f"residual from {corrector['initial_residual']:.2e} to "
                f"{corrector['residual']:.2e}, measured from the point the "
                "iterations start at rather than from the step handed in, "
                "so the estimate is close to the uncorrected step. One "
                "cause is a bound that must leave the active set with "
                "nothing else for the iterations to reduce; "
                "mode=\"fix_relax\" and mode=\"path\" decide such releases "
                "where the step shows them.")

    dx = session.scatter_x(np.asarray(step))
    x_new = session.base_x + dx

    lo, hi = np.asarray(session.nl.x_l), np.asarray(session.nl.x_u)
    if mode in ("fix_relax", "path"):
        # The refinement holds each pinned coordinate AT its bound and
        # lets the others move, so a coordinate can still be left
        # outside one, either because the pass budget ran out or because
        # the pins have exhausted the problem's degrees of freedom.
        # `clamp` decides what happens then, exactly as it does under
        # `linear`, and the warning says which of the two stopped it.
        #
        # One tolerance for "outside its bound", taken from the solve
        # rather than chosen here: it was willing to leave a converged
        # point `bound_relax_factor` outside, so anything within that is
        # on the bound. The refinement pins against the same number.
        # The comparison is absolute, as the refinement's own is, so the
        # clamp and the pins agree on a coordinate of any magnitude.
        eps = _bound_margin(
            session, bound_eps if mode == "fix_relax" else None)
        out = np.where((x_new < lo - eps) | (x_new > hi + eps))[0]
        if out.size:
            names = [session.var_names[i] for i in out]
            if mode == "fix_relax":
                did = f"pinned {len(pinned)} variable(s)"
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
                n_changes = len(segments)
                did = f"applied {n_changes} active-set change(s)"
                why = ("the limit of %d was reached, so raising "
                       "predictor_iter may finish it" % predictor_iter
                       if n_changes >= predictor_iter else
                       "the path settled the active set here")
            warnings.warn(
                f"{who}: {mode} {did} and "
                f"still leaves the bounds for {names}, because {why}."
                + (" The values were clamped, which breaks the constraints "
                   "the pins were solved against." if clamp else
                   " The values are returned unclamped."))
            if clamp:
                x_new = np.clip(x_new, lo, hi)
    elif clamp:
        # `linear` takes the step as the predictor gives it, so a
        # crossing shows up as a value outside its bound and clamping is
        # all this mode can do about it. The active set changed and the
        # step does not know, which is what `fix_relax` addresses.
        # The comparison is absolute, as the refinement's own is.
        eps = _bound_margin(session, None)
        clamped = (x_new < lo - eps) | (x_new > hi + eps)
        if clamped.any():
            names = [session.var_names[i] for i in np.where(clamped)[0]]
            warnings.warn(
                f"{who}: linear step leaves the variable bounds for "
                f"{names}; values were clamped and the active set likely "
                "changed, so the estimate is unreliable there. mode="
                "'fix_relax' pins them and re-solves instead.")
        x_new = np.clip(x_new, lo, hi)

    return x_new


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
    perturbations : list
        The held factor's inertia-correction perturbations. Any
        non-zero entry means the factor is regularized, so the step is
        taken against a modified matrix and differs from the exact
        active-set answer by that much.
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
                 refined=None):
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

    def __repr__(self):
        n = len(self.crossed) + len(self.crossed_rows)
        where = f", first {self.first_kind} {self.first}" if self.first else ""
        flags = "".join([
            ", regularized" if any(self.perturbations) else "",
            ", bounds relaxed" if self.bounds_relaxed else "",
            f", refined {len(self.refined)}" if self.refined else "",
        ])
        return (f"SolutionReport(alpha={self.alpha:.6g}{where}, "
                f"crossed={n}, violation={self.violation:.3e}, "
                f"mu={self.mu:.3e}{flags})")


#: The class the cheap classifier reports when it cannot decide, and the
#: only one the reduced normalizer is spent on. See `SolutionReport` for
#: why a genuine kink lands here.
_AMBIGUOUS = "ambiguous"


def _refine_ambiguous(session, var_status, row_status, var_names, row_names):
    """Re-classify the ambiguous entries with the reduced normalizer.

    Returns `(var_status, row_status, refined)` with the two lists
    updated in place-equivalent copies and `refined` mapping each
    changed name to `(before, after)`.

    The cheap classifier divides a variable's barrier diagonal by the
    Hessian's **diagonal** and a row's by the curvature along the row's
    own gradient. Neither is the curvature that generated the
    multiplier: the other free coordinates re-optimize, so the true
    ratio is `reduced/diagonal` for a variable and `reduced/directional`
    for a row, and both equal 1 only when the coordinate or row is
    decoupled. Couple it and a genuine kink lands in "ambiguous" -- at
    every tolerance, since that ratio does not move with mu. That is
    gh#763 and gh#804, and reading the class as "probably not a kink"
    is the inference that shipped the defect.

    The reduced normalizer answers it exactly, at one back-solve per
    entry. Spending that on every bounded variable would be `n`
    back-solves for a question almost none of them raise, so it is
    spent here on the entries that actually raised it -- which is the
    "on demand, over the ambiguous entries" shape CLAUDE.md prescribes.
    """
    refined = {}
    var_idx = [i for i, s in enumerate(var_status) if s == _AMBIGUOUS]
    row_idx = [j for j, s in enumerate(row_status) if s == _AMBIGUOUS]
    if not var_idx and not row_idx:
        return var_status, row_status, refined

    var_status = list(var_status)
    row_status = list(row_status)
    if var_idx:
        red = session.solver.reduced_activity(var_idx)
        for k, i in enumerate(var_idx):
            after = red["status"][k]
            if after != var_status[i]:
                refined[var_names[i]] = (var_status[i], after)
                var_status[i] = after
    if row_idx:
        red = session.solver.reduced_row_activity(row_idx)
        for k, j in enumerate(row_idx):
            after = red["status"][k]
            if after != row_status[j]:
                refined[row_names[j]] = (row_status[j], after)
                row_status[j] = after
    return var_status, row_status, refined


def _row_step(session, dx):
    """Jacobian at the base point times the step, in row order."""
    rows, cols = session.nl.jacobian_structure()
    vals = np.asarray(session.nl.jacobian(session.base_x))
    return np.bincount(np.asarray(rows),
                       weights=vals * dx[np.asarray(cols)],
                       minlength=len(session.con_names))



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
    `crossed` off the same predicate and the same tolerance. Deciding it
    twice is what let the two disagree, reporting that 998 times the
    perturbation fits while naming a coordinate the single step leaves
    its bound by 0.25.

    So the exclusion below only ever applies to coordinates that do not
    cross, and cannot cost a crossing. Two things drive it. `on_bound`
    carries the classification, which is exact where the classifier
    commits, applied per SIDE rather than per coordinate, since a
    coordinate held at one bound can still be carried across its other
    one.

    The distance test covers the coordinates the classifier declines to
    rule on, where the label cannot answer the question because
    `ambiguous` spans both a coordinate near its bound with room left
    and one already on it. What separates those is the size of the
    remaining gap, which is O(mu) at a strongly active bound and
    O(sqrt(mu)) at a weakly active one, against O(1) room in the
    interior. So the threshold scales with sqrt(mu), measured four
    orders of magnitude clear of the interior case. It is capped
    because it is applied relative to the coordinate's own magnitude,
    and a loose `mu` at termination would otherwise widen it without
    limit. Without a classification there is no `mu` either, and it
    falls back to a fixed threshold.

    Coordinates whose step is below the roundoff of their own magnitude
    also take no part, since dividing by one puts the crossing at an
    enormous multiple of the perturbation, which is not a finding.
    """
    scale = np.maximum(1.0, np.abs(base))
    toward_hi = step > 0
    bound = np.where(toward_hi, hi, lo)
    distance = bound - base
    present = np.abs(bound) < _NO_BOUND
    moving = np.abs(step) > 1e-12 * scale

    # The same predicate, and the same tolerance, that fills `crossed`
    # or `crossed_rows`. The caller passes the tolerance it fills with,
    # and the default is the rows' own.
    reached = base + step
    if tol is None:
        tol = 1e-9 * np.maximum(1.0, np.abs(reached))
    crosses = present & moving & np.where(
        toward_hi, reached > bound + tol, reached < bound - tol)

    floor = 1e-9 if not np.isfinite(mu) else min(
        1e-2, max(1e-9, 10.0 * mu ** 0.5))
    interior = np.abs(distance) > floor * scale
    if on_bound is not None:
        # the nearer bound is the one the coordinate is held at, and it
        # is only that side the classification rules out
        at_hi = (hi - base) <= (base - lo)
        interior &= ~(on_bound & (toward_hi == at_hi))

    ok = present & moving & (crosses | interior)
    if live is not None:
        ok &= live
    if not ok.any():
        return float("inf"), None
    fraction = np.full(base.shape, np.inf)
    # a relaxed solve can settle outside a bound, which leaves no room
    # to move rather than a negative amount of it
    fraction[ok] = np.maximum(distance[ok] / step[ok], 0.0)
    i = int(np.argmin(fraction))
    return float(fraction[i]), names[i]



#: Classifications that put a coordinate ON its bound. Both have the
#: slack going to zero, so neither can be crossed by a step. Strongly
#: active leaves an O(mu) gap and weakly active an O(sqrt(mu)) one, and
#: neither gap is room the step can move through. `ambiguous` and
#: `unidentified` are absent on purpose: the first spans coordinates on
#: their bound AND coordinates near it with room left, and the second
#: says only that the curvature is below scale, so neither label
#: answers the question. `_ratio_test`'s distance test rules on those.
_AT_BOUND = ("strongly_active", "weakly_active")


def _on_bound(statuses):
    return np.array([s in _AT_BOUND for s in statuses])


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
    if mode not in ("linear", "fix_relax", "path"):
        raise ValueError(
            f"{who}: mode must be 'linear', 'fix_relax' or "
            f"'path', got {mode!r}")
    if degeneracy not in ("directional", "one_sided", "release_all"):
        raise ValueError(
            f"{who}: degeneracy must be 'directional', "
            f"'one_sided' or 'release_all', got {degeneracy!r}")
    degeneracy_iter = _degeneracy_iter(
        degeneracy_iter, degeneracy, who)
    check_margins(bound_eps, max_pdpert, who)
    if bound_eps is not None and mode != "fix_relax":
        warnings.warn(
            f"{who}: bound_eps is the margin the fix_relax "
            f"refinement pins against and mode={mode!r} runs no "
            "refinement, so it changes nothing here.")
    refuse_on_pdpert(session, max_pdpert, who)
    if max_iter is not None:
        warnings.warn(
            f"{who}: max_iter no longer does anything "
            "here and is "
            "ignored; it used to budget the directional decision, which "
            "degeneracy_iter budgets now. Pass degeneracy_iter instead.",
            DeprecationWarning, stacklevel=2)
    pin_idx, deltas = list(pin_rows), list(deltas)
    # the same dispatch `solution()` runs, so the step measured here is
    # the step it takes for these arguments
    fell_back = False
    refine_stop = None
    if degeneracy == "directional":
        try:
            step, held_rows, _ = session.solver.parametric_step_directional(
                pin_idx, deltas, degeneracy_iter)
            if mode == "fix_relax":
                step, _, refine_stop = (
                    session.solver.parametric_step_bounded_decided(
                        pin_idx, deltas, held_rows, predictor_iter,
                        bound_eps))
            elif mode == "path":
                step, _ = session.solver.parametric_step_path_decided(
                    pin_idx, deltas, held_rows, predictor_iter)
        except RuntimeError as e:
            if "directional derivative" not in str(e):
                raise
            warnings.warn(
                f"{who}: {e}. Falling back to the one-sided "
                "step, the degeneracy='one_sided' behavior.")
            fell_back = True
    if degeneracy == "release_all":
        if mode == "fix_relax":
            step, _, refine_stop = (
                session.solver.parametric_step_bounded_decided(
                    pin_idx, deltas, [], predictor_iter, bound_eps))
        elif mode == "path":
            step, _ = session.solver.parametric_step_path_decided(
                pin_idx, deltas, [], predictor_iter)
        else:
            step, _released = session.solver.parametric_step_release_all(
                pin_idx, deltas)
    if degeneracy == "one_sided" or fell_back:
        if mode == "fix_relax":
            step, _, refine_stop = session.solver.parametric_step_bounded(
                pin_idx, deltas, predictor_iter, bound_eps)
        elif mode == "path":
            step, _ = session.solver.parametric_step_path(
                pin_idx, deltas, predictor_iter)
        else:
            step = session.solver.parametric_step(pin_idx, deltas)
    dx = session.scatter_x(np.asarray(step))
    base = np.asarray(session.base_x)
    x_new = base + dx

    lo, hi = np.asarray(session.nl.x_l), np.asarray(session.nl.x_u)
    g_l, g_u = np.asarray(session.nl.g_l), np.asarray(session.nl.g_u)

    # The margin the refinement pinned against, so `alpha` and `crossed`
    # answer with the number that decided the step. It is absolute, as
    # the refinement's own test is, so a coordinate of order 1e4 does not
    # get a tolerance the refinement never gave it. Unset keeps the
    # fixed floor rather than the solve's relaxation: `crossed` reports
    # a coordinate the SOLVE left outside its bound, and the relaxation
    # is exactly how far out that is, so taking it as the margin would
    # hide the case this report exists to name.
    eps = (1e-9 if bound_eps is None or mode != "fix_relax"
           else float(bound_eps))
    # The refinement pins variable bounds only, so a constraint row
    # keeps its floor whatever margin the caller set: a wide margin has
    # no say over a row the step carries past its limit. Its tolerance
    # is set beside the row ratio test below.

    row_names = user_row_names(session)

    # The classifier raises for a solve that relaxed its bounds, since
    # relaxed bounds shift the slacks it reads. Ask the solve what it
    # ran under instead of provoking that and reading the message, and
    # report it as the fact it is. The rest of the report is still
    # measured, where a diagnostic that raised would give the caller
    # nothing exactly when they are trying to find out why the estimate
    # disagrees with a re-solve. `mu` comes from the classifier, so it
    # is unavailable on that path too.
    mu, activity, row_status = float("nan"), {}, {}
    var_on_bound = row_on_bound = None
    bounds_relaxed = bool(session.solver.bound_relax_factor)
    refined = {}
    if not bounds_relaxed:
        act = session.solver.classify_activity()
        mu = float(act["mu"])
        var_st, row_st = list(act["var_status"]), list(act["row_status"])
        if refine_activity:
            var_st, row_st, refined = _refine_ambiguous(
                session, var_st, row_st, session.var_names, row_names)
        activity = dict(zip(session.var_names, var_st))
        row_status = dict(zip(row_names, row_st))
        # The ratio test reads "is this coordinate sitting on its bound",
        # which is the refined answer too -- a kink the cheap rule could
        # not call is still a coordinate on its bound.
        var_on_bound = _on_bound(var_st)
        row_on_bound = _on_bound(row_st)

    alpha, first = _ratio_test(base, dx, lo, hi, session.var_names, tol=eps,
                               on_bound=var_on_bound, mu=mu)
    first_kind = None if first is None else "variable"
    dg = _row_step(session, dx)
    g_base = np.asarray(session.nl.constraints(base))
    # an equality constraint is held to first order and cannot change
    # activity, and a pin constraint moves by construction
    live = g_l < g_u
    live[list(pin_idx)] = False
    g_pred = g_base + dg
    gtol = 1e-9 * np.maximum(1.0, np.abs(g_pred))
    a_row, f_row = _ratio_test(g_base, dg, g_l, g_u, row_names, live=live,
                               on_bound=row_on_bound, mu=mu, tol=gtol)
    if a_row < alpha:
        alpha, first, first_kind = a_row, f_row, "constraint"

    crossed = session.new_keymap()
    for i in np.where((x_new < lo - eps) | (x_new > hi + eps))[0]:
        ov = session.var_key(i)
        if ov is not None:
            crossed[ov] = float(max(lo[i] - x_new[i], x_new[i] - hi[i]))
    crossed_rows = session.new_keymap()
    out_of_bounds = (g_pred < g_l - gtol) | (g_pred > g_u + gtol)
    rows_out = np.where(live & out_of_bounds)[0]
    # resolved only when a row actually crossed: the resolution is
    # once per session, but the common case has nothing to resolve for
    row_data = session.user_row_data() if rows_out.size else ()
    for j in rows_out:
        oc = row_data[j]
        if oc is not None:
            crossed_rows[oc] = float(
                max(g_l[j] - g_pred[j], g_pred[j] - g_u[j]))

    # the perturbation IS the shift of the pin constraints' right-hand
    # sides, so the violation is measured against the shifted ones
    gl_p, gu_p = g_l.copy(), g_u.copy()
    for pin, d in zip(pin_idx, deltas):
        gl_p[pin] += d
        gu_p[pin] += d
    g_at = np.asarray(session.nl.constraints(x_new))
    violation = float(np.max(np.maximum.reduce(
        [gl_p - g_at, g_at - gu_p, np.zeros_like(g_at)])))

    corrector = None
    if corrector_iter:
        _, corrector = _correct(
            session, pin_idx, deltas, np.asarray(step), corrector_iter)

    return SolutionReport(
        alpha=alpha, first=first, first_kind=first_kind,
        crossed=crossed, crossed_rows=crossed_rows, violation=violation,
        mu=mu, activity=activity, row_activity=row_status,
        perturbations=np.asarray(session.solver.kkt_perturbations).tolist(),
        bounds_relaxed=bounds_relaxed, corrector=corrector,
        refine_stop=refine_stop, refined=refined,
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
    degeneracy_iter = _degeneracy_iter(
        degeneracy_iter, degeneracy, who)
    check_margins(None, max_pdpert, who)
    refuse_on_pdpert(session, max_pdpert, who)
    pin_idx, deltas = list(pin_rows), list(deltas)
    if degeneracy == "directional":
        try:
            _, held_rows, _ = (
                session.solver.parametric_step_directional(
                    pin_idx, deltas, degeneracy_iter))
            _, segments = session.solver.parametric_step_path_decided(
                pin_idx, deltas, held_rows, predictor_iter)
        except RuntimeError as e:
            if "directional derivative" not in str(e):
                raise
            warnings.warn(
                f"{who}: {e}. Falling back to the "
                "one-sided record, the degeneracy='one_sided' behavior.")
            _, segments = session.solver.parametric_step_path(
                pin_idx, deltas, predictor_iter)
    elif degeneracy == "release_all":
        _, segments = session.solver.parametric_step_path_decided(
            pin_idx, deltas, [], predictor_iter)
    else:
        _, segments = session.solver.parametric_step_path(
            pin_idx, deltas, predictor_iter)

    # A segment names its bounded quantity by PRIMAL KKT row, and the
    # primal blocks are x then s. Below n_x it is a variable, in var-x
    # (var_names is full-x, so invert the map scatter_x applies). At or
    # above it the limit is a constraint's own -- g(x) <= cap bounds
    # the row's slack, not any variable -- and resolving it against the
    # variable map would return a NEIGHBOURING variable, the gh#450
    # hazard. gh#928 is what made the second case reachable.
    n_x = session.solver.block_dims[0]
    full_of = {row: full
               for full, row in enumerate(session._primal_row_map())
               if row is not None}
    slack_of = None
    out = []
    for frac, row, lower, pinned in segments:
        if row < n_x:
            full = full_of[row]
            key = session.var_key(full)
            if key is None:
                key = session.var_names[full]
        else:
            if slack_of is None:
                # Built only when a row limit actually moves, so the
                # common all-variables path costs no extra call.
                slack_of = {
                    r: g
                    for g, r in enumerate(session.solver.slack_rows(
                        list(range(len(session.con_names)))))
                    if r is not None}
            key = session.row_key(slack_of[row])
            if key is None:
                key = session.con_names[slack_of[row]]
        out.append(ActiveSetChange(
            fraction=float(frac),
            var=key,
            bound="lower" if lower else "upper",
            action="reaches" if pinned else "leaves",
        ))
    return out