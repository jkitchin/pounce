"""Tests for pyomo_pounce.estimate_report: what the linear step does
about the bounds."""
import numpy as np

import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import (active_set_changes, declare_sens_param, estimate,
                          estimate_report)
from pyomo_pounce.sens import (_NO_BOUND, _perturbation_deltas, _ratio_test,
                               _session_for)


# ── the ratio test on its own ────────────────────────────────────────────────
#
# Three of its cases cannot be reached reliably by solving a model: the
# no-bound sentinel needs a model with no bound in the step direction AND
# a solve that leaves one, a relaxed solve settles outside its bound by an
# amount the solver chooses, and the per-side exclusion needs a coordinate
# the classifier calls on-bound while the step drives it the other way.
# All three are ordinary arguments here.

def test_ratio_test_reads_the_no_bound_sentinel_as_no_bound():
    """`read_nl` seeds absent bounds with +-1e19, which is finite. An
    `isfinite` test scores a crossing at 1e18 times the perturbation."""
    base = np.array([0.0])
    step = np.array([1.2])
    lo = np.array([-_NO_BOUND])
    hi = np.array([_NO_BOUND])
    alpha, first = _ratio_test(base, step, lo, hi, ["z"])
    assert alpha == float("inf")
    assert first is None


def test_ratio_test_clamps_a_coordinate_already_outside_its_bound():
    """A relaxed solve settles outside a declared bound, so the distance
    is negative. No room to move is 0.0, not a negative fraction."""
    base = np.array([5.000000049582308])
    step = np.array([0.5])
    lo = np.array([-5.0])
    hi = np.array([5.0])
    alpha, first = _ratio_test(base, step, lo, hi, ["y"])
    assert alpha == 0.0
    assert first == "y"


def test_ratio_test_excludes_only_the_side_a_coordinate_is_held_on():
    """x is on ub = 1 and the step drives it to -2, past lb = -1. The
    crossing is at 2/3 and belongs to the bound x is NOT held at."""
    base = np.array([1.0, 0.0])
    step = np.array([-3.0, 0.4])
    lo = np.array([-1.0, -5.0])
    hi = np.array([1.0, 5.0])
    on_bound = np.array([True, False])
    alpha, first = _ratio_test(base, step, lo, hi, ["x", "z"],
                               on_bound=on_bound, mu=1e-9)
    assert first == "x"
    assert alpha == pytest.approx(2.0 / 3.0, rel=1e-12)

    # and the side it IS held on stays excluded, for a step that does
    # not carry it past that bound: the gap left there is barrier
    # residue and x must not set the fraction
    near = np.array([1.0 - 1e-6, 0.0])
    alpha2, first2 = _ratio_test(near, np.array([5e-7, 0.4]), lo, hi,
                                 ["x", "z"], on_bound=on_bound, mu=1e-9)
    assert first2 == "z"


def test_ratio_test_scales_its_distance_floor_with_the_barrier():
    """A coordinate the classifier declines to rule on is judged by the
    size of its gap. At mu = 1e-9 a weakly active one sits O(sqrt(mu))
    from its bound, four orders inside the interior case."""
    lo, hi = np.array([-5.0]), np.array([1.0])
    # steps that stop short of the bound, so the question is the floor
    # rather than a crossing, which is scored either way
    weak = np.array([1.0 - 4.4e-5])       # O(sqrt(mu)) gap: on its bound
    alpha, first = _ratio_test(weak, np.array([2e-5]), lo, hi, ["x"],
                               mu=1e-9)
    assert alpha == float("inf") and first is None

    interior = np.array([1.0 - 1e-2])     # real room left
    alpha, first = _ratio_test(interior, np.array([5e-3]), lo, hi, ["x"],
                               mu=1e-9)
    assert first == "x"
    assert alpha == pytest.approx(2.0, rel=1e-9)

    # with no classification there is no mu, and the fixed floor applies
    alpha, first = _ratio_test(weak, np.array([2e-5]), lo, hi, ["x"])
    assert first == "x"


def bounded(ub_y=5.0, fixed=False):
    """min (x-p)^2 + (y-2p)^2, so the unconstrained solution is x = p,
    y = 2p and the step is dx/dp = 1, dy/dp = 2. From p = 1, y reaches
    ub_y after (ub_y - 2) / 2 units of p."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-5.0, ub_y), initialize=1.0)
    expr = (m.x - m.p) ** 2 + (m.y - 2 * m.p) ** 2
    if fixed:
        # a fixed variable is removed from the solve, which shifts every
        # later factor column; the report must stay in user space
        m.f = pyo.Var(bounds=(3.0, 3.0), initialize=3.0)
        expr = expr + (m.f - 3.0) ** 2
    m.obj = pyo.Objective(expr=expr)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def held_at_a_bound():
    """x is weakly active at ub = 1 after solving at p = 1, and z has
    far bounds so it cannot be the first crossing at small
    perturbations."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(None, 1.0), initialize=0.0)
    m.z = pyo.Var(bounds=(-500.0, 500.0), initialize=0.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.z - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def with_row():
    """The same objective under x + y <= 6, which binds at p = 2 while
    both variables are still interior."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(-50.0, 50.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-50.0, 50.0), initialize=2.0)
    m.c = pyo.Constraint(expr=m.x + m.y <= 6.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - 2 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def brute_force(model, param, newval):
    """Scan the unclamped step for the first variable bound reached."""
    return brute_force_multi(model, [(param, newval)])


def brute_force_multi(model, perturb):
    """Reference scan, deliberately reading the same arrays the code
    under test reads.

    An earlier version took bounds off the Pyomo model and clamped at
    zero, so it silently disagreed with the production path in exactly
    the two places that path was wrong (the +-1e19 sentinel and a
    negative distance) and the suite could not see either.
    """
    session = model.__dict__["_pounce_sens"].session
    lo, hi = np.asarray(session.nl.x_l), np.asarray(session.nl.x_u)
    base = np.asarray(session.base_x)
    est = estimate(model, perturb, clamp=False)
    alpha, who = float("inf"), None
    for i, nm in enumerate(session.var_names):
        v = model.find_component(nm)
        if v is None or v not in est:
            continue
        d = est[v] - base[i]
        if abs(d) < 1e-14:
            continue
        b = hi[i] if d > 0 else lo[i]
        if abs(b) >= _NO_BOUND:
            continue
        a = max((b - base[i]) / d, 0.0)
        if a < alpha:
            alpha, who = a, nm
    return alpha, who


def test_step_fraction_matches_the_hand_computed_crossing():
    # y goes 2 -> 8 over p = 1 -> 4 and stops at 5, which is half way
    m = bounded()
    r = estimate_report(m, [(m.p, 4.0)])
    assert r.first == "y"
    assert r.first_kind == "variable"
    assert r.alpha == pytest.approx(0.5, abs=1e-8)


def test_step_fraction_matches_a_brute_force_scan():
    m = bounded()
    alpha, who = brute_force(m, m.p, 4.0)
    r = estimate_report(m, [(m.p, 4.0)])
    assert r.first == who
    assert r.alpha == pytest.approx(alpha, rel=1e-12)


def test_a_fixed_variable_does_not_shift_the_scan():
    m = bounded(fixed=True)
    alpha, who = brute_force(m, m.p, 4.0)
    r = estimate_report(m, [(m.p, 4.0)])
    assert r.first == who == "y"
    assert r.alpha == pytest.approx(alpha, rel=1e-12)


def test_an_interior_perturbation_crosses_nothing():
    m = bounded()
    r = estimate_report(m, [(m.p, 1.2)])
    assert r.alpha > 1.0
    assert len(r.crossed) == 0
    assert r.crossed_rows == {}
    assert r.violation == pytest.approx(0.0, abs=1e-8)


def test_crossed_reports_the_distance_past_the_bound():
    m = bounded()
    r = estimate_report(m, [(m.p, 4.0)])
    est = estimate(m, [(m.p, 4.0)], clamp=False)
    assert len(r.crossed) == 1
    assert m.y in r.crossed
    assert r.crossed[m.y] == pytest.approx(est[m.y] - m.y.ub, rel=1e-9)


def test_a_constraint_row_can_bind_before_any_variable():
    m = with_row()
    r = estimate_report(m, [(m.p, 3.0)])
    # the row sits at 3 and gains 3 per unit of p, so it reaches 6 at
    # p = 2, half way to p = 3
    assert r.first_kind == "constraint"
    assert r.alpha == pytest.approx(0.5, abs=1e-8)
    assert len(r.crossed) == 0


def test_rows_are_named_as_the_model_names_them():
    m = with_row()
    r = estimate_report(m, [(m.p, 3.0)])
    assert r.first == "c"
    assert len(r.crossed_rows) == 1
    assert m.c in r.crossed_rows
    assert "c" in r.row_activity


def test_violation_matches_direct_evaluation_at_the_predicted_point():
    m = with_row()
    r = estimate_report(m, [(m.p, 3.0)])
    for v, val in estimate(m, [(m.p, 3.0)], clamp=False).items():
        v.set_value(val)
    body = pyo.value(m.x) + pyo.value(m.y)
    assert r.violation == pytest.approx(max(body - 6.0, 0.0), rel=1e-12)


def test_the_pin_row_is_not_reported_as_a_crossing():
    # the perturbation moves the pin row's right-hand side by
    # construction, so it is neither a crossing nor a violation
    m = bounded()
    r = estimate_report(m, [(m.p, 4.0)])
    assert all("paramConst" not in nm for nm in r.crossed_rows)
    assert r.violation == pytest.approx(0.0, abs=1e-8)


def test_classification_and_mu_match_the_core_classifier():
    m = bounded()
    r = estimate_report(m, [(m.p, 4.0)])
    session = m.__dict__["_pounce_sens"].session
    act = session.solver.classify_activity()
    assert r.mu == pytest.approx(float(act["mu"]), rel=1e-12)
    assert r.activity["y"] == act["var_status"][session.var_names.index("y")]
    assert set(r.activity) >= {"x", "y"}


def test_an_already_active_bound_is_not_a_crossing():
    """A variable on its bound has an O(mu) gap left and an O(mu) step
    component, and their quotient is noise. It must not set the step
    fraction: the classification is what reports it."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=4.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-5.0, 5.0), initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - 2 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    assert pyo.value(m.y) == pytest.approx(5.0, abs=1e-6)

    r = estimate_report(m, [(m.p, 5.0)])
    assert r.activity["y"] == "strongly_active"
    assert r.first == "x"          # x runs 4 -> 5 against its bound of 10
    assert r.alpha == pytest.approx(6.0, rel=1e-6)


def test_a_saturating_control_is_named_with_its_step_fraction():
    """The case the diagnostics exist for: a setpoint move large enough
    to drive a control onto its bound, where estimate() clamps and
    reports nothing about which control or how far along the move it
    happened."""
    n, a, b, r = 6, 0.8, 0.5, 0.05
    m = pyo.ConcreteModel()
    m.k = pyo.RangeSet(0, n - 1)
    m.sp = pyo.Param(initialize=0.5, mutable=True)
    m.x = pyo.Var(pyo.RangeSet(0, n), initialize=0.0)
    m.u = pyo.Var(m.k, bounds=(-1.0, 1.0), initialize=0.0)
    m.x[0].fix(0.0)

    @m.Constraint(m.k)
    def dynamics(m, k):
        return m.x[k + 1] == a * m.x[k] + b * m.u[k]

    m.obj = pyo.Objective(
        expr=sum((m.x[k + 1] - m.sp) ** 2 for k in m.k)
        + r * sum(m.u[k] ** 2 for k in m.k))
    declare_sens_param(m.sp)
    pyo.SolverFactory("pounce").solve(m)
    assert all(abs(pyo.value(m.u[k])) < 0.999 for k in m.k)

    r_small = estimate_report(m, [(m.sp, 0.55)])
    assert r_small.alpha > 1.0
    assert len(r_small.crossed) == 0

    r_big = estimate_report(m, [(m.sp, 3.0)])
    assert r_big.first_kind == "variable"
    assert r_big.first.startswith("u[")
    assert 0.0 < r_big.alpha < 1.0
    assert r_big.crossed                     # estimate() clamps these
    assert all(v.name.startswith("u[") for v in r_big.crossed)

    # the step fraction is the fraction of the setpoint move that fits,
    # so the perturbation it admits crosses nothing
    fits = 0.5 + r_big.alpha * (3.0 - 0.5)
    assert estimate_report(m, [(m.sp, fits)]).alpha == pytest.approx(
        1.0, rel=1e-6)


def test_provenance_is_reported_on_an_ordinary_solve():
    """The three things separating the predictor from the exact value
    at the perturbed active set: the barrier parameter, whether the
    factor was regularized, and whether the solve relaxed its bounds."""
    m = bounded()
    r = estimate_report(m, [(m.p, 4.0)])
    assert np.isfinite(r.mu) and r.mu > 0.0
    assert r.bounds_relaxed is False
    assert not any(r.perturbations)     # convex model, no inertia correction


def test_a_relaxed_solve_is_reported_rather_than_raising():
    """`bound_relax_factor` lets a variable settle outside its declared
    bound, so the classifier refuses the solve. The rest of the report
    is still measured, since a caller reaches for it precisely when the
    estimate and a re-solve disagree."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-5.0, 5.0), initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - 2 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(
        m, options={"bound_relax_factor": 1e-8})

    r = estimate_report(m, [(m.p, 4.0)])
    assert r.bounds_relaxed is True
    assert r.activity == {} and r.row_activity == {}
    assert np.isnan(r.mu)
    # the measured half still lands
    assert r.first == "y"
    assert r.alpha == pytest.approx(0.5, abs=1e-6)


def test_a_weakly_active_bound_is_classified_as_such():
    """min (x - 1)^2 with x <= 1 puts the bound on the unconstrained
    minimum, so the bound is active and its multiplier is zero: strict
    complementarity fails and the classification must say so rather
    than call it strongly active."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(None, 1.0), initialize=0.0)
    m.z = pyo.Var(bounds=(-5.0, 5.0), initialize=0.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.z - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    # the barrier leaves an O(sqrt(mu)) gap here, not the O(mu) gap it
    # leaves at a strongly active bound
    assert pyo.value(m.x) == pytest.approx(1.0, abs=1e-3)

    # perturb away from the bound x is held at, so the question is
    # whether the gap counts as room rather than whether x crosses
    r = estimate_report(m, [(m.p, 0.5)])
    assert r.activity["x"] == "weakly_active"
    # that gap is barrier residue, not room, so it must not set the
    # fraction: scoring it would put the crossing at a fraction of a
    # percent of a perturbation that crosses nothing
    assert len(r.crossed) == 0
    assert r.first != "x"


def test_several_parameters_perturbed_at_once():
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.q = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-5.0, 5.0), initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - 2 * m.q) ** 2)
    declare_sens_param(m.p)
    declare_sens_param(m.q)
    pyo.SolverFactory("pounce").solve(m)

    # y tracks q alone and reaches 5 at q = 2.5, half way to q = 4
    r = estimate_report(m, [(m.p, 2.0), (m.q, 4.0)])
    assert r.first == "y"
    assert r.alpha == pytest.approx(0.5, abs=1e-6)
    alpha, who = brute_force_multi(m, [(m.p, 2.0), (m.q, 4.0)])
    assert r.first == who
    assert r.alpha == pytest.approx(alpha, rel=1e-12)


def test_every_solver_route_reports_the_same():
    """`Pounce.solve` sends a model carrying declarations down the same
    in-process sensitivity route the legacy plugin uses, so one session
    serves all three entry points and the report cannot depend on which
    one ran."""
    from pyomo.contrib.solver.common.factory import SolverFactory as SF2

    reports = []
    for solve in (lambda m: pyo.SolverFactory("pounce").solve(m),
                  lambda m: pyo.SolverFactory("pounce_v2").solve(m),
                  lambda m: SF2("pounce").solve(m)):
        m = pyo.ConcreteModel()
        m.p = pyo.Param(initialize=1.0, mutable=True)
        m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
        m.y = pyo.Var(bounds=(-5.0, 5.0), initialize=1.0)
        m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - 2 * m.p) ** 2)
        declare_sens_param(m.p)
        solve(m)
        reports.append(estimate_report(m, [(m.p, 4.0)]))

    legacy, v2, contrib = reports
    for other in (v2, contrib):
        assert other.first == legacy.first == "y"
        assert other.alpha == pytest.approx(legacy.alpha, rel=1e-12)
        assert other.mu == pytest.approx(legacy.mu, rel=1e-12)
        assert other.activity == legacy.activity


def test_a_bound_the_step_crosses_is_never_missed():
    """The report must not contradict itself: anything in `crossed` was
    reached by the full step, so the fraction that fits is below one.

    Reachable whenever a two-sided coordinate is classified on-bound and
    the step drives it toward the other side, which an exclusion applied
    per coordinate rather than per side would drop.

    It holds on a solve that kept its bounds, which is why the check
    below is scoped to those. A relaxed solve can settle a coordinate
    outside a bound, and `crossed` measures both bounds at the predicted
    point while the step fraction looks only along the step direction,
    so there the two answer different questions and are not required to
    agree. The test below this one covers that case.
    """
    cases = [
        (bounded(), 4.0),
        (bounded(fixed=True), 4.0),
        (with_row(), 3.0),
        (bounded(), 1.2),
        (held_at_a_bound(), 1.5),   # pushed further past the same side
        (held_at_a_bound(), 0.5),   # and away from it
    ]
    for m, newval in cases:
        r = estimate_report(m, [(m.p, newval)])
        if (len(r.crossed) or len(r.crossed_rows)) and not r.bounds_relaxed:
            assert r.alpha < 1.0, (
                f"crossed {len(r.crossed)} variables and "
                f"{len(r.crossed_rows)} constraints, yet reported that "
                f"{r.alpha} of the perturbation fits")


def test_a_relaxed_solve_reports_a_bound_its_own_solve_left():
    """`bound_relax_factor` lets the solve settle a coordinate outside a
    declared bound. `crossed` measures both bounds at the predicted
    point, so it names that coordinate even though no step carried it
    there, while the step fraction looks only along the step direction.
    They answer different questions, which is why the invariant in
    `test_a_bound_the_step_crosses_is_never_missed` is scoped to solves
    that kept their bounds."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=4.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-5.0, 5.0), initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - 2 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(
        m, options={"bound_relax_factor": 1e-8})
    assert pyo.value(m.y) > 5.0, "the solve did not relax the bound"

    r = estimate_report(m, [(m.p, 3.0)])
    assert r.bounds_relaxed is True
    # the solve left y outside, and `crossed` reports it as such
    assert m.y in r.crossed
    # measured at the predicted point rather than the base one, so it
    # is the relaxation's scale rather than that exact number, and well
    # below anything the model itself works at
    assert r.crossed[m.y] == pytest.approx(pyo.value(m.y) - 5.0, rel=0.1)
    assert r.crossed[m.y] < 1e-6
    # and the classification is unavailable on this path
    assert r.activity == {} and np.isnan(r.mu)


def test_a_coordinate_on_one_bound_can_still_cross_the_other():
    """x is weakly active at its upper bound while the step drives it
    down past its lower one. Excluding the coordinate rather than the
    side it is held on loses that crossing."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(-1.0, 1.0), initialize=0.0)
    m.z = pyo.Var(bounds=(-5.0, 5.0), initialize=0.0)
    # minimized at x = p, so at p = 1 the upper bound is weakly active
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + 0.1 * (m.z - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    assert m.x.value == pytest.approx(1.0, abs=1e-3)
    assert estimate_report(m, [(m.p, 1.1)]).activity["x"] == "weakly_active"

    # drive x down, away from the bound it is held at and toward the
    # other one. x must be scored: the exclusion covers its upper side
    # only. (The base point is degenerate, so the step is the
    # directional derivative for this direction, the interior value,
    # and the report and the brute force agree because both take it.)
    r = estimate_report(m, [(m.p, -2.0)])
    assert r.first == "x"
    alpha, who = brute_force_multi(m, [(m.p, -2.0)])
    assert who == "x"
    assert r.alpha == pytest.approx(alpha, rel=1e-12)


def test_a_coordinate_pushed_further_past_the_bound_it_is_held_at():
    """The exclusion covers the side x is held on, which is also the
    side the one-sided step pushes it past. Excluding it there reported
    that 998 times the perturbation fits while naming x as leaving its
    bound by 0.25 in the same object. That bookkeeping lives on the
    one-sided step, so this pins it there; under the directional
    default the weak bound is decided for this direction, which holds
    x, and nothing crosses at all."""
    m = held_at_a_bound()
    r = estimate_report(m, [(m.p, 1.5)], degeneracy="one_sided")
    assert r.activity["x"] == "weakly_active"
    assert len(r.crossed) == 1 and m.x in r.crossed

    # x is on its bound and the one-sided step drives it outward, so no
    # part of the perturbation fits before the bound is reached
    assert r.first == "x"
    assert r.alpha < 1e-3

    # under the directional default the decided step holds x on its
    # bound, which is the correct one-sided derivative for this side
    assert len(estimate_report(m, [(m.p, 1.5)]).crossed) == 0

    # the classification still keeps it out when nothing crosses, which
    # is what the exclusion is for
    r2 = estimate_report(m, [(m.p, 1.0 + 1e-9)], degeneracy="one_sided")
    assert len(r2.crossed) == 0
    assert r2.first != "x"


def test_the_distance_floor_is_capped():
    """The floor is relative to the coordinate's own magnitude, so an
    uncapped `10 * sqrt(mu)` widens without limit when the solve leaves
    mu loose, and a dropped coordinate is a missed crossing."""
    lo = np.array([-1e9])
    hi = np.array([1000.4])
    base = np.array([1000.0])         # 0.4 units of genuine room
    step = np.array([1.0])

    # a loose solve: sqrt(mu) * 10 would be 3e-2 relative, 30 absolute
    alpha, first = _ratio_test(base, step, lo, hi, ["x"], mu=9.1e-6)
    assert first == "x"
    assert alpha == pytest.approx(0.4, rel=1e-9)

    # and the calibration at an ordinary mu is unchanged: a gap at
    # barrier scale is still read as being on the bound
    weak = np.array([1.0 - 4.4e-5])
    alpha, first = _ratio_test(weak, np.array([2e-5]), np.array([-5.0]),
                               np.array([1.0]), ["x"], mu=1e-9)
    assert alpha == float("inf") and first is None


@pytest.mark.parametrize("opts, label", [
    ({}, "ordinary"),
    ({"tol": 1e-2, "mu_init": 1e-1}, "loose"),
])
def test_a_loose_solve_does_not_widen_the_floor_past_a_real_gap(opts, label):
    """End to end for the cap, since the floor is relative to the
    coordinate's own magnitude and mu at termination is whatever the
    solve leaves. Uncapped, `10 * sqrt(mu)` is 0.50 absolute here on an
    ordinary solve and 13.6 on a loose one, both past the 0.4 units of
    genuine room, so x would be dropped and its crossing missed."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1000.0, mutable=True)
    m.x = pyo.Var(bounds=(None, 1000.4), initialize=999.0)
    m.z = pyo.Var(bounds=(-5000.0, 5000.0), initialize=0.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.z - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m, options=opts)

    r = estimate_report(m, [(m.p, 1002.0)])
    assert 10.0 * r.mu ** 0.5 * 1000.0 > 0.4, (
        f"{label}: the uncapped floor no longer exceeds the gap, so this "
        "no longer tests the cap")
    # dx/dp is 1 and the gap is 0.4, so a fifth of the move fits
    assert r.first == "x"
    assert r.alpha == pytest.approx(0.2, rel=1e-3)


def test_a_crossing_is_scored_even_where_the_exclusion_applies():
    """One predicate decides `crossed` and participation, so the two
    cannot disagree. A coordinate held at its bound and driven outward
    scores 0.0: no room, rather than no bound."""
    lo = np.array([-5.0])
    hi = np.array([1.0])
    base = np.array([1.0 - 4.4e-5])   # on its bound at barrier scale
    step = np.array([0.25])           # driven outward, well past it
    alpha, first = _ratio_test(base, step, lo, hi, ["x"],
                               on_bound=np.array([True]), mu=1e-9)
    assert first == "x"
    assert 0.0 <= alpha < 1.0

    # nothing crosses: the exclusion applies and x drops out
    alpha, first = _ratio_test(base, np.array([1e-6]), lo, hi, ["x"],
                               on_bound=np.array([True]), mu=1e-9)
    assert alpha == float("inf") and first is None


def test_no_bound_in_the_step_direction_reports_no_crossing():
    """Unbounded variables reach the ratio test as the reader's +-1e19
    sentinel, which is finite."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(initialize=0.0)
    m.z = pyo.Var(initialize=0.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.z - 2 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)

    r = estimate_report(m, [(m.p, 4.0)])
    assert r.alpha == float("inf")
    assert r.first is None and r.first_kind is None
    assert len(r.crossed) == 0


def test_a_relaxed_solve_never_reports_a_negative_fraction():
    """Relaxed bounds let a variable settle outside its declared bound,
    which is no room to move rather than a negative amount of it."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=4.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-5.0, 5.0), initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + (m.y - 2 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(
        m, options={"bound_relax_factor": 1e-8})

    r = estimate_report(m, [(m.p, 5.0)])
    assert r.bounds_relaxed is True
    assert r.alpha >= 0.0

    # read off the solve, not off the classifier's error message: the
    # report says the bounds were relaxed because the solve says so
    session = m.__dict__["_pounce_sens"].session
    assert session.solver.bound_relax_factor == 1e-8
    with pytest.raises(Exception, match="bound_relax_factor"):
        session.solver.classify_activity()


def test_no_session_is_a_clean_error():
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    with pytest.raises(RuntimeError, match="no sensitivity session"):
        estimate_report(m, [(m.p, 2.0)])


def test_max_iter_here_warns_instead_of_being_ignored():
    """`max_iter` used to budget the directional decision here, so
    `max_iter=0` forced the one-sided fallback. `degeneracy_iter` is
    that knob now and `max_iter` does nothing, which would silently
    change what this returns for a caller who passed it."""
    m = bounded()
    with pytest.warns(DeprecationWarning, match="max_iter no longer"):
        got = estimate_report(m, [(m.p, 4.0)], max_iter=0)
    # and it really is ignored: the report is the one the same call
    # without it produces, not the one-sided fallback max_iter=0 used
    # to force
    assert got.alpha == pytest.approx(estimate_report(m, [(m.p, 4.0)]).alpha)


def test_the_other_two_surfaces_keep_their_budget_silent():
    """`estimate` and `active_set_changes` still spend a budget on the
    mode's own work, so passing it there is not deprecated. It is named
    `predictor_iter` on those two, since what it bounds is the
    prediction rather than the correction or the degeneracy decision.
    Matched on this deprecation's own text rather than on
    DeprecationWarning as a class, so an unrelated one from pyomo or
    numpy cannot fail it."""
    import warnings as _w
    m = bounded()
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        estimate(m, [(m.p, 1.2)], predictor_iter=8)
        active_set_changes(m, [(m.p, 1.2)], predictor_iter=8)
    assert not [w for w in caught if "no longer does anything" in str(w.message)]


# ── the mode the report measures ─────────────────────────────────────────────
#
# `violation` and `corrector` are properties of the step, so they move
# with the mode. The rest either measures where the step leaves a bound,
# which only "linear" does, or comes from the base point.

def coupled():
    """A nonlinear model whose linear step leaves `x` outside its lower
    bound, so the three modes take genuinely different steps and the
    constraint is violated by a different amount at each."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.1, 5.0), initialize=1.0)
    m.y = pyo.Var(bounds=(0.1, 5.0), initialize=1.0)
    m.c = pyo.Constraint(expr=m.x * m.y == 1.0 + 0.4 * m.p)
    m.obj = pyo.Objective(expr=(m.x - 2 * m.p) ** 2 + (m.y - m.p) ** 4)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


@pytest.mark.parametrize("mode", ["linear", "fix_relax", "path"])
def test_the_violation_is_the_one_at_the_mode_s_own_point(mode):
    """The report has to describe the step `estimate()` takes for the
    same arguments. Evaluating the constraint at that mode's estimate
    is what says whether it does."""
    m = coupled()
    r = estimate_report(m, [(m.p, -1.2)], mode=mode)
    for v, val in estimate(m, [(m.p, -1.2)], clamp=False, mode=mode).items():
        v.set_value(val)
    body = pyo.value(m.x) * pyo.value(m.y)
    rhs = 1.0 + 0.4 * (-1.2)
    assert r.violation == pytest.approx(abs(body - rhs), rel=1e-9)


def test_the_violation_moves_with_the_mode():
    m = coupled()
    lin = estimate_report(m, [(m.p, -1.2)], mode="linear").violation
    fix = estimate_report(m, [(m.p, -1.2)], mode="fix_relax").violation
    pat = estimate_report(m, [(m.p, -1.2)], mode="path").violation
    assert lin > 5 * fix, (
        f"the linear step should be visibly less feasible here: "
        f"{lin:.6e} against {fix:.6e}")
    assert fix == pytest.approx(pat, rel=1e-9)


def test_the_other_modes_report_a_step_that_stops_at_the_bound():
    """Both stop the step at the bound, so there is no crossing left to
    report. That is the correct answer for such a step."""
    m = coupled()
    assert estimate_report(m, [(m.p, -1.2)]).crossed, (
        "this perturbation should cross a bound under the linear step")
    for mode in ("fix_relax", "path"):
        r = estimate_report(m, [(m.p, -1.2)], mode=mode)
        assert r.alpha == pytest.approx(1.0), f"mode={mode}: {r.alpha}"
        assert list(r.crossed) == [], f"mode={mode}: {list(r.crossed)}"
        assert list(r.crossed_rows) == []


def test_the_base_point_fields_do_not_move_with_the_mode():
    m = coupled()
    reports = [estimate_report(m, [(m.p, -1.2)], mode=mode)
               for mode in ("linear", "fix_relax", "path")]
    first = reports[0]
    for r in reports[1:]:
        assert r.mu == pytest.approx(first.mu, rel=1e-12)
        assert r.activity == first.activity
        assert r.row_activity == first.row_activity
        assert r.bounds_relaxed == first.bounds_relaxed


def test_the_corrector_measures_the_mode_s_own_step():
    """The residual the corrector starts from is the mode's step in the
    barrier system, so a mode that takes a different step reports a
    different starting residual."""
    m = coupled()
    lin = estimate_report(m, [(m.p, -1.2)], mode="linear", corrector_iter=6)
    fix = estimate_report(m, [(m.p, -1.2)], mode="fix_relax", corrector_iter=6)
    assert lin.corrector["initial_residual"] > (
        2 * fix.corrector["initial_residual"]), (
        f"{lin.corrector['initial_residual']:.4e} against "
        f"{fix.corrector['initial_residual']:.4e}")


def test_an_unknown_mode_is_refused():
    m = bounded()
    with pytest.raises(ValueError, match="mode must be"):
        estimate_report(m, [(m.p, 2.0)], mode="relax_fix")


def test_refine_stop_says_why_the_refinement_stopped():
    """A pass pins every crossing it sees, so the pin count says
    nothing about which limit was reached. Only the stop reason does."""
    m = coupled()
    r = estimate_report(m, [(m.p, -1.2)], mode="fix_relax")
    assert r.refine_stop in ("settled", "iteration_limit",
                            "degrees_of_freedom", "worse_than_plain"), (
        f"unrecognised stop reason {r.refine_stop!r}")
    for mode in ("linear", "path"):
        assert estimate_report(m, [(m.p, -1.2)], mode=mode).refine_stop is None


@pytest.mark.parametrize("mk,target", [
    (coupled, -1.2), (bounded, 4.0), (with_row, 3.0),
])
def test_settled_means_nothing_is_left_outside_a_bound(mk, target):
    """The label has to match the state it names. A pass now pins every
    crossing it sees, so these all settle in one, which is what makes
    the pin count useless as a proxy and the label worth having."""
    m = mk()
    r = estimate_report(m, [(m.p, target)], mode="fix_relax")
    assert r.refine_stop == "settled"
    assert list(r.crossed) == [], (
        f"settled but {[v.name for v in r.crossed]} is outside its bound")


# ── the two sIPOPT margins, on the pyomo surface ─────────────────────────────
#
# Both are settable through the CLI and the SensSolve builder and were
# unreachable from here (gh#736).

def test_bound_eps_sets_what_counts_as_leaving_a_bound():
    """The refinement pins a coordinate only once it is outside by more
    than the margin, so a margin wider than the crossing pins nothing
    and the step stays where the plain predictor put it. To solver
    tolerance rather than bit for bit: the release test runs at the
    solve's own margin whatever the primal one is, and the bounds it
    releases here are inactive ones, which moves the step by the
    multiplier's own size."""
    m = coupled()
    plain = estimate(m, [(m.p, -1.2)], mode="linear", clamp=False)
    tight = estimate(m, [(m.p, -1.2)], mode="fix_relax", bound_eps=1e-9)
    slack = estimate(m, [(m.p, -1.2)], mode="fix_relax", bound_eps=10.0)

    moved = max(abs(tight[v] - plain[v]) for v in plain)
    assert moved > 1e-3, (
        f"a tight margin should repair the crossing, moved {moved:g}")
    for v in plain:
        assert slack[v] == pytest.approx(plain[v], abs=1e-6), (
            f"a margin wider than the crossing should leave {v.name} alone")


def test_a_wide_primal_margin_does_not_stop_a_release():
    """`bound_eps` is a primal margin and says nothing about whether a
    multiplier has changed sign. With one number for both, a margin of
    ten stopped every release, so the refinement returned a different
    active set from the one the step asked for. Rows below `n_x` are
    pins and rows above are releases."""
    m = coupled()
    sess = _session_for(m)
    pin_idx, deltas = _perturbation_deltas(sess, [(m.p, -1.2)])
    n_x = len(sess.solver.parametric_step(pin_idx, deltas))

    _, rows, stop = sess.solver.parametric_step_bounded(
        pin_idx, deltas, 16, 10.0)
    assert stop == "settled"
    assert rows, "the step drives bound multipliers negative here"
    assert all(r >= n_x for r in rows), (
        f"a margin of ten pins nothing, got rows {list(rows)}")

    _, rows, _ = sess.solver.parametric_step_bounded(
        pin_idx, deltas, 16, 1e-9)
    assert any(r < n_x for r in rows), "the floor pins the crossing"
    assert any(r >= n_x for r in rows), "and releases what both do"


def test_bound_eps_decides_crossed_and_the_stop_reason_together():
    """`refine_stop` and `crossed` have to keep agreeing under the new
    argument. The refinement decides "outside a bound" against the
    margin and `crossed` used to decide it against a fixed 1e-9, so a
    margin wider than the crossing reported `settled` beside a
    coordinate 2.0 outside its bound."""
    m = coupled()
    for eps in (1e-9, 1e-2, 1.0, 10.0):
        r = estimate_report(m, [(m.p, -1.2)], mode="fix_relax", bound_eps=eps)
        assert r.refine_stop == "settled"
        assert list(r.crossed) == [], (
            f"settled at bound_eps={eps:g} but "
            f"{[v.name for v in r.crossed]} is reported outside its bound")


def test_bound_eps_leaves_constraint_rows_alone():
    """The refinement pins variable bounds only, so the margin has no
    say over a constraint row the step carries past its limit. A margin
    wider than everything on the model still reports the row, and the
    row is still what `alpha` reaches first."""
    m = with_row()
    r = estimate_report(m, [(m.p, 3.0)], mode="fix_relax", bound_eps=10.0)
    assert m.c in r.crossed_rows, "a margin on variables hid a row"
    assert r.crossed_rows[m.c] == pytest.approx(3.0, abs=1e-6)
    assert r.alpha == pytest.approx(0.5, abs=1e-6)
    assert r.first_kind == "constraint"


def large_coordinate():
    """One variable of order 1e4 with its bound at 1e4. A relative
    tolerance scaled by the coordinate is 1e-5 here, where the
    refinement's own test is absolute at 1e-9."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=9999.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 1e4), initialize=9999.0)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def test_the_margin_is_absolute_as_the_refinement_is():
    """`crossed` and the clamp compare the same way the refinement
    does. Scaling the tolerance by the coordinate gave a variable of
    order 1e4 a margin of 1e-5 that the refinement never gave it, so a
    step 1e-6 past the bound was pinned and refined against while the
    report said nothing had crossed."""
    m = large_coordinate()
    target = 1e4 + 1e-6

    r = estimate_report(m, [(m.p, target)], mode="linear")
    assert m.x in r.crossed, "the linear step leaves the bound by 1e-6"
    over = r.crossed[m.x]
    assert 1e-9 < over < 1e-9 * 1e4, (
        f"overshoot {over:g} is the case a relative tolerance hides")
    with pytest.warns(UserWarning, match="clamped"):
        estimate(m, [(m.p, target)], mode="linear")

    r = estimate_report(m, [(m.p, target)], mode="fix_relax")
    assert r.refine_stop == "settled"
    assert list(r.crossed) == []


def test_bound_eps_unset_is_the_solves_own_margin():
    """Unset is `bound_relax_factor` floored at 1e-9. pyomo-pounce
    solves with the relaxation off, so unset IS the floor here, and
    naming both numbers is what keeps this from comparing a value
    against itself."""
    m = coupled()
    assert _session_for(m).solver.bound_relax_factor == 0.0, (
        "this test reads the floor, which only shows through an "
        "unrelaxed solve")

    unset = estimate(m, [(m.p, -1.2)], mode="fix_relax")
    floor = estimate(m, [(m.p, -1.2)], mode="fix_relax", bound_eps=1e-9)
    wide = estimate(m, [(m.p, -1.2)], mode="fix_relax", bound_eps=10.0)

    for v in unset:
        assert unset[v] == pytest.approx(floor[v], abs=1e-12), (
            f"unset should resolve to the 1e-9 floor, {v.name} differs")
    moved = max(abs(wide[v] - unset[v]) for v in unset)
    assert moved > 1e-3, (
        f"a margin that covers the crossing should move the answer, "
        f"moved {moved:g}")


def test_bound_eps_warns_where_no_refinement_reads_it():
    """Only the fix_relax refinement pins against the margin. Passing it
    under another mode changes nothing, and a registered argument that
    looks wired and is not is what gh#677 was."""
    m = coupled()
    for mode in ("linear", "path"):
        with pytest.warns(UserWarning, match="bound_eps"):
            estimate(m, [(m.p, -1.2)], mode=mode, bound_eps=1e-3)
        with pytest.warns(UserWarning, match="bound_eps"):
            estimate_report(m, [(m.p, -1.2)], mode=mode, bound_eps=1e-3)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_bound_eps_takes_the_option_surfaces_bounds(bad):
    """`sens_bound_eps` is registered strictly above zero, so the CLI
    turns these down. Zero reinstates the roundoff pinning the floor
    prevents, and NaN makes every comparison against it false, so the
    refinement pins nothing and still reports settled."""
    m = coupled()
    for call in (estimate, estimate_report):
        with pytest.raises(ValueError,
                           match="bound_eps must be a positive number"):
            call(m, [(m.p, -1.2)], mode="fix_relax", bound_eps=bad)


def test_arguments_are_checked_before_the_factor_is():
    """A typo'd mode plus a tight cap should name the typo. The cap
    reads the factor, which is a fact about the solve rather than about
    the call, so it cannot be what a malformed call reports."""
    m = regularized()
    worst = max(abs(v) for v in _session_for(m).solver.kkt_perturbations)
    with pytest.raises(ValueError, match="mode"):
        estimate(m, [(m.p, 0.5)], mode="relax_fix", max_pdpert=worst / 10.0)
    with pytest.raises(ValueError, match="mode"):
        estimate_report(m, [(m.p, 0.5)], mode="relax_fix",
                        max_pdpert=worst / 10.0)
    with pytest.raises(ValueError, match="degeneracy"):
        active_set_changes(m, [(m.p, 0.5)], degeneracy="one-sided",
                           max_pdpert=worst / 10.0)


def regularized():
    """Two constraints stating the same row, so the factor carries an
    inertia correction and there is something for the cap to refuse."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=0.25, mutable=True)
    m.x = pyo.Var(range(3), bounds=(-10.0, 10.0), initialize=0.3)
    m.c1 = pyo.Constraint(expr=sum(m.x[i] for i in range(3)) == 1.0 - m.p)
    m.c2 = pyo.Constraint(expr=1.0 * sum(m.x[i] for i in range(3)) == 1.0 - m.p)
    m.obj = pyo.Objective(expr=sum(m.x[i] ** 2 for i in range(3)))
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def test_max_pdpert_refuses_a_factor_the_correction_perturbed():
    """Every sensitivity output inverts the converged factor, so a
    factor the inertia correction had to perturb answers for a nearby
    problem. The cap is the caller declining to accept that."""
    m = regularized()
    worst = max(abs(v) for v in _session_for(m).solver.kkt_perturbations)
    assert worst > 0.0, "this fixture is meant to be regularized"

    from pyomo_pounce import gradient
    for call in (lambda cap: gradient(m.x[0], wrt=m.p, max_pdpert=cap),
                 lambda cap: estimate(m, [(m.p, 0.5)], max_pdpert=cap),
                 lambda cap: estimate_report(m, [(m.p, 0.5)], max_pdpert=cap),
                 lambda cap: active_set_changes(m, [(m.p, 0.5)],
                                                max_pdpert=cap)):
        call(None)
        call(worst * 10.0)
        with pytest.raises(ValueError, match="max_pdpert"):
            call(worst / 10.0)


def test_max_pdpert_reads_the_same_comparison_the_option_reads():
    """`_refuse_on_pdpert` asks the solver rather than recomputing the
    threshold, so the pyomo argument and the CLI's sens_max_pdpert
    cannot drift apart on what counts as too perturbed. The boundary is
    where that shows: the comparison is strictly above, so a cap at the
    correction accepts it and a cap a hair below refuses."""
    m = regularized()
    refuse, worst = _session_for(m).solver.pdpert_verdict(0.0)
    assert refuse and worst > 0.0, "this fixture is meant to be regularized"

    estimate(m, [(m.p, 0.5)], max_pdpert=worst)
    with pytest.raises(ValueError, match="max_pdpert"):
        estimate(m, [(m.p, 0.5)], max_pdpert=worst * (1.0 - 1e-12))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_max_pdpert_takes_the_option_surfaces_bounds(bad):
    """`sens_max_pdpert` is registered strictly above zero. A negative
    cap refuses on every model, with a message that reads as though the
    factor were bad."""
    m = regularized()
    # matched on the validation wording, not on "max_pdpert": the
    # refusal message carries that too, so a cap of 0.0 or -1.0 raises
    # either way and only the message says which check fired
    with pytest.raises(ValueError,
                       match="max_pdpert must be a positive number"):
        estimate(m, [(m.p, 0.5)], max_pdpert=bad)
