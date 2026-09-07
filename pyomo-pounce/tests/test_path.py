"""Tests for sens_solution(mode="path"): apply the perturbation a little at
a time, changing the active set at the fraction where each change
happens, and for sens_active_set_changes(), the record of those changes."""
import warnings

import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import (
    sens_active_set_changes,
    declare_sens_param,
    sens_solution,
    sens_solution_report,
)


def linked(p=1.0):
    """x tracks p against a lower bound of 0, and y is tied to x by an
    equality. The same model the fix_relax tests use."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(-50.0, 50.0), initialize=1.0)
    m.link = pyo.Constraint(expr=m.y == 2 * m.x + 1)
    m.obj = pyo.Objective(
        expr=(m.x - m.p) ** 2 + 0.5 * (m.y - 3 * m.p) ** 2)
    declare_sens_param(m.p)
    return m


def solved(p=1.0):
    m = linked(p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def resolve_at(newval):
    m = linked(newval)
    pyo.SolverFactory("pounce").solve(m)
    return pyo.value(m.x), pyo.value(m.y)


def test_path_matches_a_resolve_across_a_crossing():
    m = solved()
    exact_x, exact_y = resolve_at(-2.0)
    est = sens_solution(m, [(m.p, -2.0)], mode="path")
    assert est[m.x] == pytest.approx(exact_x, abs=1e-6)
    assert est[m.y] == pytest.approx(exact_y, abs=1e-6)
    assert est[m.y] == pytest.approx(2 * est[m.x] + 1, abs=1e-6)


def test_path_agrees_with_fix_relax_where_they_settle_the_same_set():
    m = solved()
    fix = sens_solution(m, [(m.p, -2.0)], mode="fix_relax")
    path = sens_solution(m, [(m.p, -2.0)], mode="path")
    for v in (m.x, m.y):
        assert path[v] == pytest.approx(fix[v], abs=1e-9)


def test_the_modes_agree_when_nothing_crosses():
    m = solved()
    lin = sens_solution(m, [(m.p, 1.1)])
    path = sens_solution(m, [(m.p, 1.1)], mode="path")
    for v in (m.x, m.y):
        assert path[v] == pytest.approx(lin[v], rel=1e-12)
    assert sens_active_set_changes(m, [(m.p, 1.1)]) == []


def test_the_record_names_the_crossing():
    m = solved()
    rec = sens_active_set_changes(m, [(m.p, -2.0)])
    assert len(rec) == 1
    c = rec[0]
    assert c.var is m.x
    assert c.bound == "lower"
    assert c.action == "reaches"
    assert 0.0 < c.fraction < 1.0


def test_the_first_fraction_matches_the_report_alpha():
    """The report's ratio test and the path's first breakpoint answer
    the same question with the same base direction, so they must give
    the same fraction."""
    m = solved()
    r = sens_solution_report(m, [(m.p, -2.0)])
    rec = sens_active_set_changes(m, [(m.p, -2.0)])
    assert rec[0].fraction == pytest.approx(r.alpha, rel=1e-9)


def releasing(p=-1.0):
    """x sits on its lower bound at the base point. The same model the
    fix_relax release tests use."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=0.5)
    m.y = pyo.Var(bounds=(-50.0, 50.0), initialize=1.0)
    m.link = pyo.Constraint(expr=m.y == 2 * m.x + 1)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + 0.5 * (m.y - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def test_path_releases_a_bound_the_perturbation_pulls_off():
    m = releasing()
    assert pyo.value(m.x) == pytest.approx(0.0, abs=1e-6), "bound is active"

    path = sens_solution(m, [(m.p, 3.0)], mode="path")
    assert path[m.x] == pytest.approx(1.666667, abs=1e-5)
    assert path[m.y] == pytest.approx(2 * path[m.x] + 1, abs=1e-6)

    rec = sens_active_set_changes(m, [(m.p, 3.0)])
    assert len(rec) == 1
    c = rec[0]
    assert c.var is m.x
    assert c.bound == "lower"
    assert c.action == "leaves"
    assert 0.0 < c.fraction < 1.0


def test_a_bound_pushed_deeper_stays_and_records_nothing():
    """The perturbation pushes x further into its active lower bound.
    The factorization already enforces that bound, so the path must
    not hold it again through a Schur row: the multiplier grows,
    nothing changes hands, and the record is empty. Holding it a
    second time would put a wrong entry in the record at fraction
    zero and enforce the same bound twice."""
    m = releasing()
    assert pyo.value(m.x) == pytest.approx(0.0, abs=1e-6), "bound is active"
    path = sens_solution(m, [(m.p, -3.0)], mode="path")
    assert path[m.x] == pytest.approx(0.0, abs=1e-6)
    assert path[m.y] == pytest.approx(1.0, abs=1e-6)
    assert sens_active_set_changes(m, [(m.p, -3.0)]) == []


def test_a_release_on_an_upper_bound():
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=5.0, mutable=True)
    m.x = pyo.Var(bounds=(-10.0, 1.0), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    assert pyo.value(m.x) == pytest.approx(1.0, abs=1e-6), "on the upper bound"

    path = sens_solution(m, [(m.p, -3.0)], mode="path")
    assert path[m.x] == pytest.approx(-3.0, abs=1e-5)
    rec = sens_active_set_changes(m, [(m.p, -3.0)])
    assert [(c.var, c.bound, c.action) for c in rec] == [
        (m.x, "upper", "leaves")]


def crossing_qp(p=0.0):
    """A parametric QP whose solution path puts x2 on its lower bound
    partway through the change and takes it off again before the
    target, while x1 arrives at its own lower bound and stays. Found by
    a random scan over QPs of this family and verified against
    re-solves along the path."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x1 = pyo.Var(bounds=(0.0, 1.0), initialize=0.3)
    m.x2 = pyo.Var(bounds=(0.0, 10.0), initialize=0.3)
    g, a0, a1, b0, b1 = -0.22, 0.27, -1.45, -0.05, 0.15
    m.obj = pyo.Objective(
        expr=0.5 * m.x1**2 + 0.5 * m.x2**2 + g * m.x1 * m.x2
        - (a0 + a1 * m.p) * m.x1 - (b0 + b1 * m.p) * m.x2)
    declare_sens_param(m.p)
    return m


def test_a_variable_reached_partway_can_leave_again():
    """The record can contain a bound the path itself added and later
    dropped. A single decision at the base point cannot represent that,
    and the endpoint is exact here because a QP's solution path is
    piecewise linear in the parameter."""
    m = crossing_qp()
    pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-10})

    exact = crossing_qp(1.0)
    pyo.SolverFactory("pounce").solve(exact, options={"tol": 1e-10})

    path = sens_solution(m, [(m.p, 1.0)], mode="path")
    assert path[m.x1] == pytest.approx(pyo.value(exact.x1), abs=1e-8)
    assert path[m.x2] == pytest.approx(pyo.value(exact.x2), abs=1e-8)

    rec = sens_active_set_changes(m, [(m.p, 1.0)])
    x2_events = [(c.action, c.fraction) for c in rec
                 if c.var is m.x2 and c.bound == "lower"]
    assert [a for a, _ in x2_events] == ["reaches", "leaves"], (
        f"x2 should arrive and depart, record: {rec}")
    fracs = [c.fraction for c in rec]
    assert fracs == sorted(fracs), "the record is in path order"
    assert all(0.0 < f <= 1.0 for f in fracs)


def returning_qp(p=0.0):
    """A parametric QP whose solution path releases x1's upper bound
    partway through the change and returns x1 to that same bound
    before the target: x2 reaching its lower bound mid-path flips
    x1's direction. Found by a random scan over QPs of this family
    and verified against re-solves."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x1 = pyo.Var(bounds=(0.0, 1.0), initialize=0.3)
    m.x2 = pyo.Var(bounds=(0.0, 10.0), initialize=0.3)
    g, a0, a1, b0, b1 = -0.73, 0.48, 0.58, 0.4, -2.6
    m.obj = pyo.Objective(
        expr=0.5 * m.x1**2 + 0.5 * m.x2**2 + g * m.x1 * m.x2
        - (a0 + a1 * m.p) * m.x1 - (b0 + b1 * m.p) * m.x2)
    declare_sens_param(m.p)
    return m


def test_a_released_bound_can_be_reached_again():
    """Base activity is decided once at the base point, but a released
    row leaves the factorization at the fraction it releases, so the
    variable can come back to that same bound and be held through a
    Schur row. A reach scan that treated the factorization's bounds as
    fixed for the whole path would refuse this hold and miss the
    endpoint."""
    m = returning_qp()
    pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-10})

    exact = returning_qp(1.0)
    pyo.SolverFactory("pounce").solve(exact, options={"tol": 1e-10})

    path = sens_solution(m, [(m.p, 1.0)], mode="path")
    assert path[m.x1] == pytest.approx(pyo.value(exact.x1), abs=1e-8)
    assert path[m.x2] == pytest.approx(pyo.value(exact.x2), abs=1e-8)

    rec = sens_active_set_changes(m, [(m.p, 1.0)])
    x1_upper = [c.action for c in rec
                if c.var is m.x1 and c.bound == "upper"]
    assert x1_upper == ["leaves", "reaches"], (
        f"x1 should leave its upper bound and return to it, record: {rec}")
    fracs = [c.fraction for c in rec]
    assert fracs == sorted(fracs)


def test_the_cap_falls_back_to_the_clamp_with_a_warning():
    m = solved()
    with pytest.warns(UserWarning, match="predictor_iter may finish it"):
        capped = sens_solution(m, [(m.p, -2.0)], mode="path", predictor_iter=0,
                          clamp=False)
    # with no changes allowed the whole perturbation is one plain step
    lin = sens_solution(m, [(m.p, -2.0)], clamp=False)
    assert capped[m.x] == pytest.approx(lin[m.x], rel=1e-9)


def test_path_is_quiet_when_it_reaches_the_target():
    m = solved()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sens_solution(m, [(m.p, -2.0)], mode="path")
        sens_active_set_changes(m, [(m.p, -2.0)])


def test_the_record_requires_a_session():
    m = linked()
    with pytest.raises(RuntimeError, match="no sensitivity session"):
        sens_active_set_changes(m, [(m.p, 2.0)])


# ---------------------------------------------------------------------
# gh#928: the same limit written as a constraint row.
#
# `x <= 1` as a bound and `cap: x <= 1` as a row describe the same
# feasible set, but the second bounds the row's SLACK rather than any
# variable, so its multiplier is in `v_u` and its primal KKT row is in
# the `s` block. Before gh#928 the row form was in no bound row and in
# no box: the walk could not reach it, could not release it, and
# recorded nothing when it walked past. These tests are the Python end
# of that, and they are also what pins the record's own widening -- the
# entry names a Constraint, not a Var, and resolving the row against
# the variable map would return a neighbouring variable (gh#450).
# ---------------------------------------------------------------------


def capped(p=0.8):
    """`x` tracks `p` with the limit stated as a ROW, not a bound.

    Strictly convex in `x` and unbounded as a variable, so the limit is
    reachable only through `cap` and the answer is `min(p, 1)`.
    """
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(initialize=0.5)
    m.cap = pyo.Constraint(expr=m.x <= 1.0)
    m.obj = pyo.Objective(expr=0.5 * (m.x - m.p) ** 2)
    declare_sens_param(m.p)
    return m


def capped_solved(p=0.8):
    m = capped(p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def test_a_row_limit_the_step_reaches_is_named_by_its_constraint():
    """Pre-fix: x walked to 1.3 against a truth of 1.0, silently."""
    m = capped_solved(0.8)
    est = sens_solution(m, [(m.p, 1.3)], mode="path")
    assert est[m.x] == pytest.approx(1.0, abs=1e-6)
    rec = sens_active_set_changes(m, [(m.p, 1.3)])
    assert len(rec) == 1
    c = rec[0]
    # The record must name the CONSTRAINT. Resolving an `s`-block row
    # against the variable map is the gh#450 neighbouring-entry bug,
    # and here it would name `x` -- a plausible, wrong answer.
    assert c.var is m.cap
    assert c.bound == "upper"
    assert c.action == "reaches"
    assert c.fraction == pytest.approx(0.4, abs=1e-6)


def test_a_row_limit_the_step_leaves_is_released():
    """Pre-fix: x stayed pinned at 1.0 against a truth of 0.8, because
    nothing took the cap's stiffness out of the operator."""
    m = capped_solved(1.3)
    est = sens_solution(m, [(m.p, 0.8)], mode="path")
    assert est[m.x] == pytest.approx(0.8, abs=1e-6)
    rec = sens_active_set_changes(m, [(m.p, 0.8)])
    assert len(rec) == 1
    c = rec[0]
    assert c.var is m.cap
    assert c.bound == "upper"
    assert c.action == "leaves"
    assert c.fraction == pytest.approx(0.6, abs=1e-6)


def test_a_step_that_does_not_touch_the_row_limit_records_nothing():
    """The control: exact before the fix too, so a test that only
    checked answers would pass here while both cases above were broken."""
    m = capped_solved(0.8)
    est = sens_solution(m, [(m.p, 0.6)], mode="path")
    assert est[m.x] == pytest.approx(0.6, abs=1e-6)
    assert sens_active_set_changes(m, [(m.p, 0.6)]) == []


def test_slack_rows_is_the_discriminator_for_a_row_limit():
    """The accessor that tells a consumer which block a segment's row
    is in, checked against the segment the walk actually emits."""
    m = capped_solved(0.8)
    sess = m._pounce_sens.session
    n_x = sess.solver.block_dims[0]
    slack = sess.solver.slack_rows(list(range(len(sess.con_names))))
    rows = [r for r in slack if r is not None]
    assert rows, "the model has an inequality, so some row has a slack"
    assert all(r >= n_x for r in rows), (
        f"a slack row must be in the `s` block, past n_x={n_x}: {rows}")
