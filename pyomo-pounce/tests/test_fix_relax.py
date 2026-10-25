"""Tests for sens_solution(mode="fix_relax"): pin a crossing variable at its
bound and re-solve, rather than clamping it and freezing the rest."""
import warnings

import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import declare_sens_param, sens_solution, sens_solution_report


def linked(p=1.0):
    """x tracks p against a lower bound of 0, and y is tied to x by an
    equality. Clamping x leaves y at its predictor value, which breaks
    the equality outright, so the two modes are easy to tell apart."""
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
    """The exact answer: solve again at the perturbed parameter."""
    m = linked(newval)
    pyo.SolverFactory("pounce").solve(m)
    return pyo.value(m.x), pyo.value(m.y)


def test_fix_relax_matches_a_resolve_where_the_clamp_does_not():
    m = solved()
    exact_x, exact_y = resolve_at(-2.0)

    with pytest.warns(UserWarning, match="leaves the variable bounds"):
        lin = sens_solution(m, [(m.p, -2.0)])
    fix = sens_solution(m, [(m.p, -2.0)], mode="fix_relax")

    # both put x on its bound
    assert lin[m.x] == pytest.approx(exact_x, abs=1e-6)
    assert fix[m.x] == pytest.approx(exact_x, abs=1e-6)

    # the clamp leaves y at its predictor value, breaking y = 2x + 1
    assert abs(lin[m.y] - exact_y) > 1.0
    assert abs(lin[m.y] - (2 * lin[m.x] + 1)) > 1.0

    # the pinned re-solve lands on the re-solve and keeps the equality
    assert fix[m.y] == pytest.approx(exact_y, abs=1e-6)
    assert fix[m.y] == pytest.approx(2 * fix[m.x] + 1, abs=1e-6)


def test_the_modes_agree_when_nothing_crosses():
    m = solved()
    lin = sens_solution(m, [(m.p, 1.1)])
    fix = sens_solution(m, [(m.p, 1.1)], mode="fix_relax")
    assert sens_solution_report(m, [(m.p, 1.1)]).alpha > 1.0
    for v in (m.x, m.y):
        assert fix[v] == pytest.approx(lin[v], rel=1e-12)


def test_fix_relax_does_not_warn_about_clamping():
    """The pin IS the correction, so there is nothing clipped and no
    active-set change left to warn about."""
    m = solved()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sens_solution(m, [(m.p, -2.0)], mode="fix_relax")


def test_the_report_and_fix_relax_agree_on_what_crosses():
    m = solved()
    r = sens_solution_report(m, [(m.p, -2.0)])
    assert r.first == "x"
    assert 0.0 < r.alpha < 1.0
    assert m.x in r.crossed

    # the fraction the report says fits should cross nothing, and there
    # the two modes agree
    fits = 1.0 + r.alpha * (-2.0 - 1.0)
    lin = sens_solution(m, [(m.p, fits)])
    fix = sens_solution(m, [(m.p, fits)], mode="fix_relax")
    for v in (m.x, m.y):
        assert fix[v] == pytest.approx(lin[v], abs=1e-6)


def test_an_unknown_mode_is_refused():
    m = solved()
    with pytest.raises(ValueError, match="mode must be"):
        sens_solution(m, [(m.p, 2.0)], mode="newton")


def test_fix_relax_reaches_the_same_answer_through_every_route():
    from conftest import solver_routes

    # The `pounce_v2` registration needs Pyomo 6.10.1+; where it is absent the
    # remaining routes are still compared rather than the whole test skipped.
    routes = solver_routes()
    assert len(routes) >= 2, f"nothing to compare: {[n for n, _ in routes]}"

    out = []
    for _name, solve in routes:
        m = linked()
        solve(m)
        out.append(sens_solution(m, [(m.p, -2.0)], mode="fix_relax"))
    (first, *rest) = out
    for name, other in zip([n for n, _ in routes][1:], rest):
        vals_a = sorted(round(v, 9) for v in first.values())
        vals_b = sorted(round(v, 9) for v in other.values())
        assert vals_a == vals_b, name


def scaled_linked(p=1.0):
    m = linked(p)
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.x] = 1000.0
    m.scaling_factor[m.y] = 0.001
    return m


def test_fix_relax_holds_under_user_scaling():
    """The bounds the solve carries bound the algorithm's change of
    variables, while the iterate and the step are in the model's own
    units. Comparing them directly projects onto the wrong box: before
    this was handled, the scaled model returned x = -0.525 where the
    answer is 0."""
    m = scaled_linked()
    pyo.SolverFactory("pounce").solve(
        m, options={"nlp_scaling_method": "user-scaling"})
    exact_x, exact_y = resolve_at(-2.0)

    fix = sens_solution(m, [(m.p, -2.0)], mode="fix_relax")
    assert fix[m.x] == pytest.approx(exact_x, abs=1e-6)
    assert fix[m.y] == pytest.approx(exact_y, abs=1e-6)
    assert fix[m.y] == pytest.approx(2 * fix[m.x] + 1, abs=1e-6)


def test_scaling_does_not_change_the_answer():
    """The same model with and without a change of variables must give
    the same estimate."""
    plain = solved()
    m = scaled_linked()
    pyo.SolverFactory("pounce").solve(
        m, options={"nlp_scaling_method": "user-scaling"})

    a = sens_solution(plain, [(plain.p, -2.0)], mode="fix_relax")
    b = sens_solution(m, [(m.p, -2.0)], mode="fix_relax")
    assert b[m.x] == pytest.approx(a[plain.x], abs=1e-6)
    assert b[m.y] == pytest.approx(a[plain.y], abs=1e-6)


def two_bounded():
    """y = 2x + 1 with both bounded below by 0, so there is one degree
    of freedom and two bounds that a large downward move wants."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(0.0, 10.0), initialize=3.0)
    m.link = pyo.Constraint(expr=m.y == 2 * m.x + 1)
    m.obj = pyo.Objective(
        expr=(m.x - m.p) ** 2 + 0.5 * (m.y - 3 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def test_fix_relax_says_which_limit_stopped_it():
    """Pinning y at 0 forces x to -0.5 through the equality, and the
    single degree of freedom is already spent, so no step holds both.
    The warning names that limit rather than the pass budget."""
    m = two_bounded()
    with pytest.warns(UserWarning, match="degrees of freedom"):
        sens_solution(m, [(m.p, -5.0)], mode="fix_relax")


def test_clamp_decides_what_happens_to_what_is_left_outside():
    """`clamp` means the same thing in both modes: clip whatever is
    still outside a bound. Under fix_relax that is only what the pins
    could not hold, and clipping it costs the constraint the pins were
    solved against, which is why the caller can turn it off."""
    m = two_bounded()
    with pytest.warns(UserWarning, match="clamped"):
        clipped = sens_solution(m, [(m.p, -5.0)], mode="fix_relax", clamp=True)
    with pytest.warns(UserWarning, match="unclamped"):
        raw = sens_solution(m, [(m.p, -5.0)], mode="fix_relax", clamp=False)

    # clamped: inside the bounds, and the equality no longer holds
    assert clipped[m.x] == pytest.approx(0.0, abs=1e-9)
    assert abs(clipped[m.y] - (2 * clipped[m.x] + 1)) > 0.5

    # unclamped: outside a bound, and the equality still holds
    assert raw[m.x] < -1e-3
    assert raw[m.y] == pytest.approx(2 * raw[m.x] + 1, abs=1e-6)


def test_predictor_iter_is_exposed_and_bounds_the_work():
    """The cap is a budget, so a caller has to be able to set it."""
    m = solved()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        full = sens_solution(m, [(m.p, -2.0)], mode="fix_relax", predictor_iter=16)
    # one pass is enough for this model, so capping at 1 changes nothing
    one = sens_solution(m, [(m.p, -2.0)], mode="fix_relax", predictor_iter=1)
    assert one[m.x] == pytest.approx(full[m.x], rel=1e-12)

    # and a cap of zero pins nothing, leaving the plain step
    with pytest.warns(UserWarning):
        none = sens_solution(m, [(m.p, -2.0)], mode="fix_relax", predictor_iter=0,
                        clamp=False)
    assert none[m.x] < -1e-3


def test_fix_relax_is_quiet_when_it_holds_every_bound():
    m = solved()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sens_solution(m, [(m.p, -2.0)], mode="fix_relax")


def test_an_absent_bound_does_not_widen_the_tolerance():
    """A variable with no upper bound reaches the check as the reader's
    +-1e19 sentinel. Scaling the tolerance by the bound rather than by
    the coordinate puts it at 1e10, and nothing is ever reported as
    outside its bounds."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, None), initialize=1.0)   # no upper bound
    m.y = pyo.Var(bounds=(-50.0, 50.0), initialize=1.0)
    m.link = pyo.Constraint(expr=m.y == 2 * m.x + 1)
    m.obj = pyo.Objective(
        expr=(m.x - m.p) ** 2 + 0.5 * (m.y - 3 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)

    # the step drives x below 0, which must still be caught
    with pytest.warns(UserWarning, match="leaves the variable bounds"):
        sens_solution(m, [(m.p, -2.0)])
    fix = sens_solution(m, [(m.p, -2.0)], mode="fix_relax")
    assert fix[m.x] == pytest.approx(0.0, abs=1e-6)
    assert fix[m.y] == pytest.approx(2 * fix[m.x] + 1, abs=1e-6)


def releasing(p=-1.0):
    """x wants to be negative, so its lower bound is active at the base
    point and carries a positive multiplier. A large enough upward
    perturbation should release it."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=0.5)
    m.y = pyo.Var(bounds=(-50.0, 50.0), initialize=1.0)
    m.link = pyo.Constraint(expr=m.y == 2 * m.x + 1)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2 + 0.5 * (m.y - m.p) ** 2)
    declare_sens_param(m.p)
    return m


def active_at_base(p=-1.0):
    m = releasing(p)
    pyo.SolverFactory("pounce").solve(m)
    return m


def test_fix_relax_releases_a_bound_the_step_wants_to_leave():
    """The second half of upstream's fix-relax, its equation 18. A bound
    active at the base point holds the variable there under the plain
    step, because the step preserves complementarity. Releasing it lets
    the variable move.

    Verified against sIPOPT 3.14.19 driven through
    pyomo.contrib.sensitivity_toolbox on this model: with
    sens_boundcheck=no it reports x = 0, with sens_boundcheck=yes
    x = 1.666666, and a full re-solve gives 1.666667.
    """
    m = active_at_base()
    assert pyo.value(m.x) == pytest.approx(0.0, abs=1e-6), "bound is active"

    lin = sens_solution(m, [(m.p, 3.0)])
    fix = sens_solution(m, [(m.p, 3.0)], mode="fix_relax")

    # the plain step cannot leave the bound
    assert lin[m.x] == pytest.approx(0.0, abs=1e-6)
    # releasing it reaches the re-solve
    assert fix[m.x] == pytest.approx(1.666667, abs=1e-5)
    assert fix[m.y] == pytest.approx(4.333333, abs=1e-5)
    assert fix[m.y] == pytest.approx(2 * fix[m.x] + 1, abs=1e-6)


def test_a_release_holds_at_a_tight_tolerance():
    """A release is accurate in proportion to how well the solve
    converged, not inversely.

    The released system is not recoverable from the converged factor: an
    active bound puts `sigma = z/s` on the x diagonal, and the tighter
    the solve the larger that term, so a release computed from the held
    factorization degrades as the slack shrinks. It used to obey
    err * s ~ 1e-16 -- at this tolerance the answer was off by 2.2e-4,
    while at a *looser* 1e-6 it was off by 7e-9. Re-factoring without
    the bound's sigma is what removes the dependence.

    The assertion is four orders tighter than the old error, and the
    loose-tolerance tests cannot stand in for it: they pass either way.
    """
    m = releasing()
    pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-10})
    assert pyo.value(m.x) == pytest.approx(0.0, abs=1e-6), "bound is active"

    fix = sens_solution(m, [(m.p, 3.0)], mode="fix_relax")
    assert fix[m.x] == pytest.approx(5 / 3, abs=1e-8)
    assert fix[m.y] == pytest.approx(2 * fix[m.x] + 1, abs=1e-8)


def test_a_release_is_not_triggered_when_the_bound_should_stay():
    """A perturbation that pushes further INTO the bound must not
    release it: the multiplier grows rather than going negative."""
    m = active_at_base()
    fix = sens_solution(m, [(m.p, -3.0)], mode="fix_relax")
    assert fix[m.x] == pytest.approx(0.0, abs=1e-6)
    assert fix[m.y] == pytest.approx(1.0, abs=1e-6)


def scaled_releasing(p=-1.0):
    """The release model under a change of variables, with the two
    factors far apart and on both sides of 1."""
    m = releasing(p)
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.x] = 1000.0
    m.scaling_factor[m.y] = 0.001
    return m


def test_a_release_holds_under_user_scaling():
    """The release right-hand side is a bound multiplier read off the
    converged iterate, so it is in the solve's own coordinates, while
    the step it joins is in the model's units. Handing over an
    unconverted multiplier puts a scaled right-hand side into a Schur
    complement whose other rows are natural, which moves every
    coordinate of the answer rather than the released one.

    The other release tests cannot catch this: with no scaling active
    the conversion is the identity, so they pass whichever way round it
    goes -- which is how the defect survived its first review. Here
    `d` is 1000 on the released variable, so dropping the conversion
    or inverting it misses by orders of magnitude rather than by a
    tolerance.
    """
    m = scaled_releasing()
    pyo.SolverFactory("pounce").solve(
        m, options={"nlp_scaling_method": "user-scaling"})
    assert pyo.value(m.x) == pytest.approx(0.0, abs=1e-6), "bound is active"

    # the plain step still cannot leave the bound, scaled or not
    lin = sens_solution(m, [(m.p, 3.0)])
    assert lin[m.x] == pytest.approx(0.0, abs=1e-6)

    # and releasing it reaches the same re-solve the unscaled model does
    fix = sens_solution(m, [(m.p, 3.0)], mode="fix_relax")
    assert fix[m.x] == pytest.approx(1.666667, abs=1e-5)
    assert fix[m.y] == pytest.approx(4.333333, abs=1e-5)
    assert fix[m.y] == pytest.approx(2 * fix[m.x] + 1, abs=1e-6)


def nonlinear(p=1.0):
    """Nonlinear in both the objective and a constraint, so the barrier
    term actually contributes and mu shows up in the step."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=p, mutable=True)
    m.x = pyo.Var(bounds=(0.1, 10.0), initialize=1.0)
    m.y = pyo.Var(bounds=(0.1, 50.0), initialize=2.0)
    m.c = pyo.Constraint(expr=m.y * m.x >= 0.5)
    m.obj = pyo.Objective(
        expr=(m.x - m.p) ** 2 + 0.5 * (m.y - 2 * m.p) ** 2 + 0.1 / m.x)
    declare_sens_param(m.p)
    return m


def test_the_step_carries_the_barrier_correction():
    """The step is taken against a factorization held at the final mu,
    so it estimates where the BARRIER problem's solution moves, not the
    original problem's. The paper's equation 11 closes that gap.

    The expected value is sIPOPT 3.14.19's own answer for this model and
    perturbation, read from its `sens_sol_state_1` suffix through
    pyomo.contrib.sensitivity_toolbox at the same solver tolerance. A
    loose tolerance is what makes this test able to fail: without the
    correction pounce differs from sIPOPT by 8.7e-6 here.

    The band is not tighter because pounce and sIPOPT no longer factor
    the same matrix. sIPOPT factors with the Hessian left from the final
    Newton step, W(w_k); pounce re-evaluates it at the converged point,
    so its K is the KKT Jacobian at w* (crates/pounce-sensitivity/tests/
    converged_hessian_at_w_star.rs). At tol=1e-3 that last step is
    large. Measured against sIPOPT:

        pounce, K(w*), with the correction     -5.2e-7
        pounce, W(w_k), with the correction    -2.4e-7  (the old pin, 5e-7)
        pounce, K(w*), correction removed      +8.7e-6

    2e-6 passes the first with 3.8x margin and fails the third by 4.3x.
    """
    m = nonlinear()
    pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-3})
    got = sens_solution(m, [(m.p, 1.0 + 1e-3)])[m.x]
    assert got == pytest.approx(1.046643388715, abs=2e-6)


def test_the_correction_is_below_notice_at_a_converged_tolerance():
    """The same comparison at a tight tolerance, where the term is
    O(mu) and must not disturb an answer that was already right. This
    one cannot fail by removing the correction, and is here to pin that
    it stays small rather than to detect its absence."""
    m = nonlinear()
    pyo.SolverFactory("pounce").solve(m, options={"tol": 1e-8})
    got = sens_solution(m, [(m.p, 1.0 + 1e-3)])[m.x]
    assert got == pytest.approx(1.046642776555, abs=1e-8)


def test_the_correction_does_not_move_a_converged_answer():
    """At a tight tolerance the term is O(mu) and must not disturb a
    step that was already right."""
    m = solved()
    fix = sens_solution(m, [(m.p, -2.0)], mode="fix_relax")
    assert fix[m.x] == pytest.approx(0.0, abs=1e-6)
    assert fix[m.y] == pytest.approx(1.0, abs=1e-6)


def test_a_release_and_a_pin_in_the_same_step():
    """One bound must release while another activates, so both halves
    run against the same factorization. x starts on its lower bound and
    is pulled off, while z is driven onto its upper bound."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=-1.0, mutable=True)
    m.x = pyo.Var(bounds=(0.0, 10.0), initialize=0.5)
    m.z = pyo.Var(bounds=(-10.0, 1.0), initialize=0.0)
    m.obj = pyo.Objective(
        expr=(m.x - m.p) ** 2 + (m.z - 2 * m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    assert pyo.value(m.x) == pytest.approx(0.0, abs=1e-6), "x on its bound"
    assert pyo.value(m.z) == pytest.approx(-2.0, abs=1e-6), "z interior"

    # p: -1 -> 2 pulls x up off its bound and drives z past ub = 1
    exact = pyo.ConcreteModel()
    exact.p = pyo.Param(initialize=2.0, mutable=True)
    exact.x = pyo.Var(bounds=(0.0, 10.0), initialize=0.5)
    exact.z = pyo.Var(bounds=(-10.0, 1.0), initialize=0.0)
    exact.obj = pyo.Objective(
        expr=(exact.x - exact.p) ** 2 + (exact.z - 2 * exact.p) ** 2)
    pyo.SolverFactory("pounce").solve(exact)

    fix = sens_solution(m, [(m.p, 2.0)], mode="fix_relax")
    assert fix[m.x] == pytest.approx(pyo.value(exact.x), abs=1e-5)
    assert fix[m.z] == pytest.approx(pyo.value(exact.z), abs=1e-5)
    # and the released bound really moved: the plain step cannot
    lin = sens_solution(m, [(m.p, 2.0)], clamp=False)
    assert abs(lin[m.x] - fix[m.x]) > 1.0


def test_a_release_on_an_upper_bound():
    """The release half must work on either side. Here the variable
    sits on its UPPER bound and the perturbation pulls it down."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(initialize=5.0, mutable=True)
    m.x = pyo.Var(bounds=(-10.0, 1.0), initialize=0.5)
    m.obj = pyo.Objective(expr=(m.x - m.p) ** 2)
    declare_sens_param(m.p)
    pyo.SolverFactory("pounce").solve(m)
    assert pyo.value(m.x) == pytest.approx(1.0, abs=1e-6), "on the upper bound"

    fix = sens_solution(m, [(m.p, -3.0)], mode="fix_relax")
    lin = sens_solution(m, [(m.p, -3.0)], clamp=False)
    # the true answer is x = -3, interior
    assert fix[m.x] == pytest.approx(-3.0, abs=1e-5)
    assert lin[m.x] == pytest.approx(1.0, abs=1e-5), "the plain step is stuck"
