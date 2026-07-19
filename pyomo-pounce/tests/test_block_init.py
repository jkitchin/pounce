"""Tests for pyomo_pounce.block_initialize (experimental).

The 1x1 Newton path is hermetic; the multi-variable subsystem path
needs the pounce binary and is skipped when it is absent.
"""

import pyomo.environ as pyo
import pytest

import pyomo_pounce

# pyomo defers optional imports, so pyomo.contrib.incidence_analysis
# imports fine without networkx and only fails at first use — skip on
# the real dependencies.
pytest.importorskip("networkx", reason="block_initialize needs networkx")
pytest.importorskip("scipy", reason="block_initialize needs scipy")
pytest.importorskip(
    "pyomo.contrib.incidence_analysis",
    reason="block_initialize needs pyomo.contrib.incidence_analysis",
)


@pytest.fixture(scope="module")
def solver():
    s = pyo.SolverFactory("pounce")
    if not s.available(exception_flag=False):
        pytest.skip("pounce binary not found on PATH")
    return s


def test_triangular_system_solved_by_newton_only():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()
    m.c1 = pyo.Constraint(expr=m.x == 2.0)
    m.c2 = pyo.Constraint(expr=m.y == m.x + 1.0)
    m.c3 = pyo.Constraint(expr=m.z * m.y == 6.0)
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.block_initialize(m)
    assert report.ok, str(report)
    assert report.square
    assert report.n_blocks == 3
    assert report.n_1x1 == 3
    assert report.n_subsystem_solves == 0  # no solver needed
    assert m.x.value == pytest.approx(2.0)
    assert m.y.value == pytest.approx(3.0)
    assert m.z.value == pytest.approx(2.0)


def test_degrees_of_freedom_left_untouched_and_named():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.w = pyo.Var()  # free DOF: appears only in the objective
    m.c = pyo.Constraint(expr=m.x == 4.0)
    m.obj = pyo.Objective(expr=(m.w - 1.0) ** 2 + m.x)

    report = pyomo_pounce.block_initialize(m)
    assert report.ok, str(report)
    assert m.x.value == pytest.approx(4.0)
    assert m.w.value is None
    # w participates in no equality, so the equality system is square
    # over {x}; the DM partition never sees w. Nothing to name.
    assert report.square
    # ... and initialize_missing_values fills the rest.
    pyomo_pounce.initialize_missing_values(m)
    assert m.w.value == 0.0


def test_underconstrained_names_reported():
    # Two variables coupled by one equation: without a decision the
    # system is underdetermined, and the names say which variables.
    m = pyo.ConcreteModel()
    m.feed = pyo.Var()
    m.split = pyo.Var(bounds=(0.0, 1.0))
    m.out = pyo.Var()
    m.c = pyo.Constraint(expr=m.out == m.split * m.feed)
    m.obj = pyo.Objective(expr=m.out)

    report = pyomo_pounce.block_initialize(m)
    assert not report.square
    assert report.skipped_underdetermined >= 2
    named = set(report.underconstrained_variables)
    assert {"feed", "split"} & named, str(report)
    assert "underconstrained" in str(report)


def test_decisions_square_the_system_and_are_released():
    m = pyo.ConcreteModel()
    m.feed = pyo.Var(initialize=10.0)
    m.split = pyo.Var(bounds=(0.0, 1.0), initialize=0.3)
    m.out1 = pyo.Var()
    m.out2 = pyo.Var()
    m.c1 = pyo.Constraint(expr=m.out1 == m.split * m.feed)
    m.c2 = pyo.Constraint(expr=m.out2 == (1.0 - m.split) * m.feed)
    m.obj = pyo.Objective(expr=m.out1)

    report = pyomo_pounce.block_initialize(m, decisions=[m.feed, m.split])
    assert report.ok, str(report)
    assert report.square
    assert report.n_decisions_fixed == 2
    assert m.out1.value == pytest.approx(3.0)
    assert m.out2.value == pytest.approx(7.0)
    # Decisions are released afterwards (the optimizer moves them next).
    assert not m.feed.fixed and not m.split.fixed
    assert m.feed.value == pytest.approx(10.0)


def test_decisions_accept_indexed_containers():
    m = pyo.ConcreteModel()
    m.d = pyo.Var([1, 2], initialize={1: 2.0, 2: 3.0})
    m.y = pyo.Var()
    m.c = pyo.Constraint(expr=m.y == m.d[1] + m.d[2])
    m.obj = pyo.Objective(expr=m.y)

    report = pyomo_pounce.block_initialize(m, decisions=[m.d])
    assert report.ok and report.square, str(report)
    assert report.n_decisions_fixed == 2
    assert m.y.value == pytest.approx(5.0)
    assert not m.d[1].fixed and not m.d[2].fixed


def test_decision_without_value_raises():
    m = pyo.ConcreteModel()
    m.feed = pyo.Var()  # no value: cannot be held at anything
    m.out = pyo.Var()
    m.c = pyo.Constraint(expr=m.out == 2.0 * m.feed)
    m.obj = pyo.Objective(expr=m.out)

    with pytest.raises(ValueError, match="feed"):
        pyomo_pounce.block_initialize(m, decisions=[m.feed])
    assert not m.feed.fixed  # nothing leaked


def test_already_fixed_decision_stays_fixed():
    m = pyo.ConcreteModel()
    m.feed = pyo.Var()
    m.feed.fix(10.0)
    m.out = pyo.Var()
    m.c = pyo.Constraint(expr=m.out == 0.5 * m.feed)
    m.obj = pyo.Objective(expr=m.out)

    report = pyomo_pounce.block_initialize(m, decisions=[m.feed])
    assert report.ok, str(report)
    assert report.n_decisions_fixed == 0  # was already fixed, not by us
    assert m.out.value == pytest.approx(5.0)
    assert m.feed.fixed  # user's fix survives


def test_overconstrained_names_reported():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.c1 = pyo.Constraint(expr=m.x == 1.0)
    m.c2 = pyo.Constraint(expr=2.0 * m.x == 2.0)  # redundant spec
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.block_initialize(m)
    assert not report.square
    assert report.skipped_overdetermined >= 1
    assert report.overconstrained_constraints, str(report)
    assert "overconstrained" in str(report)


def test_fixed_vars_act_as_inputs():
    m = pyo.ConcreteModel()
    m.feed = pyo.Var()
    m.feed.fix(10.0)
    m.out = pyo.Var()
    m.c = pyo.Constraint(expr=m.out == 0.5 * m.feed)
    m.obj = pyo.Objective(expr=m.out)

    report = pyomo_pounce.block_initialize(m)
    assert report.ok, str(report)
    assert m.out.value == pytest.approx(5.0)


def test_coupled_block_uses_subsystem_solve(solver):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()
    # 2x2 coupled block feeding a downstream 1x1.
    m.c1 = pyo.Constraint(expr=m.x + m.y == 3.0)
    m.c2 = pyo.Constraint(expr=m.x - m.y == 1.0)
    m.c3 = pyo.Constraint(expr=m.z == m.x * m.y)
    m.obj = pyo.Objective(expr=m.z)

    report = pyomo_pounce.block_initialize(m, solver=solver)
    assert report.ok, str(report)
    assert report.n_subsystem_solves == 1
    assert m.x.value == pytest.approx(2.0, abs=1e-6)
    assert m.y.value == pytest.approx(1.0, abs=1e-6)
    assert m.z.value == pytest.approx(2.0, abs=1e-6)


def test_block_analyze_touches_nothing():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.c1 = pyo.Constraint(expr=m.x == 2.0)
    m.c2 = pyo.Constraint(expr=m.y == m.x + 1.0)
    m.obj = pyo.Objective(expr=m.y)

    report = pyomo_pounce.block_analyze(m)
    assert report.square
    assert report.n_constraints == 2 and report.n_variables == 2
    assert report.n_blocks == 2 and report.n_1x1 == 2
    # analysis only: nothing seeded, nothing solved
    assert m.x.value is None and m.y.value is None


def test_block_analyze_returns_components_uncapped():
    # 15 loose variables in one equation: more than block_initialize's
    # display cap, and block_analyze must return every one, as objects.
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(15))
    m.c = pyo.Constraint(expr=sum(m.x[i] for i in range(15)) == 1.0)
    m.obj = pyo.Objective(expr=m.x[0])

    report = pyomo_pounce.block_analyze(m)
    assert not report.square
    assert len(report.underconstrained_variables) == 15
    assert report.underconstrained_constraints == [m.c]
    assert report.n_extra_degrees_of_freedom == 14
    assert any(v is m.x[3] for v in report.underconstrained_variables)
    # the display preview is capped, the data is not
    assert "and 5 more" in str(report)
    # contrast: block_initialize's name list is display-sized
    init_report = pyomo_pounce.block_initialize(m)
    assert len(init_report.underconstrained_variables) == 10


def test_block_analyze_overconstrained_part():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.c1 = pyo.Constraint(expr=m.x == 1.0)
    m.c2 = pyo.Constraint(expr=2.0 * m.x == 2.0)  # redundant spec
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.block_analyze(m)
    assert not report.square
    assert set(report.overconstrained_constraints) == {m.c1, m.c2}
    assert report.overconstrained_variables == [m.x]
    assert report.n_extra_specifications == 1
    assert "overconstrained" in str(report)


def test_block_analyze_decisions_need_no_values():
    # Purely structural: a valueless decision is fine here (it is a
    # ValueError in block_initialize), and its fixed flag is restored.
    m = pyo.ConcreteModel()
    m.feed = pyo.Var()  # no value
    m.split = pyo.Var(bounds=(0.0, 1.0))  # no value
    m.out1 = pyo.Var()
    m.out2 = pyo.Var()
    m.c1 = pyo.Constraint(expr=m.out1 == m.split * m.feed)
    m.c2 = pyo.Constraint(expr=m.out2 == (1.0 - m.split) * m.feed)
    m.obj = pyo.Objective(expr=m.out1)

    report = pyomo_pounce.block_analyze(m, decisions=[m.feed, m.split])
    assert report.square
    assert report.n_decisions_fixed == 2
    assert not m.feed.fixed and not m.split.fixed
    assert m.feed.value is None  # still untouched


def test_block_analyze_calculation_order():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.z = pyo.Var()
    # 2x2 coupled block feeding a downstream 1x1.
    m.c1 = pyo.Constraint(expr=m.x + m.y == 3.0)
    m.c2 = pyo.Constraint(expr=m.x - m.y == 1.0)
    m.c3 = pyo.Constraint(expr=m.z == m.x * m.y)
    m.obj = pyo.Objective(expr=m.z)

    report = pyomo_pounce.block_analyze(m)
    assert report.square
    assert report.n_blocks == 2 and report.n_1x1 == 1
    assert sorted(v.name for v in report.variable_blocks[0]) == ["x", "y"]
    assert report.variable_blocks[1] == [m.z]
    assert report.constraint_blocks[1] == [m.c3]


def test_block_analyze_unfixes_decisions_on_exception(monkeypatch):
    # The finally must release our fixes even when the analysis dies
    # before producing anything.
    import pyomo.contrib.incidence_analysis as ia

    def boom(*args, **kwargs):
        raise RuntimeError("igraph construction failed")

    monkeypatch.setattr(ia, "IncidenceGraphInterface", boom)
    m = pyo.ConcreteModel()
    m.u = pyo.Var()
    m.x = pyo.Var()
    m.c = pyo.Constraint(expr=m.x == m.u)
    m.obj = pyo.Objective(expr=m.x)

    with pytest.raises(RuntimeError, match="igraph construction failed"):
        pyomo_pounce.block_analyze(m, decisions=[m.u])
    assert not m.u.fixed


def test_block_analyze_no_equalities():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.c = pyo.Constraint(expr=m.x <= 1.0)  # inequalities do not participate
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.block_analyze(m)
    assert report.square
    assert report.n_constraints == 0 and report.n_variables == 0
    assert report.variable_blocks == []


def splitter_model():
    # F = D + B: three flows, one balance. Holding all three is one
    # specification too many.
    m = pyo.ConcreteModel()
    m.F = pyo.Var(initialize=10.0)
    m.D = pyo.Var(initialize=4.0)
    m.B = pyo.Var(initialize=7.0)  # inconsistent on purpose: 4 + 7 != 10
    m.bal = pyo.Constraint(expr=m.F == m.D + m.B)
    m.obj = pyo.Objective(expr=m.D)
    return m


def drum_model():
    # A loose integrator: M appears only as the denominator of a
    # 0 == f/M row, the shape substituting d/dt = 0 into a dynamic
    # balance produces. The equalities provably cannot determine it.
    m = pyo.ConcreteModel()
    m.u = pyo.Var(initialize=2.0)
    m.x = pyo.Var()
    m.M = pyo.Var(bounds=(1.0, 3.0))  # no value
    m.c1 = pyo.Constraint(expr=0 == (m.x - m.u) / m.M)
    m.obj = pyo.Objective(expr=m.x)
    return m


def test_repair_plan_no_op_on_square_system():
    m = pyo.ConcreteModel()
    m.feed = pyo.Var(initialize=10.0)
    m.split = pyo.Var(bounds=(0.0, 1.0), initialize=0.3)
    m.out1 = pyo.Var()
    m.out2 = pyo.Var()
    m.c1 = pyo.Constraint(expr=m.out1 == m.split * m.feed)
    m.c2 = pyo.Constraint(expr=m.out2 == (1.0 - m.split) * m.feed)
    m.obj = pyo.Objective(expr=m.out1)

    plan = pyomo_pounce.block_repair_plan(m, decision_candidates=[m.feed, m.split])
    assert plan.square
    assert plan.decisions == [m.feed, m.split]
    assert not plan.pruned and not plan.pinned


def test_repair_plan_prunes_a_conflicting_candidate():
    m = splitter_model()
    plan = pyomo_pounce.block_repair_plan(m, decision_candidates=[m.F, m.D, m.B])
    assert plan.square
    assert len(plan.decisions) == 2
    assert len(plan.pruned) == 1  # the provable minimum
    assert not plan.pinned and not plan.redundant_constraints
    assert "pruned" in str(plan)
    # a plan, not an action: nothing fixed, nothing moved
    assert not m.F.fixed and not m.D.fixed and not m.B.fixed
    assert (m.F.value, m.D.value, m.B.value) == (10.0, 4.0, 7.0)


def test_repair_plan_pins_automatically():
    # No user input names M: it is identified structurally, because its
    # only edge is the denominator of a 0 == f/M row.
    m = drum_model()
    plan = pyomo_pounce.block_repair_plan(m, decision_candidates=[m.u])
    assert plan.square
    assert plan.decisions == [m.u]
    assert plan.pinned == [m.M]
    assert not plan.pruned and not plan.loose_variables
    assert m.M.value is None  # structural: no values involved
    assert "pinned" in str(plan)


def test_repair_plan_denominator_rule_scope():
    # 0 == f/g cannot determine a variable appearing only in g, but
    # f/g == 5 can (it says f = 5g), so g must NOT be pinned there.
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.0)
    m.g = pyo.Var(bounds=(0.5, 2.0))
    m.c = pyo.Constraint(expr=m.x / m.g == 5.0)
    m.obj = pyo.Objective(expr=m.x)

    plan = pyomo_pounce.block_repair_plan(m, decision_candidates=[m.x])
    assert not plan.pinned  # g is genuinely determined: g = x / 5
    assert plan.square


def test_repair_plan_loose_variable_is_a_defect():
    # Genuine underdetermination: y * M == x could determine either y
    # or M, so the plan cannot know which to pin and says so.
    m = pyo.ConcreteModel()
    m.u = pyo.Var(initialize=2.0)
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.M = pyo.Var(bounds=(1.0, 3.0))
    m.c1 = pyo.Constraint(expr=m.x == m.u)
    m.c2 = pyo.Constraint(expr=m.y * m.M == m.x)
    m.obj = pyo.Objective(expr=m.y)

    plan = pyomo_pounce.block_repair_plan(m, decision_candidates=[m.u])
    assert not plan.square
    assert len(plan.loose_variables) == 1
    assert not plan.pinned
    assert "loose" in str(plan)


def test_repair_plan_redundancy_is_a_defect():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.c1 = pyo.Constraint(expr=m.x == 1.0)
    m.c2 = pyo.Constraint(expr=2.0 * m.x == 2.0)
    m.obj = pyo.Objective(expr=m.x)

    plan = pyomo_pounce.block_repair_plan(m)
    assert not plan.square
    assert len(plan.redundant_constraints) == 1
    assert "redundant" in str(plan)


def test_repair_plan_off_graph_candidate_selected():
    # A candidate in no equality (objective only) is an uncontested input.
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.w = pyo.Var(initialize=1.0)
    m.c = pyo.Constraint(expr=m.x == 4.0)
    m.obj = pyo.Objective(expr=(m.w - 1.0) ** 2 + m.x)

    plan = pyomo_pounce.block_repair_plan(m, decision_candidates=[m.w])
    assert plan.square
    assert plan.decisions == [m.w]


def test_block_initialize_auto_repairs_and_solves():
    m = splitter_model()
    report = pyomo_pounce.block_initialize(m, decisions=[m.F, m.D, m.B])
    assert report.ok, str(report)
    assert report.square  # after the repair
    assert report.repair is not None
    assert len(report.repair.pruned) == 1
    assert report.n_decisions_fixed == 2
    # the balance now holds; the pruned flow was solved, not held
    assert m.F.value == pytest.approx(m.D.value + m.B.value)
    assert not m.F.fixed and not m.D.fixed and not m.B.fixed
    assert "spec repair" in str(report)


def test_block_initialize_pruned_decision_needs_no_value():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.u = pyo.Var()  # no value, but the repair prunes it
    m.c1 = pyo.Constraint(expr=m.x == 1.0)
    m.c2 = pyo.Constraint(expr=m.u == m.x)
    m.obj = pyo.Objective(expr=m.u)

    report = pyomo_pounce.block_initialize(m, decisions=[m.u])
    assert report.ok, str(report)
    assert report.repair is not None
    assert report.repair.pruned == [m.u]
    assert m.u.value == pytest.approx(1.0)
    assert not m.u.fixed


def test_block_initialize_pins_and_seeds_automatically():
    m = drum_model()
    report = pyomo_pounce.block_initialize(m, decisions=[m.u])
    assert report.ok, str(report)
    assert report.square
    assert report.repair.pinned == [m.M]
    assert m.M.value == pytest.approx(2.0)  # bounds-aware midpoint seed
    assert m.x.value == pytest.approx(2.0)
    assert not m.M.fixed  # the pin is call-scoped, like the decisions


def test_repair_plan_tie_break_prefers_earlier_listed():
    # Among conflicting candidates, listing order is the priority: the
    # pruned one is the latest-listed that resolves the conflict.
    m = splitter_model()
    plan = pyomo_pounce.block_repair_plan(m, decision_candidates=[m.F, m.D, m.B])
    assert plan.pruned == [m.B]
    plan = pyomo_pounce.block_repair_plan(m, decision_candidates=[m.B, m.D, m.F])
    assert plan.pruned == [m.F]


def test_repair_plan_fixed_candidate_is_an_input():
    m = splitter_model()
    m.F.fix()
    plan = pyomo_pounce.block_repair_plan(m, decision_candidates=[m.F, m.D, m.B])
    assert plan.square
    # F is an input, not part of the plan; the conflict resolves among
    # the remaining candidates by listing order.
    assert all(v is not m.F for v in plan.decisions + plan.pruned + plan.pinned)
    assert plan.decisions == [m.D]
    assert plan.pruned == [m.B]
    m.F.unfix()


def unbounded_drum_model():
    m = pyo.ConcreteModel()
    m.u = pyo.Var(initialize=2.0)
    m.x = pyo.Var()
    m.M = pyo.Var()  # no bounds, no value: _seed_var would give 0
    m.c1 = pyo.Constraint(expr=0 == (m.x - m.u) / m.M)
    m.obj = pyo.Objective(expr=m.x)
    return m


def test_pin_never_seeds_to_zero_unbounded():
    m = unbounded_drum_model()
    report = pyomo_pounce.block_initialize(m, decisions=[m.u])
    assert report.ok, str(report)
    assert report.repair.pinned == [m.M]
    assert report.n_pinned == 1
    assert m.M.value == pytest.approx(1.0)  # not the zero seed
    assert m.x.value == pytest.approx(2.0)


def test_pin_never_seeds_to_zero_symmetric_bounds():
    m = unbounded_drum_model()
    m.M.setlb(-1.0)
    m.M.setub(1.0)  # midpoint is exactly zero
    report = pyomo_pounce.block_initialize(m, decisions=[m.u])
    assert report.ok, str(report)
    assert m.M.value is not None and m.M.value != 0.0
    assert m.x.value == pytest.approx(2.0)


def test_block_initialize_repair_off_reports_instead():
    m = splitter_model()
    report = pyomo_pounce.block_initialize(
        m, decisions=[m.F, m.D, m.B], repair="off"
    )
    assert report.repair is None
    assert not report.square  # reported, not repaired
    assert report.n_pinned == 0
    # nothing pruned or solved: the user's values survive untouched
    assert (m.F.value, m.D.value, m.B.value) == (10.0, 4.0, 7.0)
    assert not m.F.fixed and not m.D.fixed and not m.B.fixed


def test_block_initialize_repair_off_does_not_pin():
    m = drum_model()
    report = pyomo_pounce.block_initialize(m, decisions=[m.u], repair="off")
    assert report.repair is None
    assert not report.square
    assert m.M.value is None  # untouched: previously surfaced, not fixed
    assert not m.M.fixed


def test_block_initialize_rejects_bad_repair_value():
    m = splitter_model()
    with pytest.raises(ValueError, match="repair"):
        pyomo_pounce.block_initialize(m, decisions=[m.F], repair="strict")


def test_partial_fix_unwound_on_valueless_decision():
    # A ValueError on the second decision must not leave the first fixed.
    m = pyo.ConcreteModel()
    m.u1 = pyo.Var(initialize=1.0)
    m.u2 = pyo.Var()  # no value
    m.x = pyo.Var()
    m.c = pyo.Constraint(expr=m.x == m.u1 + m.u2)
    m.obj = pyo.Objective(expr=m.x)

    with pytest.raises(ValueError, match="u2"):
        pyomo_pounce.block_initialize(m, decisions=[m.u1, m.u2])
    assert not m.u1.fixed and not m.u2.fixed


def test_failure_is_reported_not_raised():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0.0, 1.0))
    # No solution in the bounds: Newton cannot satisfy x == 5 inside them.
    m.c = pyo.Constraint(expr=m.x**3 == 125.0)
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.block_initialize(m)
    # calculate_variable_from_constraint ignores bounds and sets x = 5,
    # or raises depending on version — either way no exception escapes.
    assert report.n_blocks == 1
