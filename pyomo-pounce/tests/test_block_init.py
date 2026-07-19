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
