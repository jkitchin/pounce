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
    assert report.n_blocks == 3
    assert report.n_1x1 == 3
    assert report.n_subsystem_solves == 0  # no solver needed
    assert m.x.value == pytest.approx(2.0)
    assert m.y.value == pytest.approx(3.0)
    assert m.z.value == pytest.approx(2.0)


def test_degrees_of_freedom_left_untouched():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.w = pyo.Var()  # free DOF: appears only in the objective
    m.c = pyo.Constraint(expr=m.x == 4.0)
    m.obj = pyo.Objective(expr=(m.w - 1.0) ** 2 + m.x)

    report = pyomo_pounce.block_initialize(m)
    assert report.ok, str(report)
    assert m.x.value == pytest.approx(4.0)
    assert m.w.value is None
    # ... and initialize_missing_values fills the rest.
    pyomo_pounce.initialize_missing_values(m)
    assert m.w.value == 0.0


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


def test_failure_is_reported_not_raised():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0.0, 1.0))
    # No solution in the bounds: Newton cannot satisfy x == 5.
    m.c = pyo.Constraint(expr=m.x**3 == 125.0)
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.block_initialize(m)
    # calculate_variable_from_constraint ignores bounds and sets x = 5,
    # or raises depending on version — either way no exception escapes.
    assert report.n_blocks == 1
