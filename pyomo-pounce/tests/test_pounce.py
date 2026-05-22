"""Smoke tests for the pyomo-pounce solver plugin.

Run with `pytest`. The `pounce` binary must be on PATH (or bundled).
"""
import pytest

import pyomo_pounce  # noqa: F401  (registers 'pounce' with SolverFactory)
from pyomo.environ import (
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    SolverFactory,
    Var,
    value,
)


@pytest.fixture(scope="module")
def solver():
    s = SolverFactory("pounce")
    if not s.available(exception_flag=False):
        pytest.skip("pounce binary not found on PATH")
    return s


def test_registered():
    assert SolverFactory("pounce") is not None


def test_unconstrained(solver):
    """min (x - 2)^2  ->  x* = 2."""
    m = ConcreteModel()
    m.x = Var(initialize=0.5)
    m.obj = Objective(expr=(m.x - 2) ** 2)

    solver.solve(m)

    assert value(m.x) == pytest.approx(2.0, abs=1e-6)


def test_constrained(solver):
    """min (x-2)^2 + (y-3)^2  s.t. x + y <= 4  ->  (1.5, 2.5)."""
    m = ConcreteModel()
    m.x = Var(domain=NonNegativeReals, initialize=1.0)
    m.y = Var(domain=NonNegativeReals, initialize=1.0)
    m.obj = Objective(expr=(m.x - 2) ** 2 + (m.y - 3) ** 2)
    m.budget = Constraint(expr=m.x + m.y <= 4)

    solver.solve(m)

    assert value(m.x) == pytest.approx(1.5, abs=1e-5)
    assert value(m.y) == pytest.approx(2.5, abs=1e-5)


def test_options_forwarded(solver):
    """`max_iter` is forwarded; 0 iterations cannot reach optimality."""
    m = ConcreteModel()
    m.x = Var(initialize=0.5)
    m.obj = Objective(expr=(m.x - 2) ** 2)

    solver.options["max_iter"] = 0
    try:
        result = solver.solve(m)
    finally:
        del solver.options["max_iter"]

    assert str(result.solver.termination_condition) != "optimal"
