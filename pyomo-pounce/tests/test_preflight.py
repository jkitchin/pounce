"""Tests for pyomo_pounce.preflight / initialize_missing_values.

Hermetic: no solver binary required.
"""

import math

import pyomo.environ as pyo
import pytest

import pyomo_pounce


def test_unset_values_are_flagged_and_restored():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var(initialize=1.0)
    m.obj = pyo.Objective(expr=m.x**2 + m.y**2)

    report = pyomo_pounce.preflight(m)
    assert report.n_unset == 1
    assert report.unset == ["x"]
    assert any(".nl" in w for w in report.warnings)
    # The model is untouched afterwards.
    assert m.x.value is None
    assert m.y.value == 1.0
    # As written (x = 0), the objective evaluates fine.
    assert report.objective == pytest.approx(1.0)
    assert not report.fatal


def test_silent_zero_domain_error_is_fatal():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()  # unset -> written as 0 -> log(0) blows up
    m.c = pyo.Constraint(expr=pyo.log(m.x) >= 0)
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.preflight(m)
    assert report.fatal
    assert report.verdict == "FATAL"
    assert report.n_non_evaluable == 1
    assert "c" in report.non_evaluable
    assert m.x.value is None  # restored


def test_bound_and_constraint_violations_reported():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1.0, 5.0), initialize=-2.0)
    m.y = pyo.Var(bounds=(0.0, 10.0), initialize=0.0)  # on its lower bound
    m.c = pyo.Constraint(expr=m.x + m.y >= 10.0)
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.preflight(m)
    assert not report.fatal
    assert report.n_bound_violations == 1
    name, val, lo, hi, viol = report.bound_violations[0]
    assert name == "x" and viol == pytest.approx(3.0)
    assert report.n_on_bounds == 1
    assert report.n_con_violations == 1
    assert report.max_con_violation == pytest.approx(12.0)
    assert report.verdict == "WARNINGS"


def test_clean_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0.0, 4.0), initialize=2.0)
    m.c = pyo.Constraint(expr=m.x <= 3.0)
    m.obj = pyo.Objective(expr=(m.x - 1.0) ** 2)

    report = pyomo_pounce.preflight(m)
    assert report.verdict == "CLEAN"
    assert report.ok
    assert "VERDICT: CLEAN" in str(report)


def test_fixed_vars_are_ignored():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.x.fix(3.0)
    m.obj = pyo.Objective(expr=m.x**2)
    report = pyomo_pounce.preflight(m)
    assert report.n_vars == 0
    assert report.n_unset == 0


def test_initialize_missing_values_midpoint():
    m = pyo.ConcreteModel()
    m.a = pyo.Var(bounds=(1.0, 3.0))
    m.b = pyo.Var(bounds=(0.0, None))
    m.c = pyo.Var(bounds=(None, -5.0))
    m.d = pyo.Var()
    m.e = pyo.Var(initialize=7.0)

    count = pyomo_pounce.initialize_missing_values(m)
    assert count == 4
    assert m.a.value == pytest.approx(2.0)
    assert m.b.value == pytest.approx(1.0)
    assert m.c.value == pytest.approx(-6.0)
    assert m.d.value == 0.0
    assert m.e.value == 7.0  # untouched


def test_initialize_missing_values_zero():
    m = pyo.ConcreteModel()
    m.a = pyo.Var(bounds=(1.0, 3.0))
    count = pyomo_pounce.initialize_missing_values(m, strategy="zero")
    assert count == 1
    assert m.a.value == 0.0
    with pytest.raises(ValueError):
        pyomo_pounce.initialize_missing_values(m, strategy="nope")


def test_preflight_then_init_clears_warning():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1.0, 5.0))
    m.c = pyo.Constraint(expr=pyo.log(m.x) >= 0)
    m.obj = pyo.Objective(expr=m.x)

    assert pyomo_pounce.preflight(m).fatal  # log(0) as written
    pyomo_pounce.initialize_missing_values(m)  # midpoint: x = 3
    report = pyomo_pounce.preflight(m)
    assert not report.fatal
    assert math.isfinite(report.objective)
