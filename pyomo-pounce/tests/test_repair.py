"""Tests for pyomo_pounce.project_to_feasible and the initialize pipeline.

The projection is a real POUNCE solve, so most of these need the
binary and skip without it.
"""

import pyomo.environ as pyo
import pytest

import pyomo_pounce

pytest.importorskip("networkx", reason="initialize pipeline needs networkx")
pytest.importorskip("scipy", reason="initialize pipeline needs scipy")


@pytest.fixture(scope="module")
def solver():
    s = pyo.SolverFactory("pounce")
    if not s.available(exception_flag=False):
        pytest.skip("pounce binary not found on PATH")
    return s


def _mole_fraction_model():
    """The reviewer's example: midpoint fill gives x_i = 0.5 each, which
    violates sum(x) == 1."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], bounds=(0.0, 1.0))
    m.sum_to_one = pyo.Constraint(expr=m.x[1] + m.x[2] + m.x[3] == 1.0)
    m.obj = pyo.Objective(expr=m.x[1])
    return m


def test_project_repairs_inconsistent_fill(solver):
    m = _mole_fraction_model()
    n = pyomo_pounce.initialize_missing_values(m)  # x_i = 0.5, sum = 1.5
    assert n == 3
    cond = pyomo_pounce.project_to_feasible(m, solver=solver)
    assert cond in ("optimal", "locallyOptimal")
    total = sum(pyo.value(m.x[i]) for i in [1, 2, 3])
    assert total == pytest.approx(1.0, abs=1e-6)
    # Min-norm from equal anchors: the components stay equal.
    for i in [1, 2, 3]:
        assert pyo.value(m.x[i]) == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_project_restores_objective(solver):
    m = _mole_fraction_model()
    pyomo_pounce.initialize_missing_values(m)
    assert m.obj.active
    pyomo_pounce.project_to_feasible(m, solver=solver)
    assert m.obj.active  # original objective restored
    # ... and the temporary projection objective is gone.
    objs = list(m.component_data_objects(pyo.Objective, descend_into=True))
    assert len(objs) == 1


def test_project_without_anchors_raises():
    m = _mole_fraction_model()  # nothing has a value yet
    with pytest.raises(ValueError, match="initialize_missing_values"):
        pyomo_pounce.project_to_feasible(m)


def test_initialize_pipeline_end_to_end(solver):
    # fill -> repair -> block-solve on a split with a composition sum.
    m = pyo.ConcreteModel()
    m.feed = pyo.Var(initialize=10.0)
    m.split = pyo.Var(bounds=(0.0, 1.0), initialize=0.3)
    m.out1 = pyo.Var(bounds=(0.0, None))
    m.out2 = pyo.Var(bounds=(0.0, None))
    m.x = pyo.Var([1, 2], bounds=(0.0, 1.0))
    m.bal1 = pyo.Constraint(expr=m.out1 == m.split * m.feed)
    m.bal2 = pyo.Constraint(expr=m.out2 == (1.0 - m.split) * m.feed)
    m.sum_x = pyo.Constraint(expr=m.x[1] + m.x[2] == 1.0)
    m.obj = pyo.Objective(expr=m.out1 + m.x[1])

    report = pyomo_pounce.initialize(
        m, decisions=[m.feed, m.split], solver=solver
    )
    assert report.ok, str(report)
    assert report.n_filled == 4  # out1, out2, x[1], x[2]
    assert report.projection in ("optimal", "locallyOptimal")
    assert report.block is not None and report.block.ok
    # Block solve produced the physical profile with decisions held...
    assert m.out1.value == pytest.approx(3.0, abs=1e-5)
    assert m.out2.value == pytest.approx(7.0, abs=1e-5)
    # ... and the projection made the composition consistent.
    assert pyo.value(m.x[1]) + pyo.value(m.x[2]) == pytest.approx(1.0, abs=1e-5)
    # Decisions are free again for the optimizer.
    assert not m.feed.fixed and not m.split.fixed
    assert "fill -> repair -> block-solve" in str(report)


def test_initialize_skips_projection_when_asked():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1.0, 3.0))
    m.c = pyo.Constraint(expr=m.x <= 2.5)
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.initialize(m, project=False)
    assert report.projection is None
    assert report.n_filled == 1
    assert m.x.value == pytest.approx(2.0)  # midpoint fill, no repair


def test_initialize_pins_automatically(solver):
    # A loose integrator (M appears only as the denominator of a
    # 0 == f/M row) is identified with no user input: pinned, seeded,
    # held for the pipeline, released after.
    m = pyo.ConcreteModel()
    m.u = pyo.Var(initialize=2.0)
    m.x = pyo.Var()
    m.M = pyo.Var(bounds=(1.0, 3.0))  # no value
    m.c1 = pyo.Constraint(expr=0 == (m.x - m.u) / m.M)
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.initialize(m, solver=solver, decisions=[m.u])
    assert report.ok, str(report)
    assert report.repair is not None
    assert report.repair.pinned == [m.M]
    assert report.block.square
    assert m.M.value == pytest.approx(2.0)  # bounds-aware midpoint seed
    assert m.x.value == pytest.approx(2.0)
    assert not m.u.fixed and not m.M.fixed
    assert "spec repair" in str(report)


def test_initialize_repair_off_reports_instead(solver):
    # The strict path: nothing pruned, nothing pinned, the non-square
    # specification is reported rather than repaired.
    m = pyo.ConcreteModel()
    m.u = pyo.Var(initialize=2.0)
    m.x = pyo.Var()
    m.M = pyo.Var(bounds=(1.0, 3.0))
    m.c1 = pyo.Constraint(expr=0 == (m.x - m.u) / m.M)
    m.obj = pyo.Objective(expr=m.x)

    report = pyomo_pounce.initialize(
        m, solver=solver, decisions=[m.u], repair="off"
    )
    assert report.repair is None
    assert report.n_pinned == 0
    assert not report.block.square
    assert not m.M.fixed
