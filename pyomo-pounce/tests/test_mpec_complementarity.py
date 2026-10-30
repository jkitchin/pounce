"""Pyomo `mpec` models through `SolverFactory("pounce")`.

Pyomo writes a `Complementarity` component to `.nl` as complementarity rows
(`r`-segment kind 5), which pounce reads as declared pairs and solves through
its complementarity driver. Before that, the reader refused the rows and the
solve failed.

Each model is chosen so that dropping the pairing cannot pass: ralph1's
relaxation is unbounded, and qpec_small is checked against its answer. The
all-zero qpec_small start is also the end-to-end guard on the driver's
certificate-keyed acceptance: its exact-product solve exits "acceptable" at a
point that is not stationary, and only moving on to the continuation reaches
(1, 1, 0).
"""
import pyomo.environ as pyo
import pytest
from pyomo.mpec import Complementarity, complements

import pyomo_pounce  # noqa: F401  (registers 'pounce')


def solve(m):
    res = pyo.SolverFactory("pounce").solve(m)
    assert res.solver.termination_condition == pyo.TerminationCondition.optimal, res
    return res


def test_ralph1_reaches_the_origin():
    # min 2x - y, x >= 0, 0 <= y ⟂ y - x >= 0. Without the pairing, y -> inf.
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=1.0)
    m.y = pyo.Var(bounds=(0, None), initialize=2.0)
    m.obj = pyo.Objective(expr=2 * m.x - m.y)
    m.c = Complementarity(expr=complements(m.y >= 0, m.y - m.x >= 0))
    solve(m)
    assert pyo.value(m.x) == pytest.approx(0.0, abs=1e-4)
    assert pyo.value(m.y) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("start", [(0.0, 0.5, 0.5), (0.0, 0.0, 0.0), (2.0, 1.5, 0.0)])
def test_qpec_small_reaches_its_answer(start):
    # min (x-1)^2 + (y1-1)^2 + y2^2, 0 <= x <= 2,
    # 0 <= y1 ⟂ 2 y1 - 1 - x >= 0,  0 <= y2 ⟂ x + 2 y2 - 1 >= 0.  Answer (1, 1, 0).
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 2), initialize=start[0])
    m.y1 = pyo.Var(bounds=(0, None), initialize=start[1])
    m.y2 = pyo.Var(bounds=(0, None), initialize=start[2])
    m.obj = pyo.Objective(expr=(m.x - 1) ** 2 + (m.y1 - 1) ** 2 + m.y2**2)
    m.c1 = Complementarity(expr=complements(m.y1 >= 0, 2 * m.y1 - 1 - m.x >= 0))
    m.c2 = Complementarity(expr=complements(m.y2 >= 0, m.x + 2 * m.y2 - 1 >= 0))
    solve(m)
    for v, want in ((m.x, 1.0), (m.y1, 1.0), (m.y2, 0.0)):
        assert pyo.value(v) == pytest.approx(want, abs=1e-4)
