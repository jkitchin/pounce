"""`SolverFactory("pounce").solve(m, intermediate=f)`: a per-iteration callback.

`f` receives an `Iteration` with the iteration's statistics, the objective
in the model's own units and sense, and the current value of every variable.
Returning False stops the solve. Passing `intermediate` sends the solve
down the in-process path, declarations or not.

ORACLE. Two plants meet a demand D = 10 at least cost
`2 x1 + 0.1 x1^2 + 3 x2 + 0.05 x2^2`. Equal marginal costs give
x1 = (1 + 0.1 D) / 0.3 = 20/3 and x2 = 10/3.
"""

import math

import pytest
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

import pyomo_pounce  # noqa: F401  registers SolverFactory("pounce")
from pyomo_pounce import Iteration, declare_sens_param


def cost(x1, x2):
    return 2 * x1 + 0.1 * x1**2 + 3 * x2 + 0.05 * x2**2


def plants(sense=pyo.minimize, declare=True):
    m = pyo.ConcreteModel()
    m.demand = pyo.Param(initialize=10.0, mutable=True)
    m.d = pyo.Var(initialize=10.0)
    m.set_d = pyo.Constraint(expr=m.d == m.demand)
    m.x = pyo.Var([1, 2], bounds=(0, None), initialize=5.0)
    m.meet = pyo.Constraint(expr=m.x[1] + m.x[2] == m.d)
    sign = 1 if sense == pyo.minimize else -1
    m.obj = pyo.Objective(expr=sign * cost(m.x[1], m.x[2]), sense=sense)
    if declare:
        declare_sens_param(m.demand)
    return m


def solve_recording(m, stop_at=None):
    calls = []

    def record(it):
        calls.append(it)
        return it.iteration != stop_at

    res = pyo.SolverFactory("pounce").solve(m, intermediate=record)
    return res, calls


@pytest.mark.parametrize("declare", [True, False], ids=["declared", "undeclared"])
def test_one_call_per_iteration_ending_at_the_solution(declare):
    m = plants(declare=declare)
    res, calls = solve_recording(m)
    assert res.solver.termination_condition == TerminationCondition.optimal
    assert "in-process" in res.solver.name
    n = res.solver.statistics.black_box.number_of_iterations
    assert [it.iteration for it in calls] == list(range(n + 1))
    assert all(isinstance(it, Iteration) and not it.restoration for it in calls)
    last = calls[-1].values
    for v in (m.x[1], m.x[2], m.d):
        assert last[v] == pyo.value(v)
    assert pyo.value(m.x[1]) == pytest.approx(20 / 3, rel=1e-7)


@pytest.mark.parametrize("sense", [pyo.minimize, pyo.maximize])
def test_objective_is_the_models_own_at_each_iterate(sense):
    # The maximize spelling negates the cost, so its objective is -cost at
    # every iterate, though the engine minimizes +cost either way.
    m = plants(sense=sense)
    _, calls = solve_recording(m)
    sign = 1 if sense == pyo.minimize else -1
    for it in calls:
        want = sign * cost(it.values[m.x[1]], it.values[m.x[2]])
        assert it.objective == pytest.approx(want, rel=1e-12)
    assert calls[-1].objective == pytest.approx(pyo.value(m.obj), rel=1e-12)


def test_returning_false_stops_the_solve():
    m = plants()
    res, calls = solve_recording(m, stop_at=2)
    assert res.solver.termination_condition == TerminationCondition.userInterrupt
    assert calls[-1].iteration == 2
    assert not math.isclose(pyo.value(m.x[1]), 20 / 3, rel_tol=1e-3)


def test_intermediate_without_a_model_is_refused():
    with pytest.raises(ValueError, match="intermediate= given but no model"):
        pyo.SolverFactory("pounce").solve(intermediate=lambda it: None)
