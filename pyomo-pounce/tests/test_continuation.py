"""Pyomo adapter for predictor--corrector continuation (pounce#608).

Acceptance criterion 2: "A Pyomo MPC/horizon-shift example supplies a
transfer map and reuses primal/dual/barrier state."
"""

import numpy as np
import pytest

pyo = pytest.importorskip("pyomo.environ")

import pyomo_pounce  # noqa: E402
from pyomo_pounce import continuation, shift_map  # noqa: E402


NH = 8
H = 0.2
A, B = 1.0, 0.1
U_MAX = 0.5


def mpc_model():
    """Linear-quadratic MPC of a damped oscillator over a fixed horizon.

    The initial state is a declared mutable Param, so a step of the
    parameter path is a step of the pin-equality right-hand side -- the
    sIPOPT convention the tangent predictor needs.
    """
    m = pyo.ConcreteModel()
    m.K = pyo.RangeSet(0, NH)
    m.KU = pyo.RangeSet(0, NH - 1)
    m.D = pyo.RangeSet(0, 1)

    m.x0 = pyo.Param(m.D, initialize=lambda _m, i: (1.0, 0.0)[i], mutable=True)
    m.x = pyo.Var(m.K, m.D, initialize=0.0, bounds=(-10.0, 10.0))
    m.u = pyo.Var(m.KU, initialize=0.0, bounds=(-U_MAX, U_MAX))

    m.init = pyo.Constraint(m.D, rule=lambda _m, i: _m.x[0, i] == _m.x0[i])

    def dyn(_m, k, i):
        if i == 0:
            return _m.x[k + 1, 0] == _m.x[k, 0] + H * _m.x[k, 1]
        return _m.x[k + 1, 1] == _m.x[k, 1] + H * (
            -A * _m.x[k, 0] - B * _m.x[k, 1] + _m.u[k])

    m.dyn = pyo.Constraint(m.KU, m.D, rule=dyn)

    m.obj = pyo.Objective(
        expr=sum(m.x[k, 0] ** 2 + 0.1 * m.x[k, 1] ** 2 for k in m.K)
        + 0.05 * sum(m.u[k] ** 2 for k in m.KU)
    )
    return m


def circle_path(n=6, radius=1.2, dphi=0.25):
    """The parameter walks a circle in state space, so every point is
    about as hard as the last."""
    return [
        {"x0": {0: radius * np.cos(dphi * k), 1: radius * np.sin(dphi * k)}}
        for k in range(n)
    ]


def _path_for(m, n=6):
    return [{m.x0: p["x0"]} for p in circle_path(n)]


def _solve_cold(m, point):
    for idx, val in point[m.x0].items():
        m.x0[idx].value = float(val)
    for v in m.component_data_objects(pyo.Var, active=True):
        v.value = 0.0
    res = pyo.SolverFactory("pounce").solve(m)
    return str(res.solver.termination_condition), pyo.value(m.obj)


pytestmark = pytest.mark.usefixtures()


@pytest.fixture(scope="module")
def have_solver():
    try:
        pyomo_pounce.check_binary()
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"pounce solver unavailable: {exc}")


def test_continuation_traces_a_pyomo_parameter_path(have_solver):
    m = mpc_model()
    pyomo_pounce.declare_sens_param(m.x0)
    path = _path_for(m)

    trace = continuation(m, [m.x0], path)

    assert trace.status == "ok"
    assert trace.n_steps == len(path)
    assert trace.n_corrections == len(path)
    assert trace.total_evals >= 0
    assert "active-set events" in trace.report()


def test_tangent_predictor_is_recorded_after_the_anchor(have_solver):
    m = mpc_model()
    pyomo_pounce.declare_sens_param(m.x0)
    trace = continuation(m, [m.x0], _path_for(m))
    kinds = [st.predictor for st in trace.steps]
    assert kinds[0] == "cold"
    assert "tangent" in kinds[1:]


def test_zero_order_fallback_is_supported(have_solver):
    """pounce#608's documented degraded mode: no sensitivities, plain
    previous-solution transfer, same trace type."""
    m = mpc_model()
    trace = continuation(m, [m.x0], _path_for(m), predictor="zero")
    assert trace.status == "ok"
    assert [st.predictor for st in trace.steps][1:] == ["zero"] * 5


def test_continuation_matches_independent_cold_solves(have_solver):
    """The traced objective at every point must equal what a cold solve
    at that parameter finds -- a warm start may not change the answer."""
    m = mpc_model()
    pyomo_pounce.declare_sens_param(m.x0)
    path = _path_for(m)
    trace = continuation(m, [m.x0], path)

    ref = mpc_model()
    for k, point in enumerate(path):
        tc, obj = _solve_cold(ref, {ref.x0: point[m.x0]})
        assert tc in ("optimal", "locallyOptimal", "feasible")
        assert trace.steps[k].obj == pytest.approx(obj, rel=1e-6, abs=1e-8)


def test_horizon_shift_transfer_map_is_applied(have_solver):
    """Acceptance criterion 2: a transfer map for the horizon shift."""
    m = mpc_model()
    pyomo_pounce.declare_sens_param(m.x0)
    shift = shift_map(m, [m.x, m.u], shift=1)

    calls = []

    def transfer():
        calls.append(True)
        shift()

    trace = continuation(m, [m.x0], _path_for(m), transfer=transfer)
    assert trace.status == "ok"
    assert len(calls) == trace.n_steps - 1      # not on the anchor


def test_shift_map_moves_stages_back_by_one():
    m = mpc_model()
    for k in m.KU:
        m.u[k].value = float(k)
    shift_map(m, [m.u], shift=1)()
    assert [m.u[k].value for k in m.KU] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                            7.0, 7.0]


def test_shift_map_constant_fill():
    m = mpc_model()
    for k in m.KU:
        m.u[k].value = float(k)
    shift_map(m, [m.u], shift=2, fill=-1.0)()
    assert [m.u[k].value for k in m.KU] == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                            -1.0, -1.0]


def test_predictor_must_be_a_known_mode():
    m = mpc_model()
    with pytest.raises(ValueError, match="tangent.*zero"):
        continuation(m, [m.x0], _path_for(m), predictor="quadratic")


def test_tangent_without_declared_params_is_refused():
    m = mpc_model()
    with pytest.raises(ValueError, match="at least one"):
        continuation(m, [], _path_for(m), predictor="tangent")
