"""Tests for pounce.WarmStart — the unified warm-start object.

(Distinct from test_warm_start.py, which exercises the raw
lagrange/zl/zu solve kwargs and options.)
"""

import numpy as np
import pytest

import pounce


class HS071:
    def objective(self, x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def gradient(self, x):
        return np.array([
            x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ])

    def constraints(self, x):
        return np.array([np.prod(x), np.dot(x, x)])

    def jacobianstructure(self):
        return (np.repeat([0, 1], 4), np.tile([0, 1, 2, 3], 2))

    def jacobian(self, x):
        return np.array([
            x[1] * x[2] * x[3], x[0] * x[2] * x[3],
            x[0] * x[1] * x[3], x[0] * x[1] * x[2],
            2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3],
        ])


def _make():
    p = pounce.Problem(
        n=4, m=2, problem_obj=HS071(),
        lb=[1.0] * 4, ub=[5.0] * 4,
        cl=[25.0, 40.0], cu=[2e19, 40.0],
    )
    p.add_option("tol", 1e-8)
    p.add_option("print_level", 0)
    return p


X0 = np.array([1.0, 5.0, 5.0, 1.0])


def test_from_info_captures_fields():
    x, info = _make().solve(x0=X0)
    ws = pounce.WarmStart.from_info(x, info)
    np.testing.assert_allclose(ws.x, x)
    assert ws.lagrange is not None and ws.lagrange.shape == (2,)
    assert ws.zl is not None and ws.zl.shape == (4,)
    assert ws.zu is not None and ws.zu.shape == (4,)
    assert ws.mu is not None and 0 < ws.mu < 1e-4
    assert ws.working_set is None  # IPM path carries no working set
    opts = ws.options()
    assert opts["warm_start_init_point"] == "yes"
    assert opts["mu_init"] == pytest.approx(max(ws.mu, 1e-9))
    assert opts["warm_start_bound_push"] == 1e-9


def test_warm_start_cuts_iterations_and_reaches_same_solution():
    cold_x, cold_info = _make().solve(x0=X0)
    ws = pounce.WarmStart.from_info(cold_x, cold_info)

    warm_x, warm_info = _make().solve(warm_start=ws)  # x0 defaults to ws.x
    assert warm_info["status"] == cold_info["status"]
    assert warm_info["iter_count"] < cold_info["iter_count"]
    np.testing.assert_allclose(warm_x, cold_x, atol=1e-6)
    assert abs(warm_info["obj_val"] - cold_info["obj_val"]) < 1e-6


def test_explicit_x0_overrides_captured_point():
    cold_x, cold_info = _make().solve(x0=X0)
    ws = pounce.WarmStart.from_info(cold_x, cold_info)
    x0 = np.clip(cold_x + 1e-3, 1.0, 5.0)
    warm_x, warm_info = _make().solve(x0=x0, warm_start=ws)
    assert warm_info["iter_count"] < cold_info["iter_count"]
    np.testing.assert_allclose(warm_x, cold_x, atol=1e-5)


def test_save_load_round_trip(tmp_path):
    cold_x, cold_info = _make().solve(x0=X0)
    ws = pounce.WarmStart.from_info(cold_x, cold_info, bound_push=1e-8)
    path = tmp_path / "state.npz"
    ws.save(path)
    back = pounce.WarmStart.load(path)
    np.testing.assert_allclose(back.x, ws.x)
    np.testing.assert_allclose(back.lagrange, ws.lagrange)
    np.testing.assert_allclose(back.zl, ws.zl)
    np.testing.assert_allclose(back.zu, ws.zu)
    assert back.mu == pytest.approx(ws.mu)
    assert back.bound_push == 1e-8
    assert back.working_set is None

    # ... and the reloaded state warm-starts identically.
    warm_x, warm_info = _make().solve(warm_start=back)
    assert warm_info["iter_count"] < cold_info["iter_count"]


def test_solve_without_x0_or_warm_start_raises():
    with pytest.raises(TypeError):
        _make().solve()


def test_plain_solve_is_unchanged():
    # The wrapper must be a pure pass-through without warm_start.
    x, info = _make().solve(x0=X0)
    assert info["status_msg"] == "Solve_Succeeded"
    x2, info2 = _make().solve(
        x0=X0,
        lagrange=np.asarray(info["mult_g"]),
        zl=np.asarray(info["mult_x_L"]),
        zu=np.asarray(info["mult_x_U"]),
    )
    assert info2["status_msg"] == "Solve_Succeeded"


def test_minimize_warm_start():
    def rosen(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    res1 = pounce.minimize(rosen, np.array([-1.2, 1.0]), bounds=[(-2, 2), (-2, 2)])
    assert res1.success
    ws = pounce.WarmStart.from_info(res1.x, res1.info)
    res2 = pounce.minimize(rosen, res1.x, bounds=[(-2, 2), (-2, 2)], warm_start=ws)
    assert res2.success
    assert res2.nit <= res1.nit
    np.testing.assert_allclose(res2.x, res1.x, atol=1e-6)


def test_minimize_warm_start_overrides_routing_with_warning():
    def quad(x):
        return float(x @ x)

    res1 = pounce.minimize(quad, np.array([1.0, 1.0]))
    ws = pounce.WarmStart.from_info(res1.x, res1.info)
    with pytest.warns(UserWarning, match="NLP path"):
        res2 = pounce.minimize(
            quad, res1.x, warm_start=ws, solver_selection="auto"
        )
    assert res2.success


def test_sqp_working_set_round_trip():
    class Quad:
        def objective(self, x):
            return (x[0] - 2.0) ** 2

        def gradient(self, x):
            return np.array([2.0 * (x[0] - 2.0)])

        def constraints(self, x):
            return np.array([x[0]])

        def jacobianstructure(self):
            return (np.array([0]), np.array([0]))

        def jacobian(self, x):
            return np.array([1.0])

    def make():
        p = pounce.Problem(
            n=1, m=1, problem_obj=Quad(),
            lb=[0.0], ub=[10.0], cl=[0.0], cu=[1.0],
        )
        p.add_option("algorithm", "active-set-sqp")
        p.add_option("print_level", 0)
        return p

    x, info = make().solve(x0=np.array([0.5]))
    ws = pounce.WarmStart.from_info(x, info)
    assert ws.working_set is not None  # SQP path returns the working set
    assert ws.mu is None  # SQP reports mu = 0.0 -> None

    x2, info2 = make().solve(warm_start=ws)
    np.testing.assert_allclose(x2, x, atol=1e-8)
    assert info2["working_set"] is not None
