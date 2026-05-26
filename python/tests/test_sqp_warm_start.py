"""SQP working-set warm start (Phase 5c §7.3).

Drives the convex-quadratic problem `min (x − 2)²` through the
`active-set-sqp` path twice. The first solve produces a working
set in ``info["working_set"]`` and via ``get_working_set()``; the
second solve consumes it as the ``working_set=`` kwarg on
``solve()`` and converges with iteration count not exceeding the
cold-solve count.

Also exercises the validation paths (bad tuple shape, bad status
code, wrong length) on the ``set_working_set`` setter.
"""

import numpy as np
import pytest

import pounce


class Quad:
    """min (x − 2)²; one inequality `−10 ≤ x ≤ 10`."""

    def objective(self, x):
        return float((x[0] - 2.0) ** 2)

    def gradient(self, x):
        return np.array([2.0 * (x[0] - 2.0)])

    def constraints(self, x):
        return np.array([x[0]])

    def jacobianstructure(self):
        return (np.array([0]), np.array([0]))

    def jacobian(self, x):
        return np.array([1.0])

    def hessianstructure(self):
        return (np.array([0]), np.array([0]))

    def hessian(self, x, lagrange, obj_factor):
        return np.array([2.0 * obj_factor])


def _make():
    p = pounce.Problem(
        n=1, m=1, problem_obj=Quad(),
        lb=[-1e20], ub=[1e20], cl=[-10.0], cu=[10.0],
    )
    p.add_option("algorithm", "active-set-sqp")
    p.add_option("print_level", 0)
    return p


def test_sqp_working_set_round_trip():
    p = _make()
    x_cold, info_cold = p.solve(x0=np.array([0.0]))
    assert info_cold["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x_cold, np.array([2.0]), atol=1e-6)

    ws = info_cold["working_set"]
    assert ws is not None
    bounds, cons = ws
    assert len(bounds) == 1
    assert len(cons) == 1
    assert int(bounds[0]) in (0, 1, 2, 3)
    assert int(cons[0]) in (0, 1, 2, 3)

    # get_working_set() returns the same shape.
    ws_get = p.get_working_set()
    assert ws_get is not None
    np.testing.assert_array_equal(ws_get[0], bounds)
    np.testing.assert_array_equal(ws_get[1], cons)

    # Warm-restart via kwarg.
    p2 = _make()
    x_warm, info_warm = p2.solve(x0=np.array([0.0]), working_set=ws)
    assert info_warm["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x_warm, np.array([2.0]), atol=1e-6)
    assert info_warm["iter_count"] <= info_cold["iter_count"]

    # And via set_working_set() setter.
    p3 = _make()
    p3.set_working_set(ws)
    x3, info3 = p3.solve(x0=np.array([0.0]))
    assert info3["status_msg"] == "Solve_Succeeded"

    # clear_working_set() drops the pending one; subsequent solve
    # cold-starts.
    p4 = _make()
    p4.set_working_set(ws)
    p4.clear_working_set()
    x4, info4 = p4.solve(x0=np.array([0.0]))
    assert info4["status_msg"] == "Solve_Succeeded"


def test_sqp_working_set_validation_wrong_shape():
    p = _make()
    with pytest.raises(ValueError):
        # Not a tuple.
        p.set_working_set(np.array([0]))


def test_sqp_working_set_validation_bad_length():
    p = _make()
    with pytest.raises(ValueError):
        # n=1, m=1 but bounds has length 2.
        p.set_working_set((np.array([0, 0], dtype=np.int8), np.array([0], dtype=np.int8)))


def test_sqp_working_set_validation_bad_status_code():
    p = _make()
    with pytest.raises(ValueError):
        # Status code 7 outside 0..=3.
        p.set_working_set((np.array([7], dtype=np.int8), np.array([0], dtype=np.int8)))


def test_get_working_set_returns_none_on_ipm_path():
    """The IPM path does not produce an SQP working set."""
    p = pounce.Problem(
        n=1, m=1, problem_obj=Quad(),
        lb=[-1e20], ub=[1e20], cl=[-10.0], cu=[10.0],
    )
    # No `algorithm active-set-sqp` — defaults to interior-point.
    p.add_option("print_level", 0)
    x, info = p.solve(x0=np.array([0.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    assert info["working_set"] is None
    assert p.get_working_set() is None


def test_classify_working_set_helper_module_level():
    """The Phase 5c §7.5 classifier is importable as
    ``pounce.classify_working_set`` and produces the same encoding
    the SQP path consumes."""
    # 1-D problem: x ≥ 0, x ≤ 10. At the optimum x=2 (interior),
    # the lower bound is inactive.
    bounds, cons = pounce.classify_working_set(
        x=[2.0], x_l=[0.0], x_u=[10.0],
        g=[2.0], g_l=[-10.0], g_u=[10.0],
        lambda_g=[0.0], z_l=[0.0], z_u=[0.0],
        m_eq=0,
    )
    assert len(bounds) == 1
    assert len(cons) == 1
    assert int(bounds[0]) == 0  # Inactive
    assert int(cons[0]) == 0    # Inactive

    # x at the lower bound with a positive multiplier.
    bounds, _ = pounce.classify_working_set(
        x=[0.0], x_l=[0.0], x_u=[10.0],
        g=[0.0], g_l=[-10.0], g_u=[10.0],
        lambda_g=[0.0], z_l=[2.0], z_u=[0.0],
        m_eq=0,
    )
    assert int(bounds[0]) == 1  # AtLower

    # Equality constraint always classified as Equality (code 3).
    _, cons = pounce.classify_working_set(
        x=[1.0], x_l=[-10.0], x_u=[10.0],
        g=[5.0], g_l=[5.0], g_u=[5.0],
        lambda_g=[1.0], z_l=[0.0], z_u=[0.0],
        m_eq=1,
    )
    assert int(cons[0]) == 3  # Equality


def test_classify_working_set_then_feed_into_sqp_solve():
    """End-to-end Phase 5c §7.5 pattern: classify the active set
    at a known iterate, then feed the resulting working set into a
    follow-up `solve(..., working_set=ws)` call."""
    p = _make()
    x_cold, info_cold = p.solve(x0=np.array([0.0]))
    ws_from_solve = info_cold["working_set"]

    # Compute the classifier's WS from the multipliers.
    n, m = 1, 1
    bounds_arr, cons_arr = pounce.classify_working_set(
        x=x_cold,
        x_l=np.array([-1e20]),
        x_u=np.array([1e20]),
        g=info_cold["g"],
        g_l=np.array([-10.0]),
        g_u=np.array([10.0]),
        lambda_g=info_cold["mult_g"],
        z_l=info_cold["mult_x_L"],
        z_u=info_cold["mult_x_U"],
        m_eq=0,
    )
    assert len(bounds_arr) == n
    assert len(cons_arr) == m

    # Feed back into a fresh SQP solve.
    p2 = _make()
    x_warm, info_warm = p2.solve(
        x0=np.array([0.0]),
        working_set=(bounds_arr, cons_arr),
    )
    assert info_warm["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x_warm, x_cold, atol=1e-6)


def test_classify_working_set_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        # x has length 1 but x_l has length 2.
        pounce.classify_working_set(
            x=[1.0], x_l=[0.0, 0.0], x_u=[10.0],
            g=[1.0], g_l=[0.0], g_u=[2.0],
            lambda_g=[0.0], z_l=[0.0], z_u=[0.0],
            m_eq=0,
        )
