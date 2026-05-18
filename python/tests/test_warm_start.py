"""Warm-start API: passing `lagrange`, `zl`, `zu` along with `x0`.

A warm-started solve from a known optimum (with all duals) should
converge in dramatically fewer iterations than the cold start.
"""

import pytest

import numpy as np

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


def _make(opts=None):
    prob = pounce.Problem(
        n=4, m=2, problem_obj=HS071(),
        lb=[1.0] * 4, ub=[5.0] * 4,
        cl=[25.0, 40.0], cu=[2e19, 40.0],
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)
    for k, v in (opts or {}).items():
        prob.add_option(k, v)
    return prob


def test_warm_x0_skips_iterations():
    """A primal starting point near the optimum should cut iterations."""
    cold = _make().solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    x_cold, info_cold = cold
    assert info_cold["status_msg"] == "Solve_Succeeded"

    warm = _make().solve(x0=x_cold)
    x_warm, info_warm = warm
    assert info_warm["status_msg"] == "Solve_Succeeded"
    assert info_warm["iter_count"] < info_cold["iter_count"]
    np.testing.assert_allclose(x_warm, x_cold, atol=1e-6)


def test_warm_start_init_point_with_duals():
    """`warm_start_init_point=yes` with x0/lagrange/zl/zu converges.

    Previously this configuration panicked because the
    `WarmStartIterateInitializer` expected `data.curr` to be populated
    from a prior re-optimize call. The bridge now detects the
    fresh-solve case and pulls primal/dual seeds from the TNLP.
    """
    cold_x, cold_info = _make().solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    assert cold_info["status_msg"] == "Solve_Succeeded"

    prob = _make({"warm_start_init_point": "yes"})
    x, info = prob.solve(
        x0=cold_x,
        lagrange=np.asarray(cold_info["mult_g"], dtype=np.float64),
        zl=np.asarray(cold_info["mult_x_L"], dtype=np.float64),
        zu=np.asarray(cold_info["mult_x_U"], dtype=np.float64),
    )
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, cold_x, atol=1e-6)


def test_warm_start_dual_seeds_accepted():
    """Sanity: passing `lagrange` / `zl` / `zu` without the
    `warm_start_init_point` option flag is still accepted (the seeds
    sit on the TNLP unused, matching cyipopt)."""
    cold_x, cold_info = _make().solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    n, m = 4, 2
    prob = _make()
    x, info = prob.solve(
        x0=cold_x,
        lagrange=np.asarray(cold_info["mult_g"], dtype=np.float64),
        zl=np.asarray(cold_info["mult_x_L"], dtype=np.float64),
        zu=np.asarray(cold_info["mult_x_U"], dtype=np.float64),
    )
    assert info["status_msg"] == "Solve_Succeeded"
    assert len(x) == n
    assert len(info["mult_g"]) == m
