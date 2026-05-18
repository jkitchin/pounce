"""Smoke + correctness tests for the cyipopt-shaped Problem API."""

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
            x[1] * x[2] * x[3],
            x[0] * x[2] * x[3],
            x[0] * x[1] * x[3],
            x[0] * x[1] * x[2],
            2 * x[0],
            2 * x[1],
            2 * x[2],
            2 * x[3],
        ])


def test_hs071_lbfgs():
    """L-BFGS path (no hessian methods on the user object)."""
    prob = pounce.Problem(
        n=4, m=2, problem_obj=HS071(),
        lb=[1.0] * 4, ub=[5.0] * 4,
        cl=[25.0, 40.0], cu=[2e19, 40.0],
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(info["obj_val"], 17.0140172, rtol=1e-5)
    np.testing.assert_allclose(x, [1.0, 4.7430, 3.8211, 1.3794], atol=1e-3)


def test_problem_attributes():
    prob = pounce.Problem(n=2, m=0, problem_obj=type("P", (), {
        "objective": staticmethod(lambda x: float(np.sum(x * x))),
        "gradient":  staticmethod(lambda x: 2 * np.asarray(x, dtype=float)),
    })())
    assert prob.n == 2
    assert prob.m == 0
    assert prob.has_hessian is False


def test_unconstrained_quadratic():
    """min ||x - target||² → x* = target."""
    target = np.array([1.0, 2.0, -3.0, 4.5])

    class Quad:
        def objective(self, x):
            d = x - target
            return float(d @ d)

        def gradient(self, x):
            return 2.0 * (x - target)

    prob = pounce.Problem(n=4, m=0, problem_obj=Quad())
    prob.add_option("tol", 1e-10)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.zeros(4))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, target, atol=1e-6)
