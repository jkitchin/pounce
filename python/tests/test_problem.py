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


def test_diff_handoff_masks_in_info():
    """The DiffHandoff active-set masks ride out in the info dict
    (dev-notes/diff-handoff-contract.md), computed once in the producer.

    HS071's known optimum x* ≈ (1, 4.743, 3.821, 1.379) has:
      * x0 pinned at its lower bound (1.0) → pinned_vars[0] is True;
        x1..x3 interior → not pinned;
      * constraint 0 (prod ≥ 25) binding and constraint 1 (sumsq = 40)
        an equality → both active.
    """
    prob = pounce.Problem(
        n=4, m=2, problem_obj=HS071(),
        lb=[1.0] * 4, ub=[5.0] * 4,
        cl=[25.0, 40.0], cu=[2e19, 40.0],
    )
    prob.add_option("tol", 1e-8)
    prob.add_option("print_level", 0)
    x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
    assert info["status_msg"] == "Solve_Succeeded"

    pinned = np.asarray(info["pinned_vars"])
    active_cons = np.asarray(info["active_constraints"])
    assert pinned.dtype == bool and pinned.shape == (4,)
    assert active_cons.dtype == bool and active_cons.shape == (2,)
    assert info["active_tol"] == 1e-6

    # x0 sits on its lower bound; x1..x3 are interior.
    assert bool(pinned[0]) is True
    assert not pinned[1:].any()
    # Both constraints active (binding inequality + equality).
    assert active_cons.all()

    # The masks are consistent with the raw multipliers they summarize.
    tol = info["active_tol"]
    zl = np.asarray(info["mult_x_L"])
    zu = np.asarray(info["mult_x_U"])
    np.testing.assert_array_equal(pinned, (zl > tol) | (zu > tol))


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


def test_negative_obj_scaling_factor_maximizes():
    """obj_scaling_factor < 0 means maximize (upstream Ipopt semantics).

    Regression for the pounce#128 follow-up: the option was registered
    but never read, so the IPM minimized the unscaled objective and a
    concave maximization diverged (Diverging_Iterates) instead of
    converging to the maximizer.
    """

    class ConcaveBump:
        def objective(self, x):
            return -((x[0] - 1.0) ** 2)

        def gradient(self, x):
            return np.array([-2.0 * (x[0] - 1.0)])

    prob = pounce.Problem(n=1, m=0, problem_obj=ConcaveBump(),
                          lb=[-1e19], ub=[1e19])
    prob.add_option("print_level", 0)
    prob.add_option("sb", "yes")
    prob.add_option("obj_scaling_factor", -1.0)
    x, info = prob.solve(x0=np.array([0.0]))
    assert info["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(x, [1.0], atol=1e-6)
    # The reported objective is the user's (unscaled) value at the max.
    np.testing.assert_allclose(info["obj_val"], 0.0, atol=1e-8)
