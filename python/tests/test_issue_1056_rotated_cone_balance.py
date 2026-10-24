"""gh #1056: a convex QCQP whose quadratic constraint has a huge constant.

``x₀² + x₁² ≤ R`` reaches the conic solver as the rotated cone
``(s/κ + κ, s/κ − κ, √2·Fx)`` with ``s = R``. The router used ``κ = 1``, and
at ``R = 1e20`` the doubles ``1e20 + 1`` and ``1e20 − 1`` are the same
number: the declared cone was empty but for ``x = 0``. ``κ`` is now the power
of two nearest ``√|b|``, which leaves ``(s/κ)² − κ²``-style membership exact
and both factors the same size. Mirrors the CLI's ``rotated_cone_balance``.
"""

import math

import numpy as np
import pytest
from pounce import minimize
from pounce._route import _rotated_cone_balance


def _ball(r2):
    return {
        "type": "ineq",
        "fun": lambda x: r2 - (x[0] ** 2 + x[1] ** 2),
        "jac": lambda x: np.array([-2.0 * x[0], -2.0 * x[1]]),
    }


def _solve(r2, bounds, selection):
    return minimize(
        lambda x: 3.0 * x[0] + 4.0 * x[1],
        np.zeros(2),
        jac=lambda x: np.array([3.0, 4.0]),
        constraints=[_ball(r2)],
        bounds=bounds,
        solver_selection=selection,
    )


@pytest.mark.parametrize("r2", [1e8, 1e12, 1e16, 1e20])
def test_an_inactive_huge_ball_solves_on_the_conic_path(r2):
    # Box [−10, 10]²: the ball is inactive, the optimum is the corner.
    res = _solve(r2, [(-10.0, 10.0)] * 2, "socp")
    assert res.info.get("solver") == "socp"
    assert res.success, res.message
    assert res.fun == pytest.approx(-70.0, rel=1e-8)


@pytest.mark.parametrize("r2", [1e8, 1e16, 1e20])
def test_an_active_huge_ball_solves_on_the_conic_path(r2):
    res = _solve(r2, None, "socp")
    assert res.info.get("solver") == "socp"
    assert res.success, res.message
    assert res.fun == pytest.approx(-5.0 * math.sqrt(r2), rel=1e-8)


@pytest.mark.parametrize("b", [-0.5, 1.0, -2.0, -8.0, -32.0, -1e20, 3e40])
def test_the_balance_matches_the_cli(b):
    # The CLI rounds half away from zero; Python's round() would not, and
    # |b| = 2, 32, … sit exactly on the half (κ² = |b| between two powers).
    k = _rotated_cone_balance(b)
    want = 1.0 if abs(b) <= 1.0 else 2.0 ** math.floor(0.5 * math.log2(abs(b)) + 0.5)
    assert k == want
    h0, h1 = -b / k + k, -b / k - k
    assert (h0 - h1) * (h0 + h1) == pytest.approx(-4.0 * b, rel=1e-12)
