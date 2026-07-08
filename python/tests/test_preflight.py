"""Tests for pounce.preflight — the starting-point check.

Pure-Python: no solve is performed, so these run without the native
extension doing any work beyond import.
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


HS071_BOUNDS = dict(lb=[1.0] * 4, ub=[5.0] * 4, cl=[25.0, 40.0], cu=[2e19, 40.0])


class DomainTrap:
    """min 1/x0 + x1: NaN/inf at any x with x0 == 0."""

    def objective(self, x):
        return 1.0 / x[0] + x[1]

    def gradient(self, x):
        return np.array([-1.0 / x[0] ** 2, 1.0])


class Raiser:
    def objective(self, x):
        raise ValueError("domain error")


def test_clean_interior_point():
    r = pounce.preflight(HS071(), np.array([2.0, 3.0, 3.0, 2.0]), **HS071_BOUNDS)
    assert r.ok and not r.fatal
    assert r.n == 4 and r.m == 2
    assert r.n_bound_violations == 0
    assert np.isfinite(r.objective)
    # g1 = prod = 36 >= 25 ok; g2 = 4+9+9+4 = 26 != 40 -> violated
    assert r.n_con_violations == 1
    assert r.max_con_violation == pytest.approx(14.0)


def test_on_bound_start_warns_about_clamp():
    # The canonical HS071 start has x0 and x3 exactly on the lower bound.
    r = pounce.preflight(HS071(), np.array([1.0, 5.0, 5.0, 1.0]), **HS071_BOUNDS)
    assert r.ok
    assert r.n_on_bounds == 4  # x1, x2 are on the upper bound too
    assert r.n_clamp_moved == 4
    # two-sided [1, 5]: p_l = min(1e-2*1, 1e-2*4) = 0.01 at the lower bound,
    # p_u = min(1e-2*5, 1e-2*4) = 0.04 at the upper bound.
    assert r.max_clamp_move == pytest.approx(0.04, abs=1e-12)
    assert any("warm_start_bound_push" in w for w in r.warnings)
    assert r.verdict == "WARNINGS"


def test_nan_at_x0_is_fatal():
    r = pounce.preflight(DomainTrap(), np.array([0.0, 0.0]))
    assert r.fatal and not r.ok
    assert r.verdict == "FATAL"
    assert r.grad_nonfinite_count >= 1
    assert r.x0_all_zero
    assert any("Invalid_Number_Detected" in w for w in r.warnings)


def test_raising_callback_is_fatal_not_crashing():
    r = pounce.preflight(Raiser(), np.array([1.0]))
    assert r.fatal
    assert r.eval_errors and "ValueError" in r.eval_errors[0]


def test_bound_violation_reported_not_fatal():
    r = pounce.preflight(HS071(), np.array([-2.0, 3.0, 3.0, 2.0]), **HS071_BOUNDS)
    assert r.ok  # clampable, not fatal
    assert r.n_bound_violations == 1
    assert r.max_bound_violation == pytest.approx(3.0)
    assert r.n_clamp_moved >= 1


def test_unconstrained_problem():
    class Quad:
        def objective(self, x):
            return float(x @ x)

        def gradient(self, x):
            return 2.0 * x

    r = pounce.preflight(Quad(), np.array([1.0, -1.0]))
    assert r.ok and r.m == 0
    assert r.verdict == "CLEAN"


def test_report_dict_and_str():
    r = pounce.preflight(HS071(), np.array([2.0, 3.0, 3.0, 2.0]), **HS071_BOUNDS)
    d = r.to_dict()
    assert d["schema"] == "pounce.check-x0/v1"
    assert d["problem"] == {"n_vars": 4, "n_cons": 2}
    assert d["fatal"] is False
    text = str(r)
    assert "VERDICT" in text and "preflight" in text
