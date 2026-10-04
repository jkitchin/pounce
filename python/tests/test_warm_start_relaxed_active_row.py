"""gh#988 item 3: warm start after a parameter change that relaxes an
active inequality must not cost more than a cold start.

Two-product pricing under a capacity row ``d1 + d2 <= K``. Solve at
K = 14, warm-start K = 16.  The capacity row stays active but its slack
at the carried point is 2 wide, while the carried multiplier is 1.41:
the seed is coherent with stationarity and incoherent with the slack.
The old default capped the slack multiplier it rebuilt at ten times
``mu / slack`` (a barrier-sized value), left ``inf_du`` at 1.41 with a
tiny barrier, and spent 49 iterations (7 cold) in restoration cycles.
"""

import numpy as np
import pytest

pounce = pytest.importorskip("pounce")

A = np.array([1000.0, 800.0])
EPS = np.array([2.0, 2.5])
CU = np.array([4.0, 3.0])


class Pricing:
    def objective(self, p):
        return -np.sum((p - CU) * A * p ** (-EPS))

    def gradient(self, p):
        return -(A * p ** (-EPS) - (p - CU) * A * EPS * p ** (-EPS - 1))

    def constraints(self, p):
        return np.array([np.sum(A * p ** (-EPS))])

    def jacobianstructure(self):
        return np.array([0, 0]), np.array([0, 1])

    def jacobian(self, p):
        return -A * EPS * p ** (-EPS - 1)

    def hessianstructure(self):
        return np.array([0, 1]), np.array([0, 1])

    def hessian(self, p, lam, of):
        d2f = -(-2 * A * EPS * p ** (-EPS - 1) + (p - CU) * A * EPS * (EPS + 1) * p ** (-EPS - 2))
        return of * d2f + lam[0] * A * EPS * (EPS + 1) * p ** (-EPS - 2)


def _problem(k):
    pr = pounce.Problem(n=2, m=1, problem_obj=Pricing(), lb=CU + 0.01, ub=[15.0, 15.0], cl=[-1e20], cu=[k])
    pr.add_option("print_level", 0)
    return pr


def test_warm_start_after_relaxing_an_active_row_is_no_worse_than_cold():
    x14, i14 = _problem(14.0).solve(x0=np.array([10.0, 10.0]))
    assert i14["status_msg"] == "Solve_Succeeded"
    ws = pounce.WarmStart.from_info(x14, i14)

    xc, ic = _problem(16.0).solve(x0=np.array([10.0, 10.0]))
    xw, iw = _problem(16.0).solve(warm_start=ws)

    assert ic["status_msg"] == iw["status_msg"] == "Solve_Succeeded"
    assert iw["obj_val"] == pytest.approx(ic["obj_val"], rel=1e-8)
    np.testing.assert_allclose(xw, xc, rtol=1e-5)
    assert iw["iter_count"] <= ic["iter_count"], (iw["iter_count"], ic["iter_count"])
