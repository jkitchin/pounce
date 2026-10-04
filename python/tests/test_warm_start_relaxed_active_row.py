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


@pytest.fixture(scope="module")
def ws14():
    x14, i14 = _problem(14.0).solve(x0=np.array([10.0, 10.0]))
    assert i14["status_msg"] == "Solve_Succeeded"
    return pounce.WarmStart.from_info(x14, i14)


# gh#988 (review): the K sweep the reviewer ran. Tightened (12, 8: the
# carried point violates the new capacity), nearly unchanged, relaxed a
# little (16, 20: closing move), relaxed past the release (31+) and far
# (100, 1e4: closing the slack would start the solve 86 infeasible).
@pytest.mark.parametrize("k", [8.0, 12.0, 13.9, 14.5, 16.0, 20.0, 25.0, 29.0, 31.0, 40.0, 100.0, 1e4])
def test_warm_start_is_no_worse_than_cold_across_the_capacity_sweep(ws14, k):
    xc, ic = _problem(k).solve(x0=np.array([10.0, 10.0]))
    xw, iw = _problem(k).solve(warm_start=ws14)
    assert ic["status_msg"] == iw["status_msg"] == "Solve_Succeeded"
    assert iw["obj_val"] == pytest.approx(ic["obj_val"], rel=1e-8, abs=1e-8)
    # Measured: warm <= cold at every K (K=12 was 24 against 6, K=100 11 vs 8).
    assert iw["iter_count"] <= ic["iter_count"], (k, iw["iter_count"], ic["iter_count"])


def test_slack_closing_is_capped_and_the_diagnostics_describe_the_start(ws14):
    # K = 16: a 2-wide slack against |s| = 14 is closed; the start is then
    # infeasible by about the closed width, and the diagnostics say so.
    _, iw = _problem(16.0).solve(warm_start=ws14)
    d = iw["warm_start"]
    assert d["slacks_closed"] == 1 and not d["slack_close_reverted"]
    assert 1.0 < d["primal_residual"] < 3.0, d
    # K = 100: an 86-wide slack is not closed (it used to be, starting the
    # solve 86 infeasible while primal_residual reported 0.0).
    _, iw = _problem(100.0).solve(warm_start=ws14)
    d = iw["warm_start"]
    assert d["slacks_closed"] == 0, d
    assert d["primal_residual"] < 1e-6, d


def test_user_set_recentering_wins_over_the_warm_start(ws14):
    # gh#988 (review) item 4: WarmStart.options() carries
    # warm_start_recentering="residual" and used to overwrite this.
    pr = _problem(16.0)
    pr.add_option("warm_start_recentering", "none")
    _, info = pr.solve(warm_start=ws14)
    assert info["warm_start"]["recentering_disabled"] is True
    # ...and without a user setting the WarmStart's own choice applies.
    _, info = _problem(16.0).solve(warm_start=ws14)
    assert info["warm_start"]["recentering_disabled"] is False
