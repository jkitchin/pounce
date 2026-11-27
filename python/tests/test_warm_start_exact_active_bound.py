"""gh#1269: ``WarmStart.from_info`` must not refuse the exact multipliers it
captured, and ``recentering="none"`` must report the seeds it used.

The model has an active upper bound whose multiplier (2000) is far above
the converged barrier. The warm-start push holds that bound's slack at
``bound_push = 1e-9``, so ``z * s`` there read as corrupted complementarity
against ``mu_init = 1e-9`` and the block was rejected: the restart started
at ``inf_du = 66.7`` and took 4 iterations where 1-2 suffice.
"""

import numpy as np

import pounce


class _ActiveBound:
    def objective(self, x):
        return 1000 * ((x[0] - 2) ** 2 + (x[1] - 2) ** 2)

    def gradient(self, x):
        return np.array([2000 * (x[0] - 2), 2000 * (x[1] - 2)])

    def constraints(self, x):
        return np.array([x[0] + x[1]])

    def jacobian(self, x):
        return np.array([1.0, 1.0])

    def jacobianstructure(self):
        return (np.array([0, 0]), np.array([0, 1]))

    def hessianstructure(self):
        return (np.array([0, 1]), np.array([0, 1]))

    def hessian(self, x, lam, of):
        return np.array([2000.0 * of, 2000.0 * of])


def _prob():
    p = pounce.Problem(2, 1, _ActiveBound(), lb=[-10, 0], ub=[0.5, 10],
                       cl=[2], cu=[2])
    p.add_option("tol", 1e-10)
    p.add_option("print_level", 0)
    return p


def _cold():
    x, info = _prob().solve(x0=np.array([0.0, 0.0]))
    assert abs(x[0] - 0.5) < 1e-6
    assert abs(info["mult_x_U"][0] - 2000.0) < 1e-3
    return x, info


def test_from_info_recipe_keeps_exact_bound_multipliers():
    x, info = _cold()
    ws = pounce.WarmStart.from_info(x, info)
    _, warm = _prob().solve(warm_start=ws)
    _, legacy = _prob().solve(
        warm_start=pounce.WarmStart.from_info(x, info, recentering="none"))
    report = warm["warm_start"]
    assert report["bound_duals"] != "rejected", report
    assert report["bound_duals_rejected"] == 0
    assert report["dual_residual"] < 1e-6, report
    assert warm["iter_count"] <= legacy["iter_count"]


def test_recentering_none_reports_used_seeds():
    x, info = _cold()
    ws = pounce.WarmStart.from_info(x, info, recentering="none")
    _, warm = _prob().solve(warm_start=ws)
    report = warm["warm_start"]
    assert report["recentering_disabled"]
    assert report["bound_duals"] == "used", report
    assert report["eq_duals"] == "used", report
    # Not measured under the kill switch: no extra model evaluations.
    assert np.isnan(report["dual_residual"])
