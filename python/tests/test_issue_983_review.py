"""gh#983 review fixes, end to end through the Python front end.

* item 1: ``x*y == 0`` from ``(0.3, 0.3)`` -- the gh#884 retry is promoted and
  ``info["warnings"]`` describes the promoted run, not the discarded base
  attempt (which carried ``large_dual_scale ... 1.43e9``).
* item 6: ``find_minima``'s ``kkt_tol`` is relative to the problem's gradient
  scale, and an empty result caused by it is never silent.
"""
import warnings

import numpy as np
import pytest

import pounce


class _XY:
    def objective(self, v):
        return (v[0] - 1) ** 2 + (v[1] - 1) ** 2

    def gradient(self, v):
        return np.array([2 * (v[0] - 1), 2 * (v[1] - 1)])

    def constraints(self, v):
        return np.array([v[0] * v[1]])

    def jacobianstructure(self):
        return np.array([0, 0]), np.array([0, 1])

    def jacobian(self, v):
        return np.array([v[1], v[0]])

    def hessianstructure(self):
        return np.array([0, 1, 1]), np.array([0, 0, 1])

    def hessian(self, v, lam, of):
        return np.array([2 * of, lam[0], 2 * of])


def test_xy_promoted_retry_reports_its_own_warnings_only():
    p = pounce.Problem(n=2, m=1, problem_obj=_XY(), lb=[0, 0], ub=[10, 10],
                       cl=[0], cu=[0])
    p.add_option("print_level", 0)
    p.add_option("bound_relax_factor", 0.0)
    p.add_option("constr_viol_tol", 1e-8)
    x, info = p.solve(x0=np.array([0.3, 0.3]))
    assert info["status_msg"] == "Solve_Succeeded"
    assert info["final_unscaled_dual_inf"] < 1e-6
    assert abs(info["obj_val"] - 1.0) < 1e-6
    # f = 1 is the promoted retry's point; the base attempt stopped at the
    # origin with f = 2.
    assert abs(x[0] * x[1]) < 1e-8
    stale = [w for w in info["warnings"]
             if w.startswith(("large_dual_scale:", "unscaled_dual_inf_above_acceptable:"))]
    assert stale == [], info["warnings"]


def _wavy(S):
    f = lambda x: S * float(np.sum(np.sin(3 * x) + 0.1 * x**2))
    g = lambda x: S * (3 * np.cos(3 * x) + 0.2 * x)
    return f, g


@pytest.mark.parametrize("S", [1.0, 1e12, 1e14])
def test_find_minima_kkt_tol_is_relative_to_the_gradient_scale(S):
    # Measured with the absolute bound: at S = 1e12 and 1e14 eleven of twelve
    # solves were rejected (the converged residuals are 3e-4 and 3e-2 against
    # gradients of ~3e12 / 3e14 -- stationary to sixteen digits) and the one
    # "minimum" returned was the unsolved seed.
    f, g = _wavy(S)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        r = pounce.find_minima(f, np.array([0.3, -0.2]), method="multistart",
                               jac=g, bounds=[(-5, 5)] * 2, n_minima=4,
                               max_solves=12, seed=1)
    assert len(r) == 4
    assert r.n_kkt_rejected == 0


# The acceptable-level test opened so wide that every solve stops at iteration
# 0 and returns its own start as Solved_To_Acceptable_Level: a solver success
# at a non-stationary point, made deterministic. (These tests used to stub
# `_minima.minimize`; the search runs in Rust now, gh#966, so they drive the
# real veto instead.)
_ACCEPT_ANYTHING = {
    "acceptable_tol": 100.0,
    "acceptable_compl_inf_tol": 100.0,
    "acceptable_obj_change_tol": 1e20,
    "acceptable_iter": 1,
}


def test_find_minima_warns_when_kkt_tol_empties_the_result():
    f, g = _wavy(1.0)
    with pytest.warns(RuntimeWarning, match="kkt_tol"):
        r = pounce.find_minima(f, np.array([0.3, -0.2]), method="multistart",
                               jac=g, bounds=[(-5, 5)] * 2, n_minima=2,
                               max_solves=3, seed=0, options=_ACCEPT_ANYTHING)
    assert len(r) == 0
    assert r.n_kkt_rejected == 3


def test_find_minima_relative_bound_still_rejects_a_non_stationary_point():
    """The bound is relative to a *gradient* scale, never to the residual
    itself: a returned start with |grad f| ~ 2 against a gradient scale of
    ~2 is rejected, although the residual is "small" relative to itself. The
    genuine minima of the same model at gradient scales up to 1e14 are
    accepted (`test_find_minima_kkt_tol_is_relative_to_the_gradient_scale`);
    the bound's arithmetic, at a dual scale of 1e6 included, is
    `kkt_bound_is_relative_to_the_gradient_scale` in pounce-minima. (A
    scaled-up copy of this fixture does not reach the veto: gradient scaling
    engages and the solve converges for real.)"""
    f, g = _wavy(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = pounce.find_minima(f, np.array([0.3, -0.2]), method="multistart",
                               jac=g, bounds=[(-5, 5)] * 2, n_minima=2,
                               max_solves=4, seed=0, options=_ACCEPT_ANYTHING)
    assert len(r) == 0
    assert r.n_kkt_rejected == r.n_solves == 4
