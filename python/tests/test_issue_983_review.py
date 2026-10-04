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


def test_find_minima_warns_when_kkt_tol_empties_the_result():
    from pounce import _minima

    class R(dict):
        __getattr__ = dict.get

    def fake_minimize(fun, x0, **kw):
        return R(x=np.asarray(x0, float), fun=0.0, success=True, nit=1,
                 info={"final_unscaled_dual_inf": 0.5,
                       "final_unscaled_dual_scale": 1.0}, message="ok")

    orig = _minima.minimize
    _minima.minimize = fake_minimize
    try:
        with pytest.warns(RuntimeWarning, match="kkt_tol"):
            r = pounce.find_minima(lambda x: 0.0, np.zeros(1), method="multistart",
                                   bounds=[(-1, 1)], n_minima=2, max_solves=3,
                                   seed=0)
        assert len(r) == 0
        assert r.n_kkt_rejected == 3
    finally:
        _minima.minimize = orig


def test_find_minima_relative_bound_still_rejects_a_non_stationary_point():
    """The relative bound is relative to a *gradient* scale, never to the
    residual itself: a residual of 0.5 against a gradient scale of 1 is
    rejected, the same residual against 1e6 is not."""
    from pounce import _minima

    class R(dict):
        __getattr__ = dict.get

    def make(scale):
        def fake_minimize(fun, x0, **kw):
            return R(x=np.zeros(1), fun=0.0, success=True, nit=1,
                     info={"final_unscaled_dual_inf": 0.5,
                           "final_unscaled_dual_scale": scale}, message="ok")
        return fake_minimize

    orig = _minima.minimize
    try:
        _minima.minimize = make(1.0)
        ctx = _minima._Context(lambda x: 0.0, None, None, None, None, None, 1e-6)
        assert ctx.solve(ctx.fun, np.zeros(1)).success is False
        _minima.minimize = make(1e6)
        ctx = _minima._Context(lambda x: 0.0, None, None, None, None, None, 1e-6)
        assert ctx.solve(ctx.fun, np.zeros(1)).success is True
    finally:
        _minima.minimize = orig
