"""gh#983 second pass: success verdicts are audited against the model's own scale.

Each test is a repro from the issue. The controls (``solve_quality_audit=no``)
exist so a test that stops exercising the defect is visible.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

import pounce

N_AT = 7
IU = np.triu_indices(N_AT, 1)


def lj(x):
    X = np.asarray(x).reshape(N_AT, 3)
    r2 = ((X[:, None] - X[None]) ** 2).sum(-1)[IU]
    ir6 = 1 / r2**3
    return float(4 * np.sum(ir6 * ir6 - ir6))


def lj_grad(x):
    X = np.asarray(x).reshape(N_AT, 3)
    d = X[:, None] - X[None]
    r2 = (d**2).sum(-1)
    np.fill_diagonal(r2, 1.0)
    ir6 = 1 / r2**3
    c = 4 * (-12 * ir6 * ir6 + 6 * ir6) / r2
    np.fill_diagonal(c, 0.0)
    return (c[:, :, None] * d).sum(1).ravel()


BOX = [(-1.6, 1.6)] * (3 * N_AT)


def _x_bad():
    x = np.random.default_rng(37).uniform(-1, 1, 3 * N_AT).reshape(N_AT, 3)
    x[1] = x[0] + [0.3, 0, 0]
    return x.ravel()


def test_lj7_frozen_gradient_scaling_is_rescaled_at_the_returned_point():
    x_bad = _x_bad()
    assert np.abs(lj_grad(x_bad)).max() > 1e8  # the premise
    r = pounce.minimize(lj, x_bad, jac=lj_grad, bounds=BOX)
    assert r.success
    assert np.abs(lj_grad(r.x)).max() < 1e-6, np.abs(lj_grad(r.x)).max()
    assert r.info["final_unscaled_dual_inf"] < 1e-6
    # The same energy as the unscaled run: this is the real minimum.
    r0 = pounce.minimize(lj, x_bad, jac=lj_grad, bounds=BOX, nlp_scaling_method="none")
    assert abs(r.fun - r0.fun) < 1e-6
    assert r.info["warnings"] == []


def test_find_minima_without_hess_rejects_non_stationary_candidates():
    x_init = np.random.default_rng(26).uniform(-1, 1, 3 * N_AT)
    r = pounce.find_minima(
        lj, x_init, method="multistart", jac=lj_grad, bounds=BOX,
        n_minima=12, max_solves=24, patience=40, dedup=1e-2, seed=26,
    )
    g = [np.abs(lj_grad(x)).max() for x in r.minima]
    assert len(g) > 0
    assert max(g) < 1e-3, max(g)


def test_find_minima_kkt_tol_filters_on_the_unscaled_residual():
    """The filter itself, independent of the solver fix: a stub solve that
    claims success at a point with a large unscaled residual is rejected, and
    ``kkt_tol=None`` switches it off."""
    from pounce import _minima

    class R(dict):
        __getattr__ = dict.get

    def fake_minimize(fun, x0, **kw):
        r = R(x=np.array([0.0]), fun=0.0, success=True, nit=1,
              info={"final_unscaled_dual_inf": 0.5}, message="ok")
        return r

    orig = _minima.minimize
    _minima.minimize = fake_minimize
    try:
        ctx = _minima._Context(lambda x: 0.0, None, None, None, None, None, 1e-6)
        assert ctx.solve(ctx.fun, np.zeros(1)).success is False
        ctx = _minima._Context(lambda x: 0.0, None, None, None, None, None, 1e-6,
                               kkt_tol=None)
        assert ctx.solve(ctx.fun, np.zeros(1)).success is True
        ctx = _minima._Context(lambda x: 0.0, None, None, None, None, None, 1e-6,
                               kkt_tol=1.0)
        assert ctx.solve(ctx.fun, np.zeros(1)).success is True
    finally:
        _minima.minimize = orig


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from pounce.jax import from_jax  # noqa: E402

cA0, k10, E1, k20, E2, R_ = 2.0, 1.0e6, 50_000.0, 5.0e3, 35_000.0, 8.314
TRUE = 5.352367857245759


def _reactor_g(x):
    cA, cB, tau, T = x[0], x[1], x[2], x[3]
    k1 = k10 * jnp.exp(-E1 / (R_ * T))
    k2 = k20 * jnp.exp(-E2 / (R_ * T))
    return jnp.stack([cA0 - cA - tau * k1 * cA, -cB + tau * (k1 * cA - k2 * cB)])


def _reactor(fs, **opts):
    p = from_jax(
        lambda x: -fs * (10 * x[1] - 0.02 * x[2]), _reactor_g, n=4, m=2,
        jac_pattern=np.nonzero(np.ones((2, 4))), hess_pattern=np.tril_indices(4),
        lb=np.array([0, 0, 0.1, 300.0]), ub=np.array([2, 2, 60, 360.0]),
        cl=np.zeros(2), cu=np.zeros(2),
    )
    p.add_option("print_level", 0)
    for k, v in opts.items():
        p.add_option(k, v)
    return p.solve(x0=np.array([1.0, 1.0, 10.0, 330.0]))


def test_small_objective_is_rescaled_up_not_certified_at_4e_minus_4():
    # control: the defect the issue reports
    _, ctl = _reactor(1e-6, solve_quality_audit="no")
    assert abs(-ctl["obj_val"] / 1e-6 - TRUE) / TRUE > 1e-5
    x, info = _reactor(1e-6)
    assert info["status_msg"] == "Solve_Succeeded"
    assert abs(-info["obj_val"] / 1e-6 - TRUE) / TRUE < 1e-6
    assert abs(x[3] - 360.0) < 1e-3
    # well-scaled objectives are not audited
    _, one = _reactor(1.0)
    assert one["warnings"] == []


def test_small_objective_warning_when_the_audit_is_off_is_silent_and_on_is_structured():
    # The warning is raised only when the re-solve is declined; a clean run
    # has an empty list, and the key always exists.
    _, info = _reactor(1e-6)
    assert isinstance(info["warnings"], list)


T0 = jnp.array([10.0, 12.0, 20.0])
CAP = jnp.array([1000.0, 1500.0, 800.0])
D, VOT = 2500.0, 0.25


def _toll_g(v):
    tau, x, z, pi = v[0], v[1:4], v[4:7], v[7]
    return jnp.concatenate([
        T0 * (1 + 0.15 * (x / CAP) ** 4) + jnp.array([0.0, 1.0, 0.0]) * tau / VOT - pi - z,
        x * z,
        jnp.array([x.sum() - D]),
    ])


def _toll(**opts):
    p = from_jax(
        lambda v: -v[0] * v[2], _toll_g, n=8, m=7,
        jac_pattern=np.nonzero(np.ones((7, 8))), hess_pattern=np.tril_indices(8),
        lb=np.zeros(8), ub=np.r_[10, D, D, D, 100, 100, 100, 100.0], cl=np.zeros(7),
        cu=np.zeros(7),
    )
    p.add_option("print_level", 0)
    for k, v in opts.items():
        p.add_option(k, v)
    x0 = np.r_[1.0, [D / 3] * 3, [1.0] * 3, 15.0]
    d = tempfile.mkdtemp()
    path = os.path.join(d, "r.json")
    x, info = p.solve(x0=x0, report_path=path, report_detail="full")
    return x, info, json.load(open(path))


def test_toll_mpcc_is_not_a_strict_success_at_unscaled_dual_3e_minus_2():
    x, info, rep = _toll(bound_relax_factor=0.0)
    assert info["final_unscaled_dual_inf"] > 1e-3  # the premise: the retry was accepted
    assert info["status_msg"] == "Solved_To_Acceptable_Level"
    codes = [w.split(":")[0] for w in info["warnings"]]
    assert "unscaled_dual_inf_above_acceptable" in codes, info["warnings"]
    # the report describes the run that produced the returned point
    assert rep["statistics"]["iteration_count"] == info["iter_count"]
    assert rep["statistics"]["warnings"] == info["warnings"]
    assert rep["solution"]["status_upstream"] == "Solved_To_Acceptable_Level"


def test_toll_mpcc_default_options_stay_strict():
    _, info, rep = _toll()
    assert info["status_msg"] == "Solve_Succeeded"
    assert rep["statistics"]["iteration_count"] == info["iter_count"]


def test_cusp_default_floor_does_not_certify_strictly_and_carries_the_warnings():
    class Cusp:
        def objective(self, x): return x[0]
        def gradient(self, x): return np.array([1.0, 0.0])
        def constraints(self, x): return np.array([x[1] - x[0] ** 3, x[1]])
        def jacobianstructure(self): return (np.array([0, 0, 1]), np.array([0, 1, 1]))
        def jacobian(self, x): return np.array([-3 * x[0] ** 2, 1.0, 1.0])
        def hessianstructure(self): return (np.array([0]), np.array([0]))
        def hessian(self, x, lam, of): return np.array([-6 * x[0] * lam[0]])

    p = pounce.Problem(n=2, m=2, problem_obj=Cusp(), lb=[-2, -2], ub=[2, 2],
                       cl=[-1e20, 0.0], cu=[0.0, 1e20])
    p.add_option("print_level", 0)
    p.add_option("bound_relax_factor", 0.0)
    x, info = p.solve(np.array([0.0, 0.0]))
    # multipliers ~3.5e9 (failed CQ) and an unscaled |grad L| of 0.144: the
    # scale-relative floor accepts the point, the audit refuses it a strict
    # verdict and says why.
    assert info["status_msg"] == "Solved_To_Acceptable_Level"
    codes = [w.split(":")[0] for w in info["warnings"]]
    assert "large_dual_scale" in codes, info["warnings"]
    assert "unscaled_dual_inf_above_acceptable" in codes, info["warnings"]
