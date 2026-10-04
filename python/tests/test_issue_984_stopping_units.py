"""gh#984 (second pass): the convex IPM's verdict and iteration count must not
depend on the units of ``c``, and ``status == "optimal"`` must never sit beside
``residuals["kkt_error"] > tol``.

The generators are the issue's own (numpy RNG, so the instances are exactly the
ones reported); the sizes are smaller than the issue's 3328-week model to keep
the test fast.
"""

import numpy as np
import pytest
import scipy.sparse as sps

import pounce
from pounce.qp import solve_qp


def _production_lp(T, P=3, seed=9):
    rng = np.random.default_rng(seed)
    wk = np.arange(T)
    d = rng.uniform(80, 120, (P, T)) * (1 - 0.3 * np.sin(2 * np.pi * wk / 52))
    cost = np.array([[20.0], [26.0], [31.0]])[:P] * rng.uniform(0.95, 1.05, (P, T))
    hold, a = np.array([0.8, 1.0, 1.2])[:P], np.array([1.0, 1.5, 2.0])[:P]
    E = sps.eye(T, format="csr")
    Sh = sps.eye(T, k=-1, format="csr")
    Z = sps.csr_matrix((T, T))
    bal = [
        [E if k == i else (Sh - E if k == P + i else Z) for k in range(2 * P + 2)]
        for i in range(P)
    ]
    cap = [a[i] * E for i in range(P)] + [Z] * P + [-E, E]
    A = sps.bmat(bal + [cap], format="csr")
    b = np.r_[d.ravel(), np.full(T, 400.0)]
    c = np.r_[cost.ravel(), np.repeat(hold, T), np.full(T, 45.0), np.zeros(T)]
    return A, b, c


def _dispatch_lp(days, seed=4):
    rng = np.random.default_rng(seed)
    Pmax = np.array([1200, 1200, 1000, 800, 800, 600, 500, 400, 300, 200.0])
    cost = np.array([12, 13, 18, 22, 24, 30, 35, 45, 60, 90.0])
    R = np.array([120, 120, 200, 240, 240, 300, 250, 400, 300, 200.0])
    T = 24 * days
    hr = np.arange(T)
    load = 4200 + 1300 * np.sin(2 * np.pi * (hr % 24 - 8) / 24) + rng.normal(0, 80, T)
    D = (sps.eye(T, format="csr") - sps.eye(T, k=-1, format="csr"))[1:]
    Dg = sps.block_diag([D] * 10, format="csr")
    return dict(
        c=np.repeat(cost, T),
        A=sps.hstack([sps.eye(T)] * 10, format="csr"),
        b=load,
        G=sps.vstack([Dg, -Dg], format="csr"),
        h=np.tile(np.repeat(R, T - 1), 2),
        ub=np.repeat(Pmax, T),
    )


def test_item2_primal_feasibility_is_unit_invariant():
    A, b, c = _production_lp(208)
    for s in (1.0, 100.0, 1e-3):
        q = solve_qp(c=s * c, A=A.tocsc(), b=b, lb=np.zeros(A.shape[1]))
        assert q.status == "optimal"
        assert np.abs(A @ q.x - b).max() < 1e-8, s


def test_item3_optimal_never_beside_kkt_error_above_tol():
    P = np.diag([1.0, 1.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 0.0, -1.0])
    G = np.array([[-64.0, -74.0, -63.0, 1.0]])
    h = np.array([1e9 - 1021])
    ub = np.array([1e6, 1e6, 1e6, 2e9])
    for tol in (1e-6, 1e-8, 1e-10, 1e-12):
        r = solve_qp(P=P, c=c, G=G, h=h, lb=np.zeros(4), ub=ub, tol=tol)
        assert np.allclose(r.x[:3], [64.0, 74.0, 63.0], atol=1e-5)
        if r.status == "optimal":
            assert r.residuals["kkt_error"] <= tol, (tol, r.residuals)
        else:
            assert r.status == "optimal_inaccurate"
        assert "kkt_error_raw" in r.residuals


def test_item4_iteration_count_is_unit_invariant():
    d = _dispatch_lp(7)
    its = []
    for s in (1.0, 100.0, 1e-3):
        r = solve_qp(
            c=s * d["c"], A=d["A"], b=d["b"], G=d["G"], h=d["h"],
            lb=np.zeros(len(d["c"])), ub=d["ub"],
        )
        assert r.status == "optimal"
        its.append(r.iters)
    assert max(its) <= 1.5 * min(its) + 2, its

    A, b, c = _production_lp(600)
    its = []
    for s in (1.0, 1e-3):
        q = solve_qp(c=s * c, A=A.tocsc(), b=b, lb=np.zeros(A.shape[1]))
        assert q.status == "optimal"
        its.append(q.iters)
    assert max(its) <= 1.5 * min(its) + 2, its


def test_item5_per_minute_variance_weights_do_not_depend_on_the_scale():
    mu = np.array([3.0, 4.5, 6.0, 9.0, 14.0])
    sig = np.array([5.0, 8.0, 15.0, 20.0, 30.0])
    rho = np.array(
        [[1, .6, -.2, -.1, 0], [.6, 1, .3, .3, .2], [-.2, .3, 1, .3, .1],
         [-.1, .3, .3, 1, .5], [0, .2, .1, .5, 1]]
    )
    S = rho * np.outer(sig, sig)
    kw = dict(c=np.zeros(5), A=np.ones((1, 5)), b=np.array([1.0]),
              G=-mu[None, :], h=np.array([-10.0]), lb=np.zeros(5), ub=np.ones(5))
    w_ref = solve_qp(P=2 * S, tol=1e-12, **kw).x
    for scale in (1.0, 1e-4, 1e-4 / 252, 1e-4 / (252 * 390)):
        r = solve_qp(P=2 * S * scale, **kw)
        assert np.abs(r.x - w_ref).max() < 1e-6, scale


def _portfolio(scale):
    mu = np.array([3.0, 4.5, 6.0, 9.0, 14.0])
    sig = np.array([5.0, 8.0, 15.0, 20.0, 30.0])
    rho = np.array([[1, .6, -.2, -.1, 0], [.6, 1, .3, .3, .2], [-.2, .3, 1, .3, .1],
                    [-.1, .3, .3, 1, .5], [0, .2, .1, .5, 1]])
    S = rho * np.outer(sig, sig)
    return dict(P=2 * S * scale, c=np.zeros(5), A=np.ones((1, 5)), b=np.array([1.0]),
                G=-mu[None, :], h=np.array([-10.0]), lb=np.zeros(5), ub=np.ones(5))


@pytest.mark.parametrize("scale", [1.0, 1e3, 1e6, 1e9])
def test_review_optimal_never_beside_kkt_error_above_tol_cold_or_warm(scale):
    # gh#984 review item 12: the promise holds on the warm path too.
    kw = _portfolio(scale)
    cold = solve_qp(**kw)
    warm = solve_qp(**kw, warm_start=cold)
    for r in (cold, warm):
        res = r.residuals
        assert {"primal_infeasibility_raw", "dual_infeasibility_raw",
                "complementarity_raw", "kkt_error_raw"} <= set(res)
        if r.status == "optimal":
            assert res["kkt_error"] <= 1e-8, (scale, res)
        # item 9: the raw residual is not hidden. Where only the objective's
        # unit puts it within tol, scaling_warning says so.
        if r.status == "optimal" and res["kkt_error_raw"] > 1e-8:
            assert r.scaling_warning and "objective's unit" in r.scaling_warning


def test_review_well_scaled_kkt_error_is_not_reported_as_zero():
    # Item 9: per-entry floors; a model of ordinary magnitude reports its
    # residual (the second pass's global floors read 0.0 beside a raw 1.2e-9).
    A, b, c = _production_lp(52)
    r = solve_qp(c=c, A=A.tocsc(), b=b, lb=np.zeros(A.shape[1]))
    res = r.residuals
    assert r.status == "optimal"
    # The excuse on a row of magnitude ~1e3 is ~1e-11, nothing like the
    # global 5.7e-5 of the second pass.
    assert abs(res["primal_infeasibility"] - res["primal_infeasibility_raw"]) <= 1e-10, res
