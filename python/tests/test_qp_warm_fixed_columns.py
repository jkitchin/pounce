"""gh#988 item 2: a warm start on an LP whose columns are fixed (lb == ub).

The model is the issue's seeded multi-week production-distribution LP (2
products, 3 plants, 6 warehouses, K customers, T weeks): a base plan, a
scenario whose demand peak moves earlier, and then the scenario with the
production and overtime of the first FREEZE weeks pinned at the base plan.

Before the second pass the warm leg (direct infeasible-start method) ran to
the 199-iteration limit and the call returned that, or after the first-pass
fallback a cold solve on top of it (231 iterations against 32 cold).  Two
defects were behind it: pinned columns were left in as the row pair
`x <= v, -x <= -v` (no interior), and the direct driver's absolute residual
test could not be met on a primal residual that floors at `5e-8` on data of
order `1e4`, so a converged iterate ran on to the limit.  Reduced instance
(T=26, K=10, about 3500 columns, 135 pinned) takes a second.
"""

import numpy as np
import pytest
import scipy.sparse as sps

pounce = pytest.importorskip("pounce")
from pounce.qp import solve_qp  # noqa: E402


def planning_data(T, K, seed=2026, lanes=3, peak=0.6, peak_week=44):
    rng = np.random.default_rng(seed)
    P_xy = np.array([[1.5, 4.5], [5.0, 1.0], [8.5, 4.0]]); W_xy = np.array([[1, 1.5], [3, 5], [4.5, 3], [6.5, 5], [7, 1.5], [9, 2.5]])
    K_xy = rng.uniform([0, 0], [10, 6], size=(K, 2))
    dPW = np.linalg.norm(P_xy[:, None] - W_xy[None], axis=2); dWK = np.linalg.norm(W_xy[:, None] - K_xy[None], axis=2)
    near = np.sort(np.argsort(dWK, axis=0)[:lanes], axis=0); L = [(int(w), k) for k in range(K) for w in near[:, k]]
    size = rng.lognormal(np.log(40), 0.5, K); mix = np.array([0.6, 0.4]); wk = np.arange(T)
    season = 1 + peak*np.exp(-0.5*((wk - peak_week)/4.0)**2) - 0.1*np.cos(2*np.pi*wk/52)
    d = mix[:, None, None]*size[None, :, None]*season[None, None, :]*rng.lognormal(0, 0.10, (2, K, T))
    whcap = np.array([450, 600, 750, 600, 450, 400.0])
    return dict(K=K, T=T, d=d, L=L, cap=np.array([1500.0, 1300, 1200]), hrs=np.array([[0.9, 1.0, 1.1], [1.3, 1.2, 1.4]]),
                pcost=np.array([[0.42, 0.40, 0.45], [0.60, 0.62, 0.58]]), ot_max=0.25*np.array([1500.0, 1300, 1200]),
                ot_cost=np.array([0.090, 0.085, 0.095]), hold=np.array([0.003, 0.004]), whcap=whcap,
                I0=0.3*whcap[None, :]*mix[:, None], cPW=0.02 + 0.012*dPW, cWK=0.01 + 0.012*dWK, pen=np.array([2.0, 2.5]))
def plan_lp(D, G=2, P=3, W=6):
    K, T, L = D["K"], D["T"], D["L"]
    shapes = {"x": (G, P, T), "o": (P, T), "s": (G, P, W, T), "v": (G, len(L), T), "I": (G, W, T), "u": (G, K, T)}
    ix, n = {}, 0
    for k, s in shapes.items(): ix[k] = n + np.arange(int(np.prod(s))).reshape(s); n += int(np.prod(s))
    c = np.zeros(n); c[ix["x"]] = D["pcost"][:, :, None]; c[ix["o"]] = D["ot_cost"][:, None]; c[ix["s"]] = D["cPW"][None, :, :, None]
    c[ix["v"]] = np.array([D["cWK"][w, k] for (w, k) in L])[None, :, None]; c[ix["I"]] = D["hold"][:, None, None]; c[ix["u"]] = D["pen"][:, None, None]
    ub = np.full(n, 1e4); ub[ix["o"]] = D["ot_max"][:, None]
    eq, b = [], []
    for g in range(G):
        for p in range(P):
            for t in range(T): eq.append((np.r_[ix["x"][g, p, t], ix["s"][g, p, :, t]], np.r_[1.0, -np.ones(W)])); b.append(0.0)
    for g in range(G):
        for w in range(W):
            lw = [l for l, (ww, _) in enumerate(L) if ww == w]
            for t in range(T):
                cols = np.r_[ix["s"][g, :, w, t], ix["v"][g, lw, t], ix["I"][g, w, t]]; vals = np.r_[np.ones(P), -np.ones(len(lw)), -1.0]
                if t: cols, vals = np.r_[cols, ix["I"][g, w, t - 1]], np.r_[vals, 1.0]
                eq.append((cols, vals)); b.append(-D["I0"][g, w] if t == 0 else 0.0)
    for g in range(G):
        for k in range(K):
            lk = [l for l, (_, kk) in enumerate(L) if kk == k]
            for t in range(T): eq.append((np.r_[ix["v"][g, lk, t], ix["u"][g, k, t]], np.ones(len(lk) + 1))); b.append(D["d"][g, k, t])
    ineq, h = [], []
    for p in range(P):
        for t in range(T): ineq.append((np.r_[ix["x"][:, p, t], ix["o"][p, t]], np.r_[D["hrs"][:, p], -1.0])); h.append(D["cap"][p])
    for w in range(W):
        for t in range(T): ineq.append((ix["I"][:, w, t], np.ones(G))); h.append(D["whcap"][w])
    for g in range(G):
        for w in range(W): ineq.append((np.array([ix["I"][g, w, T - 1]]), np.array([-1.0]))); h.append(-D["I0"][g, w])
    def mat(rows):
        r = np.concatenate([np.full(len(cl), i) for i, (cl, _) in enumerate(rows)])
        return sps.csc_matrix((np.concatenate([v for _, v in rows]), (r, np.concatenate([cl for cl, _ in rows]))), shape=(len(rows), n))
    return dict(c=c, A=mat(eq), b=np.array(b), G=mat(ineq), h=np.array(h), lb=np.zeros(n), ub=ub), ix


@pytest.fixture(scope="module")
def instance():
    T, K, FREEZE, PW, SHIFT = 26, 10, 15, 20, 4
    base, ix = plan_lp(planning_data(T, K, peak_week=PW))
    q = solve_qp(**base)
    scen, _ = plan_lp(planning_data(T, K, peak_week=PW - SHIFT))
    r_s = solve_qp(**scen)
    assert q.status == "optimal" and r_s.status == "optimal"
    fixed = dict(scen)
    fixed["lb"] = scen["lb"].copy()
    fixed["ub"] = scen["ub"].copy()
    i = np.r_[ix["x"][..., :FREEZE].ravel(), ix["o"][..., :FREEZE].ravel()]
    fixed["lb"][i] = fixed["ub"][i] = q.x[i]
    cold = solve_qp(**fixed)
    assert cold.status == "optimal"
    return dict(fixed=fixed, q=q, r_s=r_s, cold=cold, pinned=i)


def _warm_scenario_optimum(r_s):
    return {k: np.asarray(getattr(r_s, k)) for k in ("x", "y", "z", "z_lb", "z_ub") if getattr(r_s, k, None) is not None}


@pytest.mark.parametrize("which", ["scenario_optimum", "x_only", "base_plan"])
def test_warm_start_with_fixed_columns_is_optimal_and_not_much_slower_than_cold(instance, which):
    ws = _warm_scenario_optimum(instance["r_s"])
    warm = {"scenario_optimum": ws, "x_only": {"x": ws["x"]}, "base_plan": instance["q"]}[which]
    cold = instance["cold"]
    r = solve_qp(**instance["fixed"], warm_start=warm)
    assert r.status == "optimal", (r.status, r.iters)
    assert r.obj == pytest.approx(cold.obj, rel=1e-8)
    # The pins are honoured exactly and the answer is the cold one.
    i = instance["pinned"]
    np.testing.assert_allclose(r.x[i], instance["fixed"]["lb"][i], atol=1e-8)
    # A warm start that cannot help may cost a little; it may not cost the
    # iteration limit (199) that the stalled direct leg used to burn.
    assert r.iters <= cold.iters + 25, (which, r.iters, cold.iters)
