"""Adversary cross-check: solve_qp_batch over a MIXED batch of LPs (P=None) and QPs,
including a primal-degenerate LP vertex and an LP with a non-unique optimal face.
Family: batch   Class: heterogeneous P-presence (LP items interleaved with QP items)
Source: oracles are scipy.optimize.linprog (HiGHS) for LP items and cvxpy/Clarabel
        for QP items; plus per-item single solve_qp. Closed-form values:
        LP0 (Dantzig & Thapa, Linear Programming 1, ex. 1.1-style):
            max 3x+5y s.t. x<=4, 2y<=12, 3x+2y<=18, x,y>=0 -> 36 at (2,6)
            (Hillier & Lieberman, Intro. to OR, Wyndor Glass, sec 3.1).
        LP1 degenerate vertex: min -x-y s.t. x+y<=1, x<=1, y<=1, x+2y<=1, 2x+y<=1,
            x,y>=0 -> -2/3 at (1/3,1/3) [x+2y and 2x+y both tight; 2 vars].
        LP2 non-unique face: min x+y s.t. x+y>=1, 0<=x,y<=1 -> 1 (objective only).
        QP items: random strictly convex with box + one equality (oracle only).
Known optimal: LP0 -36, LP1 -2/3, LP2 1.
"""
import time, numpy as np
from pounce import solve_qp_batch, solve_qp
from scipy.optimize import linprog
import cvxpy as cp
rng = np.random.default_rng(7)
items, oracle, known = [], [], []
# LP0 Wyndor
items.append(dict(c=[-3., -5.], G=[[1, 0], [0, 2], [3, 2]], h=[4, 12, 18], lb=[0, 0])); known.append(-36.0)
# LP1 degenerate
items.append(dict(c=[-1., -1.], G=[[1, 1], [1, 0], [0, 1], [1, 2], [2, 1]], h=[1, 1, 1, 1, 1], lb=[0, 0])); known.append(-2 / 3)
# LP2 face
items.append(dict(c=[1., 1.], G=[[-1, -1]], h=[-1], lb=[0, 0], ub=[1, 1])); known.append(1.0)
for k in range(3):
    n = 5; M = rng.normal(size=(n, n)); P = M @ M.T + 0.5 * np.eye(n)
    items.append(dict(P=P, c=rng.normal(size=n), A=np.ones((1, n)), b=[1.0], lb=-np.ones(n), ub=np.ones(n))); known.append(None)
# interleave: QP, LP, QP, LP, QP, LP
order = [3, 0, 4, 1, 5, 2]; items = [items[i] for i in order]; known = [known[i] for i in order]
t0 = time.perf_counter(); rb = solve_qp_batch(items); tb = time.perf_counter() - t0
t0 = time.perf_counter(); rs = [solve_qp(**it) for it in items]; ts = time.perf_counter() - t0
to = 0.0; ok = True
for i, it in enumerate(items):
    t1 = time.perf_counter()
    if "P" not in it:
        n = len(it["c"]); bnds = list(zip(it.get("lb", [None] * n), it.get("ub", [None] * n)))
        o = linprog(it["c"], A_ub=it["G"], b_ub=it["h"], bounds=bnds, method="highs"); oval = o.fun
    else:
        x = cp.Variable(len(it["c"]))
        pr = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, it["P"]) + it["c"] @ x), [it["A"] @ x == it["b"], x >= it["lb"], x <= it["ub"]])
        pr.solve(solver=cp.CLARABEL); oval = pr.value
    to += time.perf_counter() - t1
    eb, es = abs(rb[i].obj - oval) / max(1, abs(oval)), abs(rs[i].obj - oval) / max(1, abs(oval))
    kn = "" if known[i] is None else f" known_err={abs(rb[i].obj-known[i]):.1e}"
    xd = np.abs(np.asarray(rb[i].x) - np.asarray(rs[i].x)).max()
    print(f"item{i} {'LP' if 'P' not in it else 'QP'} batch={rb[i].status}/{rb[i].obj:.10f} single={rs[i].status}/{rs[i].obj:.10f} oracle={oval:.10f} err_b={eb:.1e} err_s={es:.1e} |xb-xs|={xd:.1e}{kn}")
    good = rb[i].status == "optimal" and eb < 1e-6 and (known[i] is None or abs(rb[i].obj - known[i]) < 1e-6)
    # x parity only meaningful where the optimum is unique (not LP2's face)
    if not (known[i] == 1.0) and xd > 1e-5: good = False
    ok &= good
print(f"t_batch={tb:.4f}s t_single={ts:.4f}s t_oracle={to:.4f}s")
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
