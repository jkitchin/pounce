"""Adversary cross-check: auto-routing on a convex QP with a FIXED variable (lb==ub)
and a DUPLICATED (scaled) equality row -- presolve-shaped structure under auto.
Family: autoroute   Class: QP -> specialized engine; routing transparency vs forced nlp
Source: closed-form Euclidean projection (KKT): min 1/2||x-y||^2, y=(1.4,0.3,0.6,0.9),
        s.t. x1+x2+x3+x4 = 2 and 2(x1+x2+x3+x4) = 4 (rank 1), x4 in [0.5,0.5],
        0<=x1..x3<=1.  x4=0.5; x_{1:3} = clip(y-tau,0,1), tau=0.2 =>
        x* = (1, 0.1, 0.4, 0.5), f* = 0.2. Strictly complementary (x1 at ub with
        multiplier 0.2; x2,x3 interior).
Known optimal: 0.2
"""
import time, numpy as np
from scipy.optimize import LinearConstraint, Bounds
from pounce import minimize
y = np.array([1.4, 0.3, 0.6, 0.9]); KX = np.array([1.0, 0.1, 0.4, 0.5]); KNOWN = 0.2
fun = lambda x: 0.5 * np.sum((x - y) ** 2); jac = lambda x: x - y; hess = lambda x: np.eye(4)
A = np.array([[1, 1, 1, 1], [2, 2, 2, 2.0]]); con = LinearConstraint(A, [2, 4], [2, 4])
bounds = Bounds([0, 0, 0, 0.5], [1, 1, 1, 0.5]); x0 = np.full(4, 0.5)
out = {}
for sel in ("auto", "nlp"):
    t0 = time.perf_counter()
    r = minimize(fun, x0, jac=jac, hess=hess, bounds=bounds, constraints=con, solver_selection=sel)
    out[sel] = (r, time.perf_counter() - t0)
    info = getattr(r, "info", {}) or {}
    print(f"[{sel}] success={r.success} status={getattr(r,'status',None)} fun={r.fun:.12f} x={np.round(r.x,9)} solver={info.get('solver')} t={out[sel][1]:.4f}s")
ra, rn = out["auto"][0], out["nlp"][0]
import cvxpy as cp
xv = cp.Variable(4)
p = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(xv - y)), [A @ xv == [2, 4], xv >= [0, 0, 0, 0.5], xv <= [1, 1, 1, 0.5]])
t0 = time.perf_counter(); p.solve(solver=cp.CLARABEL); to = time.perf_counter() - t0
print(f"cvxpy status={p.status} obj={p.value:.12f} t={to:.4f}s")
rel = lambda u, v: abs(u - v) / max(1, abs(v))
route = (getattr(ra, "info", {}) or {}).get("solver")
agree = abs(ra.fun - rn.fun) < 1e-6 and np.abs(ra.x - rn.x).max() < 1e-5
right = all(rel(r.fun, KNOWN) < 1e-6 and np.abs(r.x - KX).max() < 1e-5 for r in (ra, rn))
specialized = route is not None and "nlp" not in str(route).lower()
print(f"auto_vs_nlp agree={agree} both_right={right} auto_route={route} specialized={specialized}")
print("VERDICT: PASS" if agree and right else "VERDICT: FAIL (ROUTING_ERROR)" if not agree else "VERDICT: FAIL")
