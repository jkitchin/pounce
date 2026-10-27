"""Adversary cross-check: projection onto a polyhedral cone whose apex is a
HIGHLY primal-degenerate vertex (12 constraints active at x*=0 in R^6) with a
non-unique multiplier vector (dual degeneracy) -- the shape gh#1061 / PR #1082
reworked in the active-set QP (degenerate-vertex handling, Schur reset).
Family: qp-active-set   Class: degenerate vertex, |active| = 2n, dual non-unique
Source: Moreau decomposition (Moreau 1962, C. R. Acad. Sci. 255:238-240): for a
        closed convex cone K, proj_K(x0) = 0 iff x0 lies in the polar cone K°.
        K = {x : x >= 0, a_j.x >= 0, j=0..5}, a_j = e_j + e_{j+1} - 0.5 e_{j+2}
        (cyclic). K° = cone{-e_i, -a_j}. x0 := -(1.0 a0 + 0.7 a2 + 0.5 a4 + 0.2 e1)
        is in K° by construction  =>  x* = 0, f* = 1/2 ||x0||^2.
        A cap sum(x) <= 3 keeps the feasible set bounded (inactive).
Known optimal: 1/2 ||x0||^2  (x* = 0)
"""
import json, os, subprocess, time, numpy as np
n = 6; E = np.eye(n)
a = np.array([E[j] + E[(j + 1) % n] - 0.5 * E[(j + 2) % n] for j in range(n)])
x0 = -(1.0 * a[0] + 0.7 * a[2] + 0.5 * a[4] + 0.2 * E[1])
KNOWN = 0.5 * x0 @ x0; KX = np.zeros(n)
# standard form min 1/2 x'Px + c'x (+ const) ; G x <= h
P = np.eye(n); c = -x0; const = 0.5 * x0 @ x0
G = np.vstack([-np.eye(n), -a, np.ones((1, n))]); h = np.concatenate([np.zeros(2 * n), [3.0]])
rel = lambda u, v: abs(u - v) / max(1, abs(v))
res = {}
from pounce import solve_qp, minimize
t0 = time.perf_counter(); r = solve_qp(P=P, c=c, G=G, h=h); res["solve_qp ipm"] = (r.status, r.obj + const, np.asarray(r.x), time.perf_counter() - t0)
t0 = time.perf_counter(); r = solve_qp(P=P, c=c, G=G, h=h, method="active-set") if "active" in solve_qp.__doc__ else None
if r is not None: res["solve_qp active-set"] = (r.status, r.obj + const, np.asarray(r.x), time.perf_counter() - t0)
from scipy.optimize import LinearConstraint
fun = lambda x: 0.5 * np.sum((x - x0) ** 2); jac = lambda x: x - x0
con = LinearConstraint(G, -np.inf, h)
xs = np.full(n, 0.1)
for sel in ("qp-active-set", "auto", "nlp"):
    t0 = time.perf_counter(); rr = minimize(fun, xs, jac=jac, hess=lambda x: np.eye(n), constraints=con, solver_selection=sel)
    res[f"minimize {sel}"] = (("optimal" if rr.success else f"FAIL:{rr.message}"), rr.fun, rr.x, time.perf_counter() - t0, (getattr(rr, "info", {}) or {}).get("solver"))
# CLI on .nl (nl tape path) with solver_selection=qp-active-set
import pyomo.environ as pe
m = pe.ConcreteModel(); m.I = pe.RangeSet(0, n - 1); m.x = pe.Var(m.I, initialize=0.1)
m.obj = pe.Objective(expr=0.5 * sum((m.x[i] - x0[i]) ** 2 for i in m.I))
m.c = pe.ConstraintList()
for gi, hi in zip(G, h): m.c.add(sum(gi[i] * m.x[i] for i in m.I if gi[i] != 0) <= hi)
d = "adversary/runs/scratch_1027"; nl = f"{d}/degvert.nl"
m.write(nl, io_options={"symbolic_solver_labels": False})
for sel in ("qp-active-set", "auto"):
    js = f"{d}/degvert_{sel}.json"; sol = f"{d}/degvert_{sel}.sol"
    t0 = time.perf_counter()
    p = subprocess.run(["target/release/pounce", nl, sol, f"solver_selection={sel}", "--json-output", js], capture_output=True, text=True)
    tc = time.perf_counter() - t0
    J = json.load(open(js)); sol_ = J.get("solution", {})
    xv = np.asarray(sol_.get("x", [np.nan] * n), float)
    res[f"CLI {sel}"] = (J.get("status", sol_.get("status")), sol_.get("objective", np.nan), xv, tc, sol_.get("engine"))
    v = subprocess.run(["target/release/pounce", "verify", nl, sol], capture_output=True, text=True)
    res[f"CLI {sel}"] += (v.returncode,)
import cvxpy as cp
xv = cp.Variable(n); pr = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(xv - x0)), [G @ xv <= h])
t0 = time.perf_counter(); pr.solve(solver=cp.CLARABEL); to = time.perf_counter() - t0
ok = True
for k, v in res.items():
    st, ob, x = v[0], v[1], v[2]
    e = rel(float(ob), KNOWN); xe = np.abs(np.asarray(x, float) - KX).max()
    extra = " ".join(str(z) for z in v[4:])
    print(f"{k:24s} status={st} obj={float(ob):.10f} err={e:.1e} x_inf={xe:.1e} t={v[3]:.4f}s {extra}")
    ok &= e < 1e-6 and xe < 1e-5
print(f"cvxpy status={pr.status} obj={pr.value:.10f} t={to:.4f}s known={KNOWN:.10f} active_at_x*={int(np.sum(np.abs(G@KX-h)<1e-12))}")
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
