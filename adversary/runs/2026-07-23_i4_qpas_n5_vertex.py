"""Adversary i4: qp-active-set on a larger (n=5) QP with THREE active inequalities.
Family: qp-active-set   Class: strictly convex QP, 3 simultaneously-active ineqs.

    min 1/2 ||x - a||^2   s.t.  x0 >= 1, x1 >= 1, x2 >= 1   (x3,x4 free)
a = [0.5, 0.5, 0.5, 2.0, -3.0].  Since a0=a1=a2=0.5 < 1, the three lower-bound
inequalities all bind; x3,x4 stay at their unconstrained optimum a3,a4.
Known: x* = [1,1,1,2,-3], active set {x0>=1, x1>=1, x2>=1},
       obj = 0.5 * (0.5^2 + 0.5^2 + 0.5^2 + 0 + 0) = 0.375.
DISTINCT from logged tests: n=5 with 3 inequalities active at once (bigger
working set than any logged active-set case). Cross-check active-set vs qp-ipm
vs cvxpy/CLARABEL.
"""
import json, subprocess, time
import numpy as np
import cvxpy as cp
import pyomo.environ as pyo

CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
a = np.array([0.5, 0.5, 0.5, 2.0, -3.0]); n = 5
P = np.eye(n); c = -a
X_STAR = np.array([1.0, 1.0, 1.0, 2.0, -3.0])
KNOWN_OPTIMAL = float(0.5 * np.sum((X_STAR - a) ** 2))   # 0.375

def rel(x, y): return abs(x - y) / max(1.0, abs(y))

def build_nl(path):
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(m.I, initialize=0.0)
    m.obj = pyo.Objective(expr=0.5 * sum((m.x[i] - a[i])**2 for i in m.I))
    m.g0 = pyo.Constraint(expr=m.x[0] >= 1.0)
    m.g1 = pyo.Constraint(expr=m.x[1] >= 1.0)
    m.g2 = pyo.Constraint(expr=m.x[2] >= 1.0)
    m.write(path, format="nl")

def run_cli(selection, tag):
    nl = f"/tmp/adv_i4_n5v_{tag}.nl"; sol_f = f"/tmp/adv_i4_n5v_{tag}.sol"
    js = f"/tmp/adv_i4_n5v_{tag}.json"
    build_nl(nl)
    t0 = time.perf_counter()
    proc = subprocess.run([CLI, nl, sol_f, f"solver_selection={selection}",
                           "--json-output", js], capture_output=True, text=True, timeout=60)
    dt = time.perf_counter() - t0
    d = json.load(open(js)); s = d["solution"]
    return dict(exit=proc.returncode, status=s["status"], obj=float(s["objective"]),
                x=np.asarray(s["x"], float), t=dt, stderr=proc.stderr)

AS = run_cli("qp-active-set", "as")
IPM = run_cli("qp-ipm", "ipm")

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x - a)),
                  [x[0] >= 1, x[1] >= 1, x[2] >= 1])
t0 = time.perf_counter(); prob.solve(solver=cp.CLARABEL); t_cvx = time.perf_counter() - t0
obj_cvx = float(prob.value); x_cvx = np.asarray(x.value, float)

x_err = float(np.linalg.norm(AS["x"] - X_STAR, np.inf))
n_active = int(np.sum(np.abs(AS["x"][:3] - 1.0) < 1e-6))
print("=== n=5 QP, three active inequalities (qp-active-set) ===")
print(f"KNOWN={KNOWN_OPTIMAL:.12e}  X_STAR={X_STAR}")
print(f"-- active-set: exit={AS['exit']} status={AS['status']} obj={AS['obj']:.12e} x={AS['x']} t={AS['t']:.4f}s")
print(f"-- qp-ipm    : status={IPM['status']} obj={IPM['obj']:.12e} t={IPM['t']:.4f}s")
print(f"-- CLARABEL  : obj={obj_cvx:.12e} x={x_cvx} t={t_cvx:.4f}s")
print(f"rel AS vs_known={rel(AS['obj'],KNOWN_OPTIMAL):.3e} vs_IPM={rel(AS['obj'],IPM['obj']):.3e} "
      f"vs_CLARABEL={rel(AS['obj'],obj_cvx):.3e}  x_inf_err={x_err:.3e}  n_active_bounds={n_active}")

TOL = 1e-6
ok = (AS["exit"] == 0 and AS["status"] == "SolveSucceeded"
      and rel(AS["obj"], KNOWN_OPTIMAL) < TOL and rel(AS["obj"], IPM["obj"]) < TOL
      and rel(AS["obj"], obj_cvx) < TOL and x_err < 1e-5 and n_active == 3)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={AS['status']}, x_err={x_err:.2e}, n_active={n_active})")
