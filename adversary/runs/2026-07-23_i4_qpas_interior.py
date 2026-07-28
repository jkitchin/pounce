"""Adversary i4: qp-active-set with an EMPTY active set (interior optimum).
Family: qp-active-set   Class: strictly convex QP, loose bounds -> no constraint active.

    min 1/2 x'P x + c'x   s.t.  -10 <= x <= 10   (n=3)
P=diag(2,2,2), c=[-2,-4,-6] -> unconstrained minimizer x* = -P^{-1}c = [1,2,3],
strictly inside the box, so the OPTIMAL ACTIVE SET IS EMPTY. Tests that the
active-set engine correctly detects that no bound binds and returns the
unconstrained stationary point. DISTINCT from logged box-bounds test (there a
bound binds).
Known: x*=[1,2,3], obj = 0.5 x'Px + c'x = 0.5*(2+8+18) + (-2-8-18) = 14 - 28 = -14.
"""
import json, subprocess, time
import numpy as np
import cvxpy as cp
import pyomo.environ as pyo

CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
Pd = np.array([2.0, 2.0, 2.0]); P = np.diag(Pd)
c = np.array([-2.0, -4.0, -6.0]); n = 3
X_STAR = -np.linalg.solve(P, c)              # [1,2,3]
KNOWN_OPTIMAL = float(0.5 * X_STAR @ P @ X_STAR + c @ X_STAR)  # -14

def rel(a, b): return abs(a - b) / max(1.0, abs(b))

def build_nl(path):
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(m.I, bounds=(-10.0, 10.0), initialize=0.0)
    m.obj = pyo.Objective(expr=0.5 * sum(Pd[i] * m.x[i]**2 for i in m.I)
                          + sum(c[i] * m.x[i] for i in m.I))
    m.write(path, format="nl")

def run_cli(selection, tag):
    nl = f"/tmp/adv_i4_interior_{tag}.nl"; sol_f = f"/tmp/adv_i4_interior_{tag}.sol"
    js = f"/tmp/adv_i4_interior_{tag}.json"
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
prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + c @ x), [x >= -10, x <= 10])
t0 = time.perf_counter(); prob.solve(solver=cp.CLARABEL); t_cvx = time.perf_counter() - t0
obj_cvx = float(prob.value)

x_err = float(np.linalg.norm(AS["x"] - X_STAR, np.inf))
at_bound = bool(np.any(np.abs(np.abs(AS["x"]) - 10.0) < 1e-6))
print("=== Interior-optimum convex QP, empty active set (qp-active-set) ===")
print(f"KNOWN={KNOWN_OPTIMAL:.12e}  X_STAR={X_STAR}")
print(f"-- active-set: exit={AS['exit']} status={AS['status']} obj={AS['obj']:.12e} x={AS['x']} t={AS['t']:.4f}s")
print(f"-- qp-ipm    : status={IPM['status']} obj={IPM['obj']:.12e} t={IPM['t']:.4f}s")
print(f"-- CLARABEL  : obj={obj_cvx:.12e} t={t_cvx:.4f}s")
print(f"rel AS vs_known={rel(AS['obj'],KNOWN_OPTIMAL):.3e} vs_IPM={rel(AS['obj'],IPM['obj']):.3e} "
      f"vs_CLARABEL={rel(AS['obj'],obj_cvx):.3e}  x_inf_err={x_err:.3e}  any_bound_active={at_bound}")

TOL = 1e-6
ok = (AS["exit"] == 0 and AS["status"] == "SolveSucceeded"
      and rel(AS["obj"], KNOWN_OPTIMAL) < TOL and rel(AS["obj"], IPM["obj"]) < TOL
      and rel(AS["obj"], obj_cvx) < TOL and x_err < 1e-5 and not at_bound)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={AS['status']}, x_err={x_err:.2e})")
