"""Adversary i4: qp-active-set on a PURELY EQUALITY-constrained convex QP.
Family: qp-active-set   Class: strictly convex QP, only equalities (no ineq).

    min 1/2 x'P x + c'x   s.t.  A x = b     (P SPD diag, 2 equalities, n=4)

DISTINCT from logged active-set tests (box-only, 1eq+1ineq, HS35, 2-active-ineq
vertex): here the active set is EXACTLY the equality rows, no inequalities at
all -> the active-set engine must solve the pure equality KKT system.

Closed form (equality-only KKT):
    [P  A'][x]   [-c]
    [A  0 ][y] = [ b]
P=diag(2,3,4,5), c=[-1,-2,-3,-4], A=[[1,1,1,1],[1,-1,1,-1]], b=[1,0].
Invoked via CLI solver_selection=qp-active-set; cross-checked vs qp-ipm and the
closed-form linear solve, plus cvxpy/CLARABEL.
"""
import json, subprocess, time
import numpy as np
import cvxpy as cp
import pyomo.environ as pyo

CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
Pd = np.array([2.0, 3.0, 4.0, 5.0]); P = np.diag(Pd)
c = np.array([-1.0, -2.0, -3.0, -4.0])
A = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0]])
b = np.array([1.0, 0.0])
n = 4

# closed-form KKT
K = np.block([[P, A.T], [A, np.zeros((2, 2))]])
rhs = np.concatenate([-c, b])
sol = np.linalg.solve(K, rhs)
X_STAR = sol[:n]
KNOWN_OPTIMAL = float(0.5 * X_STAR @ P @ X_STAR + c @ X_STAR)

def rel(a, b): return abs(a - b) / max(1.0, abs(b))

def build_nl(path):
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(m.I, initialize=0.0)
    m.obj = pyo.Objective(expr=0.5 * sum(Pd[i] * m.x[i]**2 for i in m.I)
                          + sum(c[i] * m.x[i] for i in m.I))
    m.eq0 = pyo.Constraint(expr=sum(A[0, i] * m.x[i] for i in m.I) == b[0])
    m.eq1 = pyo.Constraint(expr=sum(A[1, i] * m.x[i] for i in m.I) == b[1])
    m.write(path, format="nl")

def run_cli(selection, tag):
    nl = f"/tmp/adv_i4_pureeq_{tag}.nl"; sol_f = f"/tmp/adv_i4_pureeq_{tag}.sol"
    js = f"/tmp/adv_i4_pureeq_{tag}.json"
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
prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + c @ x), [A @ x == b])
t0 = time.perf_counter(); prob.solve(solver=cp.CLARABEL); t_cvx = time.perf_counter() - t0
obj_cvx = float(prob.value); x_cvx = np.asarray(x.value, float)

x_err = float(np.linalg.norm(AS["x"] - X_STAR, np.inf))
print("=== Pure-equality convex QP (qp-active-set) ===")
print(f"KNOWN={KNOWN_OPTIMAL:.12e}  X_STAR={X_STAR}")
print(f"-- active-set: exit={AS['exit']} status={AS['status']} obj={AS['obj']:.12e} x={AS['x']} t={AS['t']:.4f}s")
print(f"-- qp-ipm    : exit={IPM['exit']} status={IPM['status']} obj={IPM['obj']:.12e} t={IPM['t']:.4f}s")
print(f"-- CLARABEL  : obj={obj_cvx:.12e} t={t_cvx:.4f}s")
print(f"rel AS vs_known={rel(AS['obj'],KNOWN_OPTIMAL):.3e} vs_IPM={rel(AS['obj'],IPM['obj']):.3e} "
      f"vs_CLARABEL={rel(AS['obj'],obj_cvx):.3e}  x_inf_err={x_err:.3e}")

TOL = 1e-6
ok = (AS["exit"] == 0 and AS["status"] == "SolveSucceeded"
      and rel(AS["obj"], KNOWN_OPTIMAL) < TOL and rel(AS["obj"], IPM["obj"]) < TOL
      and rel(AS["obj"], obj_cvx) < TOL and x_err < 1e-5)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={AS['status']}, x_err={x_err:.2e})")
