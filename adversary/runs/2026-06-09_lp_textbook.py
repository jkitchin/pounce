"""Adversary cross-check: textbook 2-variable LP
Family: lp   Class: bounded inequality-constrained linear program
Source: standard max-profit LP (e.g. Dantzig-style). Closed-form vertex.
  maximize  x + y   s.t.  x + 2y <= 4,  4x + 2y <= 12,  x,y >= 0
  Optimum at intersection of the two active constraints:
    x + 2y = 4, 4x + 2y = 12  =>  x = 8/3, y = 2/3,  obj = 10/3.
Known optimal: max = 10/3 = 3.33333...  (min -x-y = -10/3)
"""
import time
import numpy as np

KNOWN_OPTIMAL = -10.0 / 3.0  # minimization form
X_STAR = np.array([8.0 / 3.0, 2.0 / 3.0])

c = np.array([-1.0, -1.0])
G = np.array([[1.0, 2.0], [4.0, 2.0]])
h = np.array([4.0, 12.0])
lb = np.array([0.0, 0.0])

# --- pounce (LP via convex IPM, P=None) ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_qp(P=None, c=c, G=G, h=h, lb=lb)
t_pounce = time.perf_counter() - t0
x_pounce, obj_pounce, status = np.asarray(r.x), r.obj, r.status

# --- oracle 1: scipy linprog ---
from scipy.optimize import linprog
t0 = time.perf_counter()
lp = linprog(c, A_ub=G, b_ub=h, bounds=[(0, None), (0, None)])
t_oracle = time.perf_counter() - t0
x_oracle, obj_oracle = lp.x, lp.fun

# --- oracle 2: cvxpy ---
import cvxpy as cp
xv = cp.Variable(2)
prob = cp.Problem(cp.Minimize(c @ xv), [G @ xv <= h, xv >= 0])
prob.solve(solver=cp.CLARABEL)
obj_cvx = prob.value


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} x={x_pounce} t={t_pounce:.4f}s")
print("=== oracle (linprog) ===")
print(f"obj={obj_oracle:.10e} x={x_oracle} t={t_oracle:.4f}s")
print(f"cvxpy_obj={obj_cvx:.10e}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err={x_err:.2e}")

ok = (status == "optimal" or r.success) and obj_err < 1e-4 and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e})")
