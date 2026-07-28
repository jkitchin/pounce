"""Adversary cross-check: Nocedal & Wright Example 16.4 (convex QP)
Family: qp   Class: inequality-constrained convex QP
Source: Nocedal & Wright, "Numerical Optimization" (2nd ed.), Example 16.4.
  min  q(x) = (x1-1)^2 + (x2-2.5)^2
  s.t. x1 - 2 x2 + 2 >= 0
       -x1 - 2 x2 + 6 >= 0
       -x1 + 2 x2 + 2 >= 0
       x1 >= 0, x2 >= 0
Known optimal: x* = (1.4, 1.7), q* = 0.8
"""
import time
import numpy as np

KNOWN_OPTIMAL = 0.8
X_STAR = np.array([1.4, 1.7])

# q(x) = (x1-1)^2 + (x2-2.5)^2 = 0.5 x'(2I)x + (-2,-5)'x + (1+6.25)
P = 2.0 * np.eye(2)
c = np.array([-2.0, -5.0])
CONST = 1.0 + 6.25
# Gx <= h form of the three general inequalities (>=0 -> negate)
G = np.array([[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0]])
h = np.array([2.0, 6.0, 2.0])
lb = np.array([0.0, 0.0])

# --- pounce convex QP IPM ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_qp(P=P, c=c, G=G, h=h, lb=lb)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)
obj_pounce = r.obj + CONST  # pounce returns 0.5x'Px+c'x; add the constant
status = r.status

# --- oracle: cvxpy ---
import cvxpy as cp
xv = cp.Variable(2)
prob = cp.Problem(cp.Minimize(cp.sum_squares(xv - np.array([1.0, 2.5]))),
                  [G @ xv <= h, xv >= 0])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle, obj_oracle = np.asarray(xv.value), prob.value


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} x={x_pounce} t={t_pounce:.4f}s iters={r.iters}")
print("=== oracle (cvxpy/CLARABEL) ===")
print(f"obj={obj_oracle:.10e} x={x_oracle} t={t_oracle:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"x*_known={X_STAR} x_err_vs_known={np.linalg.norm(x_pounce-X_STAR, np.inf):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err={x_err:.2e}")

ok = (status == "optimal" or r.success) and obj_err < 1e-4 and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e})")
