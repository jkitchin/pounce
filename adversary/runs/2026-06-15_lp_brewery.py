"""Adversary cross-check: Vanderbei "brewery" production-planning LP
Family: lp   Class: bounded inequality-constrained production-planning LP

Source: R. J. Vanderbei, "Linear Programming: Foundations and Extensions"
(and the widely-used Princeton/Sedgewick-Wayne "brewer's problem" teaching
example). A small brewer makes ale and beer from corn, hops, and malt.

  maximize  13 A + 23 B            (profit, $)
  subject to
     5 A + 15 B <= 480            (corn,  lbs)
     4 A +  4 B <= 160            (hops,  oz)
    35 A + 20 B <= 1190           (malt,  lbs)
     A, B >= 0

Known optimum (closed form, intersection of corn & malt constraints):
   5 A + 15 B = 480
  35 A + 20 B = 1190   =>  A = 12, B = 28
  profit = 13*12 + 23*28 = 156 + 644 = 800.
  (hops slack: 4*12+4*28 = 160 <= 160, also binding; well-known vertex.)

Known optimal: max = 800  (min -13A-23B = -800).
pounce minimizes 0.5 x'Px + c'x with P=None => c'x; use c = [-13,-23].
"""
import time
import numpy as np

KNOWN_OPTIMAL = -800.0  # minimization form
X_STAR = np.array([12.0, 28.0])

c = np.array([-13.0, -23.0])
G = np.array([[5.0, 15.0],
              [4.0, 4.0],
              [35.0, 20.0]])
h = np.array([480.0, 160.0, 1190.0])
lb = np.array([0.0, 0.0])

# --- pounce (LP via convex IPM, P=None) ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_qp(P=None, c=c, G=G, h=h, lb=lb)
t_pounce = time.perf_counter() - t0
x_pounce, obj_pounce, status = np.asarray(r.x), r.obj, r.status

# --- oracle 1: scipy linprog (HiGHS) ---
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

# feasibility of pounce solution
slack = h - G @ x_pounce
feas = float(min(slack.min(), x_pounce.min()))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} x={x_pounce} t={t_pounce:.4f}s")
print(f"min feasibility (>=0 ok)={feas:.2e}")
print("=== oracle (linprog/HiGHS) ===")
print(f"status={lp.status} obj={obj_oracle:.10e} x={x_oracle} t={t_oracle:.4f}s")
print(f"cvxpy_obj={obj_cvx:.10e}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"x_star={X_STAR} obj_err_vs_oracle={obj_err:.2e} x_inf_err={x_err:.2e}")

ok = (status == "optimal" or getattr(r, "success", False)) \
    and obj_err < 1e-4 and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 \
    and feas > -1e-6
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, feas={feas:.2e})")
