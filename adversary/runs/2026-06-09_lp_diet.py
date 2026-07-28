"""Adversary cross-check: small diet/blending LP (minimize cost, >= nutrient floors)
Family: lp   Class: bounded LP with >= (floor) inequality constraints

Source: Classic diet-LP form (Dantzig diet problem, textbook 3-food variant).
Buy nonneg amounts of 3 foods to meet 2 nutrient minimums at least cost.

  minimize  c.x   (cost)
  subject to N x >= r   (nutrient floors),  x >= 0

Data (derivable, self-consistent):
  foods: x0,x1,x2  costs c = [2, 3, 1.5]
  nutrient matrix N (rows=nutrients, cols=foods):
     protein: [4, 2, 1]   floor r0 = 12
     iron:    [1, 3, 2]   floor r1 =  9
  Known optimum: solved by the oracle and confirmed by KKT/vertex.
  The unique optimum lies at intersection of the two active floors:
     4 x0 + 2 x1 + 1 x2 >= 12
     1 x0 + 3 x1 + 2 x2 >=  9
  Closed-form check below picks the vertex with x2=0 (basic feasible):
     4 x0 + 2 x1 = 12 ; x0 + 3 x1 = 9 -> x0=1.8, x1=2.4, cost=2*1.8+3*2.4=10.8
  but vertex with x1=0: 4x0+x2=12, x0+2x2=9 -> x0=15/7, x2=24/7,
     cost=2*15/7+1.5*24/7 = (30+36)/7 = 66/7 = 9.4286  (cheaper)
  vertex x0=0: 2x1+x2=12, 3x1+2x2=9 -> x1=15, x2=-18 infeasible.
  So expected optimum ~ 66/7 at x=(15/7,0,24/7). Oracle is the authority.

pounce uses G x <= h, so encode N x >= r  as  (-N) x <= (-r).
"""
import time
import numpy as np

KNOWN_OPTIMAL = 66.0 / 7.0  # 9.428571..., minimization
X_STAR = np.array([15.0 / 7.0, 0.0, 24.0 / 7.0])

c = np.array([2.0, 3.0, 1.5])
N = np.array([[4.0, 2.0, 1.0],
              [1.0, 3.0, 2.0]])
r_floor = np.array([12.0, 9.0])

# pounce form: G x <= h
G = -N
h = -r_floor
lb = np.array([0.0, 0.0, 0.0])

# --- pounce ---
import pounce
t0 = time.perf_counter()
res = pounce.solve_qp(P=None, c=c, G=G, h=h, lb=lb)
t_pounce = time.perf_counter() - t0
x_pounce, obj_pounce, status = np.asarray(res.x), res.obj, res.status

# --- oracle 1: scipy linprog (A_ub x <= b_ub) ---
from scipy.optimize import linprog
t0 = time.perf_counter()
lp = linprog(c, A_ub=-N, b_ub=-r_floor, bounds=[(0, None)] * 3)
t_oracle = time.perf_counter() - t0
x_oracle, obj_oracle = lp.x, lp.fun

# --- oracle 2: cvxpy ---
import cvxpy as cp
xv = cp.Variable(3)
prob = cp.Problem(cp.Minimize(c @ xv), [N @ xv >= r_floor, xv >= 0])
prob.solve(solver=cp.CLARABEL)
obj_cvx = prob.value


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

# feasibility of pounce solution
slack = N @ x_pounce - r_floor
feas = float(min(slack.min(), x_pounce.min()))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} x={x_pounce} t={t_pounce:.4f}s")
print(f"min feasibility (>=0 ok)={feas:.2e}")
print("=== oracle (linprog) ===")
print(f"status={lp.status} obj={obj_oracle:.10e} x={x_oracle} t={t_oracle:.4f}s")
print(f"cvxpy_obj={obj_cvx:.10e}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err={x_err:.2e}")

ok = (status == "optimal" or getattr(res, "success", False)) \
    and obj_err < 1e-4 and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 \
    and feas > -1e-6
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, feas={feas:.2e})")
