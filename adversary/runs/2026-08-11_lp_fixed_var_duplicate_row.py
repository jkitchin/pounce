"""Adversary cross-check: LP with a fixed variable + an exact-duplicate
(scaled) equality row.
Family: lp   Class: bounded LP with presolve-relevant degeneracy
Source: self-constructed instance with an analytically-derivable optimum
        (dominance argument on a simplex constraint), designed to exercise
        two presolve reductions simultaneously: a variable fixed by
        lb==ub, and an equality row that is an exact scalar multiple of
        another (redundant row).

  minimize    x1 + 2*x2 + 3*x3 + 4*x4
  subject to  x1 + x2 + x3       = 10        (row 1)
              2x1 + 2x2 + 2x3    = 20        (row 2 == 2 * row 1, redundant)
              x4 = 7                          (lb4 = ub4 = 7, fixed var)
              0 <= x1, x2, x3 <= 10

Since row 2 is redundant given row 1, the feasible region in (x1,x2,x3) is
exactly the simplex slice {x1+x2+x3=10, 0<=xi<=10}. On that slice the
objective x1+2x2+3x3 is minimized (dominance) by putting all mass on the
cheapest-cost variable x1 (coefficient 1 < 2 < 3), which is reachable
within its bound (x1<=10 covers the full required sum of 10):
    x1*=10, x2*=0, x3*=0.
x4 is pinned to 7 by its bounds regardless of cost.

Known optimal:
    x* = (10, 0, 0, 7)
    obj* = 1*10 + 2*0 + 3*0 + 4*7 = 10 + 28 = 38
"""
import time
import numpy as np

KNOWN_OPTIMAL = 38.0
X_STAR = np.array([10.0, 0.0, 0.0, 7.0])

c = np.array([1.0, 2.0, 3.0, 4.0])
A = np.array([
    [1.0, 1.0, 1.0, 0.0],
    [2.0, 2.0, 2.0, 0.0],   # redundant: 2 * row 1
])
b = np.array([10.0, 20.0])
lb = np.array([0.0, 0.0, 0.0, 7.0])
ub = np.array([10.0, 10.0, 10.0, 7.0])

# --- pounce (LP is a QP with P=None) ---
from pounce import solve_qp
t0 = time.perf_counter()
r = solve_qp(c=c, A=A, b=b, lb=lb, ub=ub)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)
obj_pounce = r.obj
status = r.status

# --- oracle: scipy.optimize.linprog ---
from scipy.optimize import linprog
bounds = list(zip(lb, ub))
t0 = time.perf_counter()
res = linprog(c=c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
t_oracle = time.perf_counter() - t0
x_oracle = res.x
obj_oracle = res.fun


def rel(a, b_):
    return abs(a - b_) / max(1.0, abs(b_))


obj_err = rel(obj_pounce, obj_oracle)
known_err = rel(obj_pounce, KNOWN_OPTIMAL)
x_err = float(np.linalg.norm(x_pounce - X_STAR, np.inf))
x_err_oracle = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} x={x_pounce} t={t_pounce:.4f}s")
print("=== oracle (scipy HiGHS) ===")
print(f"success={res.success} obj={obj_oracle:.10e} x={x_oracle} t={t_oracle:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={known_err:.2e} x_inf_err_vs_known={x_err:.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err_vs_oracle={x_err_oracle:.2e}")

ok = (status == "optimal") and obj_err < 1e-4 and known_err < 1e-4 and x_err < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, known_err={known_err:.2e}, x_err={x_err:.2e})")
