"""Adversary cross-check: LP with a free (unrestricted-sign) variable
Family: lp   Class: free-variable, equality-constrained, unique bounded optimum
Source: constructed problem (LICQ / free-variable bound handling), optimum
        derived analytically below and cross-checked with scipy.optimize.linprog
        (HiGHS) as the independent oracle.

Formulation:
    minimize   2 x + 3 y + z
    subject to x + y + z = 6
               x - y     = 1
               y >= 0, z >= 0, x free (no bound)

Derivation:
    eq2: x = y + 1
    eq1: (y+1) + y + z = 6  =>  z = 5 - 2y
    feasibility: y >= 0 and z = 5 - 2y >= 0  =>  y in [0, 2.5]
    objective:  2(y+1) + 3y + (5 - 2y) = 3y + 7
    strictly increasing in y on [0, 2.5] with coefficient +3 > 0
    => unique minimizer at y = 0  =>  x = 1, y = 0, z = 5, obj = 7

KNOWN_OPTIMAL = 7.0 at x* = (1, 0, 5)
"""
import numpy as np
from scipy.optimize import linprog
from pounce import solve_qp
import time

KNOWN_OPTIMAL = 7.0
KNOWN_X = np.array([1.0, 0.0, 5.0])

c = np.array([2.0, 3.0, 1.0])
A = np.array([[1.0, 1.0, 1.0],
              [1.0, -1.0, 0.0]])
b = np.array([6.0, 1.0])
lb = np.array([-np.inf, 0.0, 0.0])
ub = np.array([np.inf, np.inf, np.inf])

# --- pounce ---
t0 = time.perf_counter()
r = solve_qp(P=None, c=c, A=A, b=b, lb=lb, ub=ub)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)
obj_pounce = r.obj
status = r.status

# --- oracle: scipy.optimize.linprog (HiGHS), x free via bounds=(None, None) ---
t0 = time.perf_counter()
res = linprog(
    c=c,
    A_eq=A,
    b_eq=b,
    bounds=[(None, None), (0, None), (0, None)],
    method="highs",
)
t_oracle = time.perf_counter() - t0
x_oracle = res.x
obj_oracle = res.fun


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))
x_err_known = float(np.linalg.norm(x_pounce - KNOWN_X, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} x={x_pounce} t={t_pounce:.4f}s")
print("=== oracle (scipy linprog/HiGHS) ===")
print(f"success={res.success} obj={obj_oracle:.10e} x={x_oracle} t={t_oracle:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err_vs_oracle={x_err:.2e} x_inf_err_vs_known={x_err_known:.2e}")

ok = (
    status in ("optimal", "solved")
    and obj_err < 1e-6
    and x_err_known < 1e-6
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, x_err_known={x_err_known:.2e})")
