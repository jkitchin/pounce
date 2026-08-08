"""Adversary cross-check: classic degenerate-BFS LP.

Family: lp   Class: degenerate optimal vertex (two constraints bind, one
              basic variable at zero) -- stresses crossover.rs / simplex.rs
              phase transitions.
Source: Hillier & Lieberman, "Introduction to Operations Research", the
        standard textbook example used to illustrate LP degeneracy:
            max  3 x1 + 9 x2
            s.t. x1 + 4 x2 <= 8
                 x1 + 2 x2 <= 4
                 x1, x2 >= 0
        Optimal vertex (0, 2), Z = 18. In standard form (slacks s1, s2) both
        s1 = 8 - 4*2 = 0 and s2 = 4 - 2*2 = 0 at the optimum, so with x1 = 0
        as well only ONE of the two required basic variables (x2) is
        actually nonzero: the optimal basic feasible solution is degenerate.

We minimize the negated objective (pounce/scipy/cvxpy all minimize):
    min  -3 x1 - 9 x2   s.t.  x1 + 4 x2 <= 8,  x1 + 2 x2 <= 4,  x >= 0
Known optimal (min) = -18 at x* = (0, 2).
"""

import time

import numpy as np

KNOWN_OPTIMAL = -18.0
KNOWN_X = np.array([0.0, 2.0])

c = np.array([-3.0, -9.0])
G = np.array([[1.0, 4.0], [1.0, 2.0]])
h = np.array([8.0, 4.0])
lb = np.array([0.0, 0.0])
ub = np.array([np.inf, np.inf])

# --- pounce (IPM, default LP-as-QP-with-P=None auto route) ---
from pounce import solve_qp

t0 = time.perf_counter()
r_ipm = solve_qp(c=c, G=G, h=h, lb=lb, ub=ub)
t_pounce_ipm = time.perf_counter() - t0

# --- pounce (active-set path, CLI's qp-active-set engine, via method=) ---
t0 = time.perf_counter()
r_as = solve_qp(c=c, G=G, h=h, lb=lb, ub=ub, method="active-set")
t_pounce_as = time.perf_counter() - t0

# --- oracle 1: scipy.optimize.linprog (HiGHS) ---
from scipy.optimize import linprog

t0 = time.perf_counter()
res = linprog(
    c=c, A_ub=G, b_ub=h, bounds=[(0, None), (0, None)], method="highs"
)
t_scipy = time.perf_counter() - t0

# --- oracle 2: cvxpy (independent DCP formulation, Clarabel) ---
import cvxpy as cp

x = cp.Variable(2)
prob = cp.Problem(cp.Minimize(c @ x), [G @ x <= h, x >= 0])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_cvxpy = time.perf_counter() - t0


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


print("=== pounce (ipm) ===")
print(f"status={r_ipm.status} obj={r_ipm.obj:.10e} x={np.asarray(r_ipm.x)} t={t_pounce_ipm:.4f}s")
print("=== pounce (active-set) ===")
print(f"status={r_as.status} obj={r_as.obj:.10e} x={np.asarray(r_as.x)} t={t_pounce_as:.4f}s")
print("=== scipy linprog (HiGHS) ===")
print(f"status={res.status} obj={res.fun:.10e} x={res.x} t={t_scipy:.4f}s")
print("=== cvxpy/Clarabel ===")
print(f"status={prob.status} obj={prob.value:.10e} x={x.value} t={t_cvxpy:.4f}s")

obj_err_ipm = rel(r_ipm.obj, KNOWN_OPTIMAL)
obj_err_as = rel(r_as.obj, KNOWN_OPTIMAL)
x_err_ipm = float(np.linalg.norm(np.asarray(r_ipm.x) - KNOWN_X, np.inf))
x_err_as = float(np.linalg.norm(np.asarray(r_as.x) - KNOWN_X, np.inf))
obj_err_scipy = rel(res.fun, KNOWN_OPTIMAL)
obj_err_cvxpy = rel(prob.value, KNOWN_OPTIMAL)

print(f"known_optimal={KNOWN_OPTIMAL}")
print(f"ipm: obj_err={obj_err_ipm:.2e} x_err={x_err_ipm:.2e}")
print(f"active-set: obj_err={obj_err_as:.2e} x_err={x_err_as:.2e}")
print(f"scipy obj_err={obj_err_scipy:.2e}  cvxpy obj_err={obj_err_cvxpy:.2e}")

ok = (
    r_ipm.status == "optimal"
    and r_as.status == "optimal"
    and obj_err_ipm < 1e-6
    and obj_err_as < 1e-6
    and x_err_ipm < 1e-5
    and x_err_as < 1e-5
    and obj_err_scipy < 1e-8
    and obj_err_cvxpy < 1e-6
)
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
