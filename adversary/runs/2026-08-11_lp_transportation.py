"""Adversary cross-check: classic transportation LP
Family: lp   Class: equality-constrained LP (balanced transportation problem)
Source: standard transportation-problem formulation (Hillier & Lieberman,
        Introduction to Operations Research; Bertsimas & Tsitsiklis,
        Introduction to Linear Optimization, ch. 1 transportation example).
        2 supply nodes (30, 20 units), 3 demand nodes (15, 25, 10 units),
        balanced (total supply = total demand = 50).
Known optimal: 250.0 (computed independently below via scipy HiGHS, and
        matches the textbook transportation-simplex result for this cost
        matrix by hand: route s1->d1=15, s1->d2=15, s2->d2=10, s2->d3=10).
"""
import time
import numpy as np

# variables x = (x11,x12,x13,x21,x22,x23): shipment s_i -> d_j
c = np.array([4., 6., 8., 5., 7., 3.])
supply = np.array([30., 20.])
demand = np.array([15., 25., 10.])

A_eq = np.array([
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1],
], dtype=float)
b_eq = np.concatenate([supply, demand])

# --- pounce (LP: solve_qp with P=None, nonneg bounds via lb) ---
from pounce import solve_qp

t0 = time.perf_counter()
r = solve_qp(P=None, c=c, A=A_eq, b=b_eq, lb=np.zeros(6))
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)
obj_pounce = r.obj
status = r.status
print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s x={x_pounce}")

# --- oracle: scipy.optimize.linprog (HiGHS) ---
from scipy.optimize import linprog

t0 = time.perf_counter()
res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * 6, method="highs")
t_oracle = time.perf_counter() - t0
obj_oracle = res.fun
x_oracle = res.x
print("=== oracle (scipy linprog/HiGHS) ===")
print(f"status={res.status} obj={obj_oracle:.10e} t={t_oracle:.4f}s x={x_oracle}")

KNOWN_OPTIMAL = 250.0


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_oracle = rel(obj_pounce, obj_oracle)
x_err_oracle = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={obj_err_known:.2e}")
print(f"obj_err_vs_oracle={obj_err_oracle:.2e} x_inf_err_vs_oracle={x_err_oracle:.2e}")

ok = (status == "optimal") and obj_err_known < 1e-6 and obj_err_oracle < 1e-6
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err_known={obj_err_known:.2e}, obj_err_oracle={obj_err_oracle:.2e})")
