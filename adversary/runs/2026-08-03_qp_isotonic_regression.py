"""Adversary cross-check: Isotonic regression via QP with monotonicity
constraints
Family: qp   Class: box/chain-inequality-constrained convex QP
Source: closed-form optimum via the Pool-Adjacent-Violators Algorithm (PAVA),
    the classical exact algorithm for isotonic regression (Barlow, Bartholomew,
    Bremner & Brunk, "Statistical Inference under Order Restrictions", 1972;
    also Best & Chakravarti, Math. Programming 47 (1990), 425-439, for the
    QP/active-set equivalence). PAVA is an independent, non-QP-solver
    algorithm (pool-and-average on violating adjacent blocks), so it is a
    genuine oracle distinct from both pounce and cvxpy.

Problem: given y_1..y_n, find the closest (least-squares) monotone
nondecreasing sequence x_1<=x_2<=...<=x_n:

    minimize    sum_i (x_i - y_i)^2
    subject to  x_i <= x_{i+1}   for i=1..n-1

This is a convex QP: P=2I, c=-2y, G = first-difference matrix (x_i - x_{i+1}
<= 0), n=8 (non-monotone data forcing several pooled blocks).
"""
import time
import numpy as np

y = np.array([4.0, 3.0, 5.0, 6.0, 2.0, 7.0, 8.0, 1.0])
n = len(y)

# --- oracle: PAVA (pool adjacent violators), textbook exact algorithm ---
def pava(y):
    # block value / weight stack
    vals = list(y.astype(float))
    weights = [1.0] * len(vals)
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-15:
            # merge blocks i, i+1 into a single weighted-average block
            new_val = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / (weights[i] + weights[i + 1])
            new_w = weights[i] + weights[i + 1]
            vals[i:i + 2] = [new_val]
            weights[i:i + 2] = [new_w]
            i = max(i - 1, 0)
        else:
            i += 1
    # expand blocks back to length n
    out = []
    for v, w in zip(vals, weights):
        out.extend([v] * int(round(w)))
    return np.array(out)


x_pava = pava(y)
obj_pava = float(np.sum((x_pava - y) ** 2))

# --- pounce QP ---
P = 2.0 * np.eye(n)
c = -2.0 * y
G = np.zeros((n - 1, n))
for i in range(n - 1):
    G[i, i] = 1.0
    G[i, i + 1] = -1.0
h = np.zeros(n - 1)

import pounce

t0 = time.perf_counter()
r = pounce.solve_qp(P=P, c=c, G=G, h=h)
t_pounce = time.perf_counter() - t0
x_p = np.asarray(r.x, dtype=float)
obj_pounce = float(r.obj) + float(y @ y)  # solve_qp's obj is 0.5x'Px+c'x; add back sum(y^2)
status = str(r.status)

# --- oracle: cvxpy (independent QP/conic solver) ---
import cvxpy as cp

xv = cp.Variable(n)
constraints = [xv[i] <= xv[i + 1] for i in range(n - 1)]
prob = cp.Problem(cp.Minimize(cp.sum_squares(xv - y)), constraints)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
obj_cvx = float(prob.value)
x_cvx = np.asarray(xv.value, dtype=float)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err_pava = rel(obj_pounce, obj_pava)
obj_err_cvx = rel(obj_pounce, obj_cvx)
x_err_pava = float(np.linalg.norm(x_p - x_pava, np.inf))
x_err_cvx = float(np.linalg.norm(x_p - x_cvx, np.inf))
monotone_ok = bool(np.all(np.diff(x_p) >= -1e-6))

print("=== pounce (QP, isotonic regression) ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"x={np.round(x_p, 6)}")
print("=== oracle: PAVA (closed-form exact algorithm) ===")
print(f"obj={obj_pava:.10e} x={np.round(x_pava, 6)}")
print("=== oracle: cvxpy CLARABEL (same QP) ===")
print(f"obj={obj_cvx:.10e} x={np.round(x_cvx, 6)} t={t_oracle:.4f}s")
print(f"obj_err_vs_PAVA={obj_err_pava:.2e} x_inf_err_vs_PAVA={x_err_pava:.2e}")
print(f"obj_err_vs_cvxpy={obj_err_cvx:.2e} x_inf_err_vs_cvxpy={x_err_cvx:.2e}")
print(f"monotonicity_satisfied={monotone_ok}")

ok = (
    status in ("optimal", "optimal_inaccurate")
    and obj_err_pava < 1e-6
    and obj_err_cvx < 1e-6
    and x_err_pava < 1e-4
    and monotone_ok
)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err_pava={obj_err_pava:.2e}, err_cvx={obj_err_cvx:.2e}, monotone={monotone_ok})")
