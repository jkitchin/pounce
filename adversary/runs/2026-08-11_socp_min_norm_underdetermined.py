"""Adversary cross-check: minimum-norm solution of an underdetermined linear system
Family: socp   Class: equality-constrained SOC, closed-form optimum
Source: classic minimum-norm least-squares result (e.g. Boyd & Vandenberghe,
        Convex Optimization, sec. 4.3 "least-norm problems"); optimum has the
        closed form x* = A^T (A A^T)^{-1} b, obj* = sqrt(b^T (A A^T)^{-1} b).
Known optimal: computed below from the closed form (see KNOWN_OPTIMAL / X_STAR)

Formulation as SOCP: minimize t  s.t.  (t, x) in SOC (t >= ||x||_2), Ax = b.
Variables z = (t, x) in R^{1+n}. c = (1, 0,...,0). G z <= h with
G = [[-1, 0...0], [0, -I_n]], h = 0, cones=[("soc", n+1)] (slack s = h - Gz = (t, x)).
Equality: A_eq z = b where A_eq = [0 | A] (t unconstrained by equality block).
"""
import numpy as np

A = np.array([[1., 0., 1., 0., 1.],
              [0., 1., 0., 1., 0.],
              [1., 1., 0., 0., 1.]])
b = np.array([3., 4., 2.])
n = A.shape[1]

x_star = A.T @ np.linalg.solve(A @ A.T, b)
KNOWN_OPTIMAL = float(np.linalg.norm(x_star))
print(f"closed-form x*={x_star}")
print(f"closed-form obj*={KNOWN_OPTIMAL:.10e}")

# --- pounce ---
from pounce import solve_socp
import time

nz = n + 1  # (t, x)
c = np.zeros(nz); c[0] = 1.0
G = np.zeros((nz, nz))
G[0, 0] = -1.0
G[1:, 1:] = -np.eye(n)
h = np.zeros(nz)
A_eq = np.zeros((A.shape[0], nz))
A_eq[:, 1:] = A
b_eq = b

t0 = time.perf_counter()
r = solve_socp(c=c, A=A_eq, b=b_eq, G=G, h=h, cones=[("soc", nz)])
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)[1:]
obj_pounce = r.obj
status = r.status
print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")

# --- oracle: cvxpy (Clarabel) ---
import cvxpy as cp
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.norm(x, 2)), [A @ x == b])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle = x.value
obj_oracle = prob.value
print("=== oracle (cvxpy/Clarabel) ===")
print(f"status={prob.status} obj={obj_oracle:.10e} t={t_oracle:.4f}s")


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_oracle = rel(obj_pounce, obj_oracle)
x_err_known = float(np.linalg.norm(x_pounce - x_star, np.inf))
x_err_oracle = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={obj_err_known:.2e} x_inf_err_vs_known={x_err_known:.2e}")
print(f"obj_err_vs_oracle={obj_err_oracle:.2e} x_inf_err_vs_oracle={x_err_oracle:.2e}")

ok = (status == "optimal") and obj_err_known < 1e-6 and obj_err_oracle < 1e-6
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err_known={obj_err_known:.2e}, obj_err_oracle={obj_err_oracle:.2e})")
