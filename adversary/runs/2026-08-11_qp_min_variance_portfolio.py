"""Adversary cross-check: minimum-variance portfolio (Markowitz, equality-only QP)
Family: qp   Class: equality-constrained convex QP, closed-form KKT optimum
Source: classic Markowitz minimum-variance portfolio (Markowitz 1952; Boyd &
        Vandenberghe, Convex Optimization, sec 4 exercises). Long/short allowed
        (no bounds), fully-invested constraint 1^T x = 1.
        Closed form: x* = Sigma^{-1} 1 / (1^T Sigma^{-1} 1),
                     obj* = 1/2 x*^T Sigma x* = 1 / (2 * 1^T Sigma^{-1} 1).
Known optimal: computed below from the closed form.
"""
import time
import numpy as np

Sigma = np.array([
    [0.10, 0.02, 0.01],
    [0.02, 0.08, 0.03],
    [0.01, 0.03, 0.06],
])
ones = np.ones(3)
Sigma_inv = np.linalg.inv(Sigma)
x_star = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)
KNOWN_OPTIMAL = float(0.5 * x_star @ Sigma @ x_star)
print(f"closed-form x*={x_star}")
print(f"closed-form obj*={KNOWN_OPTIMAL:.10e}")

# --- pounce ---
from pounce import solve_qp

A_eq = ones.reshape(1, 3)
b_eq = np.array([1.0])

t0 = time.perf_counter()
r = solve_qp(P=Sigma, c=np.zeros(3), A=A_eq, b=b_eq)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)
obj_pounce = r.obj
status = r.status
print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s x={x_pounce}")

# --- oracle: cvxpy (Clarabel) ---
import cvxpy as cp

x = cp.Variable(3)
prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Sigma)), [ones @ x == 1])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
obj_oracle = prob.value
x_oracle = x.value
print("=== oracle (cvxpy/Clarabel) ===")
print(f"status={prob.status} obj={obj_oracle:.10e} t={t_oracle:.4f}s x={x_oracle}")


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_oracle = rel(obj_pounce, obj_oracle)
x_err_known = float(np.linalg.norm(x_pounce - x_star, np.inf))
x_err_oracle = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={obj_err_known:.2e} x_inf_err_vs_known={x_err_known:.2e}")
print(f"obj_err_vs_oracle={obj_err_oracle:.2e} x_inf_err_vs_oracle={x_err_oracle:.2e}")

ok = (status == "optimal") and obj_err_known < 1e-6 and obj_err_oracle < 1e-6 and x_err_known < 1e-5
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err_known={obj_err_known:.2e}, obj_err_oracle={obj_err_oracle:.2e})")
