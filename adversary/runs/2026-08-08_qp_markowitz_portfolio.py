"""Adversary cross-check: Markowitz minimum-variance portfolio QP.

Family: qp   Class: convex QP, equality (budget) + box (no-short, cap)
              constraints simultaneously active -- exercises the IPM's
              combined equality/inequality/bound KKT block
              (pounce-convex/ipm.rs).
Source: Markowitz (1952), "Portfolio Selection", J. Finance 7(1); textbook
        formulation as in Boyd & Vandenberghe, Convex Optimization, Sec 4.4.1
        (minimum-variance frontier), with box bounds 0 <= w_i <= 0.4 (no
        short-selling, no >40% concentration) added to force an active
        upper bound at the optimum.

4 assets, hand-picked diagonal covariance (uncorrelated) so the
minimum-variance-for-target-return solution is derivable in closed form via
KKT/Lagrange multipliers, then verified independently by cvxpy and by scipy
SLSQP.

    min   (1/2) w^T Sigma w
    s.t.  sum(w) == 1
          mu^T w == target_return
          0 <= w <= 0.4

Sigma = diag(sigma_i^2), sigma^2 = [0.04, 0.09, 0.16, 0.01]  (asset 4 is
low-variance, high-Sharpe: its unconstrained-by-cap optimal weight at the
target return below is ~0.209, so capping it at 0.15 forces that upper
bound to bind at the optimum, on top of the two equalities).
mu = [0.08, 0.12, 0.15, 0.05], target_return = 0.10.
"""

import time

import numpy as np

sigma2 = np.array([0.04, 0.09, 0.16, 0.01])
Sigma = np.diag(sigma2)
mu = np.array([0.08, 0.12, 0.15, 0.05])
target_return = 0.10
n = 4

P = Sigma
c = np.zeros(n)
A = np.array([np.ones(n), mu])
b = np.array([1.0, target_return])
lb = np.zeros(n)
ub = np.array([0.4, 0.4, 0.4, 0.15])

# --- pounce ---
from pounce import solve_qp

t0 = time.perf_counter()
r = solve_qp(P=P, c=c, A=A, b=b, lb=lb, ub=ub)
t_pounce = time.perf_counter() - t0

# --- oracle 1: cvxpy (independent DCP formulation) ---
import cvxpy as cp

w = cp.Variable(n)
prob = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(w, Sigma)),
    [cp.sum(w) == 1.0, mu @ w == target_return, w >= lb, w <= ub],
)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_cvxpy = time.perf_counter() - t0

# --- oracle 2: scipy SLSQP (different algorithm family entirely) ---
from scipy.optimize import minimize, LinearConstraint, Bounds

t0 = time.perf_counter()
res = minimize(
    lambda w: 0.5 * w @ Sigma @ w,
    x0=np.full(n, 0.25),
    jac=lambda w: Sigma @ w,
    method="SLSQP",
    bounds=Bounds(lb, ub),
    constraints=[LinearConstraint(A, b, b)],
    tol=1e-12,
)
t_scipy = time.perf_counter() - t0


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


print("=== pounce ===")
print(f"status={r.status} obj={r.obj:.10e} w={np.asarray(r.x)} t={t_pounce:.4f}s")
print("=== cvxpy/Clarabel ===")
print(f"status={prob.status} obj={prob.value:.10e} w={w.value} t={t_cvxpy:.4f}s")
print("=== scipy SLSQP ===")
print(f"success={res.success} obj={res.fun:.10e} w={res.x} t={t_scipy:.4f}s")

obj_err_cvxpy = rel(r.obj, prob.value)
obj_err_scipy = rel(r.obj, res.fun)
x_err_cvxpy = float(np.linalg.norm(np.asarray(r.x) - w.value, np.inf))
x_err_scipy = float(np.linalg.norm(np.asarray(r.x) - res.x, np.inf))
# Active-bound check: asset 4 (index 3) should sit at its upper bound.
at_ub = abs(r.x[3] - 0.15) < 1e-5

print(f"obj_err_vs_cvxpy={obj_err_cvxpy:.2e} x_err_vs_cvxpy={x_err_cvxpy:.2e}")
print(f"obj_err_vs_scipy={obj_err_scipy:.2e} x_err_vs_scipy={x_err_scipy:.2e}")
print(f"asset4_at_upper_bound(0.15)={at_ub} (w4={r.x[3]:.6f})")

ok = (
    r.status == "optimal"
    and obj_err_cvxpy < 1e-6
    and obj_err_scipy < 1e-6
    and x_err_cvxpy < 1e-5
    and x_err_scipy < 1e-5
    and at_ub
)
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
