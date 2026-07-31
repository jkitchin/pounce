"""Adversary cross-check: Euclidean projection onto a box-capped simplex
Family: qp   Class: equality+box convex QP (closed-form via water-filling)
Source: Euclidean projection onto {x : sum(x)=s, lo<=x<=hi} is a classic QP
        solvable by a monotone bisection on the equality-constraint's
        Lagrange multiplier (a "water-filling" argument) -- see e.g. Boyd &
        Vandenberghe, "Convex Optimization" (2004), Sec 5.5.3 example, and
        Duchi, Shalev-Shwartz, Singer, Chandra, "Efficient Projections onto
        the l1-Ball" (ICML 2008) Sec 3 for the box-simplex generalization.
        For fixed multiplier nu, x_i(nu) = clip(a_i - nu, lo_i, hi_i) is
        monotone decreasing in nu, so bisecting on g(nu)=sum(x_i(nu))-s to
        zero gives the exact KKT point independent of any QP solver.
Known optimal: computed analytically below (bisection oracle), not a fixed
        literature constant, but a solver-independent closed-form method.
"""
import numpy as np
import time

n = 5
a = np.array([1.0, 4.0, -2.0, 6.0, 0.5])
lo = np.zeros(n)
hi = np.full(n, 3.0)
s = 5.0

# --- closed-form oracle: bisection on the equality multiplier nu ---
def x_of_nu(nu):
    return np.clip(a - nu, lo, hi)

lo_nu, hi_nu = -20.0, 20.0
for _ in range(200):
    mid = 0.5 * (lo_nu + hi_nu)
    if np.sum(x_of_nu(mid)) > s:
        lo_nu = mid
    else:
        hi_nu = mid
nu_star = 0.5 * (lo_nu + hi_nu)
x_bisect = x_of_nu(nu_star)
obj_bisect = 0.5 * np.sum((x_bisect - a) ** 2)

# --- pounce ---
from pounce import solve_qp

P = np.eye(n)
c = -a
A = np.ones((1, n))
b = np.array([s])

t0 = time.perf_counter()
r = solve_qp(P=P, c=c, A=A, b=b, lb=lo, ub=hi)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)
obj_pounce = 0.5 * np.sum((x_pounce - a) ** 2)
status = r.status

# --- oracle: cvxpy (independent conic solver) ---
import cvxpy as cp

xv = cp.Variable(n)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(xv - a)), [cp.sum(xv) == s, xv >= lo, xv <= hi])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle = xv.value
obj_oracle = prob.value


def rel(u, v):
    return abs(u - v) / max(1.0, abs(v))


obj_err_oracle = rel(obj_pounce, obj_oracle)
obj_err_bisect = rel(obj_pounce, obj_bisect)
x_err_oracle = float(np.linalg.norm(x_pounce - x_oracle, np.inf))
x_err_bisect = float(np.linalg.norm(x_pounce - x_bisect, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s x={x_pounce}")
print("=== oracle: bisection water-filling (closed form) ===")
print(f"obj={obj_bisect:.10e} nu*={nu_star:.6f} x={x_bisect}")
print("=== oracle: cvxpy CLARABEL ===")
print(f"obj={obj_oracle:.10e} t={t_oracle:.4f}s x={x_oracle}")
print(f"obj_err_vs_bisect={obj_err_bisect:.2e} x_inf_err_vs_bisect={x_err_bisect:.2e}")
print(f"obj_err_vs_cvxpy={obj_err_oracle:.2e} x_inf_err_vs_cvxpy={x_err_oracle:.2e}")

ok = status == "optimal" and obj_err_bisect < 1e-6 and obj_err_oracle < 1e-6
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status})")
