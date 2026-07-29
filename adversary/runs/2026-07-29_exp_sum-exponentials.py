"""Adversary cross-check: unconstrained minimization of a sum of exponentials
Family: exp   Class: GP / log-sum-exp style unconstrained convex program
Source: minimize f(x) = exp(x1) + exp(-x1+0.5) + exp(x2-0.3) + exp(-x2+0.2)
        Separable in x1, x2. Each 1-D piece exp(u)+exp(c-u) is minimized where
        the two exponents are equal (u = c - u => u = c/2), by AM-GM /
        elementary calculus (d/du = exp(u) - exp(c-u) = 0):
            x1* = 0.25 (c=0.5),  f1* = 2*exp(0.25)
            x2* = 0.25 (c=0.5, shifted by -0.3/+0.2), f2* = 2*exp(-0.05)
        Known optimal f* = 2*exp(0.25) + 2*exp(-0.05), x* = (0.25, 0.25).
Known optimal: 2*exp(0.25) + 2*exp(-0.05)
"""
import time
import numpy as np

KNOWN_X = np.array([0.25, 0.25])
KNOWN_OPTIMAL = 2 * np.exp(0.25) + 2 * np.exp(-0.05)

# Terms: t_i >= exp(a_i . x + b_i), minimize sum t_i
A = np.array([
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
])
B = np.array([0.0, 0.5, -0.3, 0.2])
M = len(B)
N = 2

# --- pounce: solve_socp with 4 exp cones ---
from pounce import solve_socp

n_vars = N + M  # x1, x2, t1..t4
c = np.zeros(n_vars)
c[N:] = 1.0

rows = 3 * M
G = np.zeros((rows, n_vars))
h = np.zeros(rows)
cones = []
for i in range(M):
    r0 = 3 * i
    # s0 = a_i . x + b_i  =>  G row = -a_i (on x block), h = b_i
    G[r0, 0:N] = -A[i]
    h[r0] = B[i]
    # s1 = 1 (fixed y-slot of Kexp)
    h[r0 + 1] = 1.0
    # s2 = t_i
    G[r0 + 2, N + i] = -1.0
    cones.append(("exp", 3))

t0 = time.perf_counter()
r = solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)[:N]
obj_pounce = r.obj
status = r.status

# --- oracle 1: scipy, direct smooth unconstrained minimization (independent) ---
from scipy.optimize import minimize as scipy_minimize

def f(x):
    return np.sum(np.exp(A @ x + B))

def grad(x):
    e = np.exp(A @ x + B)
    return A.T @ e

t0 = time.perf_counter()
res_scipy = scipy_minimize(f, x0=np.zeros(N), jac=grad, method="BFGS",
                            tol=1e-14, options={"gtol": 1e-12})
t_scipy = time.perf_counter() - t0
x_scipy = res_scipy.x
obj_scipy = res_scipy.fun

# --- oracle 2: cvxpy, DCP directly via cp.exp (no explicit exp-cone construction) ---
import cvxpy as cp

x = cp.Variable(N)
obj_cvx = cp.Minimize(cp.sum(cp.exp(A @ x + B)))
prob = cp.Problem(obj_cvx)
t0 = time.perf_counter()
prob.solve(solver=cp.SCS)
t_cvx = time.perf_counter() - t0
x_cvx = np.asarray(x.value)
obj_cvx_val = prob.value


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_scipy = rel(obj_pounce, obj_scipy)
obj_err_cvx = rel(obj_pounce, obj_cvx_val)
x_err_known = float(np.linalg.norm(x_pounce - KNOWN_X, np.inf))
x_err_scipy = float(np.linalg.norm(x_pounce - x_scipy, np.inf))
x_err_cvx = float(np.linalg.norm(x_pounce - x_cvx, np.inf))

print("=== pounce (solve_socp, exp cones) ===")
print(f"status={status} obj={obj_pounce:.10e} x={x_pounce} t={t_pounce:.4f}s")
print("=== oracle: scipy BFGS (independent, smooth unconstrained) ===")
print(f"obj={obj_scipy:.10e} x={x_scipy} t={t_scipy:.4f}s")
print("=== oracle: cvxpy SCS (DCP cp.exp, not exp-cone construction) ===")
print(f"obj={obj_cvx_val:.10e} x={x_cvx} t={t_cvx:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} x*={KNOWN_X}")
print(f"obj_err_vs_known={obj_err_known:.2e} obj_err_vs_scipy={obj_err_scipy:.2e} obj_err_vs_cvxpy={obj_err_cvx:.2e}")
print(f"x_err_vs_known={x_err_known:.2e} x_err_vs_scipy={x_err_scipy:.2e} x_err_vs_cvxpy={x_err_cvx:.2e}")

ok = (status == "optimal") and obj_err_known < 1e-6 and obj_err_scipy < 1e-6 and obj_err_cvx < 1e-6
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status})")
