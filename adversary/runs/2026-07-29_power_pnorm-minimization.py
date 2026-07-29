"""Adversary cross-check: p-norm minimization over an affine hyperplane
Family: power   Class: p-norm epigraph via 3-D power cones (dual-norm identity)
Source: classic dual-norm fact (Boyd & Vandenberghe, "Convex Optimization",
        A.1.6 / example of Holder's inequality equality case):
            minimize ||x||_p   s.t.  a^T x = 1
        has optimal value 1/||a||_q  (1/p + 1/q = 1), attained at
            x_i* = sign(a_i) |a_i|^{q-1} / S,   S = sum_j |a_j|^q.
        Derivation of the value: at x*, |x_i*|^p = |a_i|^{(q-1)p}/S^p and
        (q-1)p = q since 1/p+1/q=1 => q-1 = 1/(p-1) => (q-1)p = p/(p-1) = q,
        so sum|x_i*|^p = S/S^p = S^{1-p}, giving ||x*||_p = S^{(1-p)/p} =
        S^{-1/q} = 1/||a||_q.  This is the standard p-norm/q-norm
        Holder-equality construction; verified independently below with
        cvxpy's native cp.norm(x, p) atom and a scipy SLSQP solve.
Known optimal: 1/||a||_q with p=3, q=1.5
"""
import time
import numpy as np

P = 3.0
Q = P / (P - 1.0)  # 1.5
a = np.array([2.0, -1.0, 3.0, -0.5])
n = len(a)

S = np.sum(np.abs(a) ** Q)
KNOWN_OPTIMAL = S ** (-1.0 / Q)
KNOWN_X = np.sign(a) * np.abs(a) ** (Q - 1.0) / S
# sanity: a @ KNOWN_X should be 1, ||KNOWN_X||_p should be KNOWN_OPTIMAL
assert abs(a @ KNOWN_X - 1.0) < 1e-12
assert abs(np.sum(np.abs(KNOWN_X) ** P) ** (1.0 / P) - KNOWN_OPTIMAL) < 1e-9

# --- pounce: solve_socp, variables [x_1..x_n, r_1..r_n, t] ---
# ||x||_p <= t  <=>  sum(r) = t  and  (x_i, r_i, t) in pow-cone(alpha=1/p) for each i
from pounce import solve_socp

nv = 2 * n + 1
x_idx = list(range(0, n))
r_idx = list(range(n, 2 * n))
t_idx = 2 * n

c = np.zeros(nv)
c[t_idx] = 1.0

A = np.zeros((2, nv))
b = np.zeros(2)
A[0, x_idx] = a
b[0] = 1.0
A[1, r_idx] = 1.0
A[1, t_idx] = -1.0
b[1] = 0.0

rows = 3 * n
G = np.zeros((rows, nv))
h = np.zeros(rows)
cones = []
alpha = 1.0 / P
for i in range(n):
    r0 = 3 * i
    G[r0, x_idx[i]] = -1.0       # s0 = x_i
    G[r0 + 1, r_idx[i]] = -1.0   # s1 = r_i
    G[r0 + 2, t_idx] = -1.0      # s2 = t
    cones.append(("pow", alpha))

t0 = time.perf_counter()
res = solve_socp(c=c, A=A, b=b, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(res.x)[x_idx]
obj_pounce = res.obj
status = res.status

# --- oracle 1: cvxpy native p-norm atom (independent of pow-cone construction) ---
import cvxpy as cp

xv = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.norm(xv, P)), [a @ xv == 1])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_cvx = time.perf_counter() - t0
x_cvx = np.asarray(xv.value)
obj_cvx = prob.value

# --- oracle 2: scipy SLSQP, direct nonlinear p-norm objective ---
from scipy.optimize import minimize as scipy_minimize, LinearConstraint

def f(x):
    return np.sum(np.abs(x) ** P) ** (1.0 / P)

lc = LinearConstraint(a.reshape(1, -1), 1.0, 1.0)
t0 = time.perf_counter()
res_scipy = scipy_minimize(f, x0=KNOWN_X + 0.01, constraints=[lc], method="SLSQP",
                            tol=1e-14, options={"maxiter": 500, "ftol": 1e-14})
t_scipy = time.perf_counter() - t0
x_scipy = res_scipy.x
obj_scipy = res_scipy.fun


def rel(u, v):
    return abs(u - v) / max(1.0, abs(v))


obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_cvx = rel(obj_pounce, obj_cvx)
obj_err_scipy = rel(obj_pounce, obj_scipy)
x_err_known = float(np.linalg.norm(x_pounce - KNOWN_X, np.inf))
x_err_cvx = float(np.linalg.norm(x_pounce - x_cvx, np.inf))
x_err_scipy = float(np.linalg.norm(x_pounce - x_scipy, np.inf))

print("=== pounce (solve_socp, pow cones, alpha=1/3) ===")
print(f"status={status} obj={obj_pounce:.10e} x={x_pounce} t={t_pounce:.4f}s")
print("=== oracle: cvxpy CLARABEL (native cp.norm(x,p) atom) ===")
print(f"obj={obj_cvx:.10e} x={x_cvx} t={t_cvx:.4f}s")
print("=== oracle: scipy SLSQP (direct nonlinear p-norm) ===")
print(f"obj={obj_scipy:.10e} x={x_scipy} t={t_scipy:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} x*={KNOWN_X}")
print(f"obj_err_vs_known={obj_err_known:.2e} obj_err_vs_cvxpy={obj_err_cvx:.2e} obj_err_vs_scipy={obj_err_scipy:.2e}")
print(f"x_err_vs_known={x_err_known:.2e} x_err_vs_cvxpy={x_err_cvx:.2e} x_err_vs_scipy={x_err_scipy:.2e}")

ok = (status == "optimal") and obj_err_known < 1e-6 and obj_err_cvx < 1e-6 and obj_err_scipy < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status})")
