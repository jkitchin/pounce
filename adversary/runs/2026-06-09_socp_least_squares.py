"""Adversary cross-check: minimum-norm residual as a second-order cone program
Family: socp   Class: SOC-constrained (least-squares-as-SOCP)
Source: Boyd & Vandenberghe, "Convex Optimization", SOCP modeling of
  min ||A x - b||_2.  The optimum equals the least-squares residual norm.
  A = [[1,0],[0,1],[1,1]], b = [1,2,4].
  Normal equations -> x* = (4/3, 7/3), residual norm = 1/sqrt(3) = 0.57735.
Known optimal: t* = ||A x* - b||_2 = 0.5773502692
"""
import time
import numpy as np

A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
b = np.array([1.0, 2.0, 4.0])
x_ls = np.linalg.lstsq(A, b, rcond=None)[0]
KNOWN_OPTIMAL = float(np.linalg.norm(A @ x_ls - b))

# Variables v = (t, x1, x2). minimize t.
# SOC constraint: s = (t, A x - b) in K_soc^4, where s = h - G v.
#   s0 = t           -> G row [-1,0,0], h 0
#   s1 = x1 - 1      -> G row [0,-1,0], h -1
#   s2 = x2 - 2      -> G row [0,0,-1], h -2
#   s3 = x1+x2 - 4   -> G row [0,-1,-1], h -4
c = np.array([1.0, 0.0, 0.0])
G = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, -1.0, -1.0],
])
h = np.array([0.0, -1.0, -2.0, -4.0])

# --- pounce conic IPM ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h, cones=[("soc", 4)])
t_pounce = time.perf_counter() - t0
v_pounce = np.asarray(r.x)
t_star_pounce = v_pounce[0]
x_pounce = v_pounce[1:]
status = r.status

# --- oracle: cvxpy SOC ---
import cvxpy as cp
xv = cp.Variable(2)
tv = cp.Variable()
prob = cp.Problem(cp.Minimize(tv), [cp.SOC(tv, A @ xv - b)])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
t_star_oracle = prob.value
x_oracle = np.asarray(xv.value)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

obj_err = rel(t_star_pounce, t_star_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

print("=== pounce ===")
print(f"status={status} t*={t_star_pounce:.10e} x={x_pounce} t={t_pounce:.4f}s")
print("=== oracle (cvxpy/CLARABEL) ===")
print(f"t*={t_star_oracle:.10e} x={x_oracle} t={t_oracle:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(t_star_pounce, KNOWN_OPTIMAL):.2e}")
print(f"x_ls={x_ls} x_err_vs_ls={np.linalg.norm(x_pounce-x_ls, np.inf):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err={x_err:.2e}")

ok = (status == "optimal" or r.success) and obj_err < 1e-4 and rel(t_star_pounce, KNOWN_OPTIMAL) < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e})")
