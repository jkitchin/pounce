"""Adversary cross-check: largest-eigenvalue SDP
Family: sdp   Class: small dense semidefinite program
Source: textbook SDP for lambda_max.  min t s.t. t I - A >= 0  gives
  t* = lambda_max(A).  For A = [[2,1],[1,2]], eigenvalues {1,3}, so t* = 3.
Known optimal: 3.0
"""
import time
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 2.0]])
KNOWN_OPTIMAL = float(np.max(np.linalg.eigvalsh(A)))  # 3.0

# variable v = (t,).  PSD slack block s = svec(t I - A), smat(s) >= 0.
# 2x2 svec = [M00, sqrt(2)*M10, M11]; M = t I - A = [[t-2,-1],[-1,t-2]].
#   s0 = t - 2          -> h0=-2, G0=-1
#   s1 = -sqrt(2)       -> h1=-sqrt(2), G1=0
#   s2 = t - 2          -> h2=-2, G2=-1
s2 = np.sqrt(2.0)
c = np.array([1.0])
G = np.array([[-1.0], [0.0], [-1.0]])
h = np.array([-2.0, -s2, -2.0])

# --- pounce conic IPM (PSD cone) ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h, cones=[("psd", 2)])
t_pounce = time.perf_counter() - t0
obj_pounce = np.asarray(r.x)[0]
status = r.status

# --- oracle: cvxpy SDP ---
import cvxpy as cp
tv = cp.Variable()
prob = cp.Problem(cp.Minimize(tv), [tv * np.eye(2) - A >> 0])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
obj_oracle = prob.value


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

obj_err = rel(obj_pounce, obj_oracle)

print("=== pounce ===")
print(f"status={status} t*={obj_pounce:.10e} t={t_pounce:.4f}s")
print("=== oracle (cvxpy/CLARABEL) ===")
print(f"t*={obj_oracle:.10e} t={t_oracle:.4f}s")
print(f"known_optimal(lambda_max)={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e}")

ok = (status == "optimal" or r.success) and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e})")
