"""Adversary cross-check: geometric program min x + 1/x via the exponential cone
Family: exp   Class: exponential-cone (geometric programming)
Source: pounce solve_socp docstring example + standard GP. With u = log x,
  min x + 1/x = min_u e^u + e^{-u}, whose global optimum is 2 at u = 0 (x = 1).
  Encoded with two exp cones: (u,1,t1) -> e^u <= t1 ; (-u,1,t2) -> e^{-u} <= t2.
Known optimal: 2.0
"""
import time
import numpy as np

KNOWN_OPTIMAL = 2.0

# variables (u, t1, t2); minimize t1 + t2
c = np.array([0.0, 1.0, 1.0])
G = np.zeros((6, 3))
G[0, 0] = -1.0  # s0 = u
G[2, 1] = -1.0  # s2 = t1
G[3, 0] = 1.0   # s3 = -u
G[5, 2] = -1.0  # s5 = t2
h = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])

# --- pounce conic IPM (non-symmetric exp-cone driver) ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h, cones=[("exp", 3), ("exp", 3)])
t_pounce = time.perf_counter() - t0
v = np.asarray(r.x)
obj_pounce = r.obj
u_star = v[0]
status = r.status

# --- oracle: cvxpy (exp cone via inv_pos) ---
import cvxpy as cp
x = cp.Variable(pos=True)
prob = cp.Problem(cp.Minimize(x + cp.inv_pos(x)))
t0 = time.perf_counter()
prob.solve(solver=cp.SCS)
t_oracle = time.perf_counter() - t0
obj_oracle = prob.value


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

obj_err = rel(obj_pounce, obj_oracle)

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} u*={u_star:.6e} (x=e^u={np.exp(u_star):.6f}) t={t_pounce:.4f}s")
print("=== oracle (cvxpy/SCS) ===")
print(f"obj={obj_oracle:.10e} t={t_oracle:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e}")

ok = (status == "optimal" or r.success) and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e})")
