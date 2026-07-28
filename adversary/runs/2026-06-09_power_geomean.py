"""Adversary cross-check: geometric-mean maximization via the power cone
Family: power   Class: 3-D power-cone program
Source: standard power-cone model of the geometric mean.
  maximize (x y)^{1/2}  s.t.  x + y <= 2,  x,y >= 0.
  By AM-GM the optimum is at x = y = 1, value 1.
Known optimal: 1.0  (minimization form: -1.0)
"""
import time
import numpy as np

KNOWN_OPTIMAL = 1.0

# variables (x, y, t); maximize t with (x,y,t) in pow(0.5).
# pounce convention: ("pow", a) = {(s0,s1,s2): |s0| <= s1^a s2^(1-a)} — the
# FIRST slack is the bounded one. So the cone slack order must be (t, x, y):
# |t| <= x^0.5 y^0.5.  minimize -t.
c = np.array([0.0, 0.0, -1.0])
# rows 0..2 : power cone (s0=t, s1=x, s2=y); row 3 : nonneg (x+y<=2)
G = np.array([
    [0.0, 0.0, -1.0],   # s0 = t   (bounded component)
    [-1.0, 0.0, 0.0],   # s1 = x
    [0.0, -1.0, 0.0],   # s2 = y
    [1.0, 1.0, 0.0],    # s3 = 2 - (x+y) >= 0
])
h = np.array([0.0, 0.0, 0.0, 2.0])

# --- pounce conic IPM ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h, cones=[("pow", 0.5), ("nonneg", 1)])
t_pounce = time.perf_counter() - t0
v = np.asarray(r.x)
obj_pounce = -r.obj  # maximization value
status = r.status

# --- oracle: cvxpy geo_mean ---
import cvxpy as cp
xy = cp.Variable(2, nonneg=True)
prob = cp.Problem(cp.Maximize(cp.geo_mean(xy)), [cp.sum(xy) <= 2])
t0 = time.perf_counter()
prob.solve(solver=cp.SCS)
t_oracle = time.perf_counter() - t0
obj_oracle = prob.value


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

obj_err = rel(obj_pounce, obj_oracle)

print("=== pounce ===")
print(f"status={status} max_val={obj_pounce:.10e} (x,y,t)={v} t={t_pounce:.4f}s")
print("=== oracle (cvxpy/SCS) ===")
print(f"max_val={obj_oracle:.10e} xy={xy.value} t={t_oracle:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e}")

ok = (status == "optimal" or r.success) and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e})")
