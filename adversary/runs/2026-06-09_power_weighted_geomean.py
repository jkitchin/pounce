"""Adversary cross-check: WEIGHTED geometric-mean maximization (unequal weights).
Family: power   Class: 3-D power-cone program (Cobb-Douglas)
Source: Cobb-Douglas / weighted AM-GM closed form.
  maximize  x^w1 y^w2   s.t.  x + y <= c,  x,y >= 0,  w1+w2=1.
  Optimum (Cobb-Douglas under a budget): x* = w1*c, y* = w2*c,
  value = (w1 c)^w1 (w2 c)^w2.
  Here w1=0.6, w2=0.4, c=2 -> x*=1.2, y*=0.8, val=1.2^0.6 * 0.8^0.4.
This is DISTINCT from the equal-weight geomean already tested (w=0.5).

Power-cone encoding:  maximize t  s.t.  t <= x^w1 y^w2.
  pounce cone {(s0,s1,s2): |s0| <= s1^a s2^{1-a}} with s0=t, s1=x, s2=y,
  a = w1 = 0.6.  Plus nonneg slack for x+y<=c. minimize -t.
Variables: (x, y, t) -> 3 vars.
"""
import time
import numpy as np

w1, w2, cbud = 0.6, 0.4, 2.0
x_star, y_star = w1 * cbud, w2 * cbud
KNOWN_OPTIMAL = (x_star) ** w1 * (y_star) ** w2

# variables (x, y, t)
c = np.array([0.0, 0.0, -1.0])  # minimize -t
# rows 0..2: power cone slack (s0=t, s1=x, s2=y); row 3: nonneg (c - (x+y) >= 0)
G = np.array([
    [0.0, 0.0, -1.0],   # s0 = t
    [-1.0, 0.0, 0.0],   # s1 = x
    [0.0, -1.0, 0.0],   # s2 = y
    [1.0, 1.0, 0.0],    # s3 = c - (x+y) >= 0
])
h = np.array([0.0, 0.0, 0.0, cbud])

import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h, cones=[("pow", w1), ("nonneg", 1)])
t_pounce = time.perf_counter() - t0
v = np.asarray(r.x)
obj_pounce = -r.obj
status = r.status
x_p, y_p = v[0], v[1]

import cvxpy as cp


def solve_cvxpy(solver):
    xy = cp.Variable(2, nonneg=True)
    prob = cp.Problem(cp.Maximize(cp.geo_mean(xy, p=[w1, w2])),
                      [cp.sum(xy) <= cbud])
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    return prob.value, time.perf_counter() - t0, np.asarray(xy.value)


val_scs, t_scs, xy_scs = solve_cvxpy(cp.SCS)
val_cla, t_cla, xy_cla = solve_cvxpy(cp.CLARABEL)


def rel(a_, b_):
    return abs(a_ - b_) / max(1.0, abs(b_))


err_known = rel(obj_pounce, KNOWN_OPTIMAL)
err_scs = rel(obj_pounce, val_scs)
err_cla = rel(obj_pounce, val_cla)
err_x = abs(x_p - x_star)
err_y = abs(y_p - y_star)

print("=== pounce ===")
print(f"status={status} max_val={obj_pounce:.10e} (x,y,t)={v} t={t_pounce:.4f}s")
print(f"  x_err={err_x:.2e} y_err={err_y:.2e}  (x*={x_star}, y*={y_star})")
print("=== oracle cvxpy/SCS ===")
print(f"max_val={val_scs:.10e} xy={xy_scs} t={t_scs:.4f}s")
print("=== oracle cvxpy/CLARABEL ===")
print(f"max_val={val_cla:.10e} xy={xy_cla} t={t_cla:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e}")
print(f"rel_err vs_known={err_known:.2e}  vs_SCS={err_scs:.2e}  vs_CLARABEL={err_cla:.2e}")

ok = (status == "optimal" or r.success) and err_known < 1e-5 and err_x < 1e-4 and err_y < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, err_known={err_known:.2e}, x_err={err_x:.2e})")
