"""Adversary i4: THREE-FACTOR weighted geometric-mean maximization.
Family: power   Class: chained 3-D power cones (two pow cones composed).

Problem:  maximize x^0.2 y^0.3 z^0.5  s.t.  x+y+z <= 3,  x,y,z >= 0.
Weights w=(0.2,0.3,0.5), sum=1.  Weighted AM-GM / Lagrange under a linear
budget S gives x_i = w_i * S.  S=3 -> x=0.6, y=0.9, z=1.5,
value = 0.6^0.2 * 0.9^0.3 * 1.5^0.5.
This is DISTINCT from the logged 2-factor weighted geomean: the 3-factor
mean requires COMPOSING two power cones, exercising the cone-chaining path.

Encoding t <= x^0.2 y^0.3 z^0.5 with two pow cones:
  u <= x^0.4 y^0.6         (a1 = 0.2/(0.2+0.3) = 0.4)   -> u = x^0.4 y^0.6
  t <= u^0.5 z^0.5         (a2 = 0.5)                    -> t <= x^0.2 y^0.3 z^0.5
Variables [x,y,z,u,t]. Cone1 slack (u,x,y) a=0.4; Cone2 slack (t,u,z) a=0.5;
plus nonneg 3-(x+y+z).  minimize -t.
"""
import time
import numpy as np

w = np.array([0.2, 0.3, 0.5])
S = 3.0
x_star = w * S                       # 0.6, 0.9, 1.5
KNOWN_OPTIMAL = float(np.prod(x_star ** w))

# vars: x0 y1 z2 u3 t4
nv = 5
c = np.zeros(nv); c[4] = -1.0        # min -t

rows, h = [], []
def push(row, hv):
    rows.append(row); h.append(hv)
# cone1: (s0=u, s1=x, s2=y)
r = np.zeros(nv); r[3] = -1.0; push(r, 0.0)
r = np.zeros(nv); r[0] = -1.0; push(r, 0.0)
r = np.zeros(nv); r[1] = -1.0; push(r, 0.0)
# cone2: (s0=t, s1=u, s2=z)
r = np.zeros(nv); r[4] = -1.0; push(r, 0.0)
r = np.zeros(nv); r[3] = -1.0; push(r, 0.0)
r = np.zeros(nv); r[2] = -1.0; push(r, 0.0)
# nonneg: 3 - (x+y+z) >= 0
r = np.zeros(nv); r[0] = 1.0; r[1] = 1.0; r[2] = 1.0; push(r, S)
G = np.array(rows); h = np.array(h)
cones = [("pow", 0.4), ("pow", 0.5), ("nonneg", 1)]

import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
v = np.asarray(r.x, float)
xyz = v[:3]
obj_pounce = float(np.prod(np.maximum(xyz, 0.0) ** w))
status = str(r.status)

import cvxpy as cp
def solve_cvxpy(solver):
    xv = cp.Variable(3, nonneg=True)
    prob = cp.Problem(cp.Maximize(cp.geo_mean(xv, p=list(w))), [cp.sum(xv) <= S])
    t0 = time.perf_counter(); prob.solve(solver=solver); dt = time.perf_counter() - t0
    return float(prob.value), dt, np.asarray(xv.value, float)

val_cla, t_cla, x_cla = solve_cvxpy(cp.CLARABEL)
val_scs, t_scs, x_scs = solve_cvxpy(cp.SCS)

def rel(a, b): return abs(a - b) / max(1.0, abs(b))
budget = float(np.sum(xyz))

print("=== pounce (chained power-cone IPM) ===")
print(f"status={status} max_val={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"  xyz={xyz}  sum={budget:.6f}(<=3)  x_star={x_star}")
print(f"=== cvxpy/CLARABEL max_val={val_cla:.10e} t={t_cla:.4f}s x={x_cla}")
print(f"=== cvxpy/SCS      max_val={val_scs:.10e} t={t_scs:.4f}s")
print(f"known={KNOWN_OPTIMAL:.10e}")
print(f"rel vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e} "
      f"vs_CLARABEL={rel(obj_pounce, val_cla):.2e} vs_SCS={rel(obj_pounce, val_scs):.2e}")
x_err = float(np.linalg.norm(xyz - x_star, np.inf))
print(f"x_inf_err vs x_star = {x_err:.2e}")

ok = (status == "optimal" or getattr(r, "success", False)) \
    and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 and budget <= S + 1e-6 and x_err < 1e-3
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err={rel(obj_pounce, KNOWN_OPTIMAL):.2e}, x_err={x_err:.2e})")
