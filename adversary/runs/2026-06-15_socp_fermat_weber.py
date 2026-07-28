"""Adversary cross-check: Fermat-Weber / minimize-sum-of-norms (facility location) as SOCP.
Family: socp   Class: minimize-sum-of-norms (Fermat-Weber facility location)

Source: Boyd & Vandenberghe, "Convex Optimization" (2004), Sec. 8.7 / 4.3.1
  (SOCP, "minimize sum of norms"); MOSEK Modeling Cookbook (facility location).
  The single-facility Fermat-Weber problem:
      minimize   sum_i ||x - p_i||_2
  over the facility location x in R^2 (unit weights).

Known optimum (closed form): for an EQUILATERAL triangle of side s, every
  interior angle is 60 deg < 120 deg, so the Fermat (geometric-median) point is
  the centroid, which sees each pair of vertices at 120 deg. The optimal cost is
      f* = s * sqrt(3).
  Reason: the centroid-to-vertex distance is s/sqrt(3); summed over 3 vertices
  gives 3 * s/sqrt(3) = s*sqrt(3). The minimizer is the centroid.

  We use an equilateral triangle with side s = 1, centered at the origin:
      p0 = (0, 1/sqrt(3))
      p1 = (-1/2, -1/(2*sqrt(3)))
      p2 = ( 1/2, -1/(2*sqrt(3)))
  (circumradius = 1/sqrt(3); side length = 1.) The Fermat point is (0, 0) and
      f* = sqrt(3) = 1.7320508075688772.

KNOWN_OPTIMAL f* = sqrt(3); minimizer x* = (0, 0).

Cone layout: variables v = (x1, x2, t0, t1, t2).  Minimize t0 + t1 + t2.
  For each point p_i, one SOC cone of dim 3:
     s = (t_i, x1 - p_i_x, x2 - p_i_y) in K_soc^3  with  t_i >= ||x - p_i||_2.
  s = h - G v, so for point i (t-index 2+i):
     s0 = t_i        -> G row has -1 in col (2+i),        h  0
     s1 = x1 - p_ix  -> G row has -1 in col 0,            h -p_ix
     s2 = x2 - p_iy  -> G row has -1 in col 1,            h -p_iy
  3 points -> 3 soc cones each dim 3 -> 9 total constraint rows.
"""
import time
import numpy as np

s3 = np.sqrt(3.0)
P = np.array([
    [0.0,  1.0 / s3],
    [-0.5, -1.0 / (2.0 * s3)],
    [0.5,  -1.0 / (2.0 * s3)],
])
n = P.shape[0]            # 3 points
nv = 2 + n               # x1, x2, t0..t2

# sanity: side lengths should all be 1.0
sides = [np.linalg.norm(P[i] - P[j]) for i, j in [(0, 1), (1, 2), (2, 0)]]
assert np.allclose(sides, 1.0), sides

KNOWN_OPTIMAL = s3       # sqrt(3)
KNOWN_X = np.array([0.0, 0.0])

# objective: minimize sum t_i
c = np.zeros(nv)
c[2:] = 1.0

G_rows = []
h_vals = []
cones = []
for i, (px, py) in enumerate(P):
    # s0 = t_i
    row = np.zeros(nv); row[2 + i] = -1.0
    G_rows.append(row); h_vals.append(0.0)
    # s1 = x1 - px
    row = np.zeros(nv); row[0] = -1.0
    G_rows.append(row); h_vals.append(-px)
    # s2 = x2 - py
    row = np.zeros(nv); row[1] = -1.0
    G_rows.append(row); h_vals.append(-py)
    cones.append(("soc", 3))

G = np.array(G_rows)
h = np.array(h_vals)

# --- pounce conic IPM ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
v_pounce = np.asarray(r.x)
x_pounce = v_pounce[:2]
obj_pounce = float(r.obj)
status = r.status

# --- oracle 1: cvxpy CLARABEL ---
import cvxpy as cp
def build():
    x = cp.Variable(2)
    cons = []
    t = cp.Variable(n)
    for i in range(n):
        cons.append(cp.SOC(t[i], x - P[i]))
    return cp.Problem(cp.Minimize(cp.sum(t)), cons), x

prob, x_v = build()
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
obj_oracle = prob.value
x_oracle = np.asarray(x_v.value)

# --- oracle 2: cvxpy ECOS ---
prob2, x_v2 = build()
prob2.solve(solver=cp.ECOS)
obj_oracle2 = prob2.value

# --- independent check: evaluate true objective at returned point ---
def f(x):
    return float(sum(np.linalg.norm(x - P[i]) for i in range(n)))

f_at_pounce = f(x_pounce)
f_at_known = f(KNOWN_X)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

obj_err = rel(obj_pounce, obj_oracle)
obj_err2 = rel(obj_pounce, obj_oracle2)
known_err = rel(obj_pounce, KNOWN_OPTIMAL)
x_err = float(np.linalg.norm(x_pounce - KNOWN_X, np.inf))
eval_err = rel(obj_pounce, f_at_pounce)   # epigraph tight?

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} x={x_pounce} t={t_pounce:.4f}s")
print(f"  true f(x_pounce)={f_at_pounce:.10e} (epigraph-tight err={eval_err:.2e})")
print("=== oracle (cvxpy/CLARABEL) ===")
print(f"obj={obj_oracle:.10e} x={x_oracle} t={t_oracle:.4f}s")
print("=== oracle2 (cvxpy/ECOS) ===")
print(f"obj={obj_oracle2:.10e}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} x=(0,0)  f(x*)={f_at_known:.10e}")
print(f"rel_err_vs_known={known_err:.2e} x_inf_err={x_err:.2e}")
print(f"obj_err_vs_clarabel={obj_err:.2e} obj_err_vs_ecos={obj_err2:.2e}")
print(f"clarabel_vs_ecos={rel(obj_oracle, obj_oracle2):.2e}")

ok = ((status == "optimal" or getattr(r, "success", False))
      and obj_err < 1e-4 and obj_err2 < 1e-4 and known_err < 1e-4
      and x_err < 1e-3 and eval_err < 1e-6)
if ok:
    print("VERDICT: PASS")
else:
    print(f"VERDICT: FAIL (status={status}, known_err={known_err:.2e}, "
          f"obj_err_clarabel={obj_err:.2e}, x_err={x_err:.2e}, "
          f"epigraph_err={eval_err:.2e})")
