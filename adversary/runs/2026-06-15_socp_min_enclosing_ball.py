"""Adversary cross-check: minimum enclosing ball (smallest enclosing sphere) as SOCP.
Family: socp   Class: SOC-constrained (minimum enclosing ball)

Source: Boyd & Vandenberghe, "Convex Optimization" (2004), Sec. 8.5.1
  "Smallest enclosing ball" / Chebyshev center style problem. The minimum
  enclosing ball of a point set {p_i} is:
      minimize   R
      subject to ||x_c - p_i||_2 <= R   for all i
  over (R, x_c).

Known optimum: the four corners of the unit square
  p = (0,0),(1,0),(0,1),(1,1). By symmetry the MEB is centered at (0.5,0.5)
  and its radius is the distance to any corner = sqrt(0.5^2+0.5^2)
  = sqrt(0.5) = 1/sqrt(2) = 0.70710678118654752.
  (For a set of points, the MEB of the corners of a square is the
  circumscribed circle; center = centroid, radius = half the diagonal.)

KNOWN_OPTIMAL R* = 1/sqrt(2) = 0.7071067811865476, center = (0.5, 0.5).

Cone layout: variables v = (R, x1, x2). Minimize R.
  For each point p_i, one SOC cone of dim 3:
     s = (R, x1 - p_i_x, x2 - p_i_y) in K_soc^3  with  R >= ||x_c - p_i||.
  s = h - G v, so:
     s0 = R         -> G row [-1, 0, 0],  h 0
     s1 = x1 - p_ix -> G row [ 0,-1, 0],  h -p_ix
     s2 = x2 - p_iy -> G row [ 0, 0,-1],  h -p_iy
  4 points -> 4 soc cones each dim 3 -> 12 total constraint rows.
"""
import time
import numpy as np

P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
n = P.shape[0]
KNOWN_OPTIMAL = 1.0 / np.sqrt(2.0)
KNOWN_CENTER = np.array([0.5, 0.5])

# Variables v = (R, x1, x2)
c = np.array([1.0, 0.0, 0.0])

G_rows = []
h_vals = []
cones = []
for px, py in P:
    # s0 = R
    G_rows.append([-1.0, 0.0, 0.0]); h_vals.append(0.0)
    # s1 = x1 - px
    G_rows.append([0.0, -1.0, 0.0]); h_vals.append(-px)
    # s2 = x2 - py
    G_rows.append([0.0, 0.0, -1.0]); h_vals.append(-py)
    cones.append(("soc", 3))

G = np.array(G_rows)
h = np.array(h_vals)

# --- pounce conic IPM ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
v_pounce = np.asarray(r.x)
R_pounce = v_pounce[0]
xc_pounce = v_pounce[1:]
status = r.status

# --- oracle 1: cvxpy CLARABEL ---
import cvxpy as cp
def build():
    xc = cp.Variable(2)
    R = cp.Variable()
    cons = [cp.SOC(R, xc - P[i]) for i in range(n)]
    return cp.Problem(cp.Minimize(R), cons), xc, R

prob, xc_v, R_v = build()
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
R_oracle = prob.value
xc_oracle = np.asarray(xc_v.value)

# --- oracle 2: cvxpy ECOS ---
prob2, xc_v2, R_v2 = build()
prob2.solve(solver=cp.ECOS)
R_oracle2 = prob2.value


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

obj_err = rel(R_pounce, R_oracle)
obj_err2 = rel(R_pounce, R_oracle2)
known_err = rel(R_pounce, KNOWN_OPTIMAL)
center_err = float(np.linalg.norm(xc_pounce - KNOWN_CENTER, np.inf))

print("=== pounce ===")
print(f"status={status} R*={R_pounce:.10e} center={xc_pounce} t={t_pounce:.4f}s")
print("=== oracle (cvxpy/CLARABEL) ===")
print(f"R*={R_oracle:.10e} center={xc_oracle} t={t_oracle:.4f}s")
print("=== oracle2 (cvxpy/ECOS) ===")
print(f"R*={R_oracle2:.10e}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} center=(0.5,0.5)")
print(f"rel_err_vs_known={known_err:.2e} center_inf_err={center_err:.2e}")
print(f"obj_err_vs_clarabel={obj_err:.2e} obj_err_vs_ecos={obj_err2:.2e}")
print(f"clarabel_vs_ecos={rel(R_oracle, R_oracle2):.2e}")

ok = ((status == "optimal" or getattr(r, "success", False))
      and obj_err < 1e-4 and obj_err2 < 1e-4 and known_err < 1e-4
      and center_err < 1e-3)
if ok:
    print("VERDICT: PASS")
else:
    print(f"VERDICT: FAIL (status={status}, known_err={known_err:.2e}, "
          f"obj_err_clarabel={obj_err:.2e}, center_err={center_err:.2e})")
