"""Adversary cross-check: Chebyshev center of a polytope (LP)
Family: lp   Class: Chebyshev-center LP (largest inscribed ball), known closed form

Source: Boyd & Vandenberghe, "Convex Optimization", S4.3.1 (Chebyshev center).
The Chebyshev center of a polytope P = {x : a_i^T x <= b_i} is the center of
the largest Euclidean ball inscribed in P. It is the LP

    maximize   r
    subject to a_i^T x + ||a_i||_2 * r <= b_i,   i=1..m
               r >= 0

with variables (x, r).

Concrete polytope: the 3-4-5 right triangle with vertices (0,0), (4,0), (0,3).
Edges (interior = triangle):
    -x        <= 0          (x >= 0),   ||a||=1
        -y    <= 0          (y >= 0),   ||a||=1
    3x + 4y   <= 12         (hypotenuse through (4,0),(0,3)), ||a||=5

For a triangle the Chebyshev center IS the incenter and r IS the inradius.
3-4-5 right triangle: inradius r = (leg1 + leg2 - hyp)/2 = (3 + 4 - 5)/2 = 1.
Right angle at the origin with legs on the axes => incenter = (r, r) = (1, 1).

KNOWN OPTIMAL: r* = 1, center x* = (1, 1).  We MAXIMIZE r, so pounce (which
minimizes c'x with P=None) uses c = [0, 0, -1] over variables [x, y, r];
min objective = -r* = -1.
"""
import time
import numpy as np

KNOWN_R = 1.0
CENTER_STAR = np.array([1.0, 1.0])
KNOWN_OPTIMAL = -1.0  # minimization form (min -r)

# Edge half-planes a_i^T [x,y] <= b_i and their 2-norms.
A2 = np.array([[-1.0, 0.0],
               [0.0, -1.0],
               [3.0, 4.0]])
b = np.array([0.0, 0.0, 12.0])
norms = np.linalg.norm(A2, axis=1)  # [1, 1, 5]

# Variables z = [x, y, r].  Constraint: a_i^T[x,y] + ||a_i|| r <= b_i.
G = np.hstack([A2, norms.reshape(-1, 1)])
h = b.copy()
c = np.array([0.0, 0.0, -1.0])      # maximize r
lb = np.array([-np.inf, -np.inf, 0.0])  # r >= 0; x,y free (polytope bounds them)

# --- pounce (LP via convex IPM, P=None) ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_qp(P=None, c=c, G=G, h=h, lb=lb)
t_pounce = time.perf_counter() - t0
z_pounce, obj_pounce, status = np.asarray(r.x), r.obj, r.status
x_pounce = z_pounce[:2]
r_pounce = z_pounce[2]

# --- oracle 1: scipy linprog (HiGHS) ---
from scipy.optimize import linprog
t0 = time.perf_counter()
lp = linprog(c, A_ub=G, b_ub=h,
             bounds=[(None, None), (None, None), (0, None)])
t_oracle = time.perf_counter() - t0
z_oracle, obj_oracle = lp.x, lp.fun

# --- oracle 2: cvxpy ---
import cvxpy as cp
zv = cp.Variable(3)
prob = cp.Problem(cp.Minimize(c @ zv), [G @ zv <= h, zv[2] >= 0])
prob.solve(solver=cp.CLARABEL)
obj_cvx = prob.value


def rel(a, bb):
    return abs(a - bb) / max(1.0, abs(bb))


obj_err = rel(obj_pounce, obj_oracle)
z_err = float(np.linalg.norm(z_pounce - z_oracle, np.inf))

# feasibility of pounce solution (all G z <= h, and r>=0)
slack = h - G @ z_pounce
feas = float(min(slack.min(), r_pounce))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} z=[x,y,r]={z_pounce} t={t_pounce:.4f}s")
print(f"center={x_pounce} r={r_pounce:.10f}")
print(f"min feasibility (>=0 ok)={feas:.2e}")
print("=== oracle (linprog/HiGHS) ===")
print(f"status={lp.status} obj={obj_oracle:.10e} z={z_oracle} t={t_oracle:.4f}s")
print(f"cvxpy_obj={obj_cvx:.10e}")
print(f"known_optimal(min -r)={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"center_star={CENTER_STAR} known_r={KNOWN_R}")
print(f"obj_err_vs_oracle={obj_err:.2e} z_inf_err={z_err:.2e}")
print(f"center_err_vs_star={np.linalg.norm(x_pounce - CENTER_STAR, np.inf):.2e} r_err={abs(r_pounce-KNOWN_R):.2e}")

ok = (status == "optimal" or getattr(r, "success", False)) \
    and obj_err < 1e-4 and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 \
    and float(np.linalg.norm(x_pounce - CENTER_STAR, np.inf)) < 1e-4 \
    and feas > -1e-6
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, feas={feas:.2e})")
