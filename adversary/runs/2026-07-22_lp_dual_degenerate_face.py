"""Adversary cross-check: dual-degenerate LP with a NON-UNIQUE optimal face
Family: lp   Class: dual degeneracy (objective parallel to a facet) + primal
                    degeneracy (duplicated row) => optimal FACE and non-unique DUALS

Source: standard degeneracy construction, Bertsimas & Tsitsiklis, "Introduction
to Linear Optimization" (1997) Ch. 3.2 (degeneracy) and Ch. 4.5 / Thm 4.6
(multiplicity of optimal dual solutions <-> primal degeneracy); Chvatal,
"Linear Programming" (1983) Ch. 3.  Instance constructed here so the exact
rational optimal face is derivable in closed form.

  minimize   -3 x1 - 2 x2          (i.e. maximize 3x1 + 2x2)
  s.t. r1:  3 x1 + 2 x2 <= 12      <-- objective is PARALLEL to this facet
       r2:  6 x1 + 4 x2 <= 24      <-- exact duplicate of r1 (x2) => dual non-unique
       r3:    x1 +   x2 <=  5
       r4:    x1        <=  3
       x1, x2 >= 0

Optimal value:  -12  (3x1+2x2 = 12 on the whole optimal face).
Optimal FACE:   { (x1, (12-3x1)/2) : 2 <= x1 <= 3 }
                = conv{ (2, 3), (3, 3/2) }, a 1-dimensional edge.
   (x1 >= 2 comes from r3: x1 + (12-3x1)/2 <= 5; x1 <= 3 from r4.)

Degeneracy structure:
  * DUAL degeneracy: c is parallel to r1 => a continuum of primal optima.
  * PRIMAL degeneracy: at vertex (2,3) three constraints (r1, r2, r3) are
    active in R^2 => degenerate vertex => the dual optimum set is not a
    singleton.  Because r2 = 2*r1, any (z1, z2) >= 0 with z1 + 2*z2 = 1
    (and z3 = z4 = z_lb = 0) is an optimal dual.  Dual value -h'z = -12 for
    every such split.

Grading (per the assignment): x is NON-UNIQUE, so we do NOT grade x equality.
We grade
  (a) the OBJECTIVE exactly against the closed form / exact rational oracle,
  (b) FEASIBILITY of the returned x,
  (c) that x genuinely lies ON the optimal face, and
  (d) DUAL VALIDITY: dual feasibility (z >= 0), stationarity
      c + G'z - z_lb + z_ub = 0, complementary slackness, and strong duality
      (-h'z + lb-terms == primal objective).  Any valid dual is acceptable;
      an invalid one is a bug.

Oracles: exact rational vertex enumeration (fractions.Fraction) + scipy.linprog
(HiGHS) + cvxpy (CLARABEL).
"""
import itertools
import time
from fractions import Fraction

import numpy as np

# ----------------------------------------------------------------- problem
c = np.array([-3.0, -2.0])
G = np.array([[3.0, 2.0],
              [6.0, 4.0],
              [1.0, 1.0],
              [1.0, 0.0]])
h = np.array([12.0, 24.0, 5.0, 3.0])
lb = np.array([0.0, 0.0])
n = 2

KNOWN_OPTIMAL = -12.0
FACE_V1 = np.array([2.0, 3.0])
FACE_V2 = np.array([3.0, 1.5])

# ------------------------------------------- oracle 0: exact rational LP
# All constraints as rows  a.x <= b  (bounds -x_j <= 0 included).
Fc = [Fraction(-3), Fraction(-2)]
rows = [([Fraction(3), Fraction(2)], Fraction(12)),
        ([Fraction(6), Fraction(4)], Fraction(24)),
        ([Fraction(1), Fraction(1)], Fraction(5)),
        ([Fraction(1), Fraction(0)], Fraction(3)),
        ([Fraction(-1), Fraction(0)], Fraction(0)),
        ([Fraction(0), Fraction(-1)], Fraction(0))]


def solve2(a1, b1, a2, b2):
    det = a1[0] * a2[1] - a1[1] * a2[0]
    if det == 0:
        return None
    return [(b1 * a2[1] - a1[1] * b2) / det,
            (a1[0] * b2 - b1 * a2[0]) / det]


verts = []
for (r1, r2) in itertools.combinations(range(len(rows)), 2):
    p = solve2(rows[r1][0], rows[r1][1], rows[r2][0], rows[r2][1])
    if p is None:
        continue
    if all(a[0] * p[0] + a[1] * p[1] <= b for a, b in rows):
        if p not in verts:
            verts.append(p)
obj_at = [(Fc[0] * v[0] + Fc[1] * v[1], v) for v in verts]
exact_opt = min(o for o, _ in obj_at)
opt_verts = [v for o, v in obj_at if o == exact_opt]
EXACT_OPTIMAL = float(exact_opt)

# ------------------------------------------------------------- pounce
import pounce  # noqa: E402

t0 = time.perf_counter()
res = pounce.solve_qp(P=None, c=c, G=G, h=h, lb=lb)
t_pounce = time.perf_counter() - t0
x_p = np.asarray(res.x, dtype=float)
obj_p = float(res.obj)
status = res.status
z = np.asarray(res.z, dtype=float).ravel()
y = np.asarray(res.y, dtype=float).ravel()
z_lb = np.asarray(res.z_lb, dtype=float).ravel()
z_ub = np.asarray(res.z_ub, dtype=float).ravel()
if z_lb.size == 0:
    z_lb = np.zeros(n)
if z_ub.size == 0:
    z_ub = np.zeros(n)

# ------------------------------------------------- oracle 1: scipy linprog
from scipy.optimize import linprog  # noqa: E402

t0 = time.perf_counter()
lp = linprog(c, A_ub=G, b_ub=h, bounds=[(0, None)] * n)
t_scipy = time.perf_counter() - t0

# --------------------------------------------------------- oracle 2: cvxpy
import cvxpy as cp  # noqa: E402

xv = cp.Variable(n)
con = [G @ xv <= h, xv >= 0]
prob = cp.Problem(cp.Minimize(c @ xv), con)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_cvx = time.perf_counter() - t0
obj_cvx = float(prob.value)


def rel(a, b_):
    return abs(a - b_) / max(1.0, abs(b_))


# ----------------------------------------------------- (b) feasibility
ineq_viol = float(np.max(G @ x_p - h))
bnd_viol = float(np.max(lb - x_p))
feasible = ineq_viol <= 1e-8 and bnd_viol <= 1e-8

# --------------------------------------- (c) is x on the OPTIMAL FACE?
# Face: 3x1 + 2x2 == 12 and 2 <= x1 <= 3.  Equivalently x = t*V1+(1-t)*V2.
face_eq = abs(3 * x_p[0] + 2 * x_p[1] - 12.0)
d = FACE_V2 - FACE_V1
t_par = float(np.dot(x_p - FACE_V1, d) / np.dot(d, d))
proj = FACE_V1 + min(max(t_par, 0.0), 1.0) * d
dist_to_face = float(np.linalg.norm(x_p - proj))
on_face = face_eq <= 1e-7 and dist_to_face <= 1e-7

# ------------------------------------------------ (d) DUAL VALIDITY
# stationarity for  min c'x  s.t. Gx<=h, x>=lb :  c + G'z - z_lb + z_ub = 0
stat = c + G.T @ z - z_lb + z_ub
stat_res = float(np.max(np.abs(stat)))
dual_feas = float(min(z.min() if z.size else 0.0,
                      z_lb.min() if z_lb.size else 0.0))
slack = h - G @ x_p
cs_ineq = float(np.max(np.abs(z * slack))) if z.size else 0.0
cs_lb = float(np.max(np.abs(z_lb * (x_p - lb))))
cs = max(cs_ineq, cs_lb)
# strong duality: dual objective = -h'z + lb'z_lb - ub'z_ub  (ub absent)
dual_obj = float(-h @ z + lb @ z_lb)
gap = abs(dual_obj - obj_p)
dual_valid = (stat_res <= 1e-6 and dual_feas >= -1e-8
              and cs <= 1e-6 and gap <= 1e-6)

# Is the returned dual one of the non-unique family? z1 + 2 z2 = 1, z3=z4=0
dual_family = abs(z[0] + 2 * z[1] - 1.0)

obj_err_exact = rel(obj_p, EXACT_OPTIMAL)
obj_err_scipy = rel(obj_p, float(lp.fun))
obj_err_cvx = rel(obj_p, obj_cvx)

print("=== exact rational oracle (Fraction vertex enumeration) ===")
print(f"vertices={[(str(v[0]), str(v[1])) for v in verts]}")
print(f"exact_optimal={exact_opt} = {EXACT_OPTIMAL}")
print(f"optimal_vertices={[(str(v[0]), str(v[1])) for v in opt_verts]}  "
      f"(face dim = {len(opt_verts) - 1})")
print("=== pounce ===")
print(f"status={status} iters={getattr(res, 'iters', '?')} "
      f"obj={obj_p:.12e} t={t_pounce:.4f}s")
print(f"x={x_p}")
print(f"z(ineq)={z}  z_lb={z_lb}  z_ub={z_ub}  y(eq)={y}")
print("=== oracles ===")
print(f"scipy  status={lp.status} obj={float(lp.fun):.12e} x={lp.x} t={t_scipy:.4f}s")
print(f"cvxpy  status={prob.status} obj={obj_cvx:.12e} x={xv.value} t={t_cvx:.4f}s")
print("=== grading ===")
print(f"known_optimal={KNOWN_OPTIMAL:.12e}")
print(f"(a) obj rel_err  vs exact={obj_err_exact:.3e}  vs scipy={obj_err_scipy:.3e}"
      f"  vs cvxpy={obj_err_cvx:.3e}")
print(f"(b) feasibility: max(Gx-h)={ineq_viol:.3e}  max(lb-x)={bnd_viol:.3e}"
      f"  -> {'OK' if feasible else 'VIOLATED'}")
print(f"(c) on optimal face: |3x1+2x2-12|={face_eq:.3e}  dist_to_segment={dist_to_face:.3e}"
      f"  t_param={t_par:.6f}  -> {'ON FACE' if on_face else 'OFF FACE'}")
print(f"(d) dual validity: ||c+G'z-z_lb+z_ub||inf={stat_res:.3e}  min(z,z_lb)={dual_feas:.3e}")
print(f"    compl.slack max|z*slack|={cs:.3e}  dual_obj={dual_obj:.12e}  gap={gap:.3e}"
      f"  -> {'VALID' if dual_valid else 'INVALID'}")
print(f"    dual-family residual |z1+2*z2-1|={dual_family:.3e} "
      f"(any nonneg split is optimal); z3={z[2]:.3e} z4={z[3]:.3e}")

ok = (status == "optimal" or getattr(res, "success", False)) \
    and obj_err_exact < 1e-8 and obj_err_scipy < 1e-8 and obj_err_cvx < 1e-6 \
    and feasible and on_face and dual_valid
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, obj_err={obj_err_exact:.2e}, "
      f"feasible={feasible}, on_face={on_face}, dual_valid={dual_valid})")
