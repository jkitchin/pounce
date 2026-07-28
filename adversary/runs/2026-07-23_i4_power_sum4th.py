"""Adversary i4: minimize a SUM OF FOURTH POWERS over a linear equality.
Family: power   Class: power cone as a CONVEX power epigraph (t_i >= x_i^4).

Problem:  minimize x1^4 + x2^4  s.t.  x1 + x2 = 1.
This uses the power cone in the OTHER direction vs all logged power tests
(which use it as a geometric-mean / monomial LOWER bound). Here we need the
convex epigraph of the power function |x|^4:
  t >= |x|^p   <=>   |x| <= t^{1/p} * 1^{1-1/p}
  pounce cone {(s0,s1,s2): |s0| <= s1^a s2^{1-a}} with s0=x, s1=t, s2=1const,
  a = 1/p = 1/4.  => |x| <= t^{1/4}  => t >= x^4.
Known optimum (Lagrange: 4x1^3 = 4x2^3 = lambda => x1=x2; x1+x2=1):
  x1 = x2 = 0.5, value = 2 * (0.5)^4 = 2/16 = 0.125.
Variables [x1,x2,t1,t2]; two pow cones; objective min t1+t2.
"""
import time
import numpy as np

p = 4.0
KNOWN_OPTIMAL = 0.125
X_STAR = np.array([0.5, 0.5])

# vars: x1(0) x2(1) t1(2) t2(3)
nv = 4
c = np.array([0.0, 0.0, 1.0, 1.0])   # min t1 + t2

rows, h = [], []
def push(row, hv): rows.append(row); h.append(hv)
# cone1 for x1: (s0=x1, s1=t1, s2=1)
r = np.zeros(nv); r[0] = -1.0; push(r, 0.0)
r = np.zeros(nv); r[2] = -1.0; push(r, 0.0)
r = np.zeros(nv);              push(r, 1.0)
# cone2 for x2: (s0=x2, s1=t2, s2=1)
r = np.zeros(nv); r[1] = -1.0; push(r, 0.0)
r = np.zeros(nv); r[3] = -1.0; push(r, 0.0)
r = np.zeros(nv);              push(r, 1.0)
G = np.array(rows); h = np.array(h)
cones = [("pow", 1.0 / p), ("pow", 1.0 / p)]

A = np.array([[1.0, 1.0, 0.0, 0.0]])   # x1 + x2 = 1
bvec = np.array([1.0])

import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, A=A, b=bvec, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
v = np.asarray(r.x, float)
x_sol = v[:2]
obj_pounce = float(x_sol[0] ** 4 + x_sol[1] ** 4)
status = str(r.status)

import cvxpy as cp
def solve_cvxpy(solver):
    x = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(cp.power(x[0], 4) + cp.power(x[1], 4)),
                      [x[0] + x[1] == 1])
    t0 = time.perf_counter(); prob.solve(solver=solver); dt = time.perf_counter() - t0
    return float(prob.value), dt, np.asarray(x.value, float)

val_cla, t_cla, x_cla = solve_cvxpy(cp.CLARABEL)
val_scs, t_scs, x_scs = solve_cvxpy(cp.SCS)

def rel(a, b): return abs(a - b) / max(1.0, abs(b))
eq_res = abs(x_sol[0] + x_sol[1] - 1.0)
x_err = float(np.linalg.norm(x_sol - X_STAR, np.inf))

print("=== pounce (power-cone epigraph IPM) ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s x={x_sol} eq_res={eq_res:.2e}")
print(f"=== cvxpy/CLARABEL obj={val_cla:.10e} t={t_cla:.4f}s")
print(f"=== cvxpy/SCS      obj={val_scs:.10e} t={t_scs:.4f}s")
print(f"known={KNOWN_OPTIMAL}  x_star={X_STAR}")
print(f"rel vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e} "
      f"vs_CLARABEL={rel(obj_pounce, val_cla):.2e} vs_SCS={rel(obj_pounce, val_scs):.2e}")
print(f"x_inf_err={x_err:.2e}")

ok = (status == "optimal" or getattr(r, "success", False)) \
    and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 and eq_res < 1e-6 and x_err < 1e-3
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err={rel(obj_pounce, KNOWN_OPTIMAL):.2e}, x_err={x_err:.2e})")
