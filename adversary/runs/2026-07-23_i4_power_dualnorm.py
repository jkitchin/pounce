"""Adversary i4: DUAL-NORM via linear objective over a p-norm ball (p=4).
Family: power   Class: linear objective over a p-norm-ball constraint.

Problem:  maximize c^T x  s.t. ||x||_4 <= 1.
Known optimum (Holder / dual-norm identity): max_{||x||_p<=1} c^T x = ||c||_q,
with 1/p + 1/q = 1. p=4 -> q=4/3. This is DISTINCT from the logged p-norm
MINIMIZATION over an equality (p=3, p=1.5), which is the primal direction.

Power-cone encoding of ||x||_p <= t (here t=1):
  z_i >= 0, sum_i z_i = t, |x_i| <= z_i^{1/p} t^{1-1/p}.
  pounce cone {(s0,s1,s2): |s0| <= s1^a s2^{1-a}} with s0=x_i, s1=z_i,
  s2=t(=1 const), a=1/p.  => |x_i| <= z_i^{1/p} * 1.
Variables: x(n), z(n).  t is fixed to 1 (constant slack s2=1).
Objective: max c^T x  ->  min -c^T x.
"""
import time
import numpy as np

p = 4.0
q = p / (p - 1.0)                       # 4/3
c_obj = np.array([1.0, -2.0, 3.0, 0.5]) # objective vector c
n = c_obj.size
KNOWN_OPTIMAL = np.linalg.norm(c_obj, q)

# var layout: [x0..x3, z0..z3]  -> 2n vars
nv = 2 * n
ix = lambda i: i
iz = lambda i: n + i

c = np.zeros(nv)
c[:n] = -c_obj                          # minimize -c^T x

# cone rows: for each i, (s0=x_i, s1=z_i, s2=1const)
rows, h = [], []
for i in range(n):
    r0 = np.zeros(nv); r0[ix(i)] = -1.0; rows.append(r0); h.append(0.0)   # s0 = x_i
    r1 = np.zeros(nv); r1[iz(i)] = -1.0; rows.append(r1); h.append(0.0)   # s1 = z_i
    r2 = np.zeros(nv);                    rows.append(r2); h.append(1.0)   # s2 = 1 (const)
G = np.array(rows); h = np.array(h)
cones = [("pow", 1.0 / p)] * n

# equality: sum z_i = t = 1
A = np.zeros((1, nv)); A[0, n:2 * n] = 1.0
bvec = np.array([1.0])

import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, A=A, b=bvec, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
xv = np.asarray(r.x, float)
x_sol = xv[:n]
obj_pounce = float(c_obj @ x_sol)       # the maximized value
status = str(r.status)

# oracle: cvxpy
import cvxpy as cp
def solve_cvxpy(solver):
    x = cp.Variable(n)
    prob = cp.Problem(cp.Maximize(c_obj @ x), [cp.pnorm(x, p) <= 1])
    t0 = time.perf_counter(); prob.solve(solver=solver); dt = time.perf_counter() - t0
    return float(prob.value), dt, np.asarray(x.value, float)

val_cla, t_cla, x_cla = solve_cvxpy(cp.CLARABEL)
val_scs, t_scs, x_scs = solve_cvxpy(cp.SCS)

def rel(a, b): return abs(a - b) / max(1.0, abs(b))
norm4 = np.linalg.norm(x_sol, p)

print("=== pounce (power-cone IPM) ===")
print(f"status={status} max_val={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"  x={x_sol}  ||x||_4={norm4:.8f} (<=1 feasible)")
print("=== oracle cvxpy/CLARABEL ===")
print(f"max_val={val_cla:.10e} t={t_cla:.4f}s")
print("=== oracle cvxpy/SCS ===")
print(f"max_val={val_scs:.10e} t={t_scs:.4f}s")
print(f"known ||c||_(4/3) = {KNOWN_OPTIMAL:.10e}")
print(f"rel vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e} "
      f"vs_CLARABEL={rel(obj_pounce, val_cla):.2e} vs_SCS={rel(obj_pounce, val_scs):.2e}")

ok = (status in ("optimal", "optimal_inaccurate") or getattr(r, "success", False)) \
    and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 and norm4 <= 1 + 1e-6
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err={rel(obj_pounce, KNOWN_OPTIMAL):.2e}, norm4={norm4:.6f})")
