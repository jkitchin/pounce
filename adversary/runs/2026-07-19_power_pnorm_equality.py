"""Adversary cross-check: min sum |x_i|^p s.t. a'x = b  (power cone, p = 2.5)
Family: power   Class: non-integer p-th-power minimization, ANALYTIC optimum,
                       with one component driven exactly to the cone boundary.
Source: Analytic Hoelder/Lagrange solution (Boyd & Vandenberghe, "Convex
        Optimization" (2004), Ex. 4.7 / A.1.6 dual-norm characterization;
        MOSEK Modeling Cookbook v3.3 sec. 4.2 "Power cone" for the
        |x|^p <= s  <=>  (x, s, 1) in P_{1/p}  epigraph).

Closed form.  With q = p/(p-1) (Hoelder conjugate) and S = sum_i |a_i|^q:
        x_i^* = sign(a_i) |a_i|^{q-1} (b/S),      f^* = (b/S)^p S.
Derivation: stationarity p|x_i|^{p-1}sgn(x_i) = lam a_i gives
x_i = sgn(a_i)|lam a_i/p|^{1/(p-1)}; a'x = b fixes |lam/p|^{1/(p-1)} = b/S.

BOUNDARY PROBE: a_3 = 0, so x_3^* = 0 and s_3^* = 0 exactly -- that power-cone
block sits on the boundary (x,y,z) = (0,0,1), where the barrier blows up.

CONE CONVENTION -- verified read-only against
  crates/pounce-convex/src/cones/power.rs:
      K_alpha = {(x,y,z) : |x| <= y^alpha z^(1-alpha), y,z >= 0}
  and python/pounce/qp.py::solve_socp: ("pow", alpha) -- second element is the
  EXPONENT, not a dimension; slack s = h - Gx must lie in the cone.
  So |x_i|^p <= s_i  <=>  |x_i| <= s_i^{1/p} * 1^{1-1/p}
      <=>  (x_i, s_i, 1) in K_{1/p}, i.e. alpha = 1/p.
  NOTE cvxpy's PowCone3D(u,v,w,alpha) is u^alpha v^{1-alpha} >= |w|, i.e. the
  triple order is PERMUTED relative to pounce: pounce (x,y,z) = cvxpy (w,u,v).
  This script builds the cvxpy oracle with cp.power (DCP atom) rather than a
  hand-rolled cone, and separately checks the permuted PowCone3D form.
"""

import time

import numpy as np

p = 2.5
q = p / (p - 1.0)  # 5/3
a = np.array([2.0, -3.0, 0.0, 1.0, 5.0])  # a[2] = 0 -> x[2]* = 0 (boundary)
bval = 4.0
nvar = a.size

S = np.sum(np.abs(a) ** q)
X_STAR = np.sign(a) * np.abs(a) ** (q - 1.0) * (bval / S)
KNOWN_OPTIMAL = (bval / S) ** p * S
assert abs(a @ X_STAR - bval) < 1e-12

# ------------------------------------------------------------------ pounce
# vars: x_0..x_4, s_0..s_4  (n = 10);  min sum s_i  s.t. a'x = b,
#       (x_i, s_i, 1) in K_{1/p}
n = 2 * nvar
c = np.zeros(n)
c[nvar:] = 1.0
A = np.zeros((1, n))
A[0, :nvar] = a
bvec = np.array([bval])

rows, h = [], []
for i in range(nvar):
    r0 = np.zeros(n)
    r0[i] = -1.0
    rows.append(r0)
    h.append(0.0)  # s = x_i
    r1 = np.zeros(n)
    r1[nvar + i] = -1.0
    rows.append(r1)
    h.append(0.0)  # s = s_i
    rows.append(np.zeros(n))
    h.append(1.0)  # s = 1
G = np.array(rows)
h = np.array(h)
cones = [("pow", 1.0 / p)] * nvar

from pounce import solve_socp  # noqa: E402

t0 = time.perf_counter()
r = solve_socp(c=c, A=A, b=bvec, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)[:nvar]
obj_pounce = float(r.obj)
status = r.status

# ----------------------------------------------------------------- oracles
import cvxpy as cp  # noqa: E402


def cvx_dcp(solver):
    x = cp.Variable(nvar)
    pr = cp.Problem(cp.Minimize(cp.sum(cp.power(cp.abs(x), p))), [a @ x == bval])
    t0 = time.perf_counter()
    pr.solve(solver=solver)
    return pr.value, np.asarray(x.value), time.perf_counter() - t0, pr.status


def cvx_powcone():
    """Same model via the explicit permuted PowCone3D encoding."""
    x = cp.Variable(nvar)
    s = cp.Variable(nvar, nonneg=True)
    one = np.ones(nvar)
    cons = [a @ x == bval, cp.constraints.PowCone3D(s, one, x, 1.0 / p)]
    pr = cp.Problem(cp.Minimize(cp.sum(s)), cons)
    t0 = time.perf_counter()
    pr.solve(solver=cp.CLARABEL)
    return pr.value, np.asarray(x.value), time.perf_counter() - t0, pr.status


obj_o1, x_o1, t_o1, st1 = cvx_dcp(cp.CLARABEL)
obj_o2, x_o2, t_o2, st2 = cvx_dcp(cp.SCS)
obj_o3, x_o3, t_o3, st3 = cvx_powcone()


def rel(u, v):
    return abs(u - v) / max(1.0, abs(v))


print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s iters={r.iters} kkt={r.kkt_error:.2e}")
print(f"x={x_pounce}")
print(f"feas |a'x-b|={abs(a @ x_pounce - bval):.2e}")
print("=== oracle CLARABEL (cp.power DCP) ===")
print(f"status={st1} obj={obj_o1:.10e} t={t_o1:.4f}s x={x_o1}")
print("=== oracle SCS (cp.power DCP) ===")
print(f"status={st2} obj={obj_o2:.10e} t={t_o2:.4f}s x={x_o2}")
print("=== oracle CLARABEL (explicit PowCone3D) ===")
print(f"status={st3} obj={obj_o3:.10e} t={t_o3:.4f}s x={x_o3}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e}  x*={X_STAR}")
print(f"rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"obj_err_vs_CLARABEL={rel(obj_pounce, obj_o1):.2e} obj_err_vs_SCS={rel(obj_pounce, obj_o2):.2e}")
print(f"obj_err_vs_PowCone3D={rel(obj_pounce, obj_o3):.2e}")
print(f"x_inf_err_vs_known={np.max(np.abs(x_pounce - X_STAR)):.2e}")
print(f"boundary component x[2]={x_pounce[2]:.3e} (exact 0)")

ok = (status.startswith("optimal") or r.success) and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 and rel(obj_pounce, obj_o1) < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status})")
