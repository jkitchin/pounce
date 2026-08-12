"""Adversary cross-check: minimize sum x_i^5 on the probability simplex
Family: power   Class: p-norm epigraph (p=5, odd) with a NONNEG orthant
              block MIXED with power-cone triples in one solve_socp call --
              distinct from prior power probes (weighted p=4 Hoelder-dual
              equality-only, cubic-sum p=3 with an inactive box bound,
              CES utility, dual-norm via linear objective, Cauchy-Schwarz
              quadratic-over-linear, three-factor weighted geometric mean):
              none of those combined an explicit ("nonneg", d) cone block
              with power-cone triples in the SAME G/h partition, and none
              used an ODD exponent (p=5) where x>=0 must be enforced
              separately from the cone itself (the pow cone bounds |x|, not
              x, so a naive omission of x>=0 would make the problem
              unbounded below).
Source: power-mean / Lagrange-symmetry argument (e.g. Boyd & Vandenberghe,
        "Convex Optimization" sec 3.1.5, convexity of sum of convex
        functions on a convex set): minimize sum_i x_i^5 subject to
        sum_i x_i = 1, x >= 0. The objective is strictly convex and
        separable; Lagrange stationarity on any interior-active coordinate
        gives 5*x_i^4 = lambda for all i, so all coordinates with x_i>0
        must be EQUAL. Since sum x_i=1, x* = (1/n,...,1/n) satisfies KKT
        (all constraints inactive, so no sign issue), and by convexity of
        the objective over the (convex) simplex this KKT point is the
        unique global minimizer.
Known optimal: n=4 -> x* = (0.25,0.25,0.25,0.25), obj* = 4*(0.25)^5 = 1/256
        = 3.90625e-3 (closed form, independent of any solver).
"""
import time

import numpy as np

n = 4
KNOWN_X = np.full(n, 0.25)
KNOWN_OPTIMAL = float(n * 0.25 ** 5)
assert abs(KNOWN_OPTIMAL - 1.0 / 256.0) < 1e-15

# --- pounce: variables (x_0..x_3, t_0..t_3); minimize sum t_i
# nonneg block: x_i >= 0  (4 rows)
# pow-cone triple i: (x_i, t_i, 1) with alpha=1/5  =>  |x_i| <= t_i^0.2
#   (x_i>=0 from the nonneg block makes this exactly x_i^5 <= t_i)
# equality: sum x_i = 1
from pounce import solve_socp

nv = 2 * n
rows = []
h_list = []

for i in range(n):
    row = np.zeros(nv)
    row[i] = -1.0  # s = x_i
    rows.append(row)
    h_list.append(0.0)

for i in range(n):
    ra = np.zeros(nv)
    ra[i] = -1.0  # slot x: s = x_i
    rows.append(ra)
    h_list.append(0.0)
    rb = np.zeros(nv)
    rb[n + i] = -1.0  # slot y: s = t_i
    rows.append(rb)
    h_list.append(0.0)
    rc = np.zeros(nv)  # slot z: s = 1
    rows.append(rc)
    h_list.append(1.0)

G = np.vstack(rows)
h_vec = np.array(h_list)
c = np.zeros(nv)
c[n:] = 1.0

cones = [("nonneg", n)] + [("pow", 0.2)] * n

A = np.zeros((1, nv))
A[0, 0:n] = 1.0
b = np.array([1.0])

t0 = time.perf_counter()
r = solve_socp(c=c, A=A, b=b, G=G, h=h_vec, cones=cones)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x[:n])
obj_pounce = float(np.sum(x_pounce ** 5))
status = r.status

# --- oracle: cvxpy ---
import cvxpy as cp

x = cp.Variable(n, nonneg=True)
prob = cp.Problem(cp.Minimize(cp.sum(cp.power(x, 5))), [cp.sum(x) == 1.0])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle = np.asarray(x.value)
obj_oracle = float(prob.value)


def rel(u, v):
    return abs(u - v) / max(1.0, abs(v))


obj_err = rel(obj_pounce, obj_oracle)
known_err = rel(obj_pounce, KNOWN_OPTIMAL)
x_err_oracle = float(np.linalg.norm(x_pounce - x_oracle, np.inf))
x_err_known = float(np.linalg.norm(x_pounce - KNOWN_X, np.inf))

print("=== pounce (nonneg + pow cone mix) ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s x={x_pounce}")
print("=== oracle (cvxpy/Clarabel) ===")
print(f"obj={obj_oracle:.10e} t={t_oracle:.4f}s x={x_oracle}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} (x*={KNOWN_X}) rel_err_vs_known={known_err:.2e} x_err_vs_known={x_err_known:.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err_vs_oracle={x_err_oracle:.2e}")

ok = status == "optimal" and obj_err < 1e-4 and known_err < 1e-4 and x_err_known < 1e-3
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, known_err={known_err:.2e})")
