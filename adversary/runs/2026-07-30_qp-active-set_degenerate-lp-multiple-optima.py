"""Adversary cross-check: degenerate LP (P=0) with a continuum of optima
Family: qp-active-set   Class: LP-as-QP, primal degenerate, non-unique optimum
Source: classic degenerate-LP construction (Chvatal, *Linear Programming*,
        Ch. 3 on degeneracy); optimum derived in closed form + cvxpy oracle.
Known optimal: -3 (objective), attained on a whole face (x not unique)

Adversarial intent for this PR, two-fold:

1. `P = 0` is the singular-Hessian case that forced the homotopy's `QP_0` to be
   the box relaxation with a `delta` regularization. A degenerate LP is where a
   badly-sized `delta` mispredicts the active set.
2. It is driven through BOTH surfaces -- `solve_qp(method="active-set")` and
   `minimize(solver_selection="qp-active-set")` -- which this PR unified. They
   must agree with each other and with the oracle; `nfev == 0` proves the
   convex route ran rather than the NLP fallback.

  min  -x0 - x1 - x2
  s.t. x0 + x1 + x2 <= 3,  0 <= xi <= 2

Optimum: any point on x0+x1+x2 = 3 in the box; f* = -3. Degenerate: at a vertex
such as (2,1,0) four constraints are active in R^3.
"""
import time

import numpy as np

n = 3
c = -np.ones(n)
G = np.ones((1, n))
h = np.array([3.0])
lb = np.zeros(n)
ub = np.full(n, 2.0)
KNOWN_OPTIMAL = -3.0

from pounce.qp import solve_qp

res = {}
for method in ("ipm", "active-set"):
    t0 = time.perf_counter()
    r = solve_qp(c=c, G=G, h=h, lb=lb, ub=ub, method=method)
    res[method] = (r, time.perf_counter() - t0)

import cvxpy as cp

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c @ x), [G @ x <= h, x >= lb, x <= ub])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0

# --- the second surface: minimize(solver_selection="qp-active-set") ---
import pounce

fun = lambda v: float(c @ v)
jac = lambda v: c
con = [{"type": "ineq",
        "fun": lambda v: np.array([3.0 - v.sum()]),
        "jac": lambda v: -np.ones((1, n))}]
mres = pounce.minimize(fun, np.zeros(n), jac=jac, constraints=con,
                       bounds=[(0.0, 2.0)] * n,
                       solver_selection="qp-active-set")

print(f"known_optimal={KNOWN_OPTIMAL:.6f}  (optimum is a face, x not unique)")
print(f"=== oracle CLARABEL === obj={prob.value:.10e} t={t_oracle:.4f}s")

ok = True
for method, (r, t) in res.items():
    oe = abs(r.obj - KNOWN_OPTIMAL)
    feas = float(max(0.0, (G @ np.asarray(r.x))[0] - 3.0))
    box = float(max(np.max(lb - np.asarray(r.x)), np.max(np.asarray(r.x) - ub), 0.0))
    print(f"=== pounce {method} === status={r.status} obj={r.obj:.10e} "
          f"abs_err={oe:.2e} row_viol={feas:.2e} box_viol={box:.2e} t={t:.4f}s")
    if r.status == "optimal" and (oe > 1e-6 or feas > 1e-8 or box > 1e-8):
        print(f"  !! SOLVED-BUT-WRONG on {method}")
        ok = False
    if r.status != "optimal" or oe > 1e-6:
        ok = False

merr = abs(mres.fun - KNOWN_OPTIMAL)
print(f"=== minimize(qp-active-set) === success={mres.success} fun={mres.fun:.10e} "
      f"abs_err={merr:.2e} nfev={mres.nfev}")
if mres.nfev != 0:
    print("  !! routed to a callback path, not the convex driver (surface divergence)")
    ok = False
if not mres.success or merr > 1e-6:
    ok = False

print("VERDICT: PASS" if ok else "VERDICT: FAIL")
