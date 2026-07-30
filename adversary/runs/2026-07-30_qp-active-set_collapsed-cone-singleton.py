"""Adversary cross-check: m >> n collapsed cone, feasible set a single point
Family: qp-active-set   Class: maximally degenerate (Slater fails; no interior)
Source: the geometry behind pounce issue #282 (false infeasibility certificate
        on m/n >> 1 collapsed cones), constructed here with a known optimum.
Known optimal: 0 at x = 0

Adversarial intent: the feasible set is exactly {0} -- every one of the m rows
is active there and the active set is massively rank-deficient. This is the
geometry that historically produced a *false infeasibility certificate*, and it
exercises both the rank repair and the "never assert infeasibility without a
certificate" rule this PR relies on.

  min  1/2 ||x||^2 + c'x
  s.t. a_i' x <= 0   for m directions a_i that positively span R^n

Positively spanning directions force a_i'x <= 0 for all i  =>  x = 0.
So x* = 0 and f* = 0 regardless of c.
"""
import time

import numpy as np

rng = np.random.default_rng(4242)
n = 6
# directions positively spanning R^n: +-e_i plus random extras. Their
# nonnegative combinations cover R^n, so {x : a_i'x <= 0 for all i} = {0}.
G = np.vstack([np.eye(n), -np.eye(n), rng.standard_normal((14, n))])
m = G.shape[0]
h = np.zeros(m)
P = np.eye(n)
c = rng.standard_normal(n)          # pulls away from 0; only feasibility holds it

X_STAR = np.zeros(n)
KNOWN_OPTIMAL = 0.0

from pounce.qp import solve_qp

res = {}
for method in ("ipm", "active-set"):
    t0 = time.perf_counter()
    r = solve_qp(P=P, c=c, G=G, h=h, method=method)
    res[method] = (r, time.perf_counter() - t0)

import cvxpy as cp

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x) + c @ x), [G @ x <= h])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0

print(f"n={n} m={m}  feasible set = {{0}}  known_optimal={KNOWN_OPTIMAL:.6e}")
print(f"=== oracle CLARABEL === status={prob.status} obj={prob.value:.6e} t={t_oracle:.4f}s")

ok = True
for method, (r, t) in res.items():
    oe = abs(r.obj - KNOWN_OPTIMAL)
    xe = float(np.linalg.norm(np.asarray(r.x) - X_STAR, np.inf))
    print(f"=== pounce {method} === status={r.status} obj={r.obj:.6e} "
          f"abs_err={oe:.2e} x_inf_err={xe:.2e} t={t:.4f}s")
    # the specific historical failure: claiming this feasible problem infeasible
    if "infeasible" in str(r.status).lower():
        print(f"  !! FALSE INFEASIBILITY on {method} (feasible set is {{0}})")
        ok = False
    if r.status == "optimal" and oe > 1e-6:
        print(f"  !! SOLVED-BUT-WRONG on {method}")
        ok = False
    if r.status != "optimal" or oe > 1e-6:
        ok = False

print("VERDICT: PASS" if ok else "VERDICT: FAIL")
