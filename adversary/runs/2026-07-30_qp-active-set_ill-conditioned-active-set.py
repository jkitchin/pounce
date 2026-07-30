"""Adversary cross-check: ill-conditioned QP with a large active set
Family: qp-active-set   Class: ill-conditioned, many constraints active at x*
Source: standard eigenvalue-spread construction (Nocedal & Wright §16.5 on
        ill-conditioned QP KKT systems); optimum obtained from an independent
        oracle (cvxpy/CLARABEL) and cross-checked against a second oracle (SCS).
Known optimal: oracle-derived (no published closed form)

Adversarial intent for this PR: the homotopy traces the path on `H + delta*I`
with `delta` a 1e-6 *relative* perturbation. On a Hessian whose spectrum spans
1e8, a relative perturbation is not obviously harmless -- if `delta` is large
relative to the smallest eigenvalue it can mispredict which constraints are
active, and the corrector then has to recover from a wrong working set.
"""
import time

import numpy as np

rng = np.random.default_rng(20260730)
n, m = 25, 18
COND = 1e8

Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
eigs = np.geomspace(1.0, COND, n)          # spectrum spans 8 orders
P = (Q * eigs) @ Q.T
P = 0.5 * (P + P.T)
c = rng.standard_normal(n)

# Constraints chosen so a large subset is active at the optimum: G x <= h with
# h set just below the unconstrained minimizer's row activities.
G = rng.standard_normal((m, n))
x_unc = np.linalg.solve(P, -c)
h = G @ x_unc - np.abs(rng.standard_normal(m)) * 0.05   # most rows cut the optimum

from pounce.qp import solve_qp

res = {}
for method in ("ipm", "active-set"):
    t0 = time.perf_counter()
    r = solve_qp(P=P, c=c, G=G, h=h, method=method)
    res[method] = (r, time.perf_counter() - t0)

import cvxpy as cp

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(P)) + c @ x), [G @ x <= h])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
obj_oracle, x_oracle = float(prob.value), np.asarray(x.value)

# second oracle, per the workflow's conic/ill-conditioned guidance
prob2 = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(P)) + c @ x), [G @ x <= h])
prob2.solve(solver=cp.SCS)
obj_oracle2 = float(prob2.value)

n_active = int(np.sum(G @ x_oracle >= h - 1e-7))
print(f"cond(P)={np.linalg.cond(P):.3e}  n={n} m={m}  active at x*: {n_active}/{m}")
print(f"=== oracle CLARABEL === obj={obj_oracle:.10e} t={t_oracle:.4f}s")
print(f"=== oracle SCS      === obj={obj_oracle2:.10e} (oracle agreement "
      f"{abs(obj_oracle-obj_oracle2)/max(1,abs(obj_oracle)):.2e})")


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


ok = True
for method, (r, t) in res.items():
    oe = rel(r.obj, obj_oracle)
    xe = float(np.linalg.norm(np.asarray(r.x) - x_oracle, np.inf))
    print(f"=== pounce {method} === status={r.status} obj={r.obj:.10e} "
          f"rel_err_vs_oracle={oe:.2e} x_inf_err={xe:.2e} t={t:.4f}s")
    if r.status == "optimal" and oe > 1e-4:
        print(f"  !! SOLVED-BUT-WRONG on {method}")
        ok = False
    if r.status != "optimal" or oe > 1e-4:
        ok = False

print("VERDICT: PASS" if ok else "VERDICT: FAIL")
