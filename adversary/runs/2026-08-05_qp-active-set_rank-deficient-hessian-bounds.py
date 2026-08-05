"""Adversary cross-check: rank-deficient (singular PSD) Hessian, bound-constrained
Family: qp-active-set   Class: rank-deficient Hessian, one active bound
Source: constructed; closed form by inspection (see below)
Known optimal: analytically derived (see derivation)

    minimize    f(x) = x1^2 + x2^2 - 4*x1 - 6*x2 - x3
    subject to  -5 <= x1, x2, x3 <= 5

P = diag(2, 2, 0) is positive SEMIdefinite with a genuine zero eigenvalue
(flat, unbounded-below direction along x3 if unconstrained by bounds only
-- the linear term c3=-1 makes f -> -inf as x3 -> +inf). This is the
qp-active-set path's stress case for a Hessian that is not positive
DEFINITE: the active-set/parametric-homotopy engine must resolve the null
direction via the box bound rather than curvature.

By inspection: unconstrained-in-x1,x2 minimizer is x1=2, x2=3 (both
interior to [-5,5]); x3's coefficient in f is purely linear (-x3), so f is
decreasing in x3 without bound until the upper bound x3=5 is hit. Hence
x* = (2, 3, 5), one active bound (x3 at ub), f* = (4-8) + (9-18) + (-5) =
-4 - 9 - 5 = -18.

Cross-checked against method="ipm" on the SAME pounce entry point (routing
transparency) and against cvxpy (CLARABEL + SCS, independent of pounce)
with P declared via cp.psd_wrap since it is only semidefinite.
"""
import time

import numpy as np

KNOWN_OPTIMAL = -18.0
KNOWN_X = np.array([2.0, 3.0, 5.0])

P = np.diag([2.0, 2.0, 0.0])
c = np.array([-4.0, -6.0, -1.0])
lb = np.array([-5.0, -5.0, -5.0])
ub = np.array([5.0, 5.0, 5.0])

from pounce.qp import solve_qp

res = {}
for method in ("ipm", "active-set"):
    t0 = time.perf_counter()
    r = solve_qp(P=P, c=c, lb=lb, ub=ub, method=method)
    res[method] = (r, time.perf_counter() - t0)

import cvxpy as cp

x = cp.Variable(3)
constraints = [x >= lb, x <= ub]
prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(P)) + c @ x), constraints)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
obj_oracle, x_oracle = float(prob.value), np.asarray(x.value)

prob2 = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(P)) + c @ x), constraints)
prob2.solve(solver=cp.SCS)
obj_oracle2 = float(prob2.value)

print(f"eig(P)={np.linalg.eigvalsh(P)}  (genuine zero eigenvalue -> rank-deficient)")
print(f"known_optimal={KNOWN_OPTIMAL}  known_x={KNOWN_X}")
print(f"=== oracle CLARABEL === obj={obj_oracle:.10e} x={x_oracle} t={t_oracle:.4f}s")
print(f"=== oracle SCS      === obj={obj_oracle2:.10e} "
      f"(oracle agreement {abs(obj_oracle - obj_oracle2) / max(1, abs(obj_oracle)):.2e})")


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


ok = True
for method, (r, t) in res.items():
    oe_known = rel(r.obj, KNOWN_OPTIMAL)
    oe_oracle = rel(r.obj, obj_oracle)
    xe_known = float(np.linalg.norm(np.asarray(r.x) - KNOWN_X, np.inf))
    xe_oracle = float(np.linalg.norm(np.asarray(r.x) - x_oracle, np.inf))
    print(f"=== pounce method={method} === status={r.status} obj={r.obj:.10e} "
          f"x={np.asarray(r.x)} t={t:.4f}s")
    print(f"    obj_err_vs_known={oe_known:.2e}  x_err_vs_known={xe_known:.2e}  "
          f"obj_err_vs_oracle={oe_oracle:.2e}  x_err_vs_oracle={xe_oracle:.2e}")
    this_ok = (
        r.status in ("optimal",) or getattr(r, "success", False)
    ) and oe_known < 1e-4 and oe_oracle < 1e-4
    ok = ok and this_ok

# routing transparency: both methods must agree on the objective
ipm_obj = res["ipm"][0].obj
as_obj = res["active-set"][0].obj
cross_agree = rel(ipm_obj, as_obj) < 1e-4
print(f"IPM vs active-set objective agreement: {rel(ipm_obj, as_obj):.2e} "
      f"({'OK' if cross_agree else 'MISMATCH'})")
ok = ok and cross_agree

print("VERDICT: PASS" if ok else "VERDICT: FAIL")
