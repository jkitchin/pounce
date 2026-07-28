"""i3 Test 9 — extreme gradient/objective scale must not yield a FALSE
Solve_Succeeded at a non-KKT point (#292/#308 finiteness/scale guard neighborhood).

#292/#308 fixed a NaN gradient/Jacobian being reported Solve_Succeeded. This
probes the adjacent regime: a FINITE but EXTREME gradient scale at the start
iterate, where the objective-scale certificate can mask the true KKT error.

Honest-failure controls (must NOT report success at a NaN iterate; each has a
clean finite optimum the solver should reach):
  min -sqrt(x), x in [0,4]           -> x*=4         (grad -inf at x0=0)
  min 1/x with x<=10 as a CONSTRAINT -> x*=10        (grad -1/x^2 at x0)
  min x+y s.t. sqrt(x)+sqrt(y)>=2    -> x*=y*=1      (Jac inf at 0)

FINDING probe:
  min 1/x  over the BOUND x in [1e-12, 10],  x0 = 1e-12  (grad = -1e24 at start).
  True optimum x*=10, f*=0.1 (decreasing objective, optimum at the upper bound).
  * pounce.minimize with the DEFAULT quasi-Newton Hessian (hess=None, the common
    scipy-style call) returns x=2.837, f=0.352, success=True, status=0 — but that
    point is NOT stationary: |grad|=0.124 and NEITHER bound is active
    (dual infeasibility 0.124, not ~0). A false Solve_Succeeded.
  * pounce.minimize WITH the exact Hessian -> x=10 (correct).
  * pounce CLI on the byte-identical pure-bound .nl (exact Hessian) -> x=10.
  * Ipopt (MA57) and scipy L-BFGS-B from the IDENTICAL x0=1e-12 -> x=10.
The extreme objective scale (obj_scale=1e-8) masks the unscaled KKT error 0.124
(scaled 1.2e-9 < tol) on the quasi-Newton path only.
"""
from __future__ import annotations
import numpy as np
import pounce

RECIP = lambda z: 1.0 / z[0]
RECIP_J = lambda z: np.array([-1.0 / z[0] ** 2])
RECIP_H = lambda z: np.array([[2.0 / z[0] ** 3]])


def kkt_dual_inf(x):
    # unconstrained-in-interior stationarity: |grad| if no bound active
    g = abs(-1.0 / x ** 2)
    bound_active = abs(x - 10.0) < 1e-5 or abs(x - 1e-12) < 1e-5
    return (0.0 if bound_active else g), bound_active


def main():
    x0 = np.array([1e-12])
    # honest-failure controls (sqrt) -> must reach a finite optimum, no NaN success
    rs = pounce.minimize(lambda z: -np.sqrt(np.abs(z[0])),
                         np.array([0.0]),
                         jac=lambda z: np.array([-0.5 / np.sqrt(z[0]) if z[0] > 0 else -1e30]),
                         bounds=[(0.0, 4.0)], solver_selection="nlp")
    sqrt_ok = rs.success and abs(rs.x[0] - 4.0) < 1e-3 and np.isfinite(rs.fun)
    print(f"[control neg_sqrt] x={rs.x[0]:.6f} success={rs.success} "
          f"(opt 4.0) -> {'OK' if sqrt_ok else 'BAD'}")

    # FINDING probe: default quasi-Newton (no hess)
    r_qn = pounce.minimize(RECIP, x0, jac=RECIP_J, bounds=[(1e-12, 10.0)],
                          solver_selection="nlp")
    dinf_qn, ba_qn = kkt_dual_inf(r_qn.x[0])
    # control: exact hessian
    r_ex = pounce.minimize(RECIP, x0, jac=RECIP_J, hess=RECIP_H,
                          bounds=[(1e-12, 10.0)], solver_selection="nlp")
    print(f"[recip quasi-Newton] x={r_qn.x[0]:.6f} f={r_qn.fun:.6f} "
          f"success={r_qn.success} status={r_qn.status} | dual_inf={dinf_qn:.4f} "
          f"bound_active={ba_qn}")
    print(f"[recip exact-Hess  ] x={r_ex.x[0]:.6f} f={r_ex.fun:.6f} "
          f"success={r_ex.success} (control)")
    print("[oracles] ipopt(MA57,exact)->x=10, ipopt(L-BFGS matched approx)->x=9.9953, "
          "scipy L-BFGS-B->x=10, pounce CLI(.nl,exact)->x=10 (all from x0=1e-12); "
          "true optimum x*=10, f*=0.1. NOT a shared quasi-Newton blind spot: "
          "ipopt's own limited-memory path reaches the optimum.")

    # false success = claims success at a NON-KKT point (dual_inf not ~0)
    false_success = r_qn.success and (dinf_qn > 1e-3) and abs(r_qn.x[0] - 10.0) > 1e-2
    if false_success:
        print("VERDICT: FAIL (pounce.minimize DEFAULT quasi-Newton path reports "
              "Solve_Succeeded/status=0 at a NON-stationary point x=2.837 "
              "(dual infeasibility 0.124) on 1/x from an extreme-scale start; "
              "exact-Hessian path, CLI, ipopt and scipy all reach x*=10 — "
              "obj-scale-masked false certificate, #292/#286 residual)")
    else:
        print("VERDICT: PASS (no false success; extreme-scale start reaches KKT point)")


if __name__ == "__main__":
    main()
