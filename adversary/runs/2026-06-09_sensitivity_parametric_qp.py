"""Adversary cross-check: post-optimal QP sensitivity dx/db
Family: sensitivity   Class: parametric convex QP (equality RHS as parameter)
Source: sIPOPT analog for QPs. QpSensitivity.parametric_step gives dx/db from
  the cached KKT factor; checked against a central finite-difference re-solve
  (independent), and against the analytic KKT sensitivity.
  Problem: min 0.5 x'Px + c'x s.t. a'x = b,  P=diag(2,4), c=(-1,-2), a=(1,1).
Known: dx/db = P^{-1}a / (a'P^{-1}a).  P^{-1}a=(0.5,0.25), a'P^{-1}a=0.75
  -> dx/db = (2/3, 1/3).
"""
import time
import numpy as np

P = np.array([[2.0, 0.0], [0.0, 4.0]])
c = np.array([-1.0, -2.0])
A = np.array([[1.0, 1.0]])
b0 = np.array([2.0])

Pinv_a = np.linalg.solve(P, A[0])
ANALYTIC_DXDB = Pinv_a / (A[0] @ Pinv_a)   # (2/3, 1/3)

import pounce
from pounce.qp import QpSensitivity

# --- pounce sensitivity (build-once / solve-many) ---
t0 = time.perf_counter()
sens = QpSensitivity(P=P, c=c, A=A, b=b0)
x_star = np.asarray(sens.x)
dx = np.asarray(sens.parametric_step([0], [1.0]))   # dx for db0 = +1
t_pounce = time.perf_counter() - t0

# --- oracle: central finite-difference re-solve ---
delta = 1e-5
def solve_x(bval):
    r = pounce.solve_qp(P=P, c=c, A=A, b=np.array([bval]))
    return np.asarray(r.x)

xp = solve_x(b0[0] + delta)
xm = solve_x(b0[0] - delta)
dx_fd = (xp - xm) / (2 * delta)


def ninf(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), np.inf))

dx_vs_fd = ninf(dx, dx_fd)
dx_vs_analytic = ninf(dx, ANALYTIC_DXDB)

print("=== pounce QpSensitivity ===")
print(f"x*={x_star}  dx/db0={dx}  t={t_pounce:.4f}s")
print("=== oracle (finite-difference re-solve) ===")
print(f"dx/db0_fd={dx_fd}")
print(f"analytic dx/db0={ANALYTIC_DXDB}")
print(f"dx_vs_fd={dx_vs_fd:.2e}  dx_vs_analytic={dx_vs_analytic:.2e}")

# also verify the predictor x* + dx*Δb tracks a real re-solve at Δb=0.3
db = 0.3
x_pred = x_star + dx * db
x_true = solve_x(b0[0] + db)
pred_err = ninf(x_pred, x_true)
print(f"predictor_err(x*+dx*0.3 vs resolve)={pred_err:.2e}")

ok = dx_vs_fd < 1e-5 and dx_vs_analytic < 1e-6 and pred_err < 1e-6
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (dx_vs_fd={dx_vs_fd:.2e} analytic={dx_vs_analytic:.2e} pred={pred_err:.2e})")
