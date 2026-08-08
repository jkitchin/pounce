"""Adversary cross-check: QP envelope theorem, dObj*/db_i = y_i*.

Family: sensitivity   Class: dual-multiplier-as-objective-sensitivity --
              distinct from prior sensitivity probes (near-LICQ dx/db,
              barrier-curvature, reduced_hessian, multi-parameter dx/db,
              economic-dispatch dP/dD), which all checked the PRIMAL
              sensitivity dx*/db. This probe checks the classical
              envelope/duality theorem for the OPTIMAL VALUE instead:
              d(obj*)/db_i = y_i* (the equality constraint's own optimal
              Lagrange multiplier), to first order, while the active set
              is unchanged.
Source: Nocedal & Wright, Numerical Optimization 2nd ed., the standard
        Lagrangian sensitivity/envelope theorem (see also Boyd &
        Vandenberghe Sec 5.6.3, "Sensitivity analysis"): for
        min f(x) s.t. Ax=b (+ inequalities/bounds held fixed),
            d(p*(b))/db_i = +-y_i*
        where y* is the optimal equality-constraint multiplier, sign
        depending on the solver's Lagrangian convention (L = f - y^T(Ax-b)
        vs L = f + y^T(Ax-b)). pounce's convention, determined empirically
        below (not assumed), turns out to be d(obj*)/db_i = -y_i*.

Problem: 3 assets/resources, strictly convex separable QP with TWO equality
constraints (so the multiplier is genuinely a 2-vector, not degenerate) and
a nonnegativity bound (inactive at the chosen b, so the equality-only
theorem applies cleanly):

    min  1/2(x1^2 + x2^2 + x3^2)
    s.t. x1 + x2 + x3 = 10      (b0)
         x1 - x2      = 2       (b1)
         x >= 0

y* is read from a PLAIN pounce.qp.solve_qp() solve (independent of
QpSensitivity's own internal solve/factorization). The oracle is a central
finite difference of the optimal objective, computed via fresh, independent
plain solve_qp() re-solves at b0+-delta and b1+-delta.
"""

import time

import numpy as np

P = np.eye(3)
c = np.zeros(3)
A = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]])
b = np.array([10.0, 2.0])
lb = np.zeros(3)

from pounce.qp import solve_qp, QpSensitivity

t0 = time.perf_counter()
r0 = solve_qp(P=P, c=c, A=A, b=b, lb=lb)
t_pounce = time.perf_counter() - t0
y_star = np.asarray(r0.y)
# Confirm bounds are inactive at this b (precondition for the equality-only
# envelope theorem to apply without a bound-multiplier correction term).
bounds_inactive = bool(np.all(np.asarray(r0.x) > 1e-6))

print("=== pounce plain solve_qp (base point) ===")
print(f"status={r0.status} obj={r0.obj:.10e} x={np.asarray(r0.x)} y={y_star} t={t_pounce:.4f}s")
print(f"bounds_inactive={bounds_inactive}")

# --- oracle: central finite difference of obj*(b) via independent re-solves ---
eps = 1e-5
t0 = time.perf_counter()
dObj_db_fd = np.zeros(2)
for i in range(2):
    bp = b.copy(); bp[i] += eps
    bm = b.copy(); bm[i] -= eps
    rp = solve_qp(P=P, c=c, A=A, b=bp, lb=lb)
    rm = solve_qp(P=P, c=c, A=A, b=bm, lb=lb)
    dObj_db_fd[i] = (rp.obj - rm.obj) / (2 * eps)
t_fd = time.perf_counter() - t0

print("=== oracle: central FD of obj* via independent solve_qp re-solves ===")
print(f"dObj/db_fd={dObj_db_fd} time={t_fd:.4f}s")

err = np.abs(-y_star - dObj_db_fd) / np.maximum(1.0, np.abs(dObj_db_fd))
print(f"-y* (pounce dual, sign-flipped per pounce's Lagrangian convention) = {-y_star}")
print(f"dObj/db (FD oracle) = {dObj_db_fd}")
print(f"rel_err={err}")

# --- Cross-check 2: QpSensitivity.parametric_step's dx should ALSO be
# consistent -- verify the predicted primal step matches an independent
# re-solve to first order (small delta), for both equality rows.
s = QpSensitivity(P=P, c=c, A=A, b=b, lb=lb)
dx_err = np.zeros(2)
for i in range(2):
    delta = 1e-4
    dx_pred = s.parametric_step([i], [delta])
    b2 = b.copy(); b2[i] += delta
    r2 = solve_qp(P=P, c=c, A=A, b=b2, lb=lb)
    dx_true = np.asarray(r2.x) - np.asarray(r0.x)
    dx_err[i] = float(np.linalg.norm(dx_pred - dx_true, np.inf))
print(f"parametric_step dx max abs err vs re-solve (row0, row1)={dx_err}")

ok = (
    r0.status == "optimal"
    and bounds_inactive
    and np.all(err < 1e-3)
    and np.all(dx_err < 1e-6)
)
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
