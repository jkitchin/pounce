"""Adversary cross-check: Simultaneous multi-parameter QP sensitivity (dx/db)
Family: sensitivity   Class: equality-constrained convex QP, TWO RHS
    parameters perturbed simultaneously in one QpSensitivity.parametric_step
    call (rather than the single-parameter sweeps in prior sensitivity probes)
Source: closed-form KKT block-elimination formula for an equality-constrained
    convex QP (Nocedal & Wright, "Numerical Optimization" 2nd ed., Sec 16.1,
    eq. 16.4-16.5): for
        minimize    0.5 x'Px + c'x
        subject to  Ax = b
    the KKT system is LINEAR in b for a fixed A, so
        x*(b) = -P^-1 c + P^-1 A' y*(b),   y*(b) = (A P^-1 A')^-1 (b + A P^-1 c)
    is exact for ANY perturbation size (no active-set change is possible in an
    equality-only QP), giving
        dx/db = P^-1 A' (A P^-1 A')^-1     (a constant 3x2 matrix here)
    This lets a *simultaneous* two-parameter perturbation be checked against
    an EXACT (not first-order-truncated) closed form, and independently
    against re-solving the QP outright at b+db.

Problem (n=3 vars, 2 equality constraints):
    P = diag(2, 4, 6),  c = [1, -2, 0.5]
    A = [[1, 1, 0], [0, 1, 1]],  b0 = [2, 3]
    perturb BOTH b0, b1 at once: db = [0.7, -0.4]
"""
import time
import numpy as np

P = np.diag([2.0, 4.0, 6.0])
c = np.array([1.0, -2.0, 0.5])
A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
b0 = np.array([2.0, 3.0])
db = np.array([0.7, -0.4])

import pounce
from pounce.qp import QpSensitivity

# --- base solve ---
r0 = pounce.solve_qp(P=P, c=c, A=A, b=b0)
x0 = np.asarray(r0.x, dtype=float)

# --- pounce QpSensitivity: simultaneous two-parameter step ---
sens = QpSensitivity(P=P, c=c, A=A, b=b0)
t0 = time.perf_counter()
dx_sens = np.asarray(sens.parametric_step([0, 1], db.tolist()), dtype=float)
t_pounce = time.perf_counter() - t0
x_pred_sens = x0 + dx_sens

# --- oracle 1: closed-form KKT block-elimination (exact, since equality-only) ---
Pinv = np.linalg.inv(P)
S = A @ Pinv @ A.T  # Schur complement (A P^-1 A'), 2x2
dydb = np.linalg.inv(S)  # dy/db
dxdb_closed = Pinv @ A.T @ dydb  # 3x2, exact sensitivity matrix
dx_closed = dxdb_closed @ db
x_pred_closed = x0 + dx_closed

# --- oracle 2: exact re-solve at b0+db (equality-only QP -> linear in b, so
# this re-solve should reproduce x0+dx to solver tolerance, not merely to
# first order) ---
r1 = pounce.solve_qp(P=P, c=c, A=A, b=b0 + db)
x_resolve = np.asarray(r1.x, dtype=float)

# --- oracle 3: classical small-delta central finite difference (independent
# of the "exact re-solve" framing above, uses a much smaller perturbation) ---
eps = 1e-5
fd_cols = []
for k in range(2):
    dbk = np.zeros(2)
    dbk[k] = eps
    xp = np.asarray(pounce.solve_qp(P=P, c=c, A=A, b=b0 + dbk).x, dtype=float)
    xm = np.asarray(pounce.solve_qp(P=P, c=c, A=A, b=b0 - dbk).x, dtype=float)
    fd_cols.append((xp - xm) / (2 * eps))
dxdb_fd = np.array(fd_cols).T  # 3x2
dx_fd = dxdb_fd @ db
x_pred_fd = x0 + dx_fd


def inf_err(a, b):
    return float(np.linalg.norm(a - b, np.inf))


err_sens_vs_closed = inf_err(dx_sens, dx_closed)
err_sens_vs_resolve = inf_err(x_pred_sens, x_resolve)
err_sens_vs_fd = inf_err(dxdb_fd, dxdb_closed)  # matrix-level FD vs closed form
err_closed_vs_resolve = inf_err(x_pred_closed, x_resolve)

print("=== pounce QpSensitivity.parametric_step([0,1], db) ===")
print(f"x0={x0}  dx={dx_sens}  t={t_pounce:.4f}s")
print(f"kkt_cond_estimate={sens.kkt_cond_estimate:.3e} ill_conditioned={sens.ill_conditioned} "
      f"last_step_residual={sens.last_step_residual}")
print("=== oracle: closed-form KKT block-elimination (exact) ===")
print(f"dx/db matrix=\n{dxdb_closed}\ndx={dx_closed}")
print("=== oracle: exact re-solve at b0+db ===")
print(f"x_resolve={x_resolve}  x0+dx_sens={x_pred_sens}")
print("=== oracle: central finite-difference dx/db matrix (eps=1e-5) ===")
print(f"dx/db matrix=\n{dxdb_fd}")
print(f"err(parametric_step vs closed-form dx)={err_sens_vs_closed:.2e}")
print(f"err(parametric_step-predicted x vs exact re-solve x)={err_sens_vs_resolve:.2e}")
print(f"err(FD dx/db matrix vs closed-form dx/db matrix)={err_sens_vs_fd:.2e}")
print(f"err(closed-form-predicted x vs exact re-solve x)={err_closed_vs_resolve:.2e}")

ok = (
    err_sens_vs_closed < 1e-8
    and err_sens_vs_resolve < 1e-6
    and err_sens_vs_fd < 1e-5
    and err_closed_vs_resolve < 1e-8
    and not sens.ill_conditioned
)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (err_closed={err_sens_vs_closed:.2e}, err_resolve={err_sens_vs_resolve:.2e}, "
      f"err_fd={err_sens_vs_fd:.2e}, ill_conditioned={sens.ill_conditioned})")
