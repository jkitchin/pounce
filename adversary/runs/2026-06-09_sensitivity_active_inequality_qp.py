"""Adversary cross-check: parametric QP sensitivity dx/db with a BINDING INEQUALITY
Family: sensitivity   Class: parametric QP dx/db with an active inequality in the working set
Source: hand-built convex QP whose KKT system mixes an equality parameter
        constraint with an *active* inequality constraint. Distinct from the
        already-tested pure-equality "parametric QP dx/db": here the active set
        includes a G x <= h row, exercising the active-inequality branch of the
        QpSensitivity KKT factorization.

Problem:
    min  1/2 x' P x + c' x
    s.t. A x = b              (equality; b is the PARAMETER)
         G x <= h             (inequality; the second row is BINDING at the optimum)

    P = diag(2, 2, 2)         (so 1/2 x'Px = x0^2 + x1^2 + x2^2)
    c = 0
    A = [[1, 1, 1]]   b = [3.0]          (param: perturb b[0])
    G = [[ 1, 0, 0],                      (x0 <= 0.5)   -> inactive at optimum
         [ 0, 1, -1]]                     (x1 - x2 <= -1) -> ACTIVE/binding
    h = [0.5, -1.0]

Oracle: central finite-difference re-solve of the SAME QP,
    dx/db ~= (x*(b+delta) - x*(b-delta)) / (2 delta),
solved independently with scipy.optimize.minimize (SLSQP). pounce's analytic
sensitivity (QpSensitivity.parametric_step) must match this FD re-solve, and the
first-order predictor x* + dx must match the actual re-solved optimum at b+Db
(exact while the active set is unchanged because the problem is a QP -> the
solution is piecewise-affine in b).
"""
import time
import numpy as np
from scipy.optimize import minimize

P = np.diag([2.0, 2.0, 2.0])
c = np.zeros(3)
A = np.array([[1.0, 1.0, 1.0]])
b0 = np.array([3.0])
G = np.array([[1.0, 0.0, 0.0],
              [0.0, 1.0, -1.0]])
h = np.array([0.5, -1.0])

PIN = 0          # perturb b[0]
DELTA = 1e-6     # FD step for dx/db
DB = 0.3         # finite parameter move for the predictor check


def resolve(bval):
    """Independently re-solve the QP for a given b[0] with scipy SLSQP."""
    bb = float(bval)
    cons = [
        {"type": "eq", "fun": lambda x, bb=bb: A @ x - bb},
        {"type": "ineq", "fun": lambda x: h - G @ x},   # h - Gx >= 0  <=>  Gx <= h
    ]
    r = minimize(lambda x: 0.5 * x @ P @ x + c @ x, np.zeros(3),
                 constraints=cons, method="SLSQP",
                 options={"maxiter": 1000, "ftol": 1e-14})
    return r.x


# --- pounce: build once, query sensitivity ---
from pounce.qp import QpSensitivity

t0 = time.perf_counter()
s = QpSensitivity(P=P, c=c, A=A, b=b0, G=G, h=h)
t_build = time.perf_counter() - t0

t0 = time.perf_counter()
dx_pounce = s.parametric_step([PIN], [1.0])     # dx/db[0] (unit perturbation)
t_step = time.perf_counter() - t0
x_pounce = np.asarray(s.x)

# --- oracle: central FD re-solve for dx/db[0] ---
t0 = time.perf_counter()
xp = resolve(b0[PIN] + DELTA)
xm = resolve(b0[PIN] - DELTA)
dx_fd = (xp - xm) / (2 * DELTA)
t_fd = time.perf_counter() - t0

# nominal solution cross-check (scipy at nominal b)
x_oracle_nom = resolve(b0[PIN])

# predictor check: x* + dx*DB  vs  actual re-solve at b0+DB
x_pred = x_pounce + np.asarray(s.parametric_step([PIN], [DB]))
x_actual = resolve(b0[PIN] + DB)


def relinf(a, b):
    return float(np.linalg.norm(a - b, np.inf) / max(1.0, np.linalg.norm(b, np.inf)))


nom_err = relinf(x_pounce, x_oracle_nom)
dxdb_err = relinf(dx_pounce, dx_fd)
pred_err = relinf(x_pred, x_actual)

print("=== pounce QpSensitivity ===")
print(f"x*           = {x_pounce}")
print(f"obj          = {s.obj:.10e}")
print(f"kkt_dim      = {s.kkt_dim}")
print(f"dx/db[0]     = {dx_pounce}")
print(f"t_build={t_build:.4f}s  t_step={t_step:.6f}s")
print("=== oracle (scipy SLSQP re-solve) ===")
print(f"x*_nom       = {x_oracle_nom}")
print(f"dx/db[0]_FD  = {dx_fd}   (delta={DELTA})")
print(f"t_fd={t_fd:.4f}s")
print("=== cross-checks ===")
print(f"nominal x rel_inf_err vs scipy   = {nom_err:.2e}")
print(f"dx/db analytic vs FD rel_inf_err = {dxdb_err:.2e}")
print(f"predictor x*+dx*{DB} vs re-solve  = {pred_err:.2e}")
print(f"  x_pred   = {x_pred}")
print(f"  x_actual = {x_actual}")

ok = (nom_err < 1e-6) and (dxdb_err < 1e-4) and (pred_err < 1e-4)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (nom={nom_err:.2e}, dxdb={dxdb_err:.2e}, pred={pred_err:.2e})")
