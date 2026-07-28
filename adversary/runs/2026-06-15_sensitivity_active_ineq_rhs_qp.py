"""Adversary cross-check: parametric QP sensitivity dx/dh w.r.t. the RHS of an
ACTIVE INEQUALITY constraint  (distinct data from the prior active-inequality run).

Family: sensitivity   Class: parametric convex QP, perturb RHS of a BINDING inequality.

Distinct-from-prior:
  * 2026-06-09_sensitivity_active_inequality_qp.py perturbed the EQUALITY rhs b
    while an inequality merely sat in the active set, and used a DIAGONAL P.
  * Here the PARAMETER is the RHS *h* of the binding inequality itself, with a
    COUPLED (off-diagonal) Hessian P. We validate that pounce's KKT sensitivity
    reproduces dx/dh of the true inequality-constrained QP.

Mechanism:
  pounce's QpSensitivity.parametric_step perturbs only an EQUALITY rhs (an index
  into b). At a QP optimum the binding inequality g·x <= h is satisfied with
  equality (g·x = h) and enters the KKT active set exactly as an equality. So we
  hand pounce the active set as equalities  A = [Aeq; g],  b = [beq; h]  and
  perturb the g-row rhs. The resulting dx/db[g-row] IS the inequality
  sensitivity dx/dh — *provided the active set is unchanged*, which holds for a
  small move because the multiplier on g is strictly positive (LICQ + strict
  complementarity).  We cross-check that claim by re-solving the FULL inequality
  QP (G x <= h form) with perturbed h.

Problem (binding inequality):
    min 1/2 x' P x + c' x
    s.t. Aeq x = beq
         g · x <= h        (BINDING at the optimum)
    P = [[3,1],[1,2]] (SPD, coupled),  c = (-1,-1)
    Aeq = [[1,2]], beq = 1.0
    g = (2,-1),  h0 = -0.3      (eq-only optimum has g·x≈0 > h0, so g binds)

Analytic dx/dh (active-set KKT, source: Fiacco 1983, "Introduction to
Sensitivity and Stability Analysis in Nonlinear Programming"; Boyd & Vandenberghe
§5.6 perturbation/sensitivity; Nocedal & Wright §16.5):
  With active set treated as equalities  C = [Aeq; g], rhs r = [beq; h],
  KKT  [[P, C'],[C, 0]] [x; mu] = [-c; r].
  Then  d[x;mu]/dr = [[P,C'],[C,0]]^{-1} [0; I],  and dx/dh is the x-block of the
  column of that inverse corresponding to the g-row of r.

Oracle: central finite-difference re-solve of the TRUE inequality QP via scipy
SLSQP, plus the analytic KKT value above.
"""
import time
import numpy as np
from scipy.optimize import minimize

P = np.array([[3.0, 1.0], [1.0, 2.0]])
c = np.array([-1.0, -1.0])
Aeq = np.array([[1.0, 2.0]])
beq = np.array([1.0])
g = np.array([2.0, -1.0])
h0 = -0.3

DELTA = 1e-6     # FD step for dx/dh
DH = 0.05        # finite parameter move for the predictor check (keeps active set)


def resolve_ineq(hv):
    """Independently re-solve the TRUE inequality QP for a given h via scipy SLSQP."""
    cons = [
        {"type": "eq", "fun": lambda x: Aeq @ x - beq},
        {"type": "ineq", "fun": lambda x, hv=hv: hv - g @ x},   # g x <= h
    ]
    r = minimize(lambda x: 0.5 * x @ P @ x + c @ x, np.zeros(2),
                 constraints=cons, method="SLSQP",
                 options={"maxiter": 2000, "ftol": 1e-15})
    return r.x


# --- confirm the inequality is genuinely BINDING ---
x_nom = resolve_ineq(h0)
gx = float(g @ x_nom)
cons_eqonly = [{"type": "eq", "fun": lambda x: Aeq @ x - beq}]
x_eqonly = minimize(lambda x: 0.5 * x @ P @ x + c @ x, np.zeros(2),
                    constraints=cons_eqonly, method="SLSQP",
                    options={"ftol": 1e-15}).x
binding = abs(gx - h0) < 1e-6 and (g @ x_eqonly) > h0 + 1e-6   # ineq active & restrictive

# --- analytic dx/dh via active-set KKT ---
C = np.vstack([Aeq, g])          # active constraints as equalities
n = 2
mc = C.shape[0]
K = np.block([[P, C.T], [C, np.zeros((mc, mc))]])
Kinv = np.linalg.inv(K)
ANALYTIC_DXDH = Kinv[:n, n + 1]  # x-rows, column of the g-row rhs (index n + (g is 2nd ctr))

# --- pounce: encode active set as equalities, perturb the g-row rhs ---
from pounce.qp import QpSensitivity

t0 = time.perf_counter()
s = QpSensitivity(P=P, c=c, A=C, b=np.array([beq[0], h0]))
x_pounce = np.asarray(s.x)
dx_pounce = np.asarray(s.parametric_step([1], [1.0]))   # dx/d(rhs of g-row) = dx/dh
t_pounce = time.perf_counter() - t0

# --- oracle: central FD re-solve of the TRUE inequality QP ---
t0 = time.perf_counter()
xp = resolve_ineq(h0 + DELTA)
xm = resolve_ineq(h0 - DELTA)
dx_fd = (xp - xm) / (2 * DELTA)
t_fd = time.perf_counter() - t0


def ninf(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), np.inf))


# nominal solution agreement
nom_err = ninf(x_pounce, x_nom)
dx_vs_fd = ninf(dx_pounce, dx_fd)
dx_vs_analytic = ninf(dx_pounce, ANALYTIC_DXDH)

# predictor: x* + dx*DH  vs  actual re-solve of the TRUE inequality QP at h0+DH
x_pred = x_pounce + dx_pounce * DH
x_actual = resolve_ineq(h0 + DH)
pred_err = ninf(x_pred, x_actual)
# confirm active set unchanged at h0+DH (still binding)
still_binding = abs(g @ x_actual - (h0 + DH)) < 1e-6

print("=== problem ===")
print(f"inequality binding at optimum: {binding}  (g.x={gx:.6f}, h0={h0})")
print("=== pounce QpSensitivity (active ineq encoded as equality) ===")
print(f"x*        = {x_pounce}")
print(f"dx/dh     = {dx_pounce}   t={t_pounce:.4f}s")
print("=== oracle ===")
print(f"x*_nom    = {x_nom}")
print(f"dx/dh FD  = {dx_fd}   (delta={DELTA})   t={t_fd:.4f}s")
print(f"dx/dh ana = {ANALYTIC_DXDH}")
print("=== cross-checks ===")
print(f"nominal x inf_err vs scipy      = {nom_err:.2e}")
print(f"dx/dh pounce vs FD inf_err      = {dx_vs_fd:.2e}")
print(f"dx/dh pounce vs analytic inf_err= {dx_vs_analytic:.2e}")
print(f"predictor x*+dx*{DH} vs re-solve = {pred_err:.2e}  (still_binding={still_binding})")
print(f"  x_pred   = {x_pred}")
print(f"  x_actual = {x_actual}")

ok = (binding and still_binding and nom_err < 1e-6
      and dx_vs_fd < 1e-5 and dx_vs_analytic < 1e-6 and pred_err < 1e-5)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (binding={binding} nom={nom_err:.2e} "
      f"dx_vs_fd={dx_vs_fd:.2e} dx_vs_analytic={dx_vs_analytic:.2e} pred={pred_err:.2e})")
