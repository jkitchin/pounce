"""Adversary cross-check: parametric QP sensitivity dx/db w.r.t. an EQUALITY rhs
when a VARIABLE BOUND is in the active set.

Family: sensitivity   Class: parametric convex QP, equality-rhs perturbation with
an ACTIVE LOWER BOUND entering the KKT active set.

Distinct-from-prior (avoid-list):
  * dx/db analytic (2026-06-09 parametric_qp): equality-only, no bounds.
  * dx/db with binding INEQUALITY (2026-06-09 active_inequality_qp): explicit
    G x <= h inequality in the active set, diagonal P.
  * sIPOPT CLI on .nl (2026-06-09 sipopt_cli_nl): NLP via CLI suffixes.
  * active-inequality-RHS dx/dh (2026-06-15 active_ineq_rhs_qp): perturbs the
    rhs h of an explicit binding inequality encoded as an equality.
  NEW HERE: the binding constraint is a *variable BOUND* (lb), supplied via the
  `lb=` argument, NOT an explicit inequality row. The bound enters the active-set
  KKT factorization through pounce's bound-handling path (kkt_dim = n + m_eq +
  n_active_bounds), and we verify the resulting dx/db is correct. This exercises
  the active-BOUND branch of QpSensitivity, untouched by the four prior runs.

Problem:
    min 1/2 x' P x + c' x
    s.t. x0 + x1 = b          (equality, b is the PARAMETER)
         x1 >= lb1            (LOWER BOUND, binding at the optimum)
         x0 >= lb0            (lower bound, slack/inactive)
    P = I (so objective = 1/2||x||^2 - x0 - x1),  c = (-1, -1)
    b0 = 1.0,  lb1 = 0.7,  lb0 = -10 (loose)

Why the bound binds:
  Unconstrained-by-bounds optimum of  min 1/2||x||^2 - x  s.t. x0+x1=b  is the
  symmetric split x0=x1=b/2=0.5.  Since lb1=0.7 > 0.5, the lower bound on x1 is
  restrictive and binds: x1=0.7, x0=b-0.7=0.3.

Analytic dx/db (active-set reduced problem):
  With x1 pinned at its bound (x1 = lb1, inactive direction frozen), feasibility
  x0 + x1 = b gives x0 = b - lb1.  Hence
      dx0/db = 1,  dx1/db = 0   ->  dx/db = (1, 0).
  (Source: Fiacco 1983; Nocedal & Wright 2e Sec.16.5 — active-set KKT/range-space
  sensitivity; a bound active at the optimum is treated as an equality
  x_i = bound in the active-set KKT system, with strictly positive multiplier so
  the active set is locally stable.)

Oracle: central finite-difference re-solve of the TRUE bound-constrained QP via
scipy SLSQP, dx/db ~= (x*(b+delta) - x*(b-delta)) / (2 delta), plus the analytic
value above, plus a finite-move predictor vs full re-solve.
"""
import time
import numpy as np
from scipy.optimize import minimize

P = np.eye(2)
c = np.array([-1.0, -1.0])
LB0, LB1 = -10.0, 0.7
b0 = 1.0

DELTA = 1e-6      # FD step for dx/db
DB = 0.05         # finite parameter move for the predictor check (keeps active set)


def resolve(bv):
    """Independently re-solve the TRUE bound-constrained QP for a given b."""
    cons = [{"type": "eq", "fun": lambda x, bv=bv: x[0] + x[1] - bv}]
    bnds = [(LB0, None), (LB1, None)]
    r = minimize(lambda x: 0.5 * x @ x + c @ x, np.array([0.5, 0.7]),
                 constraints=cons, bounds=bnds, method="SLSQP",
                 options={"maxiter": 2000, "ftol": 1e-16})
    return r.x


# --- confirm the lower bound on x1 is genuinely BINDING ---
x_nom = resolve(b0)
bound_active = abs(x_nom[1] - LB1) < 1e-7
# the bound-free optimum would put x1 at 0.5 < LB1, so the bound is restrictive
bound_restrictive = (b0 / 2.0) < LB1 - 1e-9

ANALYTIC_DXDB = np.array([1.0, 0.0])

# --- pounce: QpSensitivity with the lower bound supplied via lb= ---
from pounce.qp import QpSensitivity

t0 = time.perf_counter()
s = QpSensitivity(P=P, c=c, A=np.array([[1.0, 1.0]]), b=np.array([b0]),
                  lb=np.array([LB0, LB1]))
x_pounce = np.asarray(s.x)
kkt_dim = s.kkt_dim
dx_pounce = np.asarray(s.parametric_step([0], [1.0]))   # dx/db0
t_pounce = time.perf_counter() - t0

# --- oracle: central FD re-solve of the TRUE bound-constrained QP ---
t0 = time.perf_counter()
xp = resolve(b0 + DELTA)
xm = resolve(b0 - DELTA)
dx_fd = (xp - xm) / (2 * DELTA)
t_fd = time.perf_counter() - t0


def ninf(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), np.inf))


nom_err = ninf(x_pounce, x_nom)
dx_vs_fd = ninf(dx_pounce, dx_fd)
dx_vs_analytic = ninf(dx_pounce, ANALYTIC_DXDB)

# predictor: x* + dx*DB  vs  actual re-solve at b0+DB (active set must persist)
x_pred = x_pounce + dx_pounce * DB
x_actual = resolve(b0 + DB)
pred_err = ninf(x_pred, x_actual)
still_active = abs(x_actual[1] - LB1) < 1e-7   # bound still binding after the move

print("=== problem ===")
print(f"lower bound x1>={LB1} binding: {bound_active}  restrictive: {bound_restrictive}")
print(f"x*={x_nom}  (x1 at bound {LB1})")
print("=== pounce QpSensitivity (active variable BOUND) ===")
print(f"x*       = {x_pounce}   kkt_dim={kkt_dim} (n+m_eq+n_active_bound = 2+1+1=4)")
print(f"dx/db    = {dx_pounce}   t={t_pounce:.5f}s")
print("=== oracle ===")
print(f"x*_nom   = {x_nom}")
print(f"dx/db FD = {dx_fd}   (delta={DELTA})   t={t_fd:.5f}s")
print(f"dx/db ana= {ANALYTIC_DXDB}")
print("=== cross-checks ===")
print(f"nominal x inf_err vs scipy        = {nom_err:.2e}")
print(f"dx/db pounce vs FD inf_err        = {dx_vs_fd:.2e}")
print(f"dx/db pounce vs analytic inf_err  = {dx_vs_analytic:.2e}")
print(f"predictor x*+dx*{DB} vs re-solve   = {pred_err:.2e}  (still_active={still_active})")
print(f"  x_pred   = {x_pred}")
print(f"  x_actual = {x_actual}")

ok = (bound_active and bound_restrictive and still_active and kkt_dim == 4
      and nom_err < 1e-6 and dx_vs_fd < 1e-5 and dx_vs_analytic < 1e-6
      and pred_err < 1e-5)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (bound_active={bound_active} kkt_dim={kkt_dim} "
      f"nom={nom_err:.2e} dx_vs_fd={dx_vs_fd:.2e} "
      f"dx_vs_analytic={dx_vs_analytic:.2e} pred={pred_err:.2e})")
