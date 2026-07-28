"""Adversary cross-check: DEGENERATE parametric QP sensitivity — strict
complementarity FAILS (weakly active inequality, multiplier exactly zero), and
the parameter perturbation CHANGES THE ACTIVE SET.

Family: sensitivity   Class: degenerate / weakly-active-inequality parametric QP
Source: Fiacco (1983) "Introduction to Sensitivity and Stability Analysis in
        Nonlinear Programming", Ch. 2-3 (the classical dx/dp theorem requires
        LICQ + STRICT COMPLEMENTARITY + SOSC); Nocedal & Wright 2e Sec. 12.5 /
        16.5; Boyd & Vandenberghe Sec. 5.6.  This instance deliberately breaks
        strict complementarity, which is exactly the regime the theorem
        excludes, so x*(b) is only DIRECTIONALLY differentiable at b0.

Problem (parameter is the equality rhs b):
    min  1/2 (x0^2 + x1^2)
    s.t. x0 + x1     = b        <- the PARAMETER
         x0 - 2 x1  <= h = -1/2

At b0 = 1 the equality-only optimum is x = (b/2, b/2) = (0.5, 0.5) and
    g.x = x0 - 2 x1 = -b/2 = -0.5 = h   EXACTLY.
So the inequality is ACTIVE with multiplier EXACTLY ZERO (weakly active):
strict complementarity fails.

Closed-form one-sided derivatives (derived, then confirmed by FD below):
  * b > b0 : g.x_eq = -b/2 < h, inequality INACTIVE.
        x = (b/2, b/2)              ->  dx/db_+ = (1/2, 1/2)
  * b < b0 : g.x_eq = -b/2 > h, inequality VIOLATED, so it becomes ACTIVE.
        Solve [[1,1],[1,-2]] x = [b, -1/2]:
        x = (2b/3 - 1/6,  b/3 + 1/6)  ->  dx/db_- = (2/3, 1/3)

dx/db_+ != dx/db_-, so the derivative at b0 DOES NOT EXIST.  A central finite
difference straddling b0 returns the AVERAGE (7/12, 5/12) = (0.58333, 0.41667),
which is not a derivative of anything — that is the trap this run is built on.

What we test:
  1. pounce's nominal x* is correct and the inequality is genuinely weakly active
     (multiplier ~ 0) — confirmed against scipy, independently.
  2. Which value does QpSensitivity.parametric_step return?  The only defensible
     answers are dx/db_- or dx/db_+ (a one-sided derivative).  Anything else
     (in particular the straddling-FD average, or a value matching NEITHER) is a
     genuine SOLVER_BUG.
  3. ONE-SIDED FD plateau sweep (delta = 1e-3 ... 1e-9, each entirely on one
     side of b0) confirms the two one-sided derivatives numerically, so we do
     not mistake FD noise for a solver bug.
  4. Predictor honesty: does x* + dx*Delta_b track the true re-solve on BOTH
     sides, or only on the side pounce's one-sided derivative belongs to?

Oracle: independent scipy SLSQP re-solves (never pounce) + the closed forms.
"""
import time

import numpy as np
from scipy.optimize import minimize

P = np.eye(2)
c = np.zeros(2)
A = np.array([[1.0, 1.0]])
B0 = 1.0
g = np.array([1.0, -2.0])
H = -0.5

DXDB_PLUS = np.array([0.5, 0.5])            # b > b0, inequality inactive
DXDB_MINUS = np.array([2.0 / 3.0, 1.0 / 3.0])  # b < b0, inequality active
DXDB_STRADDLE = 0.5 * (DXDB_PLUS + DXDB_MINUS)  # what a naive central FD gives


def resolve(bv):
    """Independent re-solve of the TRUE inequality QP at parameter b (scipy)."""
    cons = [
        {"type": "eq", "fun": lambda x, bv=bv: np.array([x[0] + x[1] - bv])},
        {"type": "ineq", "fun": lambda x: np.array([H - (g @ x)])},  # g.x <= H
    ]
    r = minimize(lambda x: 0.5 * x @ x, np.array([0.4, 0.6]),
                 jac=lambda x: x, constraints=cons, method="SLSQP",
                 options={"maxiter": 2000, "ftol": 1e-16})
    return r.x


def ninf(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), np.inf))


# ---------------------------------------------------------------- nominal
x_nom = resolve(B0)
slack = float(H - g @ x_nom)          # 0 => active
# multiplier of the inequality at b0 (should be ~0: weakly active).
# From stationarity x + A' nu + mu g = 0 with x=(0.5,0.5), A'=(1,1)', g=(1,-2)'.
M = np.array([[1.0, 1.0], [1.0, -2.0]]).T      # columns: nu-col, mu-col
lam = np.linalg.lstsq(M, -x_nom, rcond=None)[0]
mu_ineq = float(lam[1])

# ------------------------------------------------- one-sided FD plateau sweep
DELTAS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
sweep = []
for d in DELTAS:
    # forward-only (b0, b0+d) and backward-only (b0-d, b0): each stays on ONE side
    fwd = (resolve(B0 + d) - x_nom) / d
    bwd = (x_nom - resolve(B0 - d)) / d
    ctr = (resolve(B0 + d) - resolve(B0 - d)) / (2 * d)   # straddles b0: undefined
    sweep.append((d, fwd, bwd, ctr))

# plateau picks: mid-range deltas where truncation and roundoff are both small
fd_plus = sweep[2][1]     # delta = 1e-5, forward
fd_minus = sweep[2][2]    # delta = 1e-5, backward
fd_center = sweep[2][3]

plateau_plus = max(ninf(s[1], DXDB_PLUS) for s in sweep[:5])
plateau_minus = max(ninf(s[2], DXDB_MINUS) for s in sweep[:5])

# ---------------------------------------------------------------- pounce
from pounce.qp import QpSensitivity  # noqa: E402

t0 = time.perf_counter()
s = QpSensitivity(P=P, c=c, A=A, b=np.array([B0]), G=g.reshape(1, 2), h=np.array([H]))
x_pounce = np.asarray(s.x)
dx_pounce = np.asarray(s.parametric_step([0], [1.0]))
t_pounce = time.perf_counter() - t0

nom_err = ninf(x_pounce, x_nom)
err_plus = ninf(dx_pounce, DXDB_PLUS)
err_minus = ninf(dx_pounce, DXDB_MINUS)
err_straddle = ninf(dx_pounce, DXDB_STRADDLE)

if err_minus < 1e-7:
    side = "MINUS (active-set branch, b<b0)"
elif err_plus < 1e-7:
    side = "PLUS (inequality-dropped branch, b>b0)"
elif err_straddle < 1e-7:
    side = "STRADDLE AVERAGE (not a derivative — BUG)"
else:
    side = "NEITHER one-sided derivative (BUG)"

# --------------------------------- TOLERANCE SWEEP: is the branch choice stable?
# The weakly active constraint sits exactly on the active_tol=1e-7 knife edge, so
# which branch pounce picks may depend on how tightly the IPM converged.  Sweep
# the solver tol and record both x* accuracy and the returned dx/db.
tol_sweep = []
for tolv in [None, 1e-10, 1e-12, 1e-14]:
    kw = {} if tolv is None else {"tol": tolv}
    st = QpSensitivity(P=P, c=c, A=A, b=np.array([B0]),
                       G=g.reshape(1, 2), h=np.array([H]), max_iter=500, **kw)
    xt = np.asarray(st.x)
    dxt = np.asarray(st.parametric_step([0], [1.0]))
    br = ("MINUS" if ninf(dxt, DXDB_MINUS) < 1e-6 else
          "PLUS" if ninf(dxt, DXDB_PLUS) < 1e-6 else "OTHER")
    tol_sweep.append((tolv, ninf(xt, x_nom), float(H - g @ xt), dxt, br))
branches = {t[4] for t in tol_sweep}

# ------------------------------------------------- predictor on both sides
DB = 0.05
pred_up = x_pounce + dx_pounce * DB
pred_dn = x_pounce - dx_pounce * DB
act_up = resolve(B0 + DB)
act_dn = resolve(B0 - DB)
pred_err_up = ninf(pred_up, act_up)
pred_err_dn = ninf(pred_dn, act_dn)

# ---------------------------------------------------------------- report
print("=== problem: degenerate (weakly active) parametric QP ===")
print(f"x*(b0)            = {x_nom}")
print(f"inequality slack  = {slack:.3e}   (0 => ACTIVE)")
print(f"inequality mult mu= {mu_ineq:.3e}   (0 => STRICT COMPLEMENTARITY FAILS)")
print("=== one-sided FD plateau sweep (oracle: scipy re-solve) ===")
print(f"{'delta':>8}  {'forward (b>b0)':>28}  {'backward (b<b0)':>28}  {'central (straddle)':>28}")
for d, f, bw, ct in sweep:
    print(f"{d:8.0e}  {str(np.round(f, 9)):>28}  {str(np.round(bw, 9)):>28}  {str(np.round(ct, 9)):>28}")
print(f"closed form  dx/db_+ = {DXDB_PLUS}   dx/db_- = {DXDB_MINUS}")
print(f"plateau max err (fwd vs dx/db_+, 1e-3..1e-7) = {plateau_plus:.2e}")
print(f"plateau max err (bwd vs dx/db_-, 1e-3..1e-7) = {plateau_minus:.2e}")
print("=== pounce QpSensitivity ===")
print(f"x*        = {x_pounce}   nom_inf_err_vs_scipy = {nom_err:.2e}")
print(f"dx/db     = {dx_pounce}   t={t_pounce:.4f}s")
print(f"  |dx - dx/db_+| = {err_plus:.2e}")
print(f"  |dx - dx/db_-| = {err_minus:.2e}")
print(f"  |dx - straddle average| = {err_straddle:.2e}")
print(f"  => pounce returned the {side} derivative, with NO degeneracy warning")
print("=== tolerance sweep: which one-sided branch does pounce pick? ===")
print(f"{'tol':>8}  {'|x*-x_exact|':>13}  {'ineq slack':>12}  {'dx/db':>26}  branch")
for tolv, xe, sl, dxt, br in tol_sweep:
    print(f"{str(tolv):>8}  {xe:13.3e}  {sl:12.3e}  {str(np.round(dxt, 9)):>26}  {br}")
print(f"branches observed across tol sweep: {sorted(branches)}"
      + ("   <-- UNSTABLE: the returned derivative flips with solver tol"
         if len(branches) > 1 else ""))
print("=== predictor honesty (Delta_b = +/-0.05) ===")
print(f"up:   pred={np.round(pred_up, 9)} actual={np.round(act_up, 9)} err={pred_err_up:.2e}")
print(f"down: pred={np.round(pred_dn, 9)} actual={np.round(act_dn, 9)} err={pred_err_dn:.2e}")

# PASS criteria: forward solve right, the two one-sided derivatives are confirmed
# numerically (plateau), and pounce returns ONE of them (a defensible answer for
# a point where the derivative does not exist).  Returning neither is a bug.
one_sided_ok = (err_plus < 1e-7) or (err_minus < 1e-7)
ok = (nom_err < 1e-4 and abs(slack) < 1e-7 and abs(mu_ineq) < 1e-7
      and plateau_plus < 1e-5 and plateau_minus < 1e-5 and one_sided_ok)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (nom={nom_err:.2e} slack={slack:.2e} mu={mu_ineq:.2e} "
      f"plateau+={plateau_plus:.2e} plateau-={plateau_minus:.2e} side={side})")
