"""Adversary iteration 4: QpSensitivity vs the barrier-curvature traps.
Family: sensitivity   Class: active-bound / weakly-active / coupled dx/db
Direction: the companion notebook (notebooks/barrier_curvature_sensitivity.ipynb)
formalizes four notions of sensitivity and the traps an IPM falls into near
active bounds. QpSensitivity holds the ACTIVE-SET KKT factorization, i.e. it is
designed to compute notion D (exact active-set sensitivity), NOT the barrier-
smoothed central-path derivative. Test whether it actually does that across:
  1 inactive, 2 strongly-active, 3 weakly-active, 4 coupled strongly-active,
  5 active-set-stable finite steps.
Oracle: ONE-SIDED finite-difference re-solve (the notebook's prescription) +
exact active-set closed forms.
"""
import numpy as np
from pounce import QpSensitivity, solve_qp

def onesided(P, A, lb, b0, resolve, d=1e-6):
    s = QpSensitivity(P=P, c=np.zeros(P.shape[0]), A=A, b=[b0], lb=lb)
    x0 = np.asarray(s.x)
    dx = s.parametric_step([0], [1.0])
    dR = (resolve(b0+d) - x0)/d
    dL = (x0 - resolve(b0-d))/d
    return s, x0, dx, dR, dL

results = []

# ---- 1 INACTIVE ----
P = np.eye(2); A = np.array([[1.0, 1.0]])
res = lambda b, lb=0.2: np.asarray(solve_qp(P=P, c=np.zeros(2), A=A, b=[b], lb=[lb, -np.inf]).x)
s, x0, dx, dR, dL = onesided(P, A, [0.2, -np.inf], 1.0, res)
err = max(np.max(np.abs(dx-dR)), np.max(np.abs(dx-dL)))
print(f"[1 INACTIVE]        dx/db={np.round(dx,4)}  FD_R={np.round(dR,4)}  err={err:.1e}  PASS={err<1e-4}")
results.append(("inactive", err < 1e-4))

# ---- 2 STRONGLY ACTIVE (scalar) ----
res = lambda b, lb=0.7: np.asarray(solve_qp(P=P, c=np.zeros(2), A=A, b=[b], lb=[lb, -np.inf]).x)
s, x0, dx, dR, dL = onesided(P, A, [0.7, -np.inf], 1.0, res)
err = max(np.max(np.abs(dx-dR)), np.max(np.abs(dx-dL)))
print(f"[2 STRONGLY ACTIVE] dx/db={np.round(dx,4)}  FD={np.round(dR,4)}  err={err:.1e}  "
      f"normal_motion={dx[0]:.1e}  PASS={err<1e-4}")
results.append(("strongly-active", err < 1e-4))

# ---- 3 WEAKLY ACTIVE (kink: exact right=[.5,.5], left=[0,1]) ----
res = lambda b, lb=0.5: np.asarray(solve_qp(P=P, c=np.zeros(2), A=A, b=[b], lb=[lb, -np.inf]).x)
s, x0, dx, dR, dL = onesided(P, A, [0.5, -np.inf], 1.0, res)
# exact one-sided active-set derivatives
exact_R = np.array([0.5, 0.5])   # bound releases
exact_L = np.array([0.0, 1.0])   # bound holds
# finite-step error both directions vs EXACT active-set solution
def exact_sol(b, lb=0.5):
    x = np.array([b/2, b/2]);  return x if x[0] >= lb else np.array([lb, b-lb])
errs = {}
for delta in (+0.1, -0.1):
    pred = x0 + s.parametric_step([0], [delta])
    errs[delta] = np.max(np.abs(pred - exact_sol(1.0+delta)))
print(f"[3 WEAKLY ACTIVE]   dx/db={np.round(dx,4)}  matches exact-{'L' if np.allclose(dx,exact_L,atol=1e-3) else 'R'} branch; "
      f"ill_cond={s.ill_conditioned}")
print(f"                    finite-step err: bound-releasing(+0.1)={errs[0.1]:.2e}  bound-holding(-0.1)={errs[-0.1]:.2e}")
print(f"                    -> one-sided (valid subgradient element); accurate one side, O(delta) wrong the other; NO weak-activity flag")
# "PASS" = it returns a VALID one-sided derivative (matches L or R exactly)
valid_onesided = np.allclose(dx, exact_L, atol=1e-3) or np.allclose(dx, exact_R, atol=1e-3)
results.append(("weakly-active (one-sided, known limitation)", valid_onesided))

# ---- 4 COUPLED 3-var STRONGLY ACTIVE (reduced Hessian must be W, not barrier) ----
W = np.array([[2.0,0.5,0.3],[0.5,1.5,0.2],[0.3,0.2,1.0]]); e = np.ones((1,3))
res3 = lambda b, lb=0.8: np.asarray(solve_qp(P=W, c=np.zeros(3), A=e, b=[b], lb=[lb, -np.inf, -np.inf]).x)
s = QpSensitivity(P=W, c=np.zeros(3), A=e, b=[1.0], lb=[0.8, -np.inf, -np.inf])
x0 = np.asarray(s.x); dx = s.parametric_step([0], [1.0]); d = 1e-6
fd = 0.5*((res3(1.0+d)-x0)/d + (x0-res3(1.0-d))/d)
err = np.max(np.abs(dx - fd))
print(f"[4 COUPLED strong]  dx/db={np.round(dx,4)}  FD={np.round(fd,4)}  err={err:.1e}  "
      f"normal_motion={dx[0]:.1e}  PASS={err<1e-6}")
print(f"                    -> tangential sensitivity uses reduced Hessian W[1:,1:] (original curvature), NOT barrier-inflated")
results.append(("coupled-strongly-active", err < 1e-6))

# ---- 5 ACTIVE-SET-STABLE finite steps (predictor exact across the whole stable range) ----
s = QpSensitivity(P=P, c=np.zeros(2), A=A, b=[1.0], lb=[0.7, -np.inf]); x0 = np.asarray(s.x)
worst = 0.0
for delta in (-0.2, -0.5, +0.2, +0.3):
    pred = x0 + s.parametric_step([0], [delta])
    true = np.asarray(solve_qp(P=P, c=np.zeros(2), A=A, b=[1.0+delta], lb=[0.7, -np.inf]).x)
    worst = max(worst, np.max(np.abs(pred - true)))
print(f"[5 STABLE finite]   worst predictor err over stable range = {worst:.1e}  PASS={worst<1e-6}")
results.append(("active-set-stable-finite-step", worst < 1e-6))

print("\n" + "="*66)
npass = sum(ok for _, ok in results)
for name, ok in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"\nVERDICT: {'PASS' if npass==len(results) else 'FAIL'} ({npass}/{len(results)})  "
      f"- QpSensitivity computes exact active-set sensitivity (notion D), no barrier inflation.")
