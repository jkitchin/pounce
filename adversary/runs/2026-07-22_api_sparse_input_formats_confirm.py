"""Confirmation of the two findings from 2026-07-22_api_sparse_input_formats.py

F1 (SOLVER_BUG candidate): the check_psd guard sees a DIFFERENT matrix than the
    solver when P is a COO with duplicate (i,j) entries.  python/pounce/qp.py
    `_min_eig_lower_coo` does `M[ri,ci] = vi` (assignment) while the solver
    accumulates duplicates (scipy/COO sum convention).  An indefinite P therefore
    slips past the issue-#112 guard and solve_qp returns status="optimal" at a
    SADDLE point.

F2 (documented hazard, not a bug): P given as upper triangle only is silently
    read as diagonal-only.
"""
import numpy as np
import scipy.sparse as sp
from pounce import solve_qp

n = 2
c = np.array([-1.0, -1.0])
lb, ub = -10.0 * np.ones(n), 10.0 * np.ones(n)

# Duplicate-COO P: lower-triangle (1,0) supplied as 1.5 + 1.5 -> 3.0 by the
# scipy summation convention.
P_dup = sp.coo_matrix(([2.0, 2.0, 1.5, 1.5], ([0, 1, 1, 1], [0, 1, 0, 0])),
                      shape=(2, 2))
P_summed = np.array([[2.0, 3.0], [3.0, 2.0]])      # what the SOLVER uses
P_overwritten = np.array([[2.0, 1.5], [1.5, 2.0]])  # what the GUARD checks

print("P as scipy densifies it (sum convention):\n", P_dup.toarray())
print("symmetrized-from-lower (solver's view):\n", P_summed,
      " eig =", np.linalg.eigvalsh(P_summed))
print("guard's view (last-duplicate-wins):\n", P_overwritten,
      " eig =", np.linalg.eigvalsh(P_overwritten))

print("\n--- A: dense equivalent of the SOLVER's matrix ---")
try:
    r = solve_qp(P=P_summed, c=c, lb=lb, ub=ub)
    print(f"  ACCEPTED  status={r.status} obj={r.obj:.6e} x={np.asarray(r.x)}")
except ValueError as e:
    print(f"  REJECTED  ValueError: {e}")

print("\n--- B: the duplicate-COO sparse form (same math) ---")
r2 = solve_qp(P=P_dup, c=c, lb=lb, ub=ub)
x2 = np.asarray(r2.x)
print(f"  ACCEPTED  status={r2.status} obj={r2.obj:.6e} x={x2}")

print("\n--- which matrix did the solver actually use? (stationarity test) ---")
for name, M in (("summed [[2,3],[3,2]]", P_summed),
                ("overwritten [[2,1.5],[1.5,2]]", P_overwritten)):
    g = M @ x2 + c
    print(f"  grad with {name:<32s} = {g}  ||g||={np.linalg.norm(g):.3e}")

print("\n--- is the returned point actually optimal? ---")
f = lambda x: 0.5 * x @ P_summed @ x + c @ x
print(f"  f(returned x)   = {f(x2):.6e}   (solve_qp reported {r2.obj:.6e})")
corners = [np.array([a, b]) for a in (-10.0, 10.0) for b in (-10.0, 10.0)]
fc = [(f(p), p) for p in corners]
best = min(fc, key=lambda t: t[0])
print("  box corners:", [(float(v), p.tolist()) for v, p in fc])
print(f"  true minimum over the box <= {best[0]:.6e} at x={best[1].tolist()}")
print(f"  ==> reported objective exceeds the true minimum by "
      f"{r2.obj - best[0]:.6e}")

# Independent oracle: dense grid + scipy local minimization
from scipy.optimize import minimize as smin
gr = np.linspace(-10, 10, 401)
XX, YY = np.meshgrid(gr, gr)
FF = 0.5 * (2 * XX**2 + 6 * XX * YY + 2 * YY**2) - XX - YY
k = np.unravel_index(np.argmin(FF), FF.shape)
print(f"  grid min = {FF[k]:.6e} at ({XX[k]}, {YY[k]})")
res = smin(f, np.array([9.0, -9.0]), bounds=[(-10, 10)] * 2)
print(f"  scipy L-BFGS-B min = {res.fun:.6e} at {res.x}")

print("\n--- does check_psd=True help? ---")
try:
    r3 = solve_qp(P=P_dup, c=c, lb=lb, ub=ub, check_psd=True)
    print(f"  check_psd=True STILL ACCEPTED: status={r3.status} obj={r3.obj:.6e}")
except ValueError as e:
    print(f"  check_psd=True rejected: {e}")

bug = (r2.status == "optimal") and (r2.obj > best[0] + 1e-6)
print("\nVERDICT: SOLVER_BUG (guard bypass -> silently-wrong 'optimal')" if bug
      else "\nVERDICT: PASS")
