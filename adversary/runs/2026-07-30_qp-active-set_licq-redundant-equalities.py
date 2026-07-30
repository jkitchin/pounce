"""Adversary cross-check: LICQ-violating QP with duplicated equality rows
Family: qp-active-set   Class: degenerate / rank-deficient constraint Jacobian
Source: Nocedal & Wright, *Numerical Optimization* 2nd ed., §16.1 (equality-QP
        KKT) + the standard LICQ-violation construction (a row repeated as an
        exact multiple). Optimum derived in closed form below.
Known optimal: see KNOWN_OPTIMAL

Targets this PR's rank-repair path: the active set contains linearly dependent
rows, so the active-set KKT is singular and no H-block shift repairs it. Before
the repair this class produced a hard LinearSolverFailure.

  min  1/2 (x0^2 + x1^2 + x2^2)
  s.t. x0 + x1 + x2 = 3          (r0)
       2x0 + 2x1 + 2x2 = 6       (r1 = 2*r0, redundant -> LICQ violated)
       x0 - x1 = 0               (r2)

Reduced to r0 and r2: minimize ||x||^2/2 on that affine set.
r2 => x0 = x1 = a; r0 => 2a + x2 = 3 => x2 = 3 - 2a.
f(a) = 1/2 (2a^2 + (3-2a)^2); f'(a) = 2a - 2(3-2a)*2 ... solve exactly below.
"""
import time

import numpy as np

# closed form: minimize 0.5*(2a^2 + (3-2a)^2) over a
# d/da = 2a + (3-2a)*(-2) = 2a - 6 + 4a = 6a - 6 = 0 -> a = 1
A_STAR = 1.0
X_STAR = np.array([A_STAR, A_STAR, 3.0 - 2.0 * A_STAR])
KNOWN_OPTIMAL = 0.5 * float(X_STAR @ X_STAR)

P = np.eye(3)
c = np.zeros(3)
A = np.array([[1.0, 1.0, 1.0],
              [2.0, 2.0, 2.0],      # exact duplicate -> rank deficient
              [1.0, -1.0, 0.0]])
b = np.array([3.0, 6.0, 0.0])

from pounce.qp import solve_qp

res = {}
for method in ("ipm", "active-set"):
    t0 = time.perf_counter()
    r = solve_qp(P=P, c=c, A=A, b=b, method=method)
    res[method] = (r, time.perf_counter() - t0)

import cvxpy as cp

x = cp.Variable(3)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x)), [A @ x == b])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


print(f"known_optimal={KNOWN_OPTIMAL:.10e}  x*={X_STAR}")
print(f"=== oracle (cvxpy/CLARABEL) === obj={prob.value:.10e} t={t_oracle:.4f}s")
ok = True
for method, (r, t) in res.items():
    oe = rel(r.obj, KNOWN_OPTIMAL)
    xe = float(np.linalg.norm(np.asarray(r.x) - X_STAR, np.inf))
    print(f"=== pounce {method} === status={r.status} obj={r.obj:.10e} "
          f"rel_err_vs_known={oe:.2e} x_inf_err={xe:.2e} t={t:.4f}s")
    # the safety property this PR is built around: never claim success on a
    # wrong objective
    if r.status == "optimal" and oe > 1e-4:
        print(f"  !! SOLVED-BUT-WRONG on {method}")
        ok = False
    if r.status != "optimal" or oe > 1e-4:
        ok = False

print("VERDICT: PASS" if ok else "VERDICT: FAIL")
