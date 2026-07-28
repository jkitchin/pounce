"""Adversary cross-check: min-trace SDP with a fixed off-diagonal
Family: sdp   Class: small dense semidefinite program (trace minimization)

Problem:
    minimize   Tr(X)
    subject to X >= 0  (2x2 PSD)
               X[0,0] = 1
               X[0,1] = 1

KNOWN OPTIMAL (closed form):
    PSD of a 2x2 requires X00 >= 0, X11 >= 0, X00*X11 - X01^2 >= 0.
    With X00 = 1 and X01 = 1 -> X11 >= X01^2 / X00 = 1.
    Tr(X) = X00 + X11 = 1 + X11, minimized at X11 = 1 -> Tr = 2.
    => optimum = 2.0, with X = [[1,1],[1,1]] (rank-1).
SOURCE: standard textbook PSD-cone reasoning (Schur complement / 2x2 minor).

svec layout (pounce): lower triangle, column-major, off-diagonals * sqrt(2):
    2x2 -> svec(M) = [M00, sqrt(2)*M10, M11].
    Inner product <X,Y> = svec(X).svec(Y).
"""
import time
import numpy as np

KNOWN_OPTIMAL = 2.0
r2 = np.sqrt(2.0)

# ---- Variables for pounce ----
# We need decision variables. The free entries of X are X00, X01(=X10), X11.
# Constraints X00=1, X01=1 are linear equalities (A x = b).
# Objective Tr(X) = X00 + X11.
# PSD slack: s = svec(X) must lie in psd cone, i.e. smat(s) >= 0.
#   We set s = G x ... wait: pounce form is s = h - G x in the cone.
# Let variable vector v = [X00, X01, X11].
# svec(X) = [X00, sqrt(2)*X01, X11].
#   s0 = X00      -> s0 = v0
#   s1 = sqrt(2)*X01 -> s1 = sqrt(2)*v1
#   s2 = X11      -> s2 = v2
# We want s = h - G v  ==>  h=0, G = -[[1,0,0],[0,sqrt(2),0],[0,0,1]].
c = np.array([1.0, 0.0, 1.0])          # Tr(X) = X00 + X11
A = np.array([[1.0, 0.0, 0.0],         # X00 = 1
              [0.0, 1.0, 0.0]])        # X01 = 1
b = np.array([1.0, 1.0])
G = -np.array([[1.0, 0.0, 0.0],
               [0.0, r2, 0.0],
               [0.0, 0.0, 1.0]])
h = np.array([0.0, 0.0, 0.0])

import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, A=A, b=b, G=G, h=h, cones=[("psd", 2)])
t_pounce = time.perf_counter() - t0
v = np.asarray(r.x, dtype=float)
obj_pounce = float(c @ v)
status = r.status
X_pounce = np.array([[v[0], v[1]], [v[1], v[2]]])

# ---- Oracle: cvxpy, two solvers ----
import cvxpy as cp

def solve_cvxpy(solver):
    X = cp.Variable((2, 2), symmetric=True)
    cons = [X >> 0, X[0, 0] == 1, X[0, 1] == 1]
    prob = cp.Problem(cp.Minimize(cp.trace(X)), cons)
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    return prob.value, time.perf_counter() - t0, X.value

obj_scs, t_scs, X_scs = solve_cvxpy(cp.SCS)
obj_cla, t_cla, X_cla = solve_cvxpy(cp.CLARABEL)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


print("=== pounce (PSD cone IPM) ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"X_pounce =\n{X_pounce}")
print("=== oracle cvxpy/SCS ===")
print(f"obj={obj_scs:.10e} t={t_scs:.4f}s")
print("=== oracle cvxpy/CLARABEL ===")
print(f"obj={obj_cla:.10e} t={t_cla:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL}")
print(f"rel_err pounce vs known  = {rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"rel_err pounce vs SCS    = {rel(obj_pounce, obj_scs):.2e}")
print(f"rel_err pounce vs CLARABEL = {rel(obj_pounce, obj_cla):.2e}")

# PSD sanity: smallest eigenvalue of recovered X
eig_min = float(np.linalg.eigvalsh(X_pounce)[0])
print(f"min eigenvalue of X_pounce = {eig_min:.3e}")

ok = ((status == "optimal") or getattr(r, "success", False)) \
    and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 \
    and eig_min > -1e-6
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}, eig_min={eig_min:.2e})")
