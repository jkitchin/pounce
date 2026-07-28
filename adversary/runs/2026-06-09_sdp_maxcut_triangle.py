"""Adversary cross-check: Goemans-Williamson Max-Cut SDP relaxation on K3
Family: sdp   Class: small dense semidefinite program (3x3 SDP relaxation)

Problem (Goemans-Williamson, 1995):
    maximize   (1/2) * sum_{i<j} w_ij * (1 - X_ij)
    subject to X >= 0  (3x3 PSD)
               X_ii = 1   (i = 0,1,2)
For the unit-weight triangle K3 (edges (0,1),(0,2),(1,2), all w=1).

KNOWN OPTIMAL (closed form):
    The optimal Gram matrix is the "Mercedes-Benz" configuration
        X* = [[1,-1/2,-1/2],[-1/2,1,-1/2],[-1/2,-1/2,1]]
    eigenvalues {0, 3/2, 3/2}  -> PSD, X_ii=1.
    Each (1 - X_ij) = 3/2; sum over 3 edges = 9/2; times 1/2 = 9/4.
    => optimum (max) = 9/4 = 2.25.
SOURCE: Goemans & Williamson, JACM 42(6), 1995; standard SDP relaxation of
    max-cut; triangle SDP value 9/4 is a classic worked example.

We solve the MINIMIZATION that pounce expects:
    objective_max = (1/2)*sum(1 - X_ij) = (3/2) - (1/2)*(X01 + X02 + X12)
    => minimize (1/2)*(X01 + X02 + X12), then objective_max = 3/2 - that_min.
    min part = (1/2)*(-3/2) = -3/4 ; objective_max = 3/2 + 3/4 = 9/4. check.

svec layout (pounce): lower triangle, column-major, off-diagonals * sqrt(2):
    3x3 -> svec(M) = [M00, s2*M10, s2*M20, M11, s2*M21, M22], s2 = sqrt(2).
"""
import time
import numpy as np

KNOWN_OPTIMAL_MAX = 2.25
s2 = np.sqrt(2.0)

# Decision variables v = [X00, X10, X20, X11, X21, X22]  (6 free symm entries)
# index map: 0:X00 1:X10 2:X20 3:X11 4:X21 5:X22
# Objective for pounce (minimize): (1/2)*(X10 + X20 + X21)
c = np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0.0])

# Equalities X_ii = 1
A = np.array([
    [1.0, 0, 0, 0, 0, 0],  # X00 = 1
    [0, 0, 0, 1.0, 0, 0],  # X11 = 1
    [0, 0, 0, 0, 0, 1.0],  # X22 = 1
])
b = np.array([1.0, 1.0, 1.0])

# PSD slack s = svec(X) = h - G v ; choose h=0, G maps v -> -svec(X)
# svec rows (column-major lower tri):
#   s0 = X00            = v0
#   s1 = s2 * X10       = s2 * v1
#   s2 = s2 * X20       = s2 * v2
#   s3 = X11            = v3
#   s4 = s2 * X21       = s2 * v4
#   s5 = X22            = v5
G = -np.array([
    [1.0, 0,   0,   0,   0,   0],
    [0,   s2,  0,   0,   0,   0],
    [0,   0,   s2,  0,   0,   0],
    [0,   0,   0,   1.0, 0,   0],
    [0,   0,   0,   0,   s2,  0],
    [0,   0,   0,   0,   0,   1.0],
])
h = np.zeros(6)

import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, A=A, b=b, G=G, h=h, cones=[("psd", 3)])
t_pounce = time.perf_counter() - t0
v = np.asarray(r.x, dtype=float)
min_part = float(c @ v)
obj_pounce_max = 1.5 - min_part   # objective_max = 3/2 - (1/2)*sum  (note c already has 1/2)
status = r.status
X_pounce = np.array([[v[0], v[1], v[2]],
                     [v[1], v[3], v[4]],
                     [v[2], v[4], v[5]]])

# ---- Oracle: cvxpy, two solvers (solve the MAX directly) ----
import cvxpy as cp
W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)

def solve_cvxpy(solver):
    X = cp.Variable((3, 3), symmetric=True)
    obj = 0.5 * cp.sum(cp.multiply(W, (1 - X))) / 2  # /2 because W double counts (i,j)&(j,i)
    cons = [X >> 0] + [X[i, i] == 1 for i in range(3)]
    prob = cp.Problem(cp.Maximize(obj), cons)
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    return prob.value, time.perf_counter() - t0, X.value

obj_scs, t_scs, X_scs = solve_cvxpy(cp.SCS)
obj_cla, t_cla, X_cla = solve_cvxpy(cp.CLARABEL)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


print("=== pounce (PSD cone IPM) ===")
print(f"status={status} obj_max={obj_pounce_max:.10e} t={t_pounce:.4f}s")
print(f"X_pounce =\n{X_pounce}")
print("=== oracle cvxpy/SCS ===")
print(f"obj_max={obj_scs:.10e} t={t_scs:.4f}s")
print("=== oracle cvxpy/CLARABEL ===")
print(f"obj_max={obj_cla:.10e} t={t_cla:.4f}s")
print(f"known_optimal_max={KNOWN_OPTIMAL_MAX}")
print(f"rel_err pounce vs known    = {rel(obj_pounce_max, KNOWN_OPTIMAL_MAX):.2e}")
print(f"rel_err pounce vs SCS      = {rel(obj_pounce_max, obj_scs):.2e}")
print(f"rel_err pounce vs CLARABEL = {rel(obj_pounce_max, obj_cla):.2e}")

eig_min = float(np.linalg.eigvalsh(X_pounce)[0])
diag = np.diag(X_pounce)
print(f"min eigenvalue of X_pounce = {eig_min:.3e}; diag = {diag}")

ok = ((status == "optimal") or getattr(r, "success", False)) \
    and rel(obj_pounce_max, KNOWN_OPTIMAL_MAX) < 1e-4 \
    and eig_min > -1e-6 \
    and np.allclose(diag, 1.0, atol=1e-5)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err_known={rel(obj_pounce_max, KNOWN_OPTIMAL_MAX):.2e}, eig_min={eig_min:.2e})")
