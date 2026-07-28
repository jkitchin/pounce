"""Adversary cross-check: smallest-eigenvalue SDP via trace-1 constraint
Family: sdp   Class: small dense semidefinite program (3x3, one trace equality)

Problem:
    minimize   Tr(C X)
    subject to X >= 0   (3x3 PSD)
               Tr(X) = 1

KNOWN OPTIMAL (closed form):
    The variational/Rayleigh characterization of the smallest eigenvalue states
        lambda_min(C) = min { Tr(C X) : X >= 0, Tr(X) = 1 },
    with optimizer X* = v v^T where v is the unit eigenvector of C for its
    smallest eigenvalue (a rank-1 density matrix).
    For C = [[2,-1,0],[-1,2,-1],[0,-1,2]] (the path-graph Laplacian-like matrix /
    tridiagonal Toeplitz), eigenvalues are 2 - sqrt(2), 2, 2 + sqrt(2).
        => optimum = 2 - sqrt(2) ~= 0.5857864376269050.
SOURCE: Vandenberghe & Boyd, "Semidefinite Programming", SIAM Review 38(1),
    1996, sec. 2 (the trace-1 PSD feasible set is the spectrahedron whose linear
    minimization recovers lambda_min); standard Rayleigh-Ritz / Ky Fan k=1.

This is DISTINCT from the prior max-eigenvalue run (min t s.t. tI - A >= 0):
    that run had NO matrix variable and recovered lambda_max; here the decision
    variable IS the 3x3 matrix X with an explicit trace-1 equality, recovering
    lambda_min.

svec layout (pounce, confirmed from proven runs): lower triangle, column-major,
off-diagonals * sqrt(2):
    3x3 -> svec(M) = [M00, s2*M10, s2*M20, M11, s2*M21, M22], s2 = sqrt(2).
    Inner product <X,Y> = svec(X) . svec(Y), so Tr(C X) = svec(C) . svec(X).
"""
import time
import numpy as np

s2 = np.sqrt(2.0)
C = np.array([[2.0, -1.0, 0.0],
              [-1.0, 2.0, -1.0],
              [0.0, -1.0, 2.0]])
KNOWN_OPTIMAL = float(np.linalg.eigvalsh(C)[0])  # 2 - sqrt(2)

# Decision variables v = [X00, X10, X20, X11, X21, X22]  (6 free symm entries)
# index map: 0:X00 1:X10 2:X20 3:X11 4:X21 5:X22

# Objective Tr(C X) = sum_ij C_ij X_ij.  With symmetric X, off-diagonals appear
# twice: Tr(CX) = C00 X00 + C11 X11 + C22 X22
#                 + 2 C10 X10 + 2 C20 X20 + 2 C21 X21.
c = np.array([C[0, 0],          # X00
              2 * C[1, 0],      # X10
              2 * C[2, 0],      # X20
              C[1, 1],          # X11
              2 * C[2, 1],      # X21
              C[2, 2]])         # X22

# Equality: Tr(X) = X00 + X11 + X22 = 1
A = np.array([[1.0, 0, 0, 1.0, 0, 1.0]])
b = np.array([1.0])

# PSD slack s = svec(X) = h - G v ; h=0, G maps v -> -svec(X)
#   s0 = X00       = v0
#   s1 = s2 * X10  = s2 * v1
#   s2 = s2 * X20  = s2 * v2
#   s3 = X11       = v3
#   s4 = s2 * X21  = s2 * v4
#   s5 = X22       = v5
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
obj_pounce = float(c @ v)
status = r.status
X_pounce = np.array([[v[0], v[1], v[2]],
                     [v[1], v[3], v[4]],
                     [v[2], v[4], v[5]]])

# ---- Oracle: cvxpy, two solvers ----
import cvxpy as cp


def solve_cvxpy(solver):
    X = cp.Variable((3, 3), symmetric=True)
    cons = [X >> 0, cp.trace(X) == 1]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), cons)
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
print(f"known_optimal(lambda_min)={KNOWN_OPTIMAL:.10e}")
print(f"rel_err pounce vs known     = {rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"rel_err pounce vs SCS       = {rel(obj_pounce, obj_scs):.2e}")
print(f"rel_err pounce vs CLARABEL  = {rel(obj_pounce, obj_cla):.2e}")

eig_min = float(np.linalg.eigvalsh(X_pounce)[0])
trX = float(np.trace(X_pounce))
print(f"min eigenvalue of X_pounce = {eig_min:.3e}; trace = {trX:.6f}")

ok = ((status == "optimal") or getattr(r, "success", False)) \
    and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 \
    and rel(obj_pounce, obj_cla) < 1e-4 \
    and eig_min > -1e-6 \
    and abs(trX - 1.0) < 1e-5
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}, "
      f"eig_min={eig_min:.2e}, trX={trX:.4f})")
