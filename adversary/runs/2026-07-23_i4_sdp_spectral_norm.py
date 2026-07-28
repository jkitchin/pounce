"""Adversary i4: SPECTRAL NORM (largest singular value) of a nonsymmetric
matrix via a Schur-complement SDP.
Family: sdp   Class: single dense PSD cone with a structured off-diagonal block.

Problem:  minimize t  s.t.  [[t I_m, A],[A^T, t I_n]] >= 0.
Known optimum: ||A||_2 = sigma_max(A) (largest singular value).
SOURCE: Boyd & Vandenberghe, Convex Optimization, eq. for matrix-norm SDP
(Sec. 4.6.3 / A.5.5); Schur-complement characterization ||A||_2 <= t.
This is DISTINCT from the logged max-eigenvalue SDP (that was a SYMMETRIC
eigenvalue bound tI - A >= 0); here A is nonsymmetric and the PSD block is a
2x2-block bordered matrix, testing off-diagonal svec entries with the sqrt(2)
scaling.

A = [[1,2],[3,4]] (m=n=2). The stacked matrix M is 4x4.
svec (pounce): lower triangle, column-major, off-diagonals * sqrt(2).
"""
import time
import numpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0]])
m, n = A.shape
N = m + n                                 # 4x4 stacked matrix
KNOWN_OPTIMAL = float(np.linalg.svd(A, compute_uv=False)[0])
r2 = np.sqrt(2.0)

# svec index map for NxN, lower-tri column-major
idx = {}
k = 0
for j in range(N):
    for i in range(j, N):
        idx[(i, j)] = k
        k += 1
svec_dim = k                              # N(N+1)/2 = 10

def svec_of(M):
    v = np.zeros(svec_dim)
    for j in range(N):
        for i in range(j, N):
            v[idx[(i, j)]] = M[i, j] * (1.0 if i == j else r2)
    return v

# M(t) = diag(t) on the diagonal + constant border blocks A (top-right), A^T (bottom-left)
# constant part M0: M0[0:m, m:] = A, M0[m:, 0:m] = A^T
M0 = np.zeros((N, N))
M0[0:m, m:] = A
M0[m:, 0:m] = A.T
h = svec_of(M0)                           # constant svec part
# coefficient of t: identity on diagonal
Mt = np.eye(N)
gcoef = svec_of(Mt)                       # svec of I (diag ones)

# decision var: [t]; slack s = h - G t must be svec(M(t)) = h + gcoef*t
# => G = -gcoef (column), so s = h - (-gcoef) t = h + gcoef t.  correct.
c = np.array([1.0])                        # minimize t
G = (-gcoef).reshape(svec_dim, 1)
cones = [("psd", N)]

import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
t_val = float(np.asarray(r.x, float)[0])
obj_pounce = t_val
status = str(r.status)

# reconstruct M and check PSD
Mrec = M0 + t_val * np.eye(N)
eig_min = float(np.linalg.eigvalsh(Mrec)[0])

# oracle: cvxpy building the SAME SDP + direct sigma_max
import cvxpy as cp
def solve_cvxpy(solver):
    tt = cp.Variable()
    Mv = cp.bmat([[tt * np.eye(m), A], [A.T, tt * np.eye(n)]])
    prob = cp.Problem(cp.Minimize(tt), [Mv >> 0])
    t0 = time.perf_counter(); prob.solve(solver=solver); dt = time.perf_counter() - t0
    return float(prob.value), dt

val_cla, t_cla = solve_cvxpy(cp.CLARABEL)
val_scs, t_scs = solve_cvxpy(cp.SCS)

def rel(a, b): return abs(a - b) / max(1.0, abs(b))

print("=== pounce (PSD-cone IPM, spectral-norm SDP) ===")
print(f"status={status} t*={t_val:.10e} t={t_pounce:.4f}s  min_eig(M)={eig_min:.3e}")
print(f"=== cvxpy/CLARABEL val={val_cla:.10e} t={t_cla:.4f}s")
print(f"=== cvxpy/SCS      val={val_scs:.10e} t={t_scs:.4f}s")
print(f"known sigma_max(A) = {KNOWN_OPTIMAL:.10e}")
print(f"rel vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e} "
      f"vs_CLARABEL={rel(obj_pounce, val_cla):.2e} vs_SCS={rel(obj_pounce, val_scs):.2e}")

ok = (status in ("optimal", "optimal_inaccurate") or getattr(r, "success", False)) \
    and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 and eig_min > -1e-6
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err={rel(obj_pounce, KNOWN_OPTIMAL):.2e}, eig_min={eig_min:.2e})")
