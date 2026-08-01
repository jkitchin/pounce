"""Adversary cross-check: nuclear norm of a fixed matrix via SDP
Family: sdp   Class: block-PSD-cone epigraph of the nuclear (trace) norm.
    DISTINCT from logged sdp probes (max-eig 2x2, min-trace, maxcut-triangle,
    Lyapunov-trace, min-eig-trace1, nearest-correlation, no-Slater-gap,
    strict-complementarity, ill-cond max-eig, Lovasz theta C5, spectral-norm,
    eigmax-2x2): none used the nuclear-norm epigraph block-matrix SDP.
Source: Fazel, Hindi, Boyd, "A Rank Minimization Heuristic with Application
    to Minimum Order System Approximation" (ACC 2001); the SDP
    representation of the nuclear norm ||X||_* is standard, e.g. Recht,
    Fazel, Parrilo, "Guaranteed Minimum-Rank Solutions of Linear Matrix
    Equations via Nuclear Norm Minimization" (SIAM Review 2010), Sec 2:

        ||A||_* = min_{W1,W2} 0.5*(tr(W1)+tr(W2))
                  s.t.  [[W1, A], [A^T, W2]] >= 0   (PSD)

    with equality at the optimum (this is a *tight* SDP characterization,
    not a relaxation), independent of A being fixed data here (no other
    constraints -- A is baked into the LMI, not a free variable).
Known optimal: ||A||_* = sum of singular values of A, computed independently
    via numpy.linalg.svd (a completely different algorithm from pounce's
    interior-point PSD-cone solve).
"""
import time
import numpy as np

rng_A = np.array([
    [3.0, 1.0, 0.0, 2.0],
    [0.0, 2.0, -1.0, 1.0],
    [1.0, 0.0, 4.0, -1.0],
])  # 3x4, deliberately non-square, non-symmetric, dense

m, n = rng_A.shape
sv = np.linalg.svd(rng_A, compute_uv=False)
KNOWN_OPTIMAL = float(np.sum(sv))     # independent closed-form (SVD)

# --- SDP formulation for pounce.solve_socp ---
# Block matrix Y = [[W1, A],[A^T, W2]] of size (m+n)x(m+n), PSD.
# W1: m x m symmetric (free), W2: n x n symmetric (free), off-diagonal
# block fixed = A (not a decision var -- folded into equality constraints
# tying the corresponding svec(Y) entries to A's fixed entries).
# Decision vector x = svec(Y) (dimension d(d+1)/2, d=m+n=7).
d = m + n
svec_dim = d * (d + 1) // 2
r2 = np.sqrt(2.0)

idx = {}
k = 0
for j in range(d):
    for i in range(j, d):
        idx[(i, j)] = k
        k += 1


def sidx(a, b):
    return idx[(max(a, b), min(a, b))]


# objective: 0.5*(tr(W1)+tr(W2)) = 0.5*tr(Y) restricted to the two diagonal
# blocks = 0.5 * sum_{i=0}^{d-1} Y_ii  (all diagonal entries of Y, since
# W1 occupies rows/cols 0..m-1 and W2 occupies m..d-1 -- together the full
# diagonal of Y).
c = np.zeros(svec_dim)
for i in range(d):
    c[idx[(i, i)]] = 0.5   # diag svec scale is 1

# s = h - G x in psd cone; x = svec(Y), G = -I, h = 0
G = -np.eye(svec_dim)
h = np.zeros(svec_dim)
cones = [("psd", d)]

# equalities: pin the off-diagonal block Y[0:m, m:d] = A (and its symmetric
# transpose is automatic since Y is symmetric by construction of svec).
Arows, brhs = [], []
for i in range(m):
    for j in range(n):
        row = np.zeros(svec_dim)
        col = m + j
        row[sidx(i, col)] = 1.0
        Arows.append(row)
        brhs.append(r2 * rng_A[i, j])   # svec off-diag entries are *sqrt(2)
A = np.array(Arows)
bvec = np.array(brhs)

import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, A=A, b=bvec, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
xv = np.asarray(r.x, float)
obj_pounce = float(r.obj)
status = str(r.status)

# reconstruct Y, verify PSD + the A block matches
Y = np.zeros((d, d))
for j in range(d):
    for i in range(j, d):
        val = xv[idx[(i, j)]] / (1.0 if i == j else r2)
        Y[i, j] = val
        Y[j, i] = val
eig_min = float(np.linalg.eigvalsh(Y)[0])
A_block = Y[0:m, m:d]
A_block_err = float(np.max(np.abs(A_block - rng_A)))

# --- oracle: cvxpy, native normNuc atom ---
import cvxpy as cp

Av = cp.Constant(rng_A)
t0 = time.perf_counter()
# cp.normNuc requires a Variable/Expression; wrap the fixed matrix trivially.
Xv = cp.Variable((m, n))
prob = cp.Problem(cp.Minimize(cp.normNuc(Xv)), [Xv == rng_A])
prob.solve(solver=cp.SCS)
t_cvx = time.perf_counter() - t0
val_cvx = float(prob.value)

def rel(a, ref):
    return abs(a - ref) / max(1.0, abs(ref))

print("=== pounce (PSD-cone IPM, nuclear norm via block LMI) ===")
print(f"status={status} obj=0.5*tr(Y)={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"  min_eig(Y)={eig_min:.3e} max|Y_block-A|={A_block_err:.2e}")
print(f"=== oracle: numpy SVD (closed form) sum(sigma)={KNOWN_OPTIMAL:.10e} sigma={sv}")
print(f"=== oracle: cvxpy/SCS normNuc val={val_cvx:.10e} t={t_cvx:.4f}s")
print(f"rel_err vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e} vs_cvx={rel(obj_pounce, val_cvx):.2e}")

ok = (status in ("optimal", "optimal_inaccurate") or getattr(r, "success", False)) \
    and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 and eig_min > -1e-6 and A_block_err < 1e-5
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err={rel(obj_pounce, KNOWN_OPTIMAL):.2e}, "
      f"eig_min={eig_min:.2e}, block_err={A_block_err:.2e})")
