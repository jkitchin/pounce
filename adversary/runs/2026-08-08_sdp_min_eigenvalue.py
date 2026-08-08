"""Adversary cross-check: minimum eigenvalue of a symmetric matrix via SDP.

Family: sdp   Class: 1-variable PSD-cone program, svec/smat round trip on a
              non-trivial (non-diagonal) 3x3 matrix -- distinct from the
              MAX-CUT/C4 problem already logged (2026-08-06).
Source: Boyd & Vandenberghe, Convex Optimization, Sec 4.6.2 ("Minimum
        eigenvalue via SDP"): for symmetric A,
            lambda_min(A) = max { t : A - t I >= 0 }.
        This is a standard textbook fact and is checked here two
        independent ways: (1) numpy/LAPACK's eigvalsh (a dense direct
        eigensolver, not an SDP and not pounce), (2) cvxpy's own SDP
        formulation of the same problem.

A = [[2,1,0],[1,3,1],[0,1,2]]  (symmetric tridiagonal, not diagonal, so the
svec off-diagonal sqrt(2) scaling is genuinely exercised).

pounce formulation: single variable x=[t]. Cone slack s = h - G t must be
PSD with smat(s) = A - t*I, i.e. G's single column is svec(I) and h is
svec(A). We maximize t (minimize -t).
"""

import time

import numpy as np

A = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
n_mat = 3


def svec(M):
    """Lower triangle, column-major, off-diagonals scaled by sqrt(2)."""
    out = []
    for j in range(n_mat):
        for i in range(j, n_mat):
            v = M[i, j]
            out.append(v if i == j else v * np.sqrt(2.0))
    return np.array(out)


I3 = np.eye(n_mat)
h = svec(A)
G = svec(I3).reshape(-1, 1)  # s = h - G @ [t] = svec(A) - t*svec(I) = svec(A - tI)
c = np.array([-1.0])  # minimize -t == maximize t

# --- pounce ---
from pounce import solve_socp

t0 = time.perf_counter()
r = solve_socp(c=c, G=G, h=h, cones=[("psd", n_mat)])
t_pounce = time.perf_counter() - t0
t_pounce_val = float(np.asarray(r.x)[0])

# --- oracle 1: numpy/LAPACK eigvalsh (independent direct eigensolver) ---
t0 = time.perf_counter()
eigs = np.linalg.eigvalsh(A)
t_lapack = time.perf_counter() - t0
lambda_min = float(eigs[0])

# --- oracle 2: cvxpy SDP (independent DCP/cone formulation) ---
import cvxpy as cp

t_var = cp.Variable()
prob = cp.Problem(cp.Maximize(t_var), [A - t_var * np.eye(n_mat) >> 0])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_cvxpy = time.perf_counter() - t0


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


print("=== pounce (solve_socp, psd cone) ===")
print(f"status={r.status} t*={t_pounce_val:.10e} obj={r.obj:.10e} time={t_pounce:.4f}s")
print("=== oracle: numpy/LAPACK eigvalsh ===")
print(f"lambda_min={lambda_min:.10e} time={t_lapack:.6f}s")
print("=== oracle: cvxpy/Clarabel SDP ===")
print(f"status={prob.status} t*={t_var.value:.10e} time={t_cvxpy:.4f}s")

err_lapack = rel(t_pounce_val, lambda_min)
err_cvxpy = rel(t_pounce_val, t_var.value)

# PSD residual check: A - t*I must be (numerically) PSD at the reported t.
resid_eigs = np.linalg.eigvalsh(A - t_pounce_val * np.eye(n_mat))
min_resid_eig = float(resid_eigs[0])

print(f"rel_err_vs_lapack={err_lapack:.2e} rel_err_vs_cvxpy={err_cvxpy:.2e}")
print(f"min_eig(A - t*I)={min_resid_eig:.3e} (must be >= ~0)")

ok = (
    r.status == "optimal"
    and err_lapack < 1e-6
    and err_cvxpy < 1e-6
    and min_resid_eig > -1e-6
)
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
