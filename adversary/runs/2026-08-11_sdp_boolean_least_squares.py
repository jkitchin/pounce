"""Adversary cross-check: SDP relaxation of Boolean Least Squares (BLS).
Family: sdp   Class: PSD cone, homogenized lifting with diag(Y)=1 (distinct
objective structure from prior sdp probes: max-cut/Lovasz-theta use a
{-1,+1} EDGE objective on Y itself; this uses a dense quadratic-form
objective trace(QY) built from a least-squares data matrix, and the bound
is checked against a brute-force combinatorial optimum, not a graph
symmetry argument).
Source: Boolean least squares SDP relaxation, e.g. Boyd & Vandenberghe,
        Convex Optimization (2004) sec 4.6.4 / exercises; Luo, Ma, So, Ye,
        Zhang, "Semidefinite Relaxation of Quadratic Optimization
        Problems," IEEE SP Mag 27(3), 2010.

  True (NP-hard) problem: minimize ||Ax - b||_2^2 over x in {-1, +1}^n.
  Homogenize y = (1, x) in R^{n+1}, Y = y y' (PSD, rank 1, diag(Y)=1).
    ||Ax-b||^2 = x'(A'A)x - 2(A'b)'x + b'b = trace(Q Y)
  where Q = [[b'b, -(A'b)'], [-(A'b), A'A]] (block indices 0 and 1..n).
  SDP relaxation: minimize trace(Q Y) s.t. diag(Y)=1, Y PSD (drop rank-1).
  This is a valid LOWER BOUND on the true combinatorial minimum -- for any
  feasible SDP objective and the true combinatorial optimum obtained by
  brute-force over all 2^n sign vectors, we must have
      SDP_relaxation_value <= true_combinatorial_min + tol.
  A violation of that inequality (pounce's returned "optimal" value
  EXCEEDING the true achievable minimum) would be a solver bug, not a
  looseness-of-relaxation issue.
Known optimal: SDP relaxation value = cvxpy's value (checked to tolerance);
independently, the SDP value must lower-bound the brute-force optimum.
"""
import time
from itertools import product

import numpy as np

np.random.seed(11)
m, n = 5, 4
A = np.random.randn(m, n)
b = np.random.randn(m)

dim = n + 1
dim_svec = dim * (dim + 1) // 2
SQRT2 = np.sqrt(2.0)


def svec(M):
    out = np.zeros(dim_svec)
    k = 0
    for j in range(dim):
        for i in range(j, dim):
            out[k] = M[i, j] if i == j else M[i, j] * SQRT2
            k += 1
    return out


def smat(v):
    M = np.zeros((dim, dim))
    k = 0
    for j in range(dim):
        for i in range(j, dim):
            val = v[k] if i == j else v[k] / SQRT2
            M[i, j] = val
            M[j, i] = val
            k += 1
    return M


AtA = A.T @ A
Atb = A.T @ b
Q = np.zeros((dim, dim))
Q[0, 0] = b @ b
Q[0, 1:] = -Atb
Q[1:, 0] = -Atb
Q[1:, 1:] = AtA
c_svec = svec(Q)

A_eq = np.zeros((dim, dim_svec))
b_eq = np.ones(dim)
for i in range(dim):
    e = np.zeros((dim, dim))
    e[i, i] = 1.0
    A_eq[i, :] = svec(e)

G = -np.eye(dim_svec)
h = np.zeros(dim_svec)

# --- brute-force true combinatorial optimum ---
best = np.inf
best_x = None
for signs in product([-1.0, 1.0], repeat=n):
    x = np.array(signs)
    val = np.sum((A @ x - b) ** 2)
    if val < best:
        best, best_x = val, x
TRUE_COMBINATORIAL_MIN = best

# --- pounce ---
from pounce import solve_socp
t0 = time.perf_counter()
r = solve_socp(c=c_svec, A=A_eq, b=b_eq, G=G, h=h, cones=[("psd", dim)])
t_pounce = time.perf_counter() - t0
status = r.status
obj_pounce = r.obj
Y_pounce = smat(r.x)

# --- oracle: cvxpy ---
import cvxpy as cp
Y = cp.Variable((dim, dim), symmetric=True)
constraints = [Y >> 0, cp.diag(Y) == 1]
prob = cp.Problem(cp.Minimize(cp.trace(Q @ Y)), constraints)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
obj_oracle = prob.value
Y_oracle = Y.value


def rel(u, v):
    return abs(u - v) / max(1.0, abs(v))


obj_err = rel(obj_pounce, obj_oracle)
Y_err = float(np.abs(Y_pounce - Y_oracle).max())
bound_violation = obj_pounce - TRUE_COMBINATORIAL_MIN   # must be <= ~0

print("=== pounce (SDP relaxation) ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"diag(Y)={np.diag(Y_pounce)}")
print("=== oracle (cvxpy/CLARABEL) ===")
print(f"obj={obj_oracle:.10e} t={t_oracle:.4f}s")
print(f"brute_force_true_min(x in {{-1,1}}^{n})={TRUE_COMBINATORIAL_MIN:.10e} at x={best_x}")
print(f"obj_err_vs_oracle={obj_err:.2e} Y_inf_err_vs_oracle={Y_err:.2e}")
print(f"bound_violation(pounce_obj - true_min)={bound_violation:.2e} (must be <= ~1e-6)")

ok = (
    status == "optimal"
    and obj_err < 1e-4
    and Y_err < 1e-3
    and bound_violation < 1e-6
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, bound_violation={bound_violation:.2e})")
