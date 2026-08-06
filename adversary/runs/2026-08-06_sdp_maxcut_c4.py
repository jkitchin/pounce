"""Adversary cross-check: Goemans-Williamson MAX-CUT SDP relaxation on C4
Family: sdp   Class: PSD cone, standard-form Laplacian SDP relaxation
Source: Goemans & Williamson (1995), J. ACM 42(6); SDP relaxation
            max  sum_{(i,j) in E} (1 - Y_ij)/2
            s.t. diag(Y) = 1, Y PSD
        For a BIPARTITE graph the relaxation is TIGHT (equals the true
        max-cut, achieved by Y_ij = -1 across the bipartition, +1 within
        each side -- a rank-1 +-1 solution, which is trivially PSD).
        C4 (4-cycle 0-1-2-3-0) is bipartite ({0,2} vs {1,3}), |E|=4, so
        every edge is cut by the bipartition -> max cut = 4.
Known optimal: 4.0
"""
import time
import numpy as np

edges = [(0, 1), (1, 2), (2, 3), (3, 0)]   # 4-cycle C4 (bipartite: {0,2} vs {1,3})
n = 4
KNOWN_OPTIMAL = 4.0
dim_svec = n * (n + 1) // 2
SQRT2 = np.sqrt(2.0)


def svec(M):
    """Lower triangle, column-major, off-diagonals x sqrt(2) -- matches
    pounce's solve_socp psd-cone convention exactly."""
    out = np.zeros(dim_svec)
    k = 0
    for j in range(n):
        for i in range(j, n):
            out[k] = M[i, j] if i == j else M[i, j] * SQRT2
            k += 1
    return out


def smat(v):
    M = np.zeros((n, n))
    k = 0
    for j in range(n):
        for i in range(j, n):
            val = v[k] if i == j else v[k] / SQRT2
            M[i, j] = val
            M[j, i] = val
            k += 1
    return M


# Objective: minimize f(Y) = 0.5 * sum_{edges} Y_ij  (so cut = |E|/2 - f(Y) is
# maximized). trace(C, Y) = sum_ij C_ij Y_ij = f(Y) requires C_ij = C_ji =
# 0.25 for each edge (i,j), i != j, C_ii = 0.
C = np.zeros((n, n))
for (i, j) in edges:
    C[i, j] += 0.25
    C[j, i] += 0.25
c_svec = svec(C)

# Equality constraints: diag(Y) = 1
A_eq = np.zeros((n, dim_svec))
b_eq = np.ones(n)
for i in range(n):
    e = np.zeros((n, n))
    e[i, i] = 1.0
    A_eq[i, :] = svec(e)

# G, h for the PSD cone slack: s = h - G x = x  =>  G = -I, h = 0
G = -np.eye(dim_svec)
h = np.zeros(dim_svec)

from pounce import solve_socp
t0 = time.perf_counter()
r = solve_socp(c=c_svec, A=A_eq, b=b_eq, G=G, h=h, cones=[("psd", n)])
t_pounce = time.perf_counter() - t0
status = r.status
Y_pounce = smat(r.x)
cut_pounce = len(edges) / 2.0 - r.obj

# --- oracle: cvxpy ---
import cvxpy as cp
Y = cp.Variable((n, n), symmetric=True)
constraints = [Y >> 0, cp.diag(Y) == 1]
cut_expr = sum(0.5 * (1 - Y[i, j]) for (i, j) in edges)
prob = cp.Problem(cp.Maximize(cut_expr), constraints)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
cut_oracle = prob.value
Y_oracle = Y.value


def rel(u, v):
    return abs(u - v) / max(1.0, abs(v))


obj_err = rel(cut_pounce, cut_oracle)
known_err = rel(cut_pounce, KNOWN_OPTIMAL)
Y_err = float(np.abs(Y_pounce - Y_oracle).max())

print("=== pounce ===")
print(f"status={status} cut_value={cut_pounce:.10e} t={t_pounce:.4f}s")
print(f"diag(Y)={np.diag(Y_pounce)}")
print("=== oracle (cvxpy/CLARABEL) ===")
print(f"cut_value={cut_oracle:.10e} t={t_oracle:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={known_err:.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} Y_inf_err={Y_err:.2e}")

ok = (status == "optimal") and known_err < 1e-4 and obj_err < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, known_err={known_err:.2e}, obj_err={obj_err:.2e})")
