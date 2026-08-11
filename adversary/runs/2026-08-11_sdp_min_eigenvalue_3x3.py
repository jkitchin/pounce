"""Adversary cross-check: minimum eigenvalue of a 3x3 symmetric matrix via SDP
Family: sdp   Class: min-eigenvalue SDP (PSD cone, off-diagonal svec scaling)
Source: standard SDP characterization of the minimum eigenvalue (Vandenberghe &
        Boyd, "Semidefinite Programming", SIAM Review 38(1), 1996, sec 3.1;
        Boyd & Vandenberghe, Convex Optimization, sec 5.9): for symmetric A,
        lambda_min(A) = max t  s.t.  A - t*I ⪰ 0.
        A larger (3x3, dense, non-diagonal) instance than the existing
        python/tests/test_socp.py 2x2 fixtures (test_psd_min_eigenvalue_diagonal
        / _offdiagonal), to exercise a different svec length and a fully dense
        off-diagonal block.
Known optimal: lambda_min(A) computed independently via numpy.linalg.eigvalsh.
"""
import time
import numpy as np

A = np.array([
    [3., 1., 0.],
    [1., 4., 1.],
    [0., 1., 5.],
])
KNOWN_OPTIMAL = float(np.linalg.eigvalsh(A).min())
print(f"numpy eigvalsh lambda_min={KNOWN_OPTIMAL:.10e}")


def svec(M):
    n = M.shape[0]
    out = []
    for j in range(n):
        for i in range(j, n):
            out.append(M[i, j] * (2.0 ** 0.5 if i != j else 1.0))
    return np.array(out)


I3 = np.eye(3)
G = svec(I3).reshape(-1, 1)  # s = h - G@[t] = svec(A) - t*svec(I)
h = svec(A)

# --- pounce: maximize t <=> minimize -t ---
from pounce import solve_socp

t0 = time.perf_counter()
r = solve_socp(c=[-1.0], G=G, h=h, cones=[("psd", 3)])
t_pounce = time.perf_counter() - t0
t_pounce_val = r.x[0]
obj_pounce = -r.obj  # r.obj is min(-t) = -lambda_min ; flip sign for lambda_min
status = r.status
print("=== pounce ===")
print(f"status={status} lambda_min={obj_pounce:.10e} t={t_pounce:.4f}s")

# --- oracle: cvxpy (SCS / Clarabel PSD) ---
import cvxpy as cp

t = cp.Variable()
prob = cp.Problem(cp.Maximize(t), [A - t * np.eye(3) >> 0])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
obj_oracle = prob.value
print("=== oracle (cvxpy/Clarabel) ===")
print(f"status={prob.status} lambda_min={obj_oracle:.10e} t={t_oracle:.4f}s")


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_oracle = rel(obj_pounce, obj_oracle)

print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={obj_err_known:.2e}")
print(f"obj_err_vs_oracle={obj_err_oracle:.2e}")

ok = (status == "optimal") and obj_err_known < 1e-5 and obj_err_oracle < 1e-5
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err_known={obj_err_known:.2e}, obj_err_oracle={obj_err_oracle:.2e})")
