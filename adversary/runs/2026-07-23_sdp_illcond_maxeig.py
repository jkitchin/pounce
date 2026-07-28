"""Adversary cross-check: max-eigenvalue SDP with extreme spectrum
Family: sdp   Class: ill-conditioned (eigenvalues span 1e-8 .. 1e8)
Source: Boyd & Vandenberghe: min t s.t. t*I - M >= 0  =>  t* = lambda_max(M)
Known optimal: lambda_max(M), computed by numpy eigh (exact for symmetric M).
Direction: extreme spectral scaling of the PSD-cone data.
"""
import time, numpy as np

n = 4
# Symmetric M with eigenvalues spanning 1e-8 .. 1e8 in a rotated basis.
eigs = np.array([1e8, 1.0, 1e-4, 1e-8])
Q, _ = np.linalg.qr(np.random.RandomState(1).randn(n, n))
M = (Q * eigs) @ Q.T
M = 0.5 * (M + M.T)
lam_max = float(np.linalg.eigvalsh(M)[-1])
print(f"lambda_max(M)={lam_max:.10e}  cond(M)={eigs[0]/eigs[-1]:.1e}")

# --- pounce PSD:  min t  s.t.  t*I - M  PSD ---
# variable = t (scalar). slack block s = svec(t*I - M) must be PSD.
# svec layout: lower triangle, column-by-column, off-diag * sqrt(2).
from pounce import solve_socp
sq2 = np.sqrt(2.0)
def svec(X):
    out = []
    for j in range(n):
        for i in range(j, n):
            out.append(X[i, j] if i == j else sq2 * X[i, j])
    return np.array(out)
L = n * (n + 1) // 2
# s = h - G t  with  smat(s) = t*I - M  =>  s = t*svec(I) - svec(M)
G = (-svec(np.eye(n))).reshape(L, 1)     # -svec(I) so that h - G t = svec(M)... fix below
h = svec(M)
# We need s = svec(t I - M) = t*svec(I) - svec(M).  s = h - G t  => h = -svec(M), G = -svec(I).
G = (-svec(np.eye(n))).reshape(L, 1)
h = -svec(M)
c = np.array([1.0])
t0 = time.perf_counter()
r = solve_socp(P=None, c=c, G=G, h=h, cones=[("psd", n)])
t_pounce = time.perf_counter() - t0
t_star = float(np.asarray(r.x)[0])
status = r.status

# --- oracle: cvxpy ---
import cvxpy as cp
tc = cp.Variable()
prob = cp.Problem(cp.Minimize(tc), [tc * np.eye(n) - M >> 0])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
t_oracle_val = float(tc.value)

def rel(a, b): return abs(a - b) / max(1.0, abs(b))
obj_err_ref = rel(t_star, lam_max)
obj_err_oracle = rel(t_star, t_oracle_val)

print("=== pounce ===");  print(f"status={status} t*={t_star:.10e} t={t_pounce:.4f}s")
print("=== oracle(cvxpy) ==="); print(f"t*={t_oracle_val:.10e} t={t_oracle:.4f}s")
print("=== reference(eigh) ==="); print(f"lambda_max={lam_max:.10e}")
print(f"obj_err_vs_ref={obj_err_ref:.2e}  obj_err_vs_oracle={obj_err_oracle:.2e}")

ok = (status in ("optimal",) or getattr(r, "success", False)) and obj_err_ref < 1e-6
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err_ref={obj_err_ref:.2e})")
