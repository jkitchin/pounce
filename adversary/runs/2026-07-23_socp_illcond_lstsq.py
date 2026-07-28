"""Adversary cross-check: ill-conditioned least squares via SOC epigraph
Family: socp   Class: ill-conditioned (cond(A) ~ 1e8)
Source: SOC epigraph of least squares; reference from numpy SVD lstsq (float64)
Known optimal: residual norm computed by SVD; minimizer x* from lstsq
Direction: extreme conditioning of the cone data (matrix A inside the SOC).
"""
import time, numpy as np

np.random.seed(0)
n = 4
m = 6
# Build A with a controlled condition number ~1e8 via SVD.
U, _ = np.linalg.qr(np.random.randn(m, m))
V, _ = np.linalg.qr(np.random.randn(n, n))
svals = np.array([1e4, 1e1, 1e-2, 1e-4])          # cond = 1e4/1e-4 = 1e8
S = np.zeros((m, n)); S[:n, :n] = np.diag(svals)
A = U @ S @ V.T
x_true = np.array([1.0, -2.0, 3.0, -4.0])
b = A @ x_true + np.array([0.0, 0.0, 0.0, 0.0, 0.3, -0.2])  # slight inconsistency in U-null part

# --- reference: high-accuracy least squares (SVD) ---
x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)
res_ref = float(np.linalg.norm(A @ x_ref - b))
print(f"cond(A)={np.linalg.cond(A):.3e}")

# --- pounce SOC:  min t  s.t. (t, A x - b) in SOC ---
# variables z = (x[0..n-1], t);  slack s = h - G z must be in SOC of dim m+1.
# s[0] = t ; s[1:] = -(A x - b) = b - A x   (SOC symmetric in the vector part)
from pounce import solve_socp
nv = n + 1
c = np.zeros(nv); c[-1] = 1.0
G = np.zeros((m + 1, nv))
h = np.zeros(m + 1)
G[0, -1] = -1.0                     # s0 = t
G[1:, :n] = A                       # s[1:] = h - A x = b - A x
h[1:] = b
t0 = time.perf_counter()
r = solve_socp(P=None, c=c, G=G, h=h, cones=[("soc", m + 1)])
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)[:n]
res_pounce = float(np.linalg.norm(A @ x_pounce - b))
status = r.status

# --- oracle: cvxpy ---
import cvxpy as cp
xc = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.norm(A @ xc - b, 2)))
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle = np.asarray(xc.value)
res_oracle = float(np.linalg.norm(A @ x_oracle - b))

def rel(a, b): return abs(a - b) / max(1.0, abs(b))
obj_err = rel(res_pounce, res_ref)
x_err_ref = float(np.linalg.norm(x_pounce - x_ref, np.inf))
x_err_oracle = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

print("=== pounce ===");  print(f"status={status} res={res_pounce:.10e} t={t_pounce:.4f}s x={x_pounce}")
print("=== oracle(cvxpy) ==="); print(f"res={res_oracle:.10e} t={t_oracle:.4f}s x={x_oracle}")
print("=== reference(SVD lstsq) ==="); print(f"res={res_ref:.10e} x={x_ref}")
print(f"obj_err_vs_ref={obj_err:.2e}  x_inf_err_vs_ref={x_err_ref:.2e}  x_inf_err_vs_oracle={x_err_oracle:.2e}")

ok = (status in ("optimal",) or getattr(r, "success", False)) and obj_err < 1e-4 and x_err_ref < 1e-3 * max(1.0, np.linalg.norm(x_ref, np.inf))
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, x_err_ref={x_err_ref:.2e})")
