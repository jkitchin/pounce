"""Adversary cross-check: weighted sum |x_i - a_i|^p, p=1.25, s.t. sum x = 0
Family: power   Class: fractional p<1.5 epigraph, alpha=0.8 (near the cone's alpha->1 edge)
Source: closed-form KKT (Lagrange): w_i p |d_i|^{p-1} sgn(d_i) = -lambda gives
        d_i = x_i - a_i = -k w_i^{-1/(p-1)}, k = S / sum_j w_j^{-1/(p-1)}, S = sum a_i,
        f* = S^p / (sum_j w_j^{-1/(p-1)})^{p-1}  (strictly convex => unique).
        Prior power probes used alpha in {1/2,1/3,1/4,1/5,2/3}; alpha=0.8 untested.
"""
import time, numpy as np
p = 1.25; alpha = 1 / p
a = np.array([0.7, -0.2, 1.3, 0.4, 0.9]); w = np.array([1.0, 2.0, 0.5, 3.0, 1.5])
n = len(a); S = a.sum(); e = -1 / (p - 1)
k = S / np.sum(w ** e); d = -k * w ** e
KX = a + d; KNOWN = S ** p / np.sum(w ** e) ** (p - 1)
assert abs(np.sum(w * np.abs(d) ** p) - KNOWN) < 1e-12 and abs(KX.sum()) < 1e-12
# vars z = (x_0..x_{n-1}, t_0..t_{n-1});  pow triple (x_i - a_i, t_i, 1): |x_i-a_i| <= t_i^alpha
nv = 2 * n; G = []; h = []
for i in range(n):
    r1 = np.zeros(nv); r1[i] = -1; G.append(r1); h.append(-a[i])   # s = x_i - a_i
    r2 = np.zeros(nv); r2[n + i] = -1; G.append(r2); h.append(0.0)  # s = t_i
    G.append(np.zeros(nv)); h.append(1.0)                            # s = 1
c = np.concatenate([np.zeros(n), w]); A = np.concatenate([np.ones(n), np.zeros(n)])[None, :]
from pounce import solve_socp
t0 = time.perf_counter()
r = solve_socp(c=c, A=A, b=[0.0], G=np.array(G), h=np.array(h), cones=[("pow", alpha)] * n)
tp = time.perf_counter() - t0
import cvxpy as cp
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(w @ cp.power(cp.abs(x - a), p)), [cp.sum(x) == 0])
t0 = time.perf_counter(); prob.solve(solver=cp.CLARABEL); to = time.perf_counter() - t0
rel = lambda u, v: abs(u - v) / max(1, abs(v))
xe = np.abs(r.x[:n] - KX).max()
print(f"pounce status={r.status} obj={r.obj:.12f} t={tp:.4f}s x_err={xe:.2e}")
print(f"oracle status={prob.status} obj={prob.value:.12f} t={to:.4f}s x_err={np.abs(x.value-KX).max():.2e}")
print(f"known={KNOWN:.12f} err_known={rel(r.obj,KNOWN):.2e}")
ok = r.status == "optimal" and rel(r.obj, KNOWN) < 1e-6 and xe < 1e-4
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
