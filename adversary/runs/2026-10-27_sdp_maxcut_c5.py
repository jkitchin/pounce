"""Adversary cross-check: Goemans-Williamson MAX-CUT SDP relaxation on C5
Family: sdp   Class: PSD cone with unit-diagonal equalities, NON-tight (odd cycle)
Source: Goemans & Williamson, J. ACM 42(6):1115-1145 (1995); for the cycle C_n
        the SDP value is (n/2)(1 - cos(pi (n-1)/n))  (e.g. Delorme & Poljak,
        Math. Prog. 62 (1993) 557-574, eigenvalue bound is exact on cycles).
        n=5: (5/2)(1+cos(pi/5)) = 5(5+sqrt5)/8 = 4.52254248... (an earlier draft wrote 25(5+sqrt5)/32 = 5.65, an arithmetic slip caught by both solvers)
        Distinct from the logged C4 run (bipartite, where the SDP is tight):
        here the relaxation is strictly above the integer max-cut (4).
Known optimal: 5*(5+sqrt(5))/8
"""
import time, numpy as np
n = 5
KNOWN = 5 * (5 + 5 ** 0.5) / 8
edges = [(i, (i + 1) % n) for i in range(n)]
L = np.zeros((n, n))
for i, j in edges:
    L[i, i] += 1; L[j, j] += 1; L[i, j] -= 1; L[j, i] -= 1
# max (1/4) <L, X>  s.t. X_ii = 1, X psd.  Variables x = svec(X).
idx = []
for j in range(n):
    for i in range(j, n):
        idx.append((i, j))
m = len(idx)
r2 = 2 ** 0.5
c = np.array([-(0.25 * L[i, j]) * (1.0 if i == j else r2) for i, j in idx])
A = np.zeros((n, m)); b = np.ones(n)
for k, (i, j) in enumerate(idx):
    if i == j:
        A[i, k] = 1.0
from pounce import solve_socp
t0 = time.perf_counter()
r = solve_socp(c=c, A=A, b=b, G=-np.eye(m), h=np.zeros(m), cones=[("psd", n)])
tp = time.perf_counter() - t0
val_p = -r.obj
X = np.zeros((n, n))
for k, (i, j) in enumerate(idx):
    v = r.x[k] / (1 if i == j else r2); X[i, j] = X[j, i] = v
eig_min = np.linalg.eigvalsh(X).min()
import cvxpy as cp
Xv = cp.Variable((n, n), symmetric=True)
prob = cp.Problem(cp.Maximize(0.25 * cp.trace(L @ Xv)), [Xv >> 0, cp.diag(Xv) == 1])
t0 = time.perf_counter(); prob.solve(solver=cp.CLARABEL); to = time.perf_counter() - t0
rel = lambda a, b: abs(a - b) / max(1, abs(b))
print(f"pounce status={r.status} val={val_p:.10f} t={tp:.4f}s  diag={np.diag(X).round(8)} eig_min={eig_min:.2e}")
print(f"oracle(clarabel) status={prob.status} val={prob.value:.10f} t={to:.4f}s")
print(f"known={KNOWN:.10f} err_known={rel(val_p,KNOWN):.2e} err_oracle={rel(val_p,prob.value):.2e}")
ok = r.status == "optimal" and rel(val_p, KNOWN) < 1e-6 and eig_min > -1e-7 and np.allclose(np.diag(X), 1, atol=1e-7)
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
