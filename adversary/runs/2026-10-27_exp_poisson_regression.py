"""Adversary cross-check: Poisson regression maximum likelihood via exp cones
Family: exp   Class: GLM log-likelihood, one Kexp triple per observation (m=8)
Source: Poisson GLM negative log-likelihood (McCullagh & Nelder, Generalized
        Linear Models, 2nd ed., sec 6.2):  min_beta sum_i exp(a_i.beta) - y_i a_i.beta
        (the log(y_i!) constant dropped). Strictly convex when A has full column
        rank. No closed form; oracles: cvxpy/Clarabel (exp cone) AND scipy
        Newton-CG with exact gradient/Hessian (smooth, cone-free) -- independent
        of each other. Distinct from logged exp probes (logistic regression /
        softplus, entropy, GP, Kelly, water-filling).
Known optimal: none closed-form; checked by the first-order condition
        A^T (exp(A beta) - y) = 0 at pounce's beta (oracle-free).
"""
import time, numpy as np
rng = np.random.default_rng(20261027)
m = 8
A = np.column_stack([np.ones(m), rng.uniform(-1, 1, m), rng.uniform(0, 2, m)])
y = np.array([1, 0, 3, 2, 5, 1, 0, 4], dtype=float)
nb = A.shape[1]
# z = (beta, t_1..t_m); triple i: (a_i.beta, 1, t_i) in Kexp  <=>  exp(a_i.beta) <= t_i
nv = nb + m; G = []; h = []
for i in range(m):
    r = np.zeros(nv); r[:nb] = -A[i]; G.append(r); h.append(0.0)
    G.append(np.zeros(nv)); h.append(1.0)
    r = np.zeros(nv); r[nb + i] = -1; G.append(r); h.append(0.0)
c = np.concatenate([-(A.T @ y), np.ones(m)])
from pounce import solve_socp
t0 = time.perf_counter()
r = solve_socp(c=c, G=np.array(G), h=np.array(h), cones=[("exp", 3)] * m)
tp = time.perf_counter() - t0
bp = np.asarray(r.x[:nb]); fp = float(np.sum(np.exp(A @ bp)) - y @ A @ bp)
kkt = np.abs(A.T @ (np.exp(A @ bp) - y)).max()
import cvxpy as cp
b = cp.Variable(nb)
prob = cp.Problem(cp.Minimize(cp.sum(cp.exp(A @ b)) - (A.T @ y) @ b))
t0 = time.perf_counter(); prob.solve(solver=cp.CLARABEL); to = time.perf_counter() - t0
from scipy.optimize import minimize
f = lambda bb: np.sum(np.exp(A @ bb)) - y @ A @ bb
g = lambda bb: A.T @ (np.exp(A @ bb) - y)
H = lambda bb: A.T @ (np.exp(A @ bb)[:, None] * A)
t0 = time.perf_counter(); s = minimize(f, np.zeros(nb), jac=g, hess=H, method="trust-exact", options={"gtol": 1e-12}); ts = time.perf_counter() - t0
rel = lambda u, v: abs(u - v) / max(1, abs(v))
print(f"pounce status={r.status} obj={r.obj:.12f} f(beta)={fp:.12f} t={tp:.4f}s kkt_inf={kkt:.2e}")
print(f"cvxpy  status={prob.status} obj={prob.value:.12f} t={to:.4f}s")
print(f"scipy  success={s.success} obj={s.fun:.12f} t={ts:.4f}s")
print(f"beta pounce={bp}  scipy={s.x}  inf_err={np.abs(bp-s.x).max():.2e}")
print(f"err_vs_scipy={rel(fp,s.fun):.2e} err_vs_cvxpy={rel(r.obj,prob.value):.2e}")
ok = r.status == "optimal" and rel(fp, s.fun) < 1e-6 and rel(r.obj, s.fun) < 1e-6 and np.abs(bp - s.x).max() < 1e-4
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
