"""Adversary cross-check: log-optimal (Kelly criterion) portfolio via exp cone
Family: exp   Class: exp-cone (log-utility maximization over discrete scenarios)
Source: Kelly criterion / log-optimal investment, J.L. Kelly Jr., "A New
        Interpretation of Information Rate", Bell Sys. Tech. J. 35 (1956);
        also Cover & Thomas, "Elements of Information Theory" 2nd ed., Ch 16
        (log-optimal portfolios); modeled via the exponential cone exactly as
        in the CVXPY/MOSEK "log-optimal investment" cookbook examples.
        max (1/K) sum_k log(r_k^T w)  s.t. sum(w)=1, w>=0
        Two independent bets: asset 0 = cash (gross return 1.0 always),
        asset 1 doubles/halves with prob 1/2 each, asset 2 gains 20%/loses
        40% with prob 1/2 each, independent of asset 1 -> K=4 equally-likely
        scenarios (all 4 combinations).
Known optimal: none published in closed form for this instance; oracle-only
        (cvxpy exp-cone solve + an independent smooth-NLP scipy solve).
"""
import numpy as np
import time

n = 3  # assets: cash, A, B
K = 4  # scenarios (2 bets x 2 outcomes)

# rows = scenarios, cols = assets (gross returns)
R = np.array(
    [
        [1.0, 1.5, 1.2],  # A up,  B up
        [1.0, 1.5, 0.6],  # A up,  B down
        [1.0, 0.5, 1.2],  # A down, B up
        [1.0, 0.5, 0.6],  # A down, B down
    ]
)

# --- pounce: solve_socp with exp-cone epigraph on log(r_k^T w) ---
from pounce import solve_socp

nvar = n + K  # w (n) then t_k (K)


def widx(i):
    return i


def tidx(k):
    return n + k


A_eq = np.zeros((1, nvar))
A_eq[0, :n] = 1.0
b_eq = np.array([1.0])

rows = n + 3 * K
G = np.zeros((rows, nvar))
h = np.zeros(rows)
cones = []

# block 1: nonneg, w_i >= 0  -> s_i = w_i
for i in range(n):
    G[i, widx(i)] = -1.0
    h[i] = 0.0
cones.append(("nonneg", n))

# block 2..: exp cone per scenario: (t_k, 1, r_k^T w) in Kexp
row = n
for k in range(K):
    G[row, tidx(k)] = -1.0
    h[row] = 0.0  # s = t_k
    row += 1
    h[row] = 1.0  # s = 1
    row += 1
    G[row, :n] = -R[k, :]
    h[row] = 0.0  # s = r_k^T w
    row += 1
    cones.append(("exp", 3))

c = np.zeros(nvar)
c[n:] = -1.0 / K  # minimize -(1/K) sum t_k  == maximize (1/K) sum t_k

t0 = time.perf_counter()
r = solve_socp(c=c, A=A_eq, b=b_eq, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
z_pounce = np.asarray(r.x)
w_pounce = z_pounce[:n]
growth_pounce = np.mean(np.log(R @ w_pounce))
status = r.status

# --- oracle 1: cvxpy (native exp-cone support via ECOS/SCS/CLARABEL) ---
import cvxpy as cp

w = cp.Variable(n)
prob = cp.Problem(cp.Maximize(cp.sum(cp.log(R @ w)) / K), [cp.sum(w) == 1, w >= 0])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
w_oracle = w.value
growth_oracle = prob.value

# --- oracle 2: independent smooth NLP (scipy SLSQP), no cone machinery at all ---
from scipy.optimize import minimize

def neg_growth(w_):
    return -np.mean(np.log(R @ w_))

cons = [{"type": "eq", "fun": lambda w_: np.sum(w_) - 1.0}]
bnds = [(0.0, 1.0)] * n
res = minimize(neg_growth, x0=np.full(n, 1.0 / n), method="SLSQP", bounds=bnds, constraints=cons, tol=1e-12)
w_scipy = res.x
growth_scipy = -res.fun


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


err_cvxpy = rel(growth_pounce, growth_oracle)
err_scipy = rel(growth_pounce, growth_scipy)
w_err_cvxpy = float(np.linalg.norm(w_pounce - w_oracle, np.inf))
w_err_scipy = float(np.linalg.norm(w_pounce - w_scipy, np.inf))

print("=== pounce (exp cone) ===")
print(f"status={status} growth={growth_pounce:.10e} t={t_pounce:.4f}s w={w_pounce}")
print("=== oracle: cvxpy CLARABEL ===")
print(f"growth={growth_oracle:.10e} t={t_oracle:.4f}s w={w_oracle}")
print("=== oracle: scipy SLSQP (smooth NLP) ===")
print(f"growth={growth_scipy:.10e} success={res.success} w={w_scipy}")
print(f"growth_err_vs_cvxpy={err_cvxpy:.2e} w_inf_err_vs_cvxpy={w_err_cvxpy:.2e}")
print(f"growth_err_vs_scipy={err_scipy:.2e} w_inf_err_vs_scipy={w_err_scipy:.2e}")

ok = status == "optimal" and err_cvxpy < 1e-4 and err_scipy < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status})")
