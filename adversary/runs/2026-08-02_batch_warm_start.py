"""Adversary cross-check: solve_qp_batch warm_starts correctness
Family: batch   Class: solve_qp_batch, warm_starts= parameter

Problem: a batch of K equality-constrained QPs sharing (P, A) but each with
its own linear term c_k:
    minimize   0.5 x'Px + c_k'x
    subject to sum(x) = 1

Closed-form KKT (equality-only QP -- independent of both pounce and cvxpy):
    [P  A'] [x     ]   [-c_k]
    [A  0 ] [lambda] = [ b  ]
solved directly via numpy.linalg.solve, one 5x5 linear system per item.

This tests the *warm_starts* parameter of solve_qp_batch specifically (not
yet exercised in the log): solve a "day 1" batch cold, perturb each c_k to
get a "day 2" batch, then solve day 2 twice -- once cold, once warm-started
from day 1's per-item results. Warm start must not change the answer (only
iteration count), so day-2-cold and day-2-warm must agree with each other
AND with the closed-form KKT solve of day 2.
"""
import time
import numpy as np
from pounce import solve_qp_batch, solve_qp

rng = np.random.default_rng(20260802)
n = 4
K = 6

# SPD P shared across the batch
M = rng.normal(size=(n, n))
P = M @ M.T + 2.0 * np.eye(n)
A = np.ones((1, n))
b = np.array([1.0])

C_day1 = rng.normal(size=(K, n))
C_day2 = C_day1 + 0.05 * rng.normal(size=(K, n))  # small perturbation


def kkt_solve(P, A, b, c):
    n = P.shape[0]
    m = A.shape[0]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = P
    M[:n, n:] = A.T
    M[n:, :n] = A
    rhs = np.concatenate([-c, b])
    sol = np.linalg.solve(M, rhs)
    return sol[:n]


X_day1_known = np.array([kkt_solve(P, A, b, C_day1[k]) for k in range(K)])
X_day2_known = np.array([kkt_solve(P, A, b, C_day2[k]) for k in range(K)])

problems_day1 = [dict(P=P, c=C_day1[k], A=A, b=b) for k in range(K)]
problems_day2 = [dict(P=P, c=C_day2[k], A=A, b=b) for k in range(K)]

t0 = time.perf_counter()
res_day1 = solve_qp_batch(problems_day1)
t_day1 = time.perf_counter() - t0

t0 = time.perf_counter()
res_day2_cold = solve_qp_batch(problems_day2)
t_day2_cold = time.perf_counter() - t0

t0 = time.perf_counter()
res_day2_warm = solve_qp_batch(problems_day2, warm_starts=res_day1)
t_day2_warm = time.perf_counter() - t0

X_day1_pounce = np.array([r.x for r in res_day1])
X_day2_cold_pounce = np.array([r.x for r in res_day2_cold])
X_day2_warm_pounce = np.array([r.x for r in res_day2_warm])

iters_cold = [r.iters for r in res_day2_cold]
iters_warm = [r.iters for r in res_day2_warm]

err_day1 = float(np.max(np.abs(X_day1_pounce - X_day1_known)))
err_day2_cold = float(np.max(np.abs(X_day2_cold_pounce - X_day2_known)))
err_day2_warm = float(np.max(np.abs(X_day2_warm_pounce - X_day2_known)))
cold_vs_warm = float(np.max(np.abs(X_day2_cold_pounce - X_day2_warm_pounce)))

statuses_day1 = [r.status for r in res_day1]
statuses_day2_cold = [r.status for r in res_day2_cold]
statuses_day2_warm = [r.status for r in res_day2_warm]

print("=== pounce solve_qp_batch ===")
print(f"day1 (cold): statuses={statuses_day1} err_vs_KKT={err_day1:.2e} t={t_day1:.4f}s")
print(f"day2 (cold): statuses={statuses_day2_cold} err_vs_KKT={err_day2_cold:.2e} t={t_day2_cold:.4f}s iters={iters_cold}")
print(f"day2 (warm): statuses={statuses_day2_warm} err_vs_KKT={err_day2_warm:.2e} t={t_day2_warm:.4f}s iters={iters_warm}")
print(f"cold_vs_warm x max-abs-diff={cold_vs_warm:.2e}")
print(f"mean iters cold={np.mean(iters_cold):.2f} warm={np.mean(iters_warm):.2f}")

# also cross-check one item against the single-problem solve_qp entry point
single = solve_qp(P=P, c=C_day2[0], A=A, b=b)
single_err = float(np.max(np.abs(np.asarray(single.x) - X_day2_known[0])))
batch_vs_single = float(np.max(np.abs(X_day2_cold_pounce[0] - np.asarray(single.x))))
print(f"item0: single solve_qp err_vs_KKT={single_err:.2e} batch_vs_single={batch_vs_single:.2e}")

ok = (
    all(s == "optimal" for s in statuses_day1 + statuses_day2_cold + statuses_day2_warm)
    and err_day1 < 1e-6
    and err_day2_cold < 1e-6
    and err_day2_warm < 1e-6
    and cold_vs_warm < 1e-6
    and batch_vs_single < 1e-8
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (err_day1={err_day1:.2e} err_day2_cold={err_day2_cold:.2e} err_day2_warm={err_day2_warm:.2e} cold_vs_warm={cold_vs_warm:.2e})")
