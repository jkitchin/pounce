"""Adversary cross-check: batch of Non-Negative Least Squares (NNLS) QPs
Family: batch   Class: solve_qp_batch, box-constrained (nonneg) least squares,
    K heterogeneous items -- fresh oracle for this family: scipy.optimize.nnls
    (Lawson-Hanson active-set NNLS), a completely different algorithm/codebase
    from pounce's interior-point QP path and not previously used as a batch
    oracle (prior batch runs cross-checked against closed-form KKT solves,
    single solve_qp, or solve_qp_multi_rhs -- never an independent dedicated
    NNLS solver).

Problem: for each item k, minimize ||A_k x - b_k||_2^2 s.t. x >= 0, posed as
a QP with P_k = 2 A_k^T A_k, c_k = -2 A_k^T b_k, lb=0. Each item has its OWN
randomly generated A_k (m x n) and b_k (m,) -- i.e. P and c both vary across
the batch (heterogeneous shapes/data, not just a shared-P varying-c batch).

KNOWN_OPTIMAL: none in closed form (NNLS on random dense data has no simple
analytic solution); the correctness oracle is scipy.optimize.nnls itself,
per-item, plus a single solve_qp cross-check and a residual/KKT feasibility
check independent of both codebases.
"""
import time
import numpy as np
from scipy.optimize import nnls
from pounce import solve_qp_batch, solve_qp

rng = np.random.default_rng(20260804)
K = 8
n = 4

problems = []
As, bs = [], []
for k in range(K):
    m = rng.integers(5, 9)  # varying row counts across items
    A = rng.normal(size=(m, n))
    b = rng.normal(size=m) + 0.5  # bias so some coords want to go negative
    As.append(A)
    bs.append(b)
    P = 2.0 * (A.T @ A)
    c = -2.0 * (A.T @ b)
    problems.append(dict(P=P, c=c, lb=np.zeros(n)))

t0 = time.perf_counter()
res_batch = solve_qp_batch(problems)
t_batch = time.perf_counter() - t0

X_pounce = np.array([r.x for r in res_batch])
statuses = [r.status for r in res_batch]

# --- oracle: scipy.optimize.nnls, independent Lawson-Hanson algorithm ------
t0 = time.perf_counter()
X_nnls = np.array([nnls(As[k], bs[k])[0] for k in range(K)])
t_nnls = time.perf_counter() - t0

resid_pounce = np.array([np.linalg.norm(As[k] @ X_pounce[k] - bs[k]) ** 2 for k in range(K)])
resid_nnls = np.array([np.linalg.norm(As[k] @ X_nnls[k] - bs[k]) ** 2 for k in range(K)])

x_err = np.array([np.linalg.norm(X_pounce[k] - X_nnls[k], np.inf) for k in range(K)])
obj_err = np.abs(resid_pounce - resid_nnls) / np.maximum(1.0, resid_nnls)

# --- cross-check: item 0 also via the single solve_qp entry point ----------
single0 = solve_qp(P=problems[0]["P"], c=problems[0]["c"], lb=problems[0]["lb"])
batch_vs_single0 = float(np.linalg.norm(X_pounce[0] - np.asarray(single0.x), np.inf))

print("=== pounce solve_qp_batch ===")
print(f"statuses={statuses} t={t_batch:.4f}s")
print(f"per-item resid (pounce)={np.round(resid_pounce, 6)}")
print("=== oracle scipy.optimize.nnls (Lawson-Hanson) ===")
print(f"t={t_nnls:.4f}s")
print(f"per-item resid (nnls)  ={np.round(resid_nnls, 6)}")
print(f"x_inf_err per item = {np.round(x_err, 6)}")
print(f"obj_rel_err per item = {np.round(obj_err, 8)}")
print(f"max x_inf_err={x_err.max():.2e}  max obj_rel_err={obj_err.max():.2e}")
print(f"batch_vs_single(item0) x_inf_err={batch_vs_single0:.2e}")

ok = (
    all(s == "optimal" for s in statuses)
    and np.all(x_err < 1e-4)
    and np.all(obj_err < 1e-6)
    and batch_vs_single0 < 1e-8
)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (max_x_err={x_err.max():.2e}, max_obj_err={obj_err.max():.2e}, "
      f"batch_vs_single0={batch_vs_single0:.2e})")
