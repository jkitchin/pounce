"""Adversary cross-check: batch of diagonal-covariance portfolio QPs
Family: batch   Class: solve_qp_batch vs per-item solve_qp + closed-form KKT
Source: classic Markowitz mean-variance QP, closed-form KKT for diagonal
        covariance with a single budget equality constraint (no bounds):
            minimize  0.5 x^T D x - mu^T x   s.t.  1^T x = 1
        Lagrangian: D x - mu - lambda*1 = 0  =>  x_i = (mu_i + lambda) / d_i
        Enforce 1^T x = 1:
            lambda = (1 - sum(mu_i / d_i)) / sum(1 / d_i)
Known optimal: analytic closed form per instance (see solve_closed_form below)
"""
import time
import numpy as np

rng = np.random.default_rng(20260729)

N_PROBLEMS = 8
N = 6  # variables per problem

problems = []
closed_form = []
for k in range(N_PROBLEMS):
    d = rng.uniform(0.5, 20.0, size=N)          # diagonal covariance-like weights
    mu = rng.uniform(-3.0, 3.0, size=N)         # expected returns / linear term
    D = np.diag(d)
    A = np.ones((1, N))
    b = np.array([1.0])

    lam = (1.0 - np.sum(mu / d)) / np.sum(1.0 / d)
    x_star = (mu + lam) / d
    obj_star = 0.5 * x_star @ D @ x_star - mu @ x_star

    problems.append(dict(P=D, c=-mu, A=A, b=b))
    closed_form.append((x_star, obj_star))

# --- pounce: batched ---
from pounce import solve_qp_batch, solve_qp

t0 = time.perf_counter()
batch_results = solve_qp_batch(problems)
t_batch = time.perf_counter() - t0

# --- pounce: per-item (oracle contract: batch must agree with single-item path) ---
t0 = time.perf_counter()
single_results = [solve_qp(**p) for p in problems]
t_single = time.perf_counter() - t0

# --- independent oracle: cvxpy, one instance solved independently, plus closed form for all ---
import cvxpy as cp

max_closed_err = 0.0
max_single_vs_batch_err = 0.0
max_obj_closed_err = 0.0
rows = []
for k, (p, (x_star, obj_star), rb, rs) in enumerate(
    zip(problems, closed_form, batch_results, single_results)
):
    xb = np.asarray(rb.x)
    xs = np.asarray(rs.x)
    x_err_closed = float(np.linalg.norm(xb - x_star, np.inf))
    obj_err_closed = abs(rb.obj - obj_star) / max(1.0, abs(obj_star))
    batch_vs_single = float(np.linalg.norm(xb - xs, np.inf))
    max_closed_err = max(max_closed_err, x_err_closed)
    max_obj_closed_err = max(max_obj_closed_err, obj_err_closed)
    max_single_vs_batch_err = max(max_single_vs_batch_err, batch_vs_single)
    rows.append((k, rb.status, rb.obj, obj_star, x_err_closed, batch_vs_single))

# Independent cvxpy cross-check on instance 0 and instance N_PROBLEMS-1
def cvxpy_check(idx):
    p = problems[idx]
    n = p["P"].shape[0]
    x = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.quad_form(x, p["P"]) + p["c"] @ x)
    cons = [p["A"] @ x == p["b"]]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL)
    return prob.value, np.asarray(x.value)

cvx_obj0, cvx_x0 = cvxpy_check(0)
cvx_objN, cvx_xN = cvxpy_check(N_PROBLEMS - 1)

obj_err_cvx0 = abs(batch_results[0].obj - cvx_obj0) / max(1.0, abs(cvx_obj0))
obj_err_cvxN = abs(batch_results[-1].obj - cvx_objN) / max(1.0, abs(cvx_objN))
x_err_cvx0 = float(np.linalg.norm(np.asarray(batch_results[0].x) - cvx_x0, np.inf))
x_err_cvxN = float(np.linalg.norm(np.asarray(batch_results[-1].x) - cvx_xN, np.inf))

print("=== pounce solve_qp_batch ===")
for k, status, obj, obj_star, xerr, bvs in rows:
    print(f"  [{k}] status={status} obj={obj:.8e} known={obj_star:.8e} "
          f"x_err_vs_closed={xerr:.2e} batch_vs_single={bvs:.2e}")
print(f"t_batch={t_batch:.4f}s t_single(sum)={t_single:.4f}s")

print("=== independent cvxpy check (instances 0 and N-1) ===")
print(f"  inst0: obj_err={obj_err_cvx0:.2e} x_err={x_err_cvx0:.2e}")
print(f"  instN: obj_err={obj_err_cvxN:.2e} x_err={x_err_cvxN:.2e}")

print(f"max_obj_err_vs_closed_form={max_obj_closed_err:.2e}")
print(f"max_x_err_vs_closed_form={max_closed_err:.2e}")
print(f"max_batch_vs_single_disagreement={max_single_vs_batch_err:.2e}")

all_optimal = all(r.status == "optimal" for r in batch_results)
ok = (
    all_optimal
    and max_obj_closed_err < 1e-6
    and max_closed_err < 1e-5
    and max_single_vs_batch_err < 1e-6
    and obj_err_cvx0 < 1e-6
    and obj_err_cvxN < 1e-6
)
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
