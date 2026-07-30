"""Adversary cross-check: large-batch box-constrained diagonal QPs
Family: batch   Class: solve_qp_batch faithfulness at scale (50 instances) with
BOX bounds (no equality) -- new class for batch; prior batch runs used a
budget-equality Markowitz batch (8 instances), a shared-G/h multi-RHS batch, a
varying-P eq+ineq batch, a mixed-conditioning battery, and a status-attribution
battery -- none stressed batch faithfulness at this instance COUNT (50) with
per-instance independent box bounds and no coupling constraint at all.

Problem, per instance k = 0..49 (N=6 vars each):
    minimize   sum_i 0.5 q_i x_i^2 - c_i x_i
    subject to l_i <= x_i <= u_i

Fully separable (diagonal Hessian, box only, no A/G): the closed-form optimum
is x_i = clip(c_i/q_i, l_i, u_i) per coordinate, independent of every other
instance in the batch -- a pure per-coordinate clip, no linear algebra
whatsoever. This maximizes the chance of catching any batch-indexing /
cross-instance-contamination bug (issue class: one instance's data leaking
into another's solve) since with 50 independent instances a index-off-by-one
would very likely show up as at least one instance's objective/x disagreeing
with its own closed form while matching a NEIGHBOR instance's data instead.

KNOWN_OPTIMAL: per-instance closed form (exact, no solver of any kind).
"""
import time
import numpy as np

rng = np.random.default_rng(20260730)
N_PROBLEMS = 50
N = 6

problems = []
closed_form = []
for k in range(N_PROBLEMS):
    q = rng.uniform(0.3, 6.0, size=N)
    c = rng.uniform(-10.0, 10.0, size=N)
    L = rng.uniform(-4.0, -0.5, size=N)
    U = L + rng.uniform(1.0, 6.0, size=N)

    x_star = np.clip(c / q, L, U)
    obj_star = float(np.sum(0.5 * q * x_star ** 2 - c * x_star))

    problems.append(dict(P=np.diag(q), c=-c, lb=L, ub=U))
    closed_form.append((x_star, obj_star))

from pounce import solve_qp_batch, solve_qp

t0 = time.perf_counter()
batch_results = solve_qp_batch(problems)
t_batch = time.perf_counter() - t0

t0 = time.perf_counter()
single_results = [solve_qp(**p) for p in problems]
t_single = time.perf_counter() - t0

max_closed_x_err = 0.0
max_closed_obj_err = 0.0
max_batch_vs_single = 0.0
worst_idx = -1
n_optimal = 0
for k, (p, (x_star, obj_star), rb, rs) in enumerate(
    zip(problems, closed_form, batch_results, single_results)
):
    xb = np.asarray(rb.x)
    xs = np.asarray(rs.x)
    x_err = float(np.linalg.norm(xb - x_star, np.inf))
    obj_err = abs(rb.obj - obj_star) / max(1.0, abs(obj_star))
    bvs = float(np.linalg.norm(xb - xs, np.inf))
    if x_err > max_closed_x_err:
        max_closed_x_err, worst_idx = x_err, k
    max_closed_obj_err = max(max_closed_obj_err, obj_err)
    max_batch_vs_single = max(max_batch_vs_single, bvs)
    if rb.status == "optimal":
        n_optimal += 1

# Independent oracle: cvxpy on 5 spot-checked instances spread across the batch
import cvxpy as cp

spot_idx = [0, 12, 25, 37, 49]
max_cvx_err = 0.0
for idx in spot_idx:
    p = problems[idx]
    n = p["P"].shape[0]
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, p["P"]) + p["c"] @ x),
        [x >= p["lb"], x <= p["ub"]],
    )
    prob.solve(solver=cp.CLARABEL)
    err = abs(batch_results[idx].obj - prob.value) / max(1.0, abs(prob.value))
    max_cvx_err = max(max_cvx_err, err)

print(f"=== batch of {N_PROBLEMS} independent box-constrained diagonal QPs ===")
print(f"n_optimal={n_optimal}/{N_PROBLEMS}  t_batch={t_batch:.4f}s  t_single(sum)={t_single:.4f}s")
print(f"max_obj_err_vs_closed_form   = {max_closed_obj_err:.2e}")
print(f"max_x_err_vs_closed_form     = {max_closed_x_err:.2e}  (worst instance idx={worst_idx})")
print(f"max_batch_vs_single_disagree = {max_batch_vs_single:.2e}")
print(f"max_obj_err_vs_cvxpy (5 spot-checks) = {max_cvx_err:.2e}")

TOL = 1e-6
ok = (
    n_optimal == N_PROBLEMS
    and max_closed_obj_err < TOL
    and max_closed_x_err < 1e-5
    and max_batch_vs_single < 1e-6
    and max_cvx_err < TOL
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (n_optimal={n_optimal}/{N_PROBLEMS}, max_x_err={max_closed_x_err:.2e})")
