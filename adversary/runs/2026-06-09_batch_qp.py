"""Adversary cross-check: batched QP solve vs per-item single solve
Family: batch   Class: many independent convex QPs at once
Source: internal-consistency contract — solve_qp_batch must return, item by
  item, exactly what a single solve_qp returns for that problem. Each item is
  also checked against its closed-form unconstrained/bound optimum.
Known optimal: per item (see below).
"""
import time
import numpy as np

rng = np.random.default_rng(0)
N = 8
problems = []
closed_form = []
for k in range(N):
    n = 3
    M = rng.standard_normal((n, n))
    P = M @ M.T + (0.5 + k * 0.1) * np.eye(n)   # SPD
    c = rng.standard_normal(n)
    lb = np.full(n, -1.0)
    ub = np.full(n, 1.0)
    problems.append({"P": P, "c": c, "lb": lb, "ub": ub})
    # unconstrained min is -P^{-1} c; clamp into the box for a reference point
    x_unc = -np.linalg.solve(P, c)
    closed_form.append(np.clip(x_unc, lb, ub))

import pounce

# --- batch ---
t0 = time.perf_counter()
batch = pounce.solve_qp_batch(problems)
t_batch = time.perf_counter() - t0

# --- per-item single solves (the oracle) ---
t0 = time.perf_counter()
singles = [pounce.solve_qp(**p) for p in problems]
t_single = time.perf_counter() - t0

max_x_err = 0.0
max_obj_err = 0.0
all_ok = True
for k in range(N):
    xb = np.asarray(batch[k].x)
    xs = np.asarray(singles[k].x)
    x_err = float(np.linalg.norm(xb - xs, np.inf))
    obj_err = abs(batch[k].obj - singles[k].obj) / max(1.0, abs(singles[k].obj))
    max_x_err = max(max_x_err, x_err)
    max_obj_err = max(max_obj_err, obj_err)
    ok_k = (batch[k].status == "optimal") and x_err < 1e-6 and obj_err < 1e-8
    all_ok = all_ok and ok_k
    print(f"item {k}: status={batch[k].status} obj_batch={batch[k].obj:.6e} "
          f"obj_single={singles[k].obj:.6e} x_err={x_err:.2e} obj_err={obj_err:.2e}")

print(f"=== batch t={t_batch:.4f}s ; per-item t={t_single:.4f}s "
      f"(speedup {t_single/max(t_batch,1e-9):.2f}x) ===")
print(f"max_x_err_batch_vs_single={max_x_err:.2e} max_obj_err={max_obj_err:.2e}")
print("VERDICT: PASS" if all_ok else "VERDICT: FAIL (batch disagrees with single solve)")
