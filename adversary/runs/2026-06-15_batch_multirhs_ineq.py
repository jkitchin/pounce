#!/usr/bin/env python
"""Batch adversary: solve_qp_multi_rhs with a SHARED SPD P and SHARED inequality
constraints (G x <= h), swept over many distinct linear cost vectors c_i.

This exercises a DIFFERENT batch scenario than the already-tested ones
(bound-constrained batch, equality-constrained batch, unconstrained multi-rhs):
here the shared structure is a set of *general inequality* rows G,h, and we
push at least one cost vector hard enough that its constraints are ACTIVE at
the optimum.

Internal-consistency contract: each batched item must reproduce, to numerical
tolerance, the result of a standalone pounce.solve_qp on the SAME problem
(P, c_i, G, h). External check: cvxpy on a couple of items, including the one
whose constraints are active.

  min 1/2 x'P x + c_i' x   s.t. G x <= h
"""
import time
import numpy as np
import pounce

rng = np.random.default_rng(20260615)
n = 5

# Shared SPD P (well-conditioned).
M = rng.standard_normal((n, n))
P = M @ M.T + n * np.eye(n)
P = 0.5 * (P + P.T)

# Shared inequality block: a box-ish polytope plus a couple of slanted rows.
#   x_j <=  1, -x_j <= 1  (i.e. |x_j| <= 1), and two random half-spaces.
G_rows = []
h_rows = []
for j in range(n):
    e = np.zeros(n); e[j] = 1.0
    G_rows.append(e.copy()); h_rows.append(1.0)
    G_rows.append(-e.copy()); h_rows.append(1.0)
# two slanted rows that bite for large-magnitude costs
G_rows.append(np.ones(n)); h_rows.append(1.5)
G_rows.append(-np.ones(n)); h_rows.append(1.5)
G = np.array(G_rows)
h = np.array(h_rows)

# Cost vectors: most modest, but force a few to drive the optimum to the
# polytope boundary (constraints active).
cs = []
for k in range(6):
    cs.append(rng.standard_normal(n).tolist())
# strongly negative -> wants large +x -> hits sum(x)<=1.5 and |x_j|<=1
cs.append((-8.0 * np.ones(n)).tolist())
# strongly positive -> wants large -x -> hits -sum(x)<=1.5
cs.append((+8.0 * np.ones(n)).tolist())
N = len(cs)

# --- pounce multi-RHS batch (shared P, G, h) ---
t0 = time.perf_counter()
batch = pounce.solve_qp_multi_rhs(P=P, G=G, h=h, c=cs[0], cs=cs)
t_batch = time.perf_counter() - t0

# --- oracle 1: per-rhs standalone single solves ---
t0 = time.perf_counter()
singles = [pounce.solve_qp(P=P, c=c, G=G, h=h) for c in cs]
t_loop = time.perf_counter() - t0

assert len(batch) == N, f"batch returned {len(batch)} != {N}"


def obj(x, c):
    x = np.asarray(x, float)
    return 0.5 * x @ P @ x + np.asarray(c, float) @ x


# --- oracle 2: cvxpy on the two forced-active items (indices 6,7) ---
import cvxpy as cp
cvx_x = {}
for idx in (6, 7):
    xv = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P))
                                  + np.asarray(cs[idx]) @ xv),
                      [G @ xv <= h])
    prob.solve(solver=cp.CLARABEL)
    cvx_x[idx] = np.asarray(xv.value)

max_x_err = 0.0
max_obj_err = 0.0
max_viol = 0.0
all_ok = True
n_active = 0
print(f"{'item':>4} {'status':>10} {'obj_batch':>13} {'obj_single':>13} "
      f"{'x_err':>9} {'obj_err':>9} {'maxviol':>9} {'active':>6}")
for k in range(N):
    xb = np.asarray(batch[k].x, float)
    xs = np.asarray(singles[k].x, float)
    x_err = float(np.linalg.norm(xb - xs, np.inf))
    obj_err = abs(batch[k].obj - singles[k].obj) / max(1.0, abs(singles[k].obj))
    viol = float(np.max(G @ xb - h))            # <=0 means feasible
    active = bool(np.max(G @ xb - h) > -1e-5)
    n_active += int(active)
    max_x_err = max(max_x_err, x_err)
    max_obj_err = max(max_obj_err, obj_err)
    max_viol = max(max_viol, viol)
    st = str(batch[k].status).lower()
    ok_k = (st == "optimal") and x_err < 1e-6 and obj_err < 1e-8 and viol < 1e-6
    all_ok = all_ok and ok_k
    print(f"{k:>4} {str(batch[k].status):>10} {batch[k].obj:>13.6e} "
          f"{singles[k].obj:>13.6e} {x_err:>9.2e} {obj_err:>9.2e} "
          f"{viol:>9.2e} {str(active):>6}")

# cvxpy cross-check on forced-active items
cvx_ok = True
for idx in (6, 7):
    xb = np.asarray(batch[idx].x, float)
    cvx_err = float(np.linalg.norm(xb - cvx_x[idx], np.inf))
    cvx_obj_err = abs(obj(xb, cs[idx]) - obj(cvx_x[idx], cs[idx]))
    print(f"cvxpy item {idx}: x_err={cvx_err:.2e} obj_err={cvx_obj_err:.2e}")
    cvx_ok = cvx_ok and (cvx_err < 1e-4) and (cvx_obj_err < 1e-6)

print(f"=== batch t={t_batch:.4f}s ; per-item t={t_loop:.4f}s "
      f"(speedup {t_loop/max(t_batch,1e-9):.2f}x) ===")
print(f"N={N}  items_with_active_constraints={n_active}")
print(f"max_x_err_batch_vs_single={max_x_err:.2e} max_obj_err={max_obj_err:.2e} "
      f"max_constraint_viol={max_viol:.2e}")

if not all_ok:
    print("VERDICT: FAIL (batch disagrees with single solve / infeasible)")
elif not cvx_ok:
    print("VERDICT: FAIL (batch disagrees with cvxpy on active-constraint item)")
else:
    print("VERDICT: PASS")
