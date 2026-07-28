#!/usr/bin/env python
"""Batch adversary (a): solve_qp_multi_rhs, shared SPD P, many c vectors,
UNCONSTRAINED.  Closed-form per-item optimum: x* = -P^{-1} c_i.

We sweep a family of 12 distinct linear objectives against one fixed P.
Oracle 1: closed form x* = -P^{-1} c_i.
Oracle 2: standalone pounce.solve_qp per item (batch item must match its
own single solve -- catches SOLVER_BUG in the batched path).
"""
import time
import numpy as np
import pounce

rng = np.random.default_rng(20260609)
n = 6
# Fixed SPD P (well-conditioned).
M = rng.standard_normal((n, n))
P = M @ M.T + n * np.eye(n)
P = 0.5 * (P + P.T)
Pinv = np.linalg.inv(P)

# 12 distinct cost vectors.
cs = [rng.standard_normal(n).tolist() for _ in range(12)]

# Closed-form optima.
x_star = [(-Pinv @ np.asarray(c)) for c in cs]

# --- pounce multi-RHS batch ---
t0 = time.perf_counter()
batch = pounce.solve_qp_multi_rhs(P=P.tolist(), c=cs[0], cs=cs)
t_batch = time.perf_counter() - t0

# --- oracle: loop of standalone single solves ---
t0 = time.perf_counter()
singles = [pounce.solve_qp(P=P.tolist(), c=c) for c in cs]
t_loop = time.perf_counter() - t0

assert len(batch) == len(cs), f"batch returned {len(batch)} != {len(cs)}"

max_rel_cf = 0.0      # vs closed form
max_rel_single = 0.0  # vs standalone solve
for i, (res, single, xs) in enumerate(zip(batch, singles, x_star)):
    xb = np.asarray(res.x)
    denom = max(1.0, np.linalg.norm(xs))
    rel_cf = np.linalg.norm(xb - xs) / denom
    rel_s = np.linalg.norm(xb - np.asarray(single.x)) / max(1.0, np.linalg.norm(single.x))
    max_rel_cf = max(max_rel_cf, rel_cf)
    max_rel_single = max(max_rel_single, rel_s)
    if res.status != "Optimal" and res.status.lower() != "optimal":
        print(f"item {i}: status={res.status}")

print(f"batch size           : {len(cs)}  (n={n} each)")
print(f"pounce status (item0): {batch[0].status}")
print(f"max rel-err vs closed-form  : {max_rel_cf:.3e}")
print(f"max rel-err vs standalone   : {max_rel_single:.3e}")
print(f"pounce_time (batch)  : {t_batch:.4f}s")
print(f"oracle_time (loop)   : {t_loop:.4f}s")

tol = 1e-6
ok = max_rel_cf < tol and max_rel_single < tol
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
