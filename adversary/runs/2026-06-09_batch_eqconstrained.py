#!/usr/bin/env python
"""Batch adversary (b): solve_qp_batch of EQUALITY-constrained QPs, each with a
DIFFERENT P, c, and constraint A x = b.

Per item: min 1/2 x'P x + c'x  s.t. A x = b, with P SPD, A full row-rank.
KKT closed form (equality-only, convex):
    [P  A'] [x]   [-c]
    [A  0 ] [l] = [ b]
solving the (n+m) linear system gives the unique optimum x*.

Oracle 1: KKT linear-system solve (closed form).
Oracle 2: standalone pounce.solve_qp per item (batch item vs its own solve).
"""
import time
import numpy as np
import pounce

rng = np.random.default_rng(424242)
N = 10  # batch size (distinct problems)

problems = []
x_star = []
metas = []
for k in range(N):
    n = 4 + (k % 3)        # vary dims 4..6
    m = 1 + (k % 2)        # 1 or 2 equality constraints
    M = rng.standard_normal((n, n))
    P = M @ M.T + n * np.eye(n)
    P = 0.5 * (P + P.T)
    c = rng.standard_normal(n)
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)

    # KKT closed form.
    KKT = np.block([[P, A.T], [A, np.zeros((m, m))]])
    rhs = np.concatenate([-c, b])
    sol = np.linalg.solve(KKT, rhs)
    xs = sol[:n]
    x_star.append(xs)
    metas.append((n, m))
    problems.append(dict(P=P.tolist(), c=c.tolist(), A=A.tolist(), b=b.tolist()))

# --- pounce batch ---
t0 = time.perf_counter()
batch = pounce.solve_qp_batch(problems)
t_batch = time.perf_counter() - t0

# --- oracle: loop of standalone single solves ---
t0 = time.perf_counter()
singles = [pounce.solve_qp(**p) for p in problems]
t_loop = time.perf_counter() - t0

assert len(batch) == N, f"batch returned {len(batch)} != {N}"

max_rel_cf = 0.0
max_rel_single = 0.0
max_feas = 0.0  # equality-constraint residual ||A x - b||
for i, (res, single, xs, p) in enumerate(zip(batch, singles, x_star, problems)):
    xb = np.asarray(res.x)
    A = np.asarray(p["A"]); b = np.asarray(p["b"])
    rel_cf = np.linalg.norm(xb - xs) / max(1.0, np.linalg.norm(xs))
    rel_s = np.linalg.norm(xb - np.asarray(single.x)) / max(1.0, np.linalg.norm(single.x))
    feas = np.linalg.norm(A @ xb - b)
    max_rel_cf = max(max_rel_cf, rel_cf)
    max_rel_single = max(max_rel_single, rel_s)
    max_feas = max(max_feas, feas)
    if res.status.lower() != "optimal":
        print(f"item {i}: status={res.status}")

print(f"batch size           : {N}  (dims n,m = {metas})")
print(f"pounce status (item0): {batch[0].status}")
print(f"max rel-err vs KKT closed-form : {max_rel_cf:.3e}")
print(f"max rel-err vs standalone       : {max_rel_single:.3e}")
print(f"max equality feas residual      : {max_feas:.3e}")
print(f"pounce_time (batch)  : {t_batch:.4f}s")
print(f"oracle_time (loop)   : {t_loop:.4f}s")

tol = 1e-6
ok = max_rel_cf < tol and max_rel_single < tol and max_feas < 1e-6
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
