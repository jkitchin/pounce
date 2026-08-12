"""Adversary cross-check: batched bound-constrained ridge regression vs an
oracle INDEPENDENT of pounce's own QP path (scipy.optimize.lsq_linear)
Family: batch   Class: heterogeneous-shape solve_qp_batch vs per-item
              solve_qp (per the family table's sanctioned oracle) PLUS a
              genuinely independent third-party check on a representative
              item -- distinct from prior batch probes (fixed-geometry
              varying-objective, multi-RHS box QP, mixed-status failure
              attribution, mixed-conditioning, status-attribution under
              adversarial ordering, 50-instance large-scale, LQR/Riccati,
              warm-starts, NNLS, KKT-multiplier-parity): none of those
              cross-checked against a solver outside the pounce QP family
              at all (the table sanctions solve_qp as the batch oracle, but
              a bug shared by solve_qp and solve_qp_batch would be invisible
              to that comparison alone).
Source: bound-constrained ridge / Tikhonov-regularized least squares is a
        standard convex QP: min 0.5*||Ax-b||_2^2 + 0.5*lambda*||x||_2^2 s.t.
        lb <= x <= ub. This is EXACTLY equivalent (same stationarity
        conditions) to the *augmented*, *unregularized*, bound-constrained
        least-squares problem
            min 0.5*||[A; sqrt(lambda)*I] x - [b; 0]||_2^2  s.t. lb<=x<=ub
        which scipy.optimize.lsq_linear solves directly via a trust-region
        reflective algorithm that shares no code with pounce's QP IPM --
        a solver-independent, formulation-independent oracle.
Known optimal: none published; cross-checked against solve_qp per item
        (batch-vs-single parity) AND scipy.optimize.lsq_linear on one
        representative item (third-party numerical oracle).
"""
import time

import numpy as np
from scipy.optimize import lsq_linear

rng = np.random.default_rng(20260812)

problems = []
raw = []
for k in range(10):
    m = int(rng.integers(4, 9))
    n = int(rng.integers(3, 7))
    A_mat = rng.normal(size=(m, n))
    b_vec = rng.normal(size=m)
    lam = float(rng.uniform(0.1, 3.0))
    lb = -2.0 - np.abs(rng.normal(size=n))
    ub = 2.0 + np.abs(rng.normal(size=n))
    P = A_mat.T @ A_mat + lam * np.eye(n)
    c = -(A_mat.T @ b_vec)
    problems.append({"P": P, "c": c, "lb": lb, "ub": ub})
    raw.append((A_mat, b_vec, lam, lb, ub))

from pounce import solve_qp, solve_qp_batch

t0 = time.perf_counter()
batch_results = solve_qp_batch(problems)
t_batch = time.perf_counter() - t0

t0 = time.perf_counter()
single_results = [solve_qp(**p) for p in problems]
t_single = time.perf_counter() - t0


def rel(u, v):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    denom = max(1.0, float(np.linalg.norm(v, np.inf)))
    return float(np.linalg.norm(u - v, np.inf) / denom)


max_x_err = max_obj_err = 0.0
all_optimal = True
for i, (rb, rs) in enumerate(zip(batch_results, single_results)):
    if rb.status != "optimal" or rs.status != "optimal":
        all_optimal = False
        print(f"item {i}: status batch={rb.status} single={rs.status}")
        continue
    max_x_err = max(max_x_err, rel(rb.x, rs.x))
    max_obj_err = max(max_obj_err, abs(rb.obj - rs.obj) / max(1.0, abs(rs.obj)))

print("=== pounce solve_qp_batch vs per-item solve_qp (10 heterogeneous ridge QPs) ===")
print(f"t_batch={t_batch:.4f}s t_single_sum={t_single:.4f}s all_optimal={all_optimal}")
print(f"max_x_err(batch vs single)={max_x_err:.2e} max_obj_err={max_obj_err:.2e}")

# --- independent third-party oracle on every item: scipy.optimize.lsq_linear
# via the augmented, unregularized formulation ---
max_x_err_lsq = 0.0
max_obj_err_lsq = 0.0
for i, (A_mat, b_vec, lam, lb, ub) in enumerate(raw):
    m, n = A_mat.shape
    A_aug = np.vstack([A_mat, np.sqrt(lam) * np.eye(n)])
    b_aug = np.concatenate([b_vec, np.zeros(n)])
    res = lsq_linear(A_aug, b_aug, bounds=(lb, ub), tol=1e-12, method="bvls")
    x_lsq = res.x
    # True ridge objective 0.5||Ax-b||^2 + 0.5*lam*||x||^2 expands to
    # 0.5 x'(A'A+lam*I)x - (A'b)'x + 0.5||b||^2 -- i.e. the QP form (P,c)
    # passed to solve_qp/solve_qp_batch (0.5 x'Px + c'x) omits the constant
    # 0.5||b||^2 term. Add it back so both objectives are on the same
    # footing (caught by the FIRST run of this script reporting a spurious
    # ~3.3 objective gap while x matched to 1e-10 -- a formulation bug in
    # this script's oracle comparison, not a pounce defect).
    obj_lsq_qpform = (
        0.5 * float(np.sum((A_mat @ x_lsq - b_vec) ** 2))
        + 0.5 * lam * float(np.sum(x_lsq ** 2))
        - 0.5 * float(np.sum(b_vec ** 2))
    )
    rb = batch_results[i]
    if rb.status != "optimal":
        continue
    max_x_err_lsq = max(max_x_err_lsq, rel(rb.x, x_lsq))
    max_obj_err_lsq = max(max_obj_err_lsq, abs(rb.obj - obj_lsq_qpform) / max(1.0, abs(obj_lsq_qpform)))

print("=== oracle (scipy.optimize.lsq_linear, augmented formulation, all 10 items) ===")
print(f"max_x_err_vs_batch={max_x_err_lsq:.2e} max_obj_err_vs_batch={max_obj_err_lsq:.2e}")

ok = all_optimal and max(max_x_err, max_obj_err) < 1e-6 and max(max_x_err_lsq, max_obj_err_lsq) < 1e-4
print("VERDICT: PASS" if ok else "VERDICT: FAIL (batch/single or batch/lsq_linear mismatch)")
