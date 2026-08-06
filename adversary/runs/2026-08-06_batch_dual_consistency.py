"""Adversary cross-check: solve_qp_batch dual/multiplier consistency
Family: batch   Class: KKT multiplier (y, z, z_lb, z_ub) parity vs single solve
Source: no external oracle needed for this class -- per the family table the
        batch oracle IS the per-item single solve_qp() call. Prior batch
        adversary runs checked x*/obj*/status parity across the batch API;
        this run checks the full KKT multiplier vectors (equality y,
        inequality z, bound z_lb/z_ub), which a batched interior-point
        implementation could plausibly mis-index or fail to unscale
        per-instance even while getting x* right.
Known optimal: none (cross-checked against solve_qp per item, not a
        published reference).
"""
import numpy as np
import time

rng = np.random.default_rng(20260806)

problems = []
for k in range(7):
    n = rng.integers(3, 6)
    m_eq = rng.integers(0, 2)
    m_ineq = rng.integers(1, 3)
    Araw = rng.normal(size=(n, n))
    P = Araw @ Araw.T + 0.5 * np.eye(n)   # SPD
    c = rng.normal(size=n)
    prob = {"P": P, "c": c}
    if m_eq:
        A = rng.normal(size=(m_eq, n))
        x_feas = rng.normal(size=n)
        b = A @ x_feas
        prob["A"] = A
        prob["b"] = b
    G = rng.normal(size=(m_ineq, n))
    x_feas2 = rng.normal(size=n)
    h = G @ x_feas2 + np.abs(rng.normal(size=m_ineq)) + 0.5   # slack margin
    prob["G"] = G
    prob["h"] = h
    lb = -5.0 - np.abs(rng.normal(size=n))
    ub = 5.0 + np.abs(rng.normal(size=n))
    prob["lb"] = lb
    prob["ub"] = ub
    problems.append(prob)

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


max_x_err = max_obj_err = max_y_err = max_z_err = max_zlb_err = max_zub_err = 0.0
all_optimal = True
for i, (rb, rs) in enumerate(zip(batch_results, single_results)):
    if rb.status != "optimal" or rs.status != "optimal":
        all_optimal = False
        print(f"item {i}: status batch={rb.status} single={rs.status}")
        continue
    max_x_err = max(max_x_err, rel(rb.x, rs.x))
    max_obj_err = max(max_obj_err, abs(rb.obj - rs.obj) / max(1.0, abs(rs.obj)))
    max_y_err = max(max_y_err, rel(rb.y, rs.y))
    max_z_err = max(max_z_err, rel(rb.z, rs.z))
    max_zlb_err = max(max_zlb_err, rel(rb.z_lb, rs.z_lb))
    max_zub_err = max(max_zub_err, rel(rb.z_ub, rs.z_ub))

print("=== pounce solve_qp_batch vs per-item solve_qp (7 random QPs) ===")
print(f"t_batch={t_batch:.4f}s t_single_sum={t_single:.4f}s")
print(f"all_optimal={all_optimal}")
print(f"max_x_err={max_x_err:.2e} max_obj_err={max_obj_err:.2e}")
print(f"max_y_err(eq mult)={max_y_err:.2e} max_z_err(ineq mult)={max_z_err:.2e}")
print(f"max_zlb_err={max_zlb_err:.2e} max_zub_err={max_zub_err:.2e}")

# Independent KKT check on the BATCH result itself (not comparing to
# pounce's own single-solve path, but re-deriving stationarity directly):
# stationarity: P x + c - A^T y - G^T z... wait sign convention: for
# solve_qp, KKT is P x + c + A^T y + G^T z - z_lb + z_ub = 0 (verify via
# residuals field instead, which pounce reports independently per-solve).
max_kkt_err = 0.0
for rb in batch_results:
    if rb.residuals is not None:
        max_kkt_err = max(max_kkt_err, rb.residuals.get("kkt_error", 0.0))
print(f"max_reported_kkt_error(batch)={max_kkt_err:.2e}")

ok = all_optimal and max(max_x_err, max_obj_err, max_y_err, max_z_err, max_zlb_err, max_zub_err) < 1e-6
print("VERDICT: PASS" if ok else "VERDICT: FAIL (batch/single multiplier mismatch)")
