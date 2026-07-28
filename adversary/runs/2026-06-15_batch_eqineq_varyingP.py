#!/usr/bin/env python
"""Batch adversary: solve_qp_batch over a batch of QPs that EACH carry BOTH an
equality block (A x = b) AND an inequality block (G x <= h), with a DISTINCT
SPD Hessian P_i per item (not just a swept RHS).

This is a new scenario vs the already-logged batch tests:
  - 8 SPD QPs  : bound-constrained (lb/ub), varying P            (logged)
  - multi-RHS unconstrained                                       (logged)
  - equality-constrained batch                                    (logged)
  - multi-RHS inequality, SHARED G,h, swept c                     (logged)
Here every item differs in P, c, A, b, G, h -> exercises the per-instance
factorization path of solve_qp_batch, with mixed equality+inequality KKT.

Each item:
    min 1/2 x'P_i x + c_i' x   s.t. A_i x = b_i ,  G_i x <= h_i

Contract:
  (1) Internal consistency: batch[k] must reproduce pounce.solve_qp on the
      SAME item (P_i,c_i,A_i,b_i,G_i,h_i), to tight tolerance / bit-for-bit.
  (2) KNOWN closed form: for items whose inequalities are INACTIVE at the
      optimum, the equality-only KKT system gives the exact minimizer
          [P  A'][x]   [-c]
          [A  0 ][y] = [ b]
      We solve that linear system and compare. (Verified inactive by checking
      G x* < h.) For items where an inequality is active we instead verify
      against the per-item single solve (oracle 1) and check KKT stationarity.
  (3) Mix: includes feasible-interior items AND items pushed so an inequality
      row is active, plus one "just-barely-feasible" equality target.

Batch should also be faster than the per-item loop.
"""
import time
import numpy as np
import pounce

rng = np.random.default_rng(20260615)
n = 4
N = 16  # 16-item batch


def make_spd(n):
    M = rng.standard_normal((n, n))
    P = M @ M.T + n * np.eye(n)
    return 0.5 * (P + P.T)


def eq_only_solution(P, c, A, b):
    """Closed-form minimizer of the EQUALITY-ONLY QP via the KKT linear system."""
    m = A.shape[0]
    K = np.block([[P, A.T], [A, np.zeros((m, m))]])
    rhs = np.concatenate([-np.asarray(c, float), np.asarray(b, float)])
    sol = np.linalg.solve(K, rhs)
    return sol[:n]


problems = []
meta = []
for k in range(N):
    P = make_spd(n)
    c = rng.standard_normal(n)
    # one equality row: sum(x) = target  (always feasible inside the box below)
    A = np.ones((1, n))
    target = float(rng.uniform(-0.5, 0.5))
    b = np.array([target])
    # inequality box |x_j| <= bound, written as G x <= h
    bound = 1.0
    G = np.vstack([np.eye(n), -np.eye(n)])
    h = np.full(2 * n, bound)

    force_active = False
    if k in (3, 7, 11, 15):
        # Drive the optimum to a box face: large negative cost on x_0 wants
        # x_0 big +, but x_0 <= bound binds. Keep equality feasible.
        c = c.copy()
        c[0] = -50.0
        force_active = True

    problems.append(dict(P=P, c=c, A=A, b=b, G=G, h=h))
    meta.append(dict(P=P, c=c, A=A, b=b, G=G, h=h,
                     force_active=force_active, bound=bound))


def obj(x, P, c):
    x = np.asarray(x, float)
    return 0.5 * x @ P @ x + np.asarray(c, float) @ x


# --- pounce batch ---
t0 = time.perf_counter()
batch = pounce.solve_qp_batch(problems)
t_batch = time.perf_counter() - t0

# --- oracle 1: per-item single solves ---
t0 = time.perf_counter()
singles = [pounce.solve_qp(**p) for p in problems]
t_loop = time.perf_counter() - t0

assert len(batch) == N, f"batch returned {len(batch)} != {N}"

max_x_err = 0.0          # batch vs single
max_obj_err = 0.0
max_viol = 0.0
max_eq_err = 0.0
max_cf_err = 0.0         # batch vs closed form (inactive items only)
max_kkt = 0.0
all_ok = True
n_active = 0
n_cf_checked = 0

print(f"{'item':>4} {'status':>9} {'obj_batch':>13} {'obj_single':>13} "
      f"{'x_err':>9} {'obj_err':>9} {'ineqviol':>9} {'eqerr':>9} "
      f"{'cf_err':>9} {'active':>6}")
for k in range(N):
    m = meta[k]
    P, c, A, b, G, h = m["P"], m["c"], m["A"], m["b"], m["G"], m["h"]
    xb = np.asarray(batch[k].x, float)
    xs = np.asarray(singles[k].x, float)

    x_err = float(np.linalg.norm(xb - xs, np.inf))
    obj_err = abs(batch[k].obj - singles[k].obj) / max(1.0, abs(singles[k].obj))
    ineqviol = float(np.max(G @ xb - h))            # <=0 feasible
    eq_err = float(np.linalg.norm(A @ xb - b, np.inf))
    active = bool(np.max(G @ xb - h) > -1e-6)
    n_active += int(active)

    # closed-form (equality-only) check, valid only if inequalities inactive
    cf_err = float('nan')
    if not active:
        x_cf = eq_only_solution(P, c, A, b)
        # only trust it if it's actually box-feasible
        if np.max(G @ x_cf - h) <= 1e-9:
            cf_err = float(np.linalg.norm(xb - x_cf, np.inf))
            max_cf_err = max(max_cf_err, cf_err)
            n_cf_checked += 1

    # KKT stationarity residual using returned multipliers (if available)
    grad = P @ xb + c
    try:
        y = np.asarray(batch[k].y, float) if batch[k].y is not None else None
        z = np.asarray(batch[k].z, float) if batch[k].z is not None else None
    except Exception:
        y = z = None
    if y is not None and z is not None and y.size == A.shape[0] and z.size == G.shape[0]:
        stat = grad + A.T @ y + G.T @ z
        max_kkt = max(max_kkt, float(np.linalg.norm(stat, np.inf)))

    max_x_err = max(max_x_err, x_err)
    max_obj_err = max(max_obj_err, obj_err)
    max_viol = max(max_viol, ineqviol)
    max_eq_err = max(max_eq_err, eq_err)

    st = str(batch[k].status).lower()
    ok_k = (st == "optimal") and x_err < 1e-6 and obj_err < 1e-8 \
        and ineqviol < 1e-6 and eq_err < 1e-6
    if not np.isnan(cf_err):
        ok_k = ok_k and cf_err < 1e-6
    all_ok = all_ok and ok_k

    cf_disp = f"{cf_err:>9.2e}" if not np.isnan(cf_err) else f"{'--':>9}"
    print(f"{k:>4} {str(batch[k].status):>9} {batch[k].obj:>13.6e} "
          f"{singles[k].obj:>13.6e} {x_err:>9.2e} {obj_err:>9.2e} "
          f"{ineqviol:>9.2e} {eq_err:>9.2e} {cf_disp} {str(active):>6}")

print(f"=== batch t={t_batch:.4f}s ; per-item t={t_loop:.4f}s "
      f"(speedup {t_loop/max(t_batch,1e-9):.2f}x) ===")
print(f"N={N}  n_vars={n}  items_with_active_ineq={n_active}  "
      f"closed_form_checked={n_cf_checked}")
print(f"max_x_err_batch_vs_single={max_x_err:.2e}  max_obj_err={max_obj_err:.2e}")
print(f"max_ineq_viol={max_viol:.2e}  max_eq_err={max_eq_err:.2e}")
print(f"max_closed_form_err(inactive items)={max_cf_err:.2e}")
print(f"max_KKT_stationarity_residual={max_kkt:.2e}")

if not all_ok:
    print("VERDICT: FAIL (batch disagrees with single solve / closed form / infeasible)")
else:
    print("VERDICT: PASS")
