"""Adversary i4: solve_qp_multi_rhs with BOX bounds and a DIAGONAL Hessian,
where the active bound set CHANGES across right-hand sides.
Family: batch   Class: multi-RHS box-constrained QP vs per-item single + closed form.

Shared P=diag, lb, ub; swept linear term c across many RHS. For a diagonal QP
with only box bounds the minimizer is separable and EXACT in closed form:
    x_i = clip(-c_i / P_ii, lb_i, ub_i).
DISTINCT from logged multi-RHS tests (unconstrained; shared-G,h inequality):
this one uses variable BOUNDS (lb/ub) and is deliberately swept so DIFFERENT
bounds bind for different RHS (active set varies item to item).

Oracles: (1) per-item pounce.solve_qp with the same bounds; (2) separable
closed-form box projection. Also checks the active bound set actually varies.
"""
import time
import numpy as np
import pounce

rng = np.random.default_rng(20260723)
n = 5
Pd = np.array([1.0, 2.0, 3.0, 4.0, 5.0]); P = np.diag(Pd)
lb = np.full(n, -1.0); ub = np.full(n, 1.0)

# Sweep c so the unconstrained x_i = -c_i/P_ii lands inside/below/above the box
N = 12
cs = []
for k in range(N):
    cs.append(rng.uniform(-8.0, 8.0, size=n))
cs = np.array(cs)

def box_closed_form(c):
    return np.clip(-c / Pd, lb, ub)

# --- pounce multi-RHS ---
t0 = time.perf_counter()
multi = pounce.solve_qp_multi_rhs(P=P, lb=lb, ub=ub, cs=cs)
t_multi = time.perf_counter() - t0
assert len(multi) == N, f"multi returned {len(multi)}"

# --- oracle 1: per-item single ---
t0 = time.perf_counter()
singles = [pounce.solve_qp(P=P, c=cs[k], lb=lb, ub=ub) for k in range(N)]
t_loop = time.perf_counter() - t0

def obj(x, c): return float(0.5 * x @ P @ x + c @ x)

max_x_ms = max_obj_ms = 0.0        # multi vs single
max_x_cf = max_obj_cf = 0.0        # multi vs closed form
max_viol = 0.0
active_patterns = set()
all_ok = True
print(f"{'k':>3} {'status':>10} {'obj_multi':>13} {'x_err_single':>12} {'x_err_cf':>10} {'nbind':>5}")
for k in range(N):
    xm = np.asarray(multi[k].x, float)
    xs = np.asarray(singles[k].x, float)
    xcf = box_closed_form(cs[k])
    x_ms = float(np.linalg.norm(xm - xs, np.inf))
    x_cf = float(np.linalg.norm(xm - xcf, np.inf))
    o_ms = abs(multi[k].obj - singles[k].obj) / max(1.0, abs(singles[k].obj))
    o_cf = abs(multi[k].obj - obj(xcf, cs[k])) / max(1.0, abs(obj(xcf, cs[k])))
    viol = float(max(np.max(xm - ub), np.max(lb - xm)))
    nbind = int(np.sum((np.abs(xm - lb) < 1e-6) | (np.abs(xm - ub) < 1e-6)))
    active_patterns.add(tuple(((np.abs(xm - lb) < 1e-6).astype(int)
                               - (np.abs(xm - ub) < 1e-6).astype(int)).tolist()))
    max_x_ms = max(max_x_ms, x_ms); max_obj_ms = max(max_obj_ms, o_ms)
    max_x_cf = max(max_x_cf, x_cf); max_obj_cf = max(max_obj_cf, o_cf)
    max_viol = max(max_viol, viol)
    st = str(multi[k].status).lower()
    ok_k = st == "optimal" and x_ms < 1e-6 and x_cf < 1e-6 and viol < 1e-7
    all_ok = all_ok and ok_k
    print(f"{k:>3} {str(multi[k].status):>10} {multi[k].obj:>13.6e} {x_ms:>12.2e} {x_cf:>10.2e} {nbind:>5}")

print(f"=== multi t={t_multi:.4f}s ; per-item t={t_loop:.4f}s (speedup {t_loop/max(t_multi,1e-9):.2f}x) ===")
print(f"N={N} n={n} distinct_active_patterns={len(active_patterns)}")
print(f"max_x_err multi_vs_single={max_x_ms:.2e} multi_vs_closedform={max_x_cf:.2e}")
print(f"max_obj_err vs_single={max_obj_ms:.2e} vs_closedform={max_obj_cf:.2e} max_box_viol={max_viol:.2e}")

ok = all_ok and len(active_patterns) >= 3    # ensure active set really varies
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (x_ms={max_x_ms:.2e}, x_cf={max_x_cf:.2e}, patterns={len(active_patterns)})")
