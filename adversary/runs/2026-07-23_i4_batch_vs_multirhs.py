"""Adversary i4: THREE-WAY agreement solve_qp_batch vs solve_qp_multi_rhs vs
per-item solve_qp on the SAME shared-structure inequality QP with swept c.
Family: batch   Class: cross-check of the two batch entry points against each other.

No logged test compares solve_qp_batch and solve_qp_multi_rhs directly. Both are
supposed to reproduce the per-item single solve; here we build the identical set
of problems (shared SPD P, shared inequality G x <= h, one equality A x = b,
varying only c) THREE ways and require all three to agree:
    (1) solve_qp_multi_rhs(P,A,b,G,h, cs=[...])
    (2) solve_qp_batch([{P,c_k,A,b,G,h}, ...])
    (3) [solve_qp(P,c_k,A,b,G,h) for k]
For items with all inequalities inactive, the equality-only KKT linear system
gives an independent closed form; we check those too.
"""
import time
import numpy as np
import pounce

rng = np.random.default_rng(70723)
n = 4
M = rng.standard_normal((n, n)); P = M @ M.T + n * np.eye(n); P = 0.5 * (P + P.T)
A = np.ones((1, n)); b = np.array([0.25])
G = np.vstack([np.eye(n), -np.eye(n)]); h = np.full(2 * n, 1.0)   # box |x|<=1
N = 10
cs = np.array([rng.standard_normal(n) for _ in range(N)])
# push a couple of items to a box face
cs[2, 0] = -40.0
cs[7, 1] = 40.0

def eq_only(c):
    K = np.block([[P, A.T], [A, np.zeros((1, 1))]])
    rhs = np.concatenate([-c, b])
    return np.linalg.solve(K, rhs)[:n]

# (1) multi_rhs
t0 = time.perf_counter()
multi = pounce.solve_qp_multi_rhs(P=P, A=A, b=b, G=G, h=h, cs=cs)
t_multi = time.perf_counter() - t0
# (2) batch
probs = [dict(P=P, c=cs[k], A=A, b=b, G=G, h=h) for k in range(N)]
t0 = time.perf_counter()
batch = pounce.solve_qp_batch(probs)
t_batch = time.perf_counter() - t0
# (3) singles
t0 = time.perf_counter()
singles = [pounce.solve_qp(**probs[k]) for k in range(N)]
t_loop = time.perf_counter() - t0

assert len(multi) == N and len(batch) == N and len(singles) == N

max_mb = max_ms = max_bs = 0.0       # pairwise x-inf errors
max_obj = 0.0
max_viol = max_eq = 0.0
max_cf = 0.0; n_cf = 0
all_ok = True
print(f"{'k':>3} {'st_multi':>9} {'obj_multi':>13} {'|m-b|':>9} {'|m-s|':>9} {'|b-s|':>9} {'cf':>9} {'active':>6}")
for k in range(N):
    xm = np.asarray(multi[k].x, float)
    xb = np.asarray(batch[k].x, float)
    xs = np.asarray(singles[k].x, float)
    e_mb = float(np.linalg.norm(xm - xb, np.inf))
    e_ms = float(np.linalg.norm(xm - xs, np.inf))
    e_bs = float(np.linalg.norm(xb - xs, np.inf))
    o = (abs(multi[k].obj - batch[k].obj) + abs(multi[k].obj - singles[k].obj)) \
        / max(1.0, abs(singles[k].obj))
    viol = float(np.max(G @ xm - h)); eq = float(np.linalg.norm(A @ xm - b, np.inf))
    active = viol > -1e-6
    cf_err = float('nan')
    if not active:
        xcf = eq_only(cs[k])
        if np.max(G @ xcf - h) <= 1e-9:
            cf_err = float(np.linalg.norm(xm - xcf, np.inf)); max_cf = max(max_cf, cf_err); n_cf += 1
    max_mb = max(max_mb, e_mb); max_ms = max(max_ms, e_ms); max_bs = max(max_bs, e_bs)
    max_obj = max(max_obj, o); max_viol = max(max_viol, viol); max_eq = max(max_eq, eq)
    st = str(multi[k].status).lower()
    ok_k = (st == "optimal" and e_mb < 1e-7 and e_ms < 1e-6 and e_bs < 1e-6
            and viol < 1e-6 and eq < 1e-6)
    if not np.isnan(cf_err): ok_k = ok_k and cf_err < 1e-6
    all_ok = all_ok and ok_k
    cf_disp = f"{cf_err:>9.2e}" if not np.isnan(cf_err) else f"{'--':>9}"
    print(f"{k:>3} {str(multi[k].status):>9} {multi[k].obj:>13.6e} {e_mb:>9.2e} {e_ms:>9.2e} "
          f"{e_bs:>9.2e} {cf_disp} {str(active):>6}")

print(f"=== multi t={t_multi:.4f}s batch t={t_batch:.4f}s per-item t={t_loop:.4f}s ===")
print(f"N={N} n={n} closed_form_checked={n_cf}")
print(f"max_x_err multi_vs_batch={max_mb:.2e} multi_vs_single={max_ms:.2e} batch_vs_single={max_bs:.2e}")
print(f"max_obj_err={max_obj:.2e} max_ineq_viol={max_viol:.2e} max_eq_err={max_eq:.2e} max_cf_err={max_cf:.2e}")

print("VERDICT: PASS" if all_ok else
      f"VERDICT: FAIL (m-b={max_mb:.2e}, m-s={max_ms:.2e}, b-s={max_bs:.2e}, cf={max_cf:.2e})")
