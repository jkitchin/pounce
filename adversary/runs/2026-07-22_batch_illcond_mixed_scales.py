"""Adversary cross-check: batch of box-constrained convex QPs with WILDLY
different per-item conditioning and objective scales.

Family: batch      Class: ill-conditioning & bad scaling
Source: internal-consistency contract (batch == per-item single solve_qp)
        + cvxpy/CLARABEL as an independent oracle, per item.
Dimension under test: a batch-level SHARED tolerance / shared scaling that is
right for one item and wrong for another.  Item k has Hessian condition number
10^(2k) (1e0 ... 1e10) and objective magnitude scaled over ~1e12.

Reported metric: the PER-ITEM max relative error (a single bad item is the
finding), not the mean.
"""
import time
import numpy as np

np.set_printoptions(precision=4, suppress=False)

N = 6          # variables per item
K = 6          # items: k = 0..5
SEED = 20260722


def make_item(k, rng):
    """Item k: cond(P) = 10^(2k), objective scaled by 10^(2.4k)."""
    cond = 10.0 ** (2 * k)
    scale = 10.0 ** (2.4 * k)
    # random orthogonal basis
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    eig = np.logspace(0.0, np.log10(cond), N)
    P = scale * (Q @ np.diag(eig) @ Q.T)
    P = 0.5 * (P + P.T)
    # unconstrained minimizer would be x_target; box clips some coords so the
    # active set is nontrivial.
    x_target = rng.standard_normal(N) * 2.0
    c = -P @ x_target
    lb = -np.ones(N)
    ub = np.ones(N)
    return dict(P=P, c=c, lb=lb, ub=ub), x_target, cond, scale


rng = np.random.default_rng(SEED)
items, targets, conds, scales = [], [], [], []
for k in range(K):
    it, xt, cd, sc = make_item(k, rng)
    items.append(it)
    targets.append(xt)
    conds.append(cd)
    scales.append(sc)


def obj_of(P, c, x):
    x = np.asarray(x, float)
    return float(0.5 * x @ P @ x + c @ x)


# ---------------- pounce: BATCH ----------------
from pounce import solve_qp, solve_qp_batch

t0 = time.perf_counter()
batch_res = solve_qp_batch(items)
t_batch = time.perf_counter() - t0

# ---------------- pounce: PER-ITEM single solve ----------------
t0 = time.perf_counter()
single_res = [solve_qp(**it) for it in items]
t_single = time.perf_counter() - t0

# ---------------- oracle: cvxpy per item ----------------
import cvxpy as cp

cvx_x, cvx_obj, t_cvx = [], [], 0.0
for it in items:
    P, c, lb, ub = it["P"], it["c"], it["lb"], it["ub"]
    # normalize the objective for cvxpy's own conditioning; the argmin is
    # invariant to a positive scaling, and we recompute obj in raw units.
    s = np.max(np.abs(P))
    x = cp.Variable(N)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap((P + P.T) / (2 * s))) + (c / s) @ x),
        [x >= lb, x <= ub],
    )
    t0 = time.perf_counter()
    prob.solve(solver=cp.CLARABEL)
    t_cvx += time.perf_counter() - t0
    cvx_x.append(np.asarray(x.value, float))
    cvx_obj.append(obj_of(P, c, x.value))


def relx(xa, xb):
    """Relative solution error, scale-free (box is [-1,1] so denom>=1)."""
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    return float(np.max(np.abs(xa - xb)) / max(1.0, np.max(np.abs(xb))))


def relobj(a, b):
    return abs(a - b) / max(1.0, abs(b))


def kkt_resid(P, c, lb, ub, x):
    """Projected-gradient stationarity residual, relative to ||grad||."""
    x = np.asarray(x, float)
    g = P @ x + c
    # project -g onto the tangent cone of the box at x
    tol = 1e-9
    d = -g.copy()
    d[(x <= lb + tol) & (d < 0)] = 0.0
    d[(x >= ub - tol) & (d > 0)] = 0.0
    return float(np.max(np.abs(d)) / max(1.0, np.max(np.abs(g))))


print(f"{'k':>2} {'cond':>8} {'scale':>8} {'status':>10} "
      f"{'obj_batch':>14} {'obj_cvx':>14} "
      f"{'relobj_bc':>10} {'relx_bc':>10} {'relx_bs':>10} "
      f"{'kkt_b':>9} {'kkt_c':>9}")

max_relobj_bc = max_relx_bc = max_relx_bs = 0.0
worst = None
rows = []
for k in range(K):
    P, c, lb, ub = items[k]["P"], items[k]["c"], items[k]["lb"], items[k]["ub"]
    rb, rs = batch_res[k], single_res[k]
    xb = np.asarray(rb.x, float)
    xs = np.asarray(rs.x, float)
    xc = cvx_x[k]
    ob, oc = obj_of(P, c, xb), cvx_obj[k]
    e_obj = relobj(ob, oc)
    e_x_bc = relx(xb, xc)
    e_x_bs = relx(xb, xs)
    kb, kc = kkt_resid(P, c, lb, ub, xb), kkt_resid(P, c, lb, ub, xc)
    rows.append((k, conds[k], scales[k], rb.status, ob, oc, e_obj, e_x_bc, e_x_bs, kb, kc))
    if e_x_bc > max_relx_bc:
        max_relx_bc, worst = e_x_bc, k
    max_relobj_bc = max(max_relobj_bc, e_obj)
    max_relx_bs = max(max_relx_bs, e_x_bs)
    print(f"{k:>2} {conds[k]:>8.0e} {scales[k]:>8.0e} {str(rb.status):>10} "
          f"{ob:>14.6e} {oc:>14.6e} "
          f"{e_obj:>10.2e} {e_x_bc:>10.2e} {e_x_bs:>10.2e} "
          f"{kb:>9.2e} {kc:>9.2e}")

print()
print(f"t_batch={t_batch:.4f}s  t_single_total={t_single:.4f}s  t_cvxpy={t_cvx:.4f}s")
print(f"PER-ITEM MAX relobj(batch vs cvxpy) = {max_relobj_bc:.3e}")
print(f"PER-ITEM MAX relx  (batch vs cvxpy) = {max_relx_bc:.3e}  (worst item k={worst})")
print(f"PER-ITEM MAX relx  (batch vs single) = {max_relx_bs:.3e}")
statuses = [str(r.status) for r in batch_res]
print(f"batch statuses = {statuses}")
print(f"single statuses = {[str(r.status) for r in single_res]}")

# --- secondary probe: does a TIGHTER shared tol change the badly-scaled items?
tight = solve_qp_batch(items, tol=1e-12, max_iter=500)
drift = [relx(np.asarray(tight[k].x, float), np.asarray(batch_res[k].x, float))
         for k in range(K)]
print(f"tol=1e-12 per-item drift vs default batch = "
      f"{[f'{d:.2e}' for d in drift]}")

# --- EXACT oracle: exhaustive active-set enumeration over the box (3^N).
# n=6 -> 729 patterns; this is the ground truth, independent of every solver.
import itertools

exact_x, exact_obj = [], []
for k in range(K):
    P, c, lb, ub = items[k]["P"], items[k]["c"], items[k]["lb"], items[k]["ub"]
    best = (np.inf, None)
    for pat in itertools.product([-1, 0, 1], repeat=N):
        pat = np.array(pat)
        free = np.where(pat == 0)[0]
        fixed = np.where(pat != 0)[0]
        x = np.where(pat == -1, lb, np.where(pat == 1, ub, 0.0)).astype(float)
        if len(free):
            try:
                x[free] = np.linalg.solve(
                    P[np.ix_(free, free)],
                    -(c[free] + P[np.ix_(free, fixed)] @ x[fixed]),
                )
            except np.linalg.LinAlgError:
                continue
        if max(np.max(lb - x), np.max(x - ub), 0.0) > 1e-9:
            continue
        o = obj_of(P, c, x)
        if o < best[0]:
            best = (o, x.copy())
    exact_x.append(best[1])
    exact_obj.append(best[0])

print()
print(f"{'k':>2} {'relx_vs_exact':>14} {'relobj_vs_exact':>16} {'boxviol_pounce':>15} "
      f"{'cvx_relx':>10}")
max_relx_exact = 0.0
for k in range(K):
    lb, ub = items[k]["lb"], items[k]["ub"]
    xb = np.asarray(batch_res[k].x, float)
    v = float(max(np.max(lb - xb), np.max(xb - ub), 0.0))
    e = relx(xb, exact_x[k])
    max_relx_exact = max(max_relx_exact, e)
    print(f"{k:>2} {e:>14.2e} "
          f"{abs(obj_of(items[k]['P'], items[k]['c'], xb) - exact_obj[k]) / abs(exact_obj[k]):>16.2e} "
          f"{v:>15.2e} {relx(cvx_x[k], exact_x[k]):>10.2e}")
print(f"PER-ITEM MAX relx (batch vs EXACT enumeration) = {max_relx_exact:.3e}")

# --- can pounce reach the exact answer on the hard item with more budget?
for mi in (200, 1000, 5000):
    r = solve_qp(**items[K - 1], max_iter=mi, tol=1e-10)
    x = np.asarray(r.x, float)
    lb, ub = items[K - 1]["lb"], items[K - 1]["ub"]
    v = float(max(np.max(lb - x), np.max(x - ub), 0.0))
    print(f"item {K-1} max_iter={mi:>5} tol=1e-10 -> {str(r.status):>16} "
          f"relx_vs_exact={relx(x, exact_x[K-1]):.2e} boxviol={v:.2e}")

# --- secondary probe: solo batch (batch of 1) per item, to isolate whether
# the presence of OTHER-scaled items in the same batch changes an item.
solo = [solve_qp_batch([items[k]])[0] for k in range(K)]
contam = [relx(np.asarray(batch_res[k].x, float), np.asarray(solo[k].x, float))
          for k in range(K)]
print(f"mixed-batch vs solo-batch per-item drift = "
      f"{[f'{d:.2e}' for d in contam]}")
max_contam = max(contam)

all_opt = all(s == "optimal" for s in statuses)
ok = all_opt and max_relx_bc < 1e-4 and max_relx_bs < 1e-4 and max_contam < 1e-6
if ok:
    print("VERDICT: PASS")
else:
    print(f"VERDICT: FAIL (all_optimal={all_opt}, max_relx_vs_cvxpy={max_relx_bc:.2e}, "
          f"max_relx_vs_single={max_relx_bs:.2e}, max_batch_contamination={max_contam:.2e})")
