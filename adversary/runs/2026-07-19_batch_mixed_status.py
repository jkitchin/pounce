"""Adversary cross-check: mixed-status batch (infeasible + unbounded + fine items)
Family: batch   Class: heterogeneous status / contamination / size-1 / duplicates
Source: internal-consistency contract (batch == per-item single solve_qp),
        plus closed-form KKT for the unconstrained/equality items.
Known optimal: per item (closed form where available)

Adversarial angles:
  A. A batch where ONE element is primal-INFEASIBLE and one is UNBOUNDED while
     the rest are well-posed. Does the failure contaminate its neighbours, or
     get misattributed to the wrong index? (Order is shuffled deliberately.)
  B. Heterogeneous conditioning: cond(P) spanning 1 .. 1e10 in one batch.
  C. Batch of size 1.
  D. Duplicated elements (same dict object repeated; and same data, new objects).
"""

import time
import numpy as np

np.set_printoptions(precision=6, suppress=True)
rng = np.random.default_rng(20260719)

from pounce import solve_qp, solve_qp_batch

FAILS = []


def note(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def spd(n, cond):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    d = np.logspace(0, np.log10(cond), n)
    return Q @ np.diag(d) @ Q.T


def sig(r):
    """Comparable signature of a QpResult, robust to missing fields."""
    x = getattr(r, "x", None)
    return (
        str(getattr(r, "status", None)),
        None if x is None else np.asarray(x, float).copy(),
        float(getattr(r, "obj", np.nan)),
    )


def cmp_res(rb, rs, tag):
    sb, ss = sig(rb), sig(rs)
    if sb[0] != ss[0]:
        note(False, f"{tag}: status batch={sb[0]!r} single={ss[0]!r}")
        return
    if sb[1] is None or ss[1] is None:
        note(sb[1] is ss[1] or (sb[1] is None) == (ss[1] is None), f"{tag}: x presence")
        return
    dx = float(np.max(np.abs(sb[1] - ss[1])))
    do = abs(sb[2] - ss[2]) / max(1.0, abs(ss[2]))
    note(dx < 1e-9 and do < 1e-9, f"{tag}: status={sb[0]} dx={dx:.2e} dobj={do:.2e}")


# ------------------------------------------------------------------
# A. mixed-status batch
# ------------------------------------------------------------------
print("=== A: mixed-status batch (infeasible + unbounded interleaved) ===")
n = 3

good = []
for c_ in (1.0, 1e3, 1e6, 1e10):
    P = spd(n, c_)
    cvec = rng.standard_normal(n)
    good.append(dict(P=P, c=cvec, lb=-2.0 * np.ones(n), ub=2.0 * np.ones(n)))

# primal infeasible: x0 <= -1 and x0 >= 1 (as G rows), with a bounded P
Pi = np.eye(n)
infeas = dict(
    P=Pi,
    c=np.zeros(n),
    G=np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
    h=np.array([-1.0, -1.0]),
)

# unbounded: P singular (PSD, zero curvature along e2), linear term drives it
Pu = np.diag([1.0, 1.0, 0.0])
unb = dict(P=Pu, c=np.array([0.0, 0.0, 1.0]))

probs = [good[0], infeas, good[1], good[2], unb, good[3]]
labels = ["good_c1", "INFEAS", "good_c1e3", "good_c1e6", "UNBOUNDED", "good_c1e10"]

t0 = time.perf_counter()
try:
    rb = solve_qp_batch(probs)
    batch_exc = None
except Exception as e:  # noqa: BLE001
    rb, batch_exc = None, e
t_batch = time.perf_counter() - t0

singles, sing_exc = [], []
t0 = time.perf_counter()
for p in probs:
    try:
        singles.append(solve_qp(**p))
        sing_exc.append(None)
    except Exception as e:  # noqa: BLE001
        singles.append(None)
        sing_exc.append(e)
t_single = time.perf_counter() - t0

print(f"batch t={t_batch:.4f}s  single-loop t={t_single:.4f}s")
if batch_exc is not None:
    print(f"batch raised: {type(batch_exc).__name__}: {batch_exc}")
    n_sing_exc = sum(e is not None for e in sing_exc)
    note(
        n_sing_exc > 0,
        f"batch raised but {n_sing_exc}/{len(probs)} singles raised too",
    )
else:
    note(len(rb) == len(probs), f"batch returned {len(rb)} results for {len(probs)} inputs")
    for i, lab in enumerate(labels):
        if sing_exc[i] is not None:
            print(f"  [{i}] {lab}: single raised {type(sing_exc[i]).__name__}; "
                  f"batch status={getattr(rb[i], 'status', None)!r}")
            continue
        print(f"  [{i}] {lab}: batch={getattr(rb[i],'status',None)!r} "
              f"single={getattr(singles[i],'status',None)!r} "
              f"obj_b={getattr(rb[i],'obj',float('nan')):.8e} "
              f"obj_s={getattr(singles[i],'obj',float('nan')):.8e}")
        cmp_res(rb[i], singles[i], f"[{i}] {lab}")

    # misattribution check: the bad statuses must sit at index 1 and 4
    bad_idx = [i for i, r in enumerate(rb)
               if str(getattr(r, "status", "")).lower() not in ("optimal", "solved")]
    print(f"  non-optimal indices in batch: {bad_idx}  (expected subset of [1, 4])")
    note(set(bad_idx) <= {1, 4}, f"bad statuses not misattributed (got {bad_idx})")
    good_idx = [0, 2, 3, 5]
    note(
        all(str(getattr(rb[i], "status", "")).lower() in ("optimal", "solved")
            for i in good_idx),
        "good items uncontaminated by their bad neighbours",
    )

# ------------------------------------------------------------------
# A2. control: the same good items solved WITHOUT the bad neighbours
# ------------------------------------------------------------------
print("=== A2: contamination control (good-only batch must be identical) ===")
rb2 = solve_qp_batch(good)
for k, i in enumerate([0, 2, 3, 5]):
    if batch_exc is None:
        cmp_res(rb[i], rb2[k], f"good[{k}] with-bad vs without-bad")
    cmp_res(rb2[k], solve_qp(**good[k]), f"good[{k}] cleanbatch vs single")

# ------------------------------------------------------------------
# B. closed-form check on the well-posed unconstrained-interior items
# ------------------------------------------------------------------
print("=== B: closed-form KKT on interior items ===")
for k, p in enumerate(good):
    xs = np.linalg.solve(p["P"], -p["c"])
    if np.all(np.abs(xs) < 2.0 - 1e-9):  # unconstrained optimum is interior
        got = np.asarray(rb2[k].x, float)
        err = float(np.max(np.abs(got - xs)))
        note(err < 1e-7, f"good[{k}] cond={np.linalg.cond(p['P']):.2e} "
                         f"closed-form inf-err={err:.2e}")
    else:
        # box-active: verify KKT stationarity with bound multipliers
        got = np.asarray(rb2[k].x, float)
        g = p["P"] @ got + p["c"]
        free = (np.abs(got) < 2.0 - 1e-7)
        r = float(np.max(np.abs(g[free]))) if free.any() else 0.0
        sgn_ok = all(
            (g[i] >= -1e-7) if got[i] <= -2 + 1e-7 else (g[i] <= 1e-7)
            for i in range(n) if not free[i]
        )
        note(r < 1e-6 and sgn_ok,
             f"good[{k}] cond={np.linalg.cond(p['P']):.2e} box-active KKT "
             f"stat={r:.2e} signs_ok={sgn_ok}")

# ------------------------------------------------------------------
# C. batch of size 1
# ------------------------------------------------------------------
print("=== C: batch of size 1 ===")
one = solve_qp_batch([good[1]])
note(len(one) == 1, f"len={len(one)}")
cmp_res(one[0], solve_qp(**good[1]), "size-1 batch")

print("=== C2: batch of size 0 ===")
try:
    zero = solve_qp_batch([])
    note(isinstance(zero, list) and len(zero) == 0, f"empty batch -> {zero!r}")
except Exception as e:  # noqa: BLE001
    note(False, f"empty batch raised {type(e).__name__}: {e}")

# ------------------------------------------------------------------
# D. duplicates
# ------------------------------------------------------------------
print("=== D: duplicated elements ===")
p0 = good[2]
dup_same_obj = [p0] * 5
rd = solve_qp_batch(dup_same_obj)
ref = solve_qp(**p0)
for i in range(5):
    cmp_res(rd[i], ref, f"dup-sameobj[{i}]")

dup_copies = [dict((k, np.array(v, dtype=float) if isinstance(v, np.ndarray) else v)
                   for k, v in p0.items()) for _ in range(5)]
rd2 = solve_qp_batch(dup_copies)
for i in range(5):
    cmp_res(rd2[i], ref, f"dup-copy[{i}]")

# duplicates mixed with the infeasible one, repeated
mix = [p0, infeas, p0, infeas, p0]
rm = solve_qp_batch(mix)
bad = [i for i, r in enumerate(rm)
       if str(getattr(r, "status", "")).lower() not in ("optimal", "solved")]
print(f"  dup-mixed non-optimal indices: {bad} (expected [1, 3])")
note(bad == [1, 3], f"dup-mixed bad indices = {bad}")
for i in (0, 2, 4):
    cmp_res(rm[i], ref, f"dup-mixed good[{i}]")

# ------------------------------------------------------------------
# E. heterogeneous conditioning, larger batch, full per-item oracle
# ------------------------------------------------------------------
print("=== E: 24-item heterogeneous-conditioning batch vs per-item single ===")
big = []
for i in range(24):
    nn = 2 + (i % 4)
    c_ = 10.0 ** (i % 11)
    P = spd(nn, c_)
    cv = rng.standard_normal(nn)
    d = dict(P=P, c=cv)
    if i % 3 == 0:
        G = rng.standard_normal((2, nn))
        d.update(G=G, h=G @ rng.standard_normal(nn) + 0.5)
    if i % 4 == 1:
        A = rng.standard_normal((1, nn))
        d.update(A=A, b=np.array([0.3]))
    big.append(d)

t0 = time.perf_counter()
rbig = solve_qp_batch(big)
t_b = time.perf_counter() - t0
t0 = time.perf_counter()
sbig = [solve_qp(**p) for p in big]
t_s = time.perf_counter() - t0
worst = 0.0
for i in range(24):
    cmp_res(rbig[i], sbig[i], f"het[{i}] cond=1e{i%11}")
    if getattr(rbig[i], "x", None) is not None and getattr(sbig[i], "x", None) is not None:
        worst = max(worst, float(np.max(np.abs(np.asarray(rbig[i].x) - np.asarray(sbig[i].x)))))
print(f"  worst |x_batch - x_single| = {worst:.3e}   "
      f"t_batch={t_b:.4f}s t_single={t_s:.4f}s speedup={t_s/max(t_b,1e-12):.2f}x")

print()
print(f"n_checks_failed={len(FAILS)}")
for f in FAILS:
    print("  !", f)
print("VERDICT: PASS" if not FAILS else f"VERDICT: FAIL ({len(FAILS)} checks)")
