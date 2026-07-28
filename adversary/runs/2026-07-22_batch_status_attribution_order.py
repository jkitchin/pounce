"""Adversary cross-check: batch STATUS ATTRIBUTION under adversarial ordering
Family: batch   Class: infeasibility / unboundedness / status-reporting correctness
Source: internal-consistency contract (batch[i] == isolated solve_qp(items[i]))
        + closed-form KKT per item + cvxpy (CLARABEL) per feasible item.
Known optimal: analytic per item (see build_* below)

Extends the 2026-07-19 "Mixed-status batch" run.  There the mix was fixed;
here the ORDER is the attack surface.

  (a) ORDER PERMUTATIONS.  Every one of the 5!=120 permutations of a 5-item
      multiset {F,F,F,INFEAS,UNBD} is submitted as a batch.  For each we assert
      result[i] matches the ITEM at position i (status, x, obj), never the
      position.  Includes infeasible-first / middle / last explicitly.
  (b) ALL items infeasible.
  (c) exactly ONE infeasible among many feasible, sweeping its position; and
      the mirror image, exactly ONE feasible among many infeasible.
  (d) batch of size 1 and size 0.
  (e) NEIGHBOUR PURITY: every feasible item's solution compared bit-for-bit
      against an isolated single solve_qp, and to cvxpy + closed form.

Smoking-gun detector: every feasible item is built with a UNIQUE analytic
minimiser, so a solution assigned to the wrong index is identifiable by
nearest-neighbour matching against the table of true minimisers.
"""

import itertools
import time

import numpy as np

from pounce import solve_qp, solve_qp_batch

np.set_printoptions(precision=6, suppress=True)

N = 3
BIG = 50.0  # box kept strictly inactive for feas_unc (see feas_box for the active case)
FAILS = []
NCHECK = 0


def note(ok, msg):
    global NCHECK
    NCHECK += 1
    if not ok:
        FAILS.append(msg)
        print("  FAIL " + msg)


# ------------------------------------------------------------------
# Item factory.  Each item carries its analytic truth.
# ------------------------------------------------------------------
def feas_unc(k):
    """P=(k+1)I, c=-(k+1)t  ->  x* = t (unique per k), box inactive."""
    t = np.array([1.0 + 0.5 * k, -0.5 - 0.25 * k, 0.3 * k - 1.0])
    s = float(k + 1)
    P = s * np.eye(N)
    c = -s * t
    kw = dict(P=P, c=c, lb=-BIG * np.ones(N), ub=BIG * np.ones(N))
    obj = 0.5 * t @ (P @ t) + c @ t
    return dict(kind=f"feas_unc{k}", kw=kw, xstar=t, obj=obj, feasible=True)


def feas_box(k):
    """Diagonal P, unconstrained min pushed outside the box -> clamped x*."""
    d = np.array([1.0, 2.0, 4.0]) * (1.0 + 0.3 * k)
    u = np.array([3.0 + k, -4.0 - k, 0.5 * k])
    P = np.diag(d)
    c = -d * u
    lb = -np.ones(N)
    ub = 2.0 * np.ones(N)
    x = np.clip(u, lb, ub)
    kw = dict(P=P, c=c, lb=lb, ub=ub)
    obj = 0.5 * x @ (P @ x) + c @ x
    return dict(kind=f"feas_box{k}", kw=kw, xstar=x, obj=obj, feasible=True)


def feas_eq(k):
    """min 1/2 x'x - t'x  s.t. sum(x) = k+1.  x* = t + (b - sum t)/n."""
    t = np.array([0.2 * k, 1.0 - k, 2.0 + 0.5 * k])
    b = float(k + 1)
    P = np.eye(N)
    c = -t
    A = np.ones((1, N))
    x = t + (b - t.sum()) / N
    kw = dict(P=P, c=c, A=A, b=np.array([b]))
    obj = 0.5 * x @ x - t @ x
    return dict(kind=f"feas_eq{k}", kw=kw, xstar=x, obj=obj, feasible=True)


def infeas(k):
    """x0 <= -(1+k) and -x0 <= -(1+k)  ->  empty feasible set."""
    g = float(1 + k)
    kw = dict(
        P=np.eye(N),
        c=np.zeros(N),
        G=np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
        h=np.array([-g, -g]),
    )
    return dict(kind=f"infeas{k}", kw=kw, xstar=None, obj=None, feasible=False)


def unbd(k):
    """PSD-singular P=diag(1,1,0) with a linear term down the null direction."""
    kw = dict(
        P=np.diag([1.0, 1.0, 0.0]),
        c=np.array([0.0, 0.0, -(1.0 + k)]),
    )
    return dict(kind=f"unbd{k}", kw=kw, xstar=None, obj=None, feasible=False)


# ------------------------------------------------------------------
# Oracles
# ------------------------------------------------------------------
ISO = {}  # kind -> isolated single solve_qp result signature


def sig(r):
    x = getattr(r, "x", None)
    return (
        str(getattr(r, "status", None)),
        None if x is None else np.asarray(x, float).copy(),
        float(getattr(r, "obj", np.nan)),
    )


def isolated(item):
    """Oracle 1: the same item solved ALONE by solve_qp."""
    if item["kind"] not in ISO:
        ISO[item["kind"]] = sig(solve_qp(**item["kw"]))
    return ISO[item["kind"]]


def cvxpy_check(item):
    """Oracle 2: cvxpy/CLARABEL on the same data."""
    import cvxpy as cp

    kw = item["kw"]
    x = cp.Variable(N)
    P = kw["P"]
    obj = 0.5 * cp.quad_form(x, cp.psd_wrap(P)) + kw["c"] @ x
    cons = []
    if "A" in kw:
        cons.append(kw["A"] @ x == kw["b"])
    if "G" in kw:
        cons.append(kw["G"] @ x <= kw["h"])
    if "lb" in kw:
        cons += [x >= kw["lb"], x <= kw["ub"]]
    p = cp.Problem(cp.Minimize(obj), cons)
    p.solve(solver=cp.CLARABEL)
    return p.status, p.value, (None if x.value is None else np.array(x.value))


def check_batch(items, tag, res=None):
    """Core attribution assertion: res[i] must equal the ORACLE for items[i]."""
    if res is None:
        res = solve_qp_batch([it["kw"] for it in items])
    note(len(res) == len(items), f"{tag}: len {len(res)} != {len(items)}")
    truths = [it["xstar"] for it in items]
    for i, (it, r) in enumerate(zip(items, res)):
        st, xb, ob = sig(r)
        sti, xi, oi = isolated(it)
        note(st == sti, f"{tag}[{i}]={it['kind']}: status batch={st} isolated={sti}")
        if it["feasible"]:
            note(st == "optimal", f"{tag}[{i}]={it['kind']}: expected optimal, got {st}")
            # (e) bit-identity vs isolated solve
            dx = float(np.max(np.abs(xb - xi)))
            note(dx == 0.0, f"{tag}[{i}]={it['kind']}: dx vs isolated = {dx:.3e} (not bit-identical)")
            # analytic ground truth
            da = float(np.max(np.abs(xb - it["xstar"])))
            note(da < 1e-7, f"{tag}[{i}]={it['kind']}: dx vs analytic = {da:.3e}")
            do = abs(ob - it["obj"]) / max(1.0, abs(it["obj"]))
            note(do < 1e-8, f"{tag}[{i}]={it['kind']}: dobj vs analytic = {do:.3e}")
            # smoking gun: is this solution actually some OTHER item's answer?
            cand = [(float(np.max(np.abs(xb - t))), j) for j, t in enumerate(truths) if t is not None]
            best = min(cand)[1]
            note(best == i, f"{tag}[{i}]={it['kind']}: x best-matches item {best}, not {i}")
        else:
            note(st != "optimal", f"{tag}[{i}]={it['kind']}: infeasible item reported {st}")
    return res


# ------------------------------------------------------------------
t_start = time.perf_counter()
print("=== oracle table (isolated single solves + cvxpy + analytic) ===")
POOL = [feas_unc(0), feas_unc(1), feas_box(0), feas_box(1), feas_eq(0), feas_eq(1),
        infeas(0), infeas(1), unbd(0), unbd(1)]
for it in POOL:
    st, x, o = isolated(it)
    if it["feasible"]:
        cst, cval, cx = cvxpy_check(it)
        dxc = float(np.max(np.abs(x - cx)))
        note(cst == "optimal", f"cvxpy status {cst} for {it['kind']}")
        note(dxc < 1e-6, f"cvxpy dx {dxc:.2e} for {it['kind']}")
        note(float(np.max(np.abs(x - it['xstar']))) < 1e-7, f"analytic dx for {it['kind']}")
        print(f"  {it['kind']:>11}: {st:>8}  x*={x}  obj={o:+.6f}  cvxpy dx={dxc:.1e}")
    else:
        cst, cval, cx = cvxpy_check(it)
        print(f"  {it['kind']:>11}: {st:>18}  (cvxpy: {cst})")
        note("infeasible" in cst or "unbounded" in cst,
             f"cvxpy calls {it['kind']} {cst} (expected infeasible/unbounded)")

# pairwise-distinct minimisers, so misattribution is detectable
tr = [it["xstar"] for it in POOL if it["feasible"]]
sep = min(float(np.max(np.abs(a - b))) for a, b in itertools.combinations(tr, 2))
print(f"  min pairwise separation of true minimisers: {sep:.3f}")
note(sep > 0.1, f"minimisers not well separated ({sep:.3f})")

# ------------------------------------------------------------------
# (a) ALL 120 orderings of {F,F,F,INFEAS,UNBD}
# ------------------------------------------------------------------
print("\n=== (a) all 120 permutations of {feas,feas,feas,infeas,unbd} ===")
base = [POOL[0], POOL[2], POOL[4], POOL[6], POOL[8]]
perm_stat = {}
t0 = time.perf_counter()
for p in itertools.permutations(range(5)):
    items = [base[j] for j in p]
    res = check_batch(items, f"perm{p}")
    bad = tuple(sorted(i for i, r in enumerate(res) if str(r.status) != "optimal"))
    want = tuple(sorted(i for i, it in enumerate(items) if not it["feasible"]))
    note(bad == want, f"perm{p}: non-optimal idx {bad} != infeasible idx {want}")
    perm_stat.setdefault(tuple(str(r.status) for r in res), []).append(p)
print(f"  120 permutations in {time.perf_counter()-t0:.3f}s; "
      f"{len(perm_stat)} distinct status vectors (expected 20 = 5!/3!)")
for p in ((3, 0, 1, 2, 4), (0, 1, 3, 2, 4), (0, 1, 2, 4, 3)):
    items = [base[j] for j in p]
    res = solve_qp_batch([it["kw"] for it in items])
    lbl = {(3, 0, 1, 2, 4): "infeasible FIRST", (0, 1, 3, 2, 4): "infeasible MIDDLE",
           (0, 1, 2, 4, 3): "infeasible LAST"}[p]
    print(f"  {lbl:<18}: " + " ".join(f"{it['kind']}={r.status}" for it, r in zip(items, res)))

# ------------------------------------------------------------------
# (b) ALL infeasible
# ------------------------------------------------------------------
print("\n=== (b) batch where ALL items fail ===")
allbad = [infeas(k) for k in range(4)] + [unbd(k) for k in range(4)]
res = check_batch(allbad, "allbad")
print("  " + " ".join(f"{it['kind']}={r.status}" for it, r in zip(allbad, res)))
note(all(str(r.status) != "optimal" for r in res), "allbad: some item reported optimal")

# ------------------------------------------------------------------
# (c) exactly ONE bad among many good, sweeping position; and mirror
# ------------------------------------------------------------------
print("\n=== (c) one-of-many, position sweep ===")
M = 12
goods = [feas_unc(k) for k in range(M)]
for pos in range(M + 1):
    items = goods[:pos] + [infeas(7)] + goods[pos:]
    res = check_batch(items, f"one-bad@{pos}")
    bad = [i for i, r in enumerate(res) if str(r.status) != "optimal"]
    note(bad == [pos], f"one-bad@{pos}: non-optimal at {bad}, expected [{pos}]")
print(f"  13 placements of a single infeasible item among {M} feasible: ok")

bads = [infeas(k) for k in range(M)]
for pos in range(M + 1):
    items = bads[:pos] + [feas_unc(3)] + bads[pos:]
    res = check_batch(items, f"one-good@{pos}")
    good = [i for i, r in enumerate(res) if str(r.status) == "optimal"]
    note(good == [pos], f"one-good@{pos}: optimal at {good}, expected [{pos}]")
print(f"  13 placements of a single feasible item among {M} infeasible: ok")

# mirror with unbounded as the singleton
for pos in (0, 6, M):
    items = goods[:pos] + [unbd(3)] + goods[pos:]
    res = check_batch(items, f"one-unbd@{pos}")
    bad = [i for i, r in enumerate(res) if str(r.status) != "optimal"]
    note(bad == [pos], f"one-unbd@{pos}: non-optimal at {bad}, expected [{pos}]")

# ------------------------------------------------------------------
# (d) size 1 and size 0
# ------------------------------------------------------------------
print("\n=== (d) edge sizes ===")
for it in (POOL[0], POOL[6], POOL[8]):
    res = check_batch([it], f"size1-{it['kind']}")
    print(f"  size-1 {it['kind']:>11}: {res[0].status}")
empty = solve_qp_batch([])
note(isinstance(empty, list) and len(empty) == 0, f"size-0 returned {empty!r}")
print(f"  size-0: {empty!r}")

# ------------------------------------------------------------------
# (e) neighbour purity: reversed / rotated orders must not move any answer
# ------------------------------------------------------------------
print("\n=== (e) neighbour purity under rotation ===")
mix = [POOL[0], POOL[6], POOL[2], POOL[8], POOL[4], POOL[7], POOL[1], POOL[9], POOL[3], POOL[5]]
ref = {}
for rot in range(len(mix)):
    items = mix[rot:] + mix[:rot]
    res = check_batch(items, f"rot{rot}")
    for it, r in zip(items, res):
        st, x, o = sig(r)
        if it["kind"] in ref:
            prev = ref[it["kind"]]
            same = prev[0] == st and (x is None) == (prev[1] is None) and (
                x is None or float(np.max(np.abs(x - prev[1]))) == 0.0)
            note(same, f"rot{rot}: {it['kind']} changed with position")
        else:
            ref[it["kind"]] = (st, x, o)
print(f"  10 rotations x 10 items: every answer position-invariant "
      f"({'bit-identical' if not FAILS else 'SEE FAILURES'})")

# ------------------------------------------------------------------
t_all = time.perf_counter() - t_start
print(f"\nchecks={NCHECK} failures={len(FAILS)} wall={t_all:.2f}s")
for f in FAILS[:20]:
    print("  !! " + f)
print("VERDICT: PASS" if not FAILS else f"VERDICT: FAIL ({len(FAILS)} checks)")
