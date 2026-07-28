"""Adversary cross-check: LP status-discrimination battery.

Family: lp    Class: infeasibility / unboundedness / status reporting
Source: LP duality + Farkas' lemma (Schrijver, "Theory of Linear and Integer
        Programming", Cor. 7.1d; Bertsimas & Tsitsiklis, "Introduction to
        Linear Optimization", Thm 4.6 & S4.10 "the four cases").
        The (c) instance is the classic primal-infeasible-AND-dual-infeasible
        cell of the 2x2 primal/dual status table (B&T Table 4.2).

Ground truth is established by EXACT rational certificates (fractions.Fraction),
checked against the exact float64 data pounce receives:

  * infeasible  <- Farkas certificate y >= 0, G^T y >= 0, h^T y < 0
                   (proves {Gx <= h, x >= 0} = {})
  * unbounded   <- feasible point x0 + recession ray d >= 0, G d <= 0, c^T d < 0
  * optimal     <- primal feasible x, dual feasible z >= 0 with
                   c + G^T z >= 0, complementary slackness, c^T x = -h^T z

Every certificate below is VERIFIED exactly at runtime; nothing is asserted on
faith.  All instances are in pounce's solve_qp form:

    min c^T x   s.t.  G x <= h,  x >= 0     (P = None => LP)
"""

from __future__ import annotations

import time
from fractions import Fraction as F

import numpy as np
import scipy.optimize as sopt

from pounce import solve_qp

# ---------------------------------------------------------------- exact oracle


def _fr(M):
    return [[F(float(v)) for v in row] for row in M]


def _fv(v):
    return [F(float(t)) for t in v]


def cert_infeasible(G, h, y):
    """Farkas: y >= 0, G^T y >= 0, h^T y < 0  =>  {Gx<=h, x>=0} is empty."""
    G, h, y = _fr(G), _fv(h), _fv(y)
    n = len(G[0])
    if any(t < 0 for t in y):
        return False, "y not >= 0"
    for j in range(n):
        s = sum(y[i] * G[i][j] for i in range(len(G)))
        if s < 0:
            return False, f"(G^T y)_{j} = {s} < 0"
    hy = sum(y[i] * h[i] for i in range(len(h)))
    if hy >= 0:
        return False, f"h^T y = {hy} >= 0"
    return True, f"Farkas OK: G^T y >= 0, h^T y = {hy} < 0"


def cert_unbounded(c, G, h, x0, d):
    """x0 feasible, d a recession ray with c^T d < 0  =>  objective -> -inf."""
    c, G, h, x0, d = _fv(c), _fr(G), _fv(h), _fv(x0), _fv(d)
    if any(t < 0 for t in x0):
        return False, "x0 not >= 0"
    for i, row in enumerate(G):
        if sum(row[j] * x0[j] for j in range(len(x0))) > h[i]:
            return False, f"x0 violates row {i}"
    if any(t < 0 for t in d):
        return False, "d not >= 0 (would leave x >= 0)"
    for i, row in enumerate(G):
        if sum(row[j] * d[j] for j in range(len(d))) > 0:
            return False, f"d not a recession direction at row {i}"
    cd = sum(c[j] * d[j] for j in range(len(c)))
    if cd >= 0:
        return False, f"c^T d = {cd} >= 0"
    return True, f"ray OK: G d <= 0, d >= 0, c^T d = {cd} < 0"


def cert_dual_infeasible_system(c, G):
    """No z >= 0 with c + G^T z >= 0  <=>  dual {max -h^Tz : -G^Tz <= c, z>=0} empty.

    Detected structurally: if column j of G is identically zero and c_j < 0,
    then (c + G^T z)_j = c_j < 0 for every z, so the dual is infeasible.
    """
    c, G = _fv(c), _fr(G)
    for j in range(len(c)):
        if all(G[i][j] == 0 for i in range(len(G))) and c[j] < 0:
            return True, f"col {j} of G is zero and c_{j} = {c[j]} < 0"
    return False, "no structural proof (not necessarily dual-feasible)"


def cert_optimal(c, G, h, x, z):
    """Exact primal/dual optimality certificate for min c^Tx, Gx<=h, x>=0."""
    c, G, h, x, z = _fv(c), _fr(G), _fv(h), _fv(x), _fv(z)
    n, m = len(x), len(h)
    if any(t < 0 for t in x):
        return False, "x not >= 0"
    slack = []
    for i in range(m):
        s = h[i] - sum(G[i][j] * x[j] for j in range(n))
        if s < 0:
            return False, f"x violates row {i} by {-s}"
        slack.append(s)
    if any(t < 0 for t in z):
        return False, "z not >= 0"
    red = [c[j] + sum(z[i] * G[i][j] for i in range(m)) for j in range(n)]
    if any(t < 0 for t in red):
        return False, f"reduced cost negative: {red}"
    for i in range(m):
        if z[i] * slack[i] != 0:
            return False, f"complementarity fails at row {i}"
    for j in range(n):
        if red[j] * x[j] != 0:
            return False, f"bound complementarity fails at var {j}"
    p = sum(c[j] * x[j] for j in range(n))
    d = -sum(h[i] * z[i] for i in range(m))
    if p != d:
        return False, f"duality gap {p - d}"
    return True, f"optimal, exact value = {p} = {float(p)!r}"


# ------------------------------------------------------------------ instances

EPS = 1e-12
H_FEAS = -(1.0 - EPS)   # x1+x2 >= 1-1e-12   (feasible strip, width 1e-12)
H_INFEAS = -(1.0 + EPS)  # x1+x2 >= 1+1e-12   (infeasible by 1e-12)

CASES = []

# (a) plainly infeasible; dual is feasible (z=0) so this is UNAMBIGUOUSLY
#     primal_infeasible -- no "infeasible or unbounded" excuse is available.
CASES.append(dict(
    tag="a_infeasible",
    desc="x1+x2 <= 1 and x1+x2 >= 2, x >= 0; c = (1,1)",
    c=[1.0, 1.0],
    G=[[1.0, 1.0], [-1.0, -1.0]],
    h=[1.0, -2.0],
    truth="primal_infeasible",
    proof=lambda C: cert_infeasible(C["G"], C["h"], [1.0, 1.0]),
))

# (b) plainly unbounded; primal feasible at 0, ray (1,1).
CASES.append(dict(
    tag="b_unbounded",
    desc="min -x1-x2 s.t. x1-x2 <= 1, x2-x1 <= 1, x >= 0",
    c=[-1.0, -1.0],
    G=[[1.0, -1.0], [-1.0, 1.0]],
    h=[1.0, 1.0],
    truth="dual_infeasible",
    proof=lambda C: cert_unbounded(C["c"], C["G"], C["h"], [0.0, 0.0], [1.0, 1.0]),
))

# (c) primal infeasible AND dual infeasible (B&T Table 4.2, the "both empty"
#     cell).  x1 <= -1 with x1 >= 0 kills the primal; column 2 of G is zero
#     while c_2 = -1 < 0 kills the dual.
CASES.append(dict(
    tag="c_both_infeasible",
    desc="min -x2 s.t. x1 <= -1, x >= 0  (primal empty AND dual empty)",
    c=[0.0, -1.0],
    G=[[1.0, 0.0]],
    h=[-1.0],
    truth="primal_infeasible|dual_infeasible",
    proof=lambda C: cert_infeasible(C["G"], C["h"], [1.0]),
    proof2=lambda C: cert_dual_infeasible_system(C["c"], C["G"]),
))

# (d) feasible set is the single point (1/2, 1/2): empty interior, Slater fails.
CASES.append(dict(
    tag="d_singleton",
    desc="x1+x2 = 1 and x1 = x2 written as 4 inequalities; unique point (1/2,1/2)",
    c=[1.0, 2.0],
    G=[[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]],
    h=[1.0, -1.0, 0.0, 0.0],
    truth="optimal",
    known_obj=1.5,
    proof=lambda C: cert_optimal(C["c"], C["G"], C["h"],
                                 [0.5, 0.5], [0.0, 1.5, 0.5, 0.0]),
))

# (e) MARGINAL PAIR: identical but for a 1e-12 shift of one rhs.
CASES.append(dict(
    tag="e_marginal_feasible",
    desc=f"x1+x2 <= 1, x1+x2 >= 1-1e-12  (feasible strip of width {EPS:g})",
    c=[1.0, 1.0],
    G=[[1.0, 1.0], [-1.0, -1.0]],
    h=[1.0, H_FEAS],
    truth="optimal",
    known_obj=-float(F(H_FEAS)),
    proof=lambda C: cert_optimal(
        C["c"], C["G"], C["h"],
        [-F(H_FEAS), F(0)], [F(0), F(1)]),
))
CASES.append(dict(
    tag="e_marginal_infeasible",
    desc=f"x1+x2 <= 1, x1+x2 >= 1+1e-12  (infeasible by {EPS:g})",
    c=[1.0, 1.0],
    G=[[1.0, 1.0], [-1.0, -1.0]],
    h=[1.0, H_INFEAS],
    truth="primal_infeasible",
    proof=lambda C: cert_infeasible(C["G"], C["h"], [1.0, 1.0]),
))


# -------------------------------------------------------------------- oracles

def run_scipy(C):
    r = sopt.linprog(C["c"], A_ub=np.array(C["G"]), b_ub=np.array(C["h"]),
                     bounds=[(0, None)] * len(C["c"]), method="highs")
    # 0 optimal, 2 infeasible, 3 unbounded, 4 numerical
    m = {0: "optimal", 1: "iteration_limit", 2: "primal_infeasible",
         3: "dual_infeasible", 4: "numerical_failure"}
    return m.get(r.status, f"status_{r.status}"), (r.fun if r.fun is not None else float("nan"))


def run_cvxpy(C):
    import cvxpy as cp
    x = cp.Variable(len(C["c"]), nonneg=True)
    prob = cp.Problem(cp.Minimize(np.array(C["c"]) @ x),
                      [np.array(C["G"]) @ x <= np.array(C["h"])])
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception as e:  # pragma: no cover
        return f"error:{type(e).__name__}", float("nan")
    m = {"optimal": "optimal", "infeasible": "primal_infeasible",
         "unbounded": "dual_infeasible",
         "infeasible_inaccurate": "primal_infeasible?",
         "unbounded_inaccurate": "dual_infeasible?",
         "optimal_inaccurate": "optimal?"}
    return m.get(prob.status, prob.status), (prob.value if prob.value is not None else float("nan"))


# ----------------------------------------------------------------------- main

def acceptable(truth, got):
    return got in truth.split("|")


print("=" * 78)
print("EXACT GROUND TRUTH (Fraction certificates on the float64 data)")
print("=" * 78)
for C in CASES:
    ok, msg = C["proof"](C)
    assert ok, f"{C['tag']}: certificate FAILED to verify: {msg}"
    print(f"  {C['tag']:24s} {C['truth']:38s} {msg}")
    if "proof2" in C:
        ok2, msg2 = C["proof2"](C)
        assert ok2, f"{C['tag']}: second certificate FAILED: {msg2}"
        print(f"  {'':24s} {'(dual side)':38s} {msg2}")

print()
print("=" * 78)
print("SOLVER STATUSES")
print("=" * 78)
hdr = f"{'case':24s} {'truth':22s} {'pounce':20s} {'scipy/HiGHS':20s} {'cvxpy/CLARABEL':20s}"
print(hdr)
print("-" * len(hdr))

rows, failures, tolerance_notes = [], [], []
for C in CASES:
    t0 = time.perf_counter()
    r = solve_qp(P=None, c=np.array(C["c"]), G=np.array(C["G"]),
                 h=np.array(C["h"]), lb=np.zeros(len(C["c"])))
    t_p = time.perf_counter() - t0
    s_sp, o_sp = run_scipy(C)
    s_cp, o_cp = run_cvxpy(C)
    ok_p = acceptable(C["truth"], r.status)
    rows.append((C["tag"], C["truth"], r.status, r.obj, r.iters, t_p, s_sp, o_sp, s_cp, o_cp, ok_p))
    print(f"{C['tag']:24s} {C['truth']:22s} {r.status:20s} {s_sp:20s} {s_cp:20s}")
    if not ok_p:
        (tolerance_notes if C["tag"].startswith("e_") else failures).append(
            f"{C['tag']}: truth={C['truth']} pounce={r.status}")

print()
print("=" * 78)
print("DETAIL")
print("=" * 78)
for (tag, truth, sp, ob, it, tp, s_sp, o_sp, s_cp, o_cp, ok) in rows:
    C = next(c for c in CASES if c["tag"] == tag)
    known = C.get("known_obj")
    extra = ""
    if known is not None and sp == "optimal":
        extra = f" obj={ob!r} known={known!r} abs_err={abs(ob - known):.3e}"
    print(f"{tag:24s} pounce={sp:20s} iters={it:3d} t={tp:.4f}s{extra}")
    print(f"{'':24s} scipy={s_sp:20s} obj={o_sp!r}")
    print(f"{'':24s} cvxpy={s_cp:20s} obj={o_cp!r}")

# objective accuracy for the two 'optimal' cases
obj_bad = []
for C in CASES:
    if C.get("known_obj") is None:
        continue
    row = next(r for r in rows if r[0] == C["tag"])
    if row[2] == "optimal" and abs(row[3] - C["known_obj"]) > 1e-6:
        obj_bad.append(f"{C['tag']}: obj={row[3]!r} known={C['known_obj']!r}")

print()
print("=" * 78)
# the marginal pair: did pounce FLIP?
mf = next(r for r in rows if r[0] == "e_marginal_feasible")
mi = next(r for r in rows if r[0] == "e_marginal_infeasible")
print(f"MARGINAL PAIR (1e-12 apart): feasible->{mf[2]}   infeasible->{mi[2]}   "
      f"{'DISCRIMINATED' if mf[2] != mi[2] else 'MERGED (same status)'}")
c_row = next(r for r in rows if r[0] == "c_both_infeasible")
print(f"BOTH-INFEASIBLE case: pounce says {c_row[2]!r} "
      f"(either primal_infeasible or dual_infeasible is TRUE here)")
print()

# ---- how far apart must the marginal pair be before pounce discriminates? ----
print("=" * 78)
print("MARGINAL SWEEP: min x1+x2 s.t. x1+x2 <= 1, x1+x2 >= 1+eps, x >= 0")
print("  (INFEASIBLE for every eps > 0; Farkas y=(1,1) verified exactly each row)")
print("=" * 78)
Gm = np.array([[1.0, 1.0], [-1.0, -1.0]])
cm = np.array([1.0, 1.0])
print(f"{'eps':>10s} {'pounce':>18s} {'p_infeas':>11s} {'scipy':>18s} {'clarabel':>22s}")
for e in [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
    hm = np.array([1.0, -(1.0 + e)])
    Cm = dict(c=cm, G=Gm, h=hm)
    ok, _ = cert_infeasible(Gm, hm, [1.0, 1.0])
    assert ok, f"eps={e}: Farkas certificate did not verify"
    r = solve_qp(P=None, c=cm, G=Gm, h=hm, lb=np.zeros(2))
    s_sp, _ = run_scipy(Cm)
    s_cp, _ = run_cvxpy(Cm)
    print(f"{e:10.0e} {r.status:>18s} "
          f"{r.residuals['primal_infeasibility']:11.2e} {s_sp:>18s} {s_cp:>22s}")

print()
print("=" * 78)
print("TOLERANCE SWEEP on the 1e-12 pair (does a tighter tol let pounce flip?)")
print("=" * 78)
for tol in [None, 1e-10, 1e-12, 1e-14]:
    out = []
    for lbl, e in [("FEASIBLE", -EPS), ("INFEASIBLE", +EPS)]:
        hm = np.array([1.0, -(1.0 + e)])
        kw = {} if tol is None else {"tol": tol}
        r = solve_qp(P=None, c=cm, G=Gm, h=hm, lb=np.zeros(2), max_iter=500, **kw)
        out.append(f"{lbl}->{r.status}(it={r.iters})")
    print(f"  tol={str(tol):8s}  " + "   ".join(out))
print()

if failures or obj_bad:
    print("VERDICT: FAIL " + "; ".join(failures + obj_bad))
elif tolerance_notes:
    print("VERDICT: TOLERANCE " + "; ".join(tolerance_notes))
else:
    print("VERDICT: PASS")
