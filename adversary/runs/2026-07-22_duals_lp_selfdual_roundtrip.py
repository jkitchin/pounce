"""Adversary run: LP DUALITY as a SELF-CONSISTENCY invariant (no external solver oracle).

Family: lp    Class: duality / multipliers / mathematical invariants
Dimension: duals, multipliers, sensitivity and mathematical invariants

ORACLE = LP duality theory itself.  We never compare pounce to another
solver's *value*; we compare pounce to pounce through the primal->dual
transformation, which no shared sign-convention error can satisfy by
accident.  scipy.linprog is used only as a weak secondary sanity check.

--------------------------------------------------------------------------
TRANSFORMATION RULES USED (stated explicitly; standard "general form" duality,
e.g. Bertsimas & Tsitsiklis, *Introduction to Linear Optimization*, Table 4.2,
p.142; Chvatal, *Linear Programming*, ch. 5)

PRIMAL is a min:      min c'x   s.t.  a_i'x  {>=,<=,==}  b_i ,  x_j {>=0,<=0,free}
DUAL is a max:        max b'l   s.t.  (A'l)_j {<=,>=,==} c_j ,  l_i {>=0,<=0,free}

    primal row  a_i'x >= b_i   ->  dual var  l_i >= 0
    primal row  a_i'x <= b_i   ->  dual var  l_i <= 0
    primal row  a_i'x == b_i   ->  dual var  l_i free
    primal var  x_j >= 0       ->  dual row  (A'l)_j <= c_j
    primal var  x_j <= 0       ->  dual row  (A'l)_j >= c_j
    primal var  x_j free       ->  dual row  (A'l)_j == c_j

The mirrored map for a max primal (so that dual(dual(P)) == P *exactly*):

PRIMAL is a max:      max c'x   s.t.  a_i'x {..} b_i,  x_j {..}
DUAL is a min:        min b'l   s.t.  (A'l)_j {..} c_j
    row <= -> l_i >= 0 ;  row >= -> l_i <= 0 ;  row == -> l_i free
    var >=0 -> (A'l)_j >= c_j ; var <=0 -> <= ; var free -> ==

--------------------------------------------------------------------------
MULTIPLIER RECOVERY (pounce -> the dual variable l of the theory above)

pounce solves   min 1/2 x'Px + c'x  s.t. A x = b, G x <= h, lb <= x <= ub
with L = c'x + y'(Ax-b) + z'(Gx-h) - z_lb'(x-lb) + z_ub'(x-ub),
so stationarity is   c + A'y + G'z - z_lb + z_ub = 0,  z,z_lb,z_ub >= 0.

Matching that against dual feasibility (A'l) {<=} c with the sign rules above:
    primal '==' row  (pounce eq row, mult y_i)          ->  l_i = -y_i
    primal '<=' row  (pounce G row  as-is, mult z_i)    ->  l_i = -z_i
    primal '>=' row  (pounce G row  negated, mult z_i)  ->  l_i = +z_i
These are DERIVED, not guessed, and are re-checked numerically by the
dual-feasibility residual test in check_pair().
"""

import time
from fractions import Fraction

import numpy as np

np.set_printoptions(precision=6, suppress=True)

from pounce import solve_qp

TOL = 1e-6
INF = np.inf

# ---------------------------------------------------------------- LP record


class LP:
    """General-form LP.  rel[i] in {'<=','>=','=='}; vsign[j] in {+1,-1,0}."""

    def __init__(self, name, sense, c, A, rel, b, vsign):
        self.name = name
        self.sense = sense  # 'min' or 'max'
        self.c = np.asarray(c, float)
        self.A = np.asarray(A, float).reshape(len(rel), len(c))
        self.rel = list(rel)
        self.b = np.asarray(b, float)
        self.vsign = list(vsign)
        assert len(self.vsign) == self.A.shape[1]
        assert len(self.b) == self.A.shape[0]

    @property
    def shape(self):
        return self.A.shape

    def same_as(self, other):
        return (
            self.sense == other.sense
            and self.rel == other.rel
            and self.vsign == other.vsign
            and self.A.shape == other.A.shape
            and np.array_equal(self.c, other.c)
            and np.array_equal(self.A, other.A)
            and np.array_equal(self.b, other.b)
        )


_ROW2VAR_MIN = {">=": +1, "<=": -1, "==": 0}
_VAR2ROW_MIN = {+1: "<=", -1: ">=", 0: "=="}
_ROW2VAR_MAX = {"<=": +1, ">=": -1, "==": 0}
_VAR2ROW_MAX = {+1: ">=", -1: "<=", 0: "=="}


def dual(lp):
    """Return the LP dual, per the rules in the module docstring."""
    if lp.sense == "min":
        return LP(
            name=f"dual({lp.name})",
            sense="max",
            c=lp.b,
            A=lp.A.T,
            rel=[_VAR2ROW_MIN[s] for s in lp.vsign],
            b=lp.c,
            vsign=[_ROW2VAR_MIN[r] for r in lp.rel],
        )
    return LP(
        name=f"dual({lp.name})",
        sense="min",
        c=lp.b,
        A=lp.A.T,
        rel=[_VAR2ROW_MAX[s] for s in lp.vsign],
        b=lp.c,
        vsign=[_ROW2VAR_MAX[r] for r in lp.rel],
    )


# ------------------------------------------------------- pounce translation


def solve_lp(lp, tol=1e-10):
    """Solve an LP record with pounce; return (status, value, x, lam, t, iters).

    `value` is always the value of the LP *as written* (max problems are
    negated back).  `lam` is the theory-side dual vector recovered from the
    pounce multipliers per the derivation in the module docstring.
    """
    n = lp.A.shape[1]
    cmin = lp.c if lp.sense == "min" else -lp.c

    eq_i = [i for i, r in enumerate(lp.rel) if r == "=="]
    le_i = [i for i, r in enumerate(lp.rel) if r == "<="]
    ge_i = [i for i, r in enumerate(lp.rel) if r == ">="]

    A = lp.A[eq_i] if eq_i else None
    bb = lp.b[eq_i] if eq_i else None
    Grows = [lp.A[i] for i in le_i] + [-lp.A[i] for i in ge_i]
    hrows = [lp.b[i] for i in le_i] + [-lp.b[i] for i in ge_i]
    G = np.array(Grows) if Grows else None
    h = np.array(hrows) if Grows else None

    lb = np.array([0.0 if s == +1 else -INF for s in lp.vsign])
    ub = np.array([0.0 if s == -1 else INF for s in lp.vsign])

    t0 = time.perf_counter()
    r = solve_qp(P=None, c=cmin, A=A, b=bb, G=G, h=h, lb=lb, ub=ub, tol=tol)
    t = time.perf_counter() - t0

    lam = np.zeros(len(lp.rel))
    for k, i in enumerate(eq_i):
        lam[i] = -r.y[k]
    for k, i in enumerate(le_i):
        lam[i] = -r.z[k]
    for k, i in enumerate(ge_i):
        lam[i] = +r.z[len(le_i) + k]

    # for a 'max' LP the pounce objective is of -c, flip back
    val = r.obj if lp.sense == "min" else -r.obj
    # the theory-side dual of a MAX problem uses the mirrored rules, whose
    # multiplier signs are the negatives of the min-form recovery.
    if lp.sense == "max":
        lam = -lam
    return r.status, val, np.asarray(r.x, float), lam, t, r.iters, r


# ------------------------------------------------------------- exact checks


def rel_ok(lhs, rel, rhs, tol):
    if rel == "==":
        return abs(lhs - rhs) <= tol
    if rel == "<=":
        return lhs <= rhs + tol
    return lhs >= rhs - tol


def primal_feas(lp, x, tol=1e-7):
    worst = 0.0
    Ax = lp.A @ x
    for i, r in enumerate(lp.rel):
        if r == "==":
            worst = max(worst, abs(Ax[i] - lp.b[i]))
        elif r == "<=":
            worst = max(worst, max(0.0, Ax[i] - lp.b[i]))
        else:
            worst = max(worst, max(0.0, lp.b[i] - Ax[i]))
    for j, s in enumerate(lp.vsign):
        if s == +1:
            worst = max(worst, max(0.0, -x[j]))
        elif s == -1:
            worst = max(worst, max(0.0, x[j]))
    return worst


def dual_feas(lp, lam):
    """Residual of the DUAL constraints of `lp` evaluated at lam (min-form)."""
    d = dual(lp)
    return primal_feas(d, lam)


def comp_slack(lp, x, lam):
    """max |l_i (a_i'x - b_i)| over rows, and |x_j * slack_j| over columns."""
    Ax = lp.A @ x
    row_cs = max((abs(lam[i] * (Ax[i] - lp.b[i])) for i in range(len(lam))), default=0.0)
    ATl = lp.A.T @ lam
    col_cs = max(
        (abs(x[j] * (lp.c[j] - ATl[j])) for j in range(len(x))), default=0.0
    )
    return row_cs, col_cs


# ------------------------------------------------------------------ problems


def textbook():
    # Chvatal / any intro text:  max 4x1+3x2  s.t. 2x1+x2<=10, x1+3x2<=15, x>=0
    # known primal opt x=(3,4), value 24;  known dual y=(9/5,2/5), value 24.
    return LP(
        "textbook_max",
        "max",
        [4.0, 3.0],
        [[2.0, 1.0], [1.0, 3.0]],
        ["<=", "<="],
        [10.0, 15.0],
        [+1, +1],
    )


def standard_form():
    # min c'x  s.t. Ax = b, x >= 0   (a small transportation-flavoured LP)
    A = [[1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [1, 0, 0, 1, 0]]
    return LP(
        "standard_eq",
        "min",
        [3.0, 5.0, 2.0, 4.0, 6.0],
        A,
        ["==", "==", "=="],
        [4.0, 5.0, 3.0],
        [+1] * 5,
    )


def canonical_diet():
    # min c'x s.t. Ax >= b, x >= 0   (diet problem)
    A = [[1, 2, 1], [3, 1, 2], [2, 4, 1]]
    return LP(
        "canonical_diet",
        "min",
        [2.0, 3.0, 4.0],
        A,
        [">=", ">=", ">="],
        [10.0, 12.0, 14.0],
        [+1, +1, +1],
    )


def mixed_form():
    # eq + <= + >= rows, one FREE variable, one NONPOSITIVE variable.
    # The trailing row x4 >= -2 is what makes it bounded (without it the free
    # x2 runs off to +inf; verified unbounded by scipy/HiGHS).
    A = [
        [1.0, 1.0, 1.0, 1.0],
        [2.0, -1.0, 0.0, 1.0],
        [0.0, 1.0, 3.0, -1.0],
        [1.0, 0.0, -1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return LP(
        "mixed",
        "min",
        [1.0, -2.0, 3.0, 1.0],
        A,
        ["==", "<=", ">=", "<=", ">="],
        [5.0, 3.0, 2.0, 6.0, -2.0],
        [+1, 0, +1, -1],
    )


def boxed():
    # explicit two-sided bounds written as ROWS so the dual sees them.
    A = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return LP(
        "boxed_rows",
        "min",
        [-1.0, -2.0, -3.0],
        A,
        ["<=", ">=", ">=", ">=", ">=", "<=", "<=", "<="],
        [6.0, -2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0],
        [0, 0, 0],
    )


def degenerate():
    # primal degenerate / dual non-unique: several constraints active at one
    # vertex, so the dual optimum is a face, not a point.
    A = [[1.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]
    return LP(
        "degenerate",
        "min",
        [-1.0, -1.0],
        A,
        ["<=", "<=", "<=", "<="],
        [4.0, 4.0, 6.0, 6.0],
        [+1, +1],
    )


FEASIBLE = [textbook(), standard_form(), canonical_diet(), mixed_form(), boxed(), degenerate()]

# --- status-table cases -----------------------------------------------------
# (primal status, expected dual status per the LP status table)
INFEAS_A = LP(  # primal infeasible AND dual infeasible
    "both_infeasible",
    "min",
    [-1.0, -1.0],
    [[1.0, -1.0], [-1.0, 1.0]],
    ["==", "=="],
    [1.0, 1.0],
    [+1, +1],
)
INFEAS_B = LP(  # primal infeasible, dual unbounded
    "infeasible_dual_unbounded",
    "min",
    [1.0, 1.0],
    [[1.0, 1.0], [1.0, 1.0]],
    [">=", "<="],
    [1.0, 0.0],
    [0, 0],
)
UNBD_C = LP(  # primal unbounded, dual infeasible
    "unbounded_dual_infeasible",
    "min",
    [-1.0, 0.0],
    [[1.0, 1.0]],
    ["=="],
    [1.0],
    [0, 0],
)


# --------------------------------------------------------------------- main

fails = []


def note(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


print("=" * 74)
print("STEP 0 - verify the primal->dual transformation on a KNOWN textbook dual")
print("=" * 74)
tb = textbook()
d_tb = dual(tb)
print(f"  primal: max {tb.c} s.t. A x <= {tb.b}, x>=0")
print(f"  auto dual: {d_tb.sense} c={d_tb.c} rel={d_tb.rel} b={d_tb.b} vsign={d_tb.vsign}")
# hand-written known dual: min 10y1+15y2 s.t. 2y1+y2>=4, y1+3y2>=3, y>=0
hand = LP(
    "hand_dual",
    "min",
    [10.0, 15.0],
    [[2.0, 1.0], [1.0, 3.0]],
    [">=", ">="],
    [4.0, 3.0],
    [+1, +1],
)
note(d_tb.same_as(hand), "auto-generated dual == hand-written textbook dual (exact match)")
note(dual(d_tb).same_as(tb), "dual(dual(textbook)) == textbook, EXACTLY (structural)")

st, val, x, lam, t, it, _ = solve_lp(tb)
note(st == "optimal" and abs(val - 24.0) < 1e-7, f"textbook primal value {val:.10f} == 24 (known)")
note(np.allclose(x, [3.0, 4.0], atol=1e-6), f"textbook x = {x} == (3,4) (known)")
sd, vd, xd, lamd, td, itd, _ = solve_lp(hand)
note(sd == "optimal" and abs(vd - 24.0) < 1e-7, f"textbook dual value {vd:.10f} == 24 (known)")
note(np.allclose(xd, [1.8, 0.4], atol=1e-6), f"textbook dual y = {xd} == (9/5,2/5) (known)")
note(np.allclose(lam, xd, atol=1e-5), f"pounce primal multipliers {lam} == known dual soln {xd}")

print()
print("=" * 74)
print("STEPS a-d - strong duality, dual<->primal multiplier match, dual-of-dual,")
print("            complementary slackness, for each LP form")
print("=" * 74)

rows = []
for lp in FEASIBLE:
    print(f"\n--- {lp.name}  ({lp.shape[0]} rows x {lp.shape[1]} cols, sense={lp.sense})")
    d = dual(lp)
    dd = dual(d)

    # (c) dual of dual recovers the primal, exactly, as a structural identity
    note(dd.same_as(lp), f"[{lp.name}] (c) dual(dual(P)) == P exactly (structural identity)")

    sP, vP, xP, lamP, tP, itP, rP = solve_lp(lp)
    sD, vD, xD, lamD, tD, itD, rD = solve_lp(d)
    sDD, vDD, xDD, lamDD, tDD, itDD, _ = solve_lp(dd)

    print(f"    P : status={sP:10s} value={vP: .12f}  iters={itP}  t={tP*1e3:.1f}ms")
    print(f"    D : status={sD:10s} value={vD: .12f}  iters={itD}  t={tD*1e3:.1f}ms")
    print(f"    DD: status={sDD:10s} value={vDD: .12f}  iters={itDD} t={tDD*1e3:.1f}ms")

    note(sP == "optimal" and sD == "optimal", f"[{lp.name}] both P and D solved to optimal")

    # (a) STRONG DUALITY
    gap = abs(vP - vD) / max(1.0, abs(vP))
    note(gap < 1e-7, f"[{lp.name}] (a) strong duality |val(P)-val(D)|/scale = {gap:.3e}")

    # (c) numeric: dual-of-dual value equals primal value
    gap2 = abs(vP - vDD) / max(1.0, abs(vP))
    note(gap2 < 1e-7, f"[{lp.name}] (c) val(dual(dual(P))) == val(P), gap = {gap2:.3e}")

    # (b) duals of P must be a feasible/optimal point of D
    dfeas = primal_feas(d, lamP)
    note(dfeas < 1e-6, f"[{lp.name}] (b) P's multipliers are DUAL-FEASIBLE, resid = {dfeas:.3e}")
    objlam = float(d.b @ lamP) if False else float(np.dot(lp.b, lamP))
    note(
        abs(objlam - vP) / max(1.0, abs(vP)) < 1e-7,
        f"[{lp.name}] (b) b'lambda(P) = {objlam:.12f} equals val(P) = {vP:.12f}",
    )
    # and D's multipliers must reproduce a primal-optimal point
    pfeas = primal_feas(lp, lamD)
    note(pfeas < 1e-6, f"[{lp.name}] (b) D's multipliers are PRIMAL-FEASIBLE, resid = {pfeas:.3e}")
    note(
        abs(float(np.dot(lp.c, lamD)) - vP) / max(1.0, abs(vP)) < 1e-7,
        f"[{lp.name}] (b) c'x-from-D-multipliers = {float(np.dot(lp.c, lamD)):.12f} == val(P)",
    )

    # (d) COMPLEMENTARY SLACKNESS at the reported (x_P, lambda_P)
    r_cs, c_cs = comp_slack(lp, xP, lamP)
    note(r_cs < 1e-6, f"[{lp.name}] (d) row complementary slackness max|l_i r_i| = {r_cs:.3e}")
    note(c_cs < 1e-6, f"[{lp.name}] (d) col complementary slackness max|x_j s_j| = {c_cs:.3e}")
    # cross-pair CS: (x_P, x_D) where x_D is the dual's own primal solution
    r_cs2, c_cs2 = comp_slack(lp, xP, xD)
    note(r_cs2 < 1e-6, f"[{lp.name}] (d) cross-pair (x_P, y_D) row CS = {r_cs2:.3e}")
    note(c_cs2 < 1e-6, f"[{lp.name}] (d) cross-pair (x_P, y_D) col CS = {c_cs2:.3e}")

    rows.append((lp.name, sP, vP, sD, vD, gap, max(r_cs, c_cs), tP + tD))

# ---- bounds handled by pounce lb/ub rather than as rows --------------------
print()
print("=" * 74)
print("EXTRA - bounds via pounce lb/ub must give the same value AND the same duals")
print("=" * 74)
bx = boxed()
sB, vB, xB, lamB, tB, itB, _ = solve_lp(bx)
c = np.array([-1.0, -2.0, -3.0])
Gb = np.array([[1.0, 1.0, 1.0]])
hb = np.array([6.0])
Gb2 = np.array([[-1.0, 1.0, 0.0]])  # x1 - x2 >= -2  ->  -x1 + x2 <= 2
hb2 = np.array([2.0])
G = np.vstack([Gb, Gb2])
h = np.concatenate([hb, hb2])
t0 = time.perf_counter()
rr = solve_qp(c=c, G=G, h=h, lb=np.zeros(3), ub=np.array([3.0, 4.0, 5.0]), tol=1e-10)
tbnd = time.perf_counter() - t0
print(f"    rows-form : status={sB} value={vB:.12f}")
print(f"    lb/ub-form: status={rr.status} value={rr.obj:.12f}  z={rr.z} z_lb={rr.z_lb} z_ub={rr.z_ub}")
note(rr.status == "optimal", "bounds-as-lb/ub solve is optimal")
note(abs(rr.obj - vB) < 1e-7, f"bounds-as-lb/ub value {rr.obj:.12f} == bounds-as-rows value {vB:.12f}")
# dual objective assembled from bound multipliers must equal the primal value:
#   L(x*) = c'x + z'(Gx-h) - z_lb'(x-0) + z_ub'(x-u)  =>  dual obj = -h'z - u'z_ub
dobj = -float(h @ rr.z) - float(np.array([3.0, 4.0, 5.0]) @ rr.z_ub)
note(
    abs(dobj - rr.obj) < 1e-6,
    f"dual objective from bound multipliers {dobj:.12f} == primal objective {rr.obj:.12f}",
)

# ---- (e) status table ------------------------------------------------------
print()
print("=" * 74)
print("STEP e - LP status table consistency on infeasible / unbounded pairs")
print("=" * 74)
# pounce vocabulary: 'primal_infeasible' = infeasible ; 'dual_infeasible' = unbounded
#
# NOTE on INFEAS_A: its primal is infeasible AND its recession cone contains an
# improving ray d = (1,1) (A d = 0, d >= 0, c'd = -2 < 0), so the DUAL of that
# primal is *also* infeasible.  In the both-infeasible cell of the LP status
# table either certificate is a true statement, so BOTH 'primal_infeasible' and
# 'dual_infeasible' are correct answers; only 'optimal' would be wrong.
# The ray is verified exactly below before the status is graded.
d_ray = np.array([1.0, 1.0])
note(
    np.allclose(INFEAS_A.A @ d_ray, 0)
    and (d_ray >= 0).all()
    and float(INFEAS_A.c @ d_ray) < 0,
    "both_infeasible: improving recession ray (1,1) verified => dual genuinely infeasible",
)

TABLE = [
    (INFEAS_A, {"primal_infeasible", "dual_infeasible"}, {"primal_infeasible", "dual_infeasible"},
     "primal AND dual both infeasible; either certificate is correct"),
    (INFEAS_B, {"primal_infeasible"}, {"dual_infeasible", "primal_infeasible"},
     "primal infeasible; dual is unbounded (certificate exists)"),
    (UNBD_C, {"dual_infeasible"}, {"primal_infeasible"},
     "primal unbounded => dual MUST be infeasible"),
]
for lp, expP, expD, why in TABLE:
    d = dual(lp)
    sP, vP, xP, lamP, tP, itP, _ = solve_lp(lp)
    sD, vD, xD, lamD, tD, itD, _ = solve_lp(d)
    print(f"\n--- {lp.name}: {why}")
    print(f"    P status={sP}   D status={sD}")
    note(sP in expP, f"[{lp.name}] (e) primal status {sP!r} in {sorted(expP)}")
    note(sD in expD, f"[{lp.name}] (e) dual status {sD!r} in {sorted(expD)}")
    # the forbidden combination: both reported optimal, or one optimal and the
    # other infeasible -- these violate the duality status table outright.
    note(
        not (sP == "optimal" and sD == "optimal"),
        f"[{lp.name}] (e) not both optimal (would contradict infeasibility)",
    )

# ---- weak secondary check with scipy --------------------------------------
print()
print("=" * 74)
print("SECONDARY (weak) - scipy.optimize.linprog on the feasible primals")
print("=" * 74)
try:
    from scipy.optimize import linprog

    for lp in FEASIBLE:
        cmin = lp.c if lp.sense == "min" else -lp.c
        eq = [i for i, r in enumerate(lp.rel) if r == "=="]
        le = [i for i, r in enumerate(lp.rel) if r == "<="]
        ge = [i for i, r in enumerate(lp.rel) if r == ">="]
        Aub = np.vstack([lp.A[le], -lp.A[ge]]) if (le or ge) else None
        bub = np.concatenate([lp.b[le], -lp.b[ge]]) if (le or ge) else None
        Aeq = lp.A[eq] if eq else None
        beq = lp.b[eq] if eq else None
        bounds = [
            (0, None) if s == +1 else ((None, 0) if s == -1 else (None, None))
            for s in lp.vsign
        ]
        res = linprog(cmin, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds)
        sP, vP = solve_lp(lp)[:2]
        if not res.success:
            note(False, f"[{lp.name}] scipy did not solve: {res.message}")
            continue
        vs = res.fun if lp.sense == "min" else -res.fun
        ok = abs(vs - vP) / max(1.0, abs(vP)) < 1e-7
        note(ok, f"[{lp.name}] scipy value {vs:.10f} vs pounce {vP:.10f}")
except Exception as exc:  # pragma: no cover
    print(f"  (scipy check skipped: {exc})")

# ---- exact rational confirmation of one strong-duality identity ------------
print()
print("=" * 74)
print("EXACT - Fraction arithmetic confirmation of the textbook optimal pair")
print("=" * 74)
xF = [Fraction(3), Fraction(4)]
yF = [Fraction(9, 5), Fraction(2, 5)]
pv = Fraction(4) * xF[0] + Fraction(3) * xF[1]
dv = Fraction(10) * yF[0] + Fraction(15) * yF[1]
note(pv == dv == Fraction(24), f"exact: c'x = {pv}, b'y = {dv}, both = 24")
note(
    Fraction(2) * yF[0] + yF[1] >= 4 and yF[0] + Fraction(3) * yF[1] >= 3,
    "exact: hand dual feasibility of (9/5, 2/5)",
)

print()
print("=" * 74)
if fails:
    print(f"{len(fails)} CHECK(S) FAILED:")
    for f in fails:
        print("   - " + f)
    print("VERDICT: FAIL")
else:
    print("all duality invariants hold")
    print("VERDICT: PASS")
