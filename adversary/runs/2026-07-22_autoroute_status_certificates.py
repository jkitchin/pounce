"""Adversary cross-check: STATUS survival across the auto-routing boundary.

Family: autoroute   Class: infeasibility / unboundedness / status reporting

Question: for infeasible and unbounded instances of each detectable structure
(LP, convex QP, convex QCQP/SOCP), does ``solver_selection="auto"`` report the
SAME semantic status as forced ``solver_selection="nlp"`` and as an analytic
Farkas / recession-direction certificate (plus scipy / cvxpy oracles)?

Specific risk under test: a specialized engine's infeasibility/unboundedness
certificate is mistranslated at the routing boundary -- e.g. the LP-IPM emits
"primal infeasible" and the facade reports "unbounded", or a merged
"infeasible_or_unbounded" is flattened into a definite (and wrong) status.

Analytic ground truth for every case is written into the CASES table below:
  * INFEASIBLE cases carry an explicit Farkas certificate y >= 0 with
    A^T y = 0 and b^T y < 0 (empty feasible set), verified numerically here.
  * UNBOUNDED cases carry an explicit recession direction d with A d <= 0 and
    c^T d < 0 from a feasible point x0 (objective -> -inf along x0 + t d),
    verified numerically here.

Semantic buckets used for comparison (raw integer codes intentionally differ
between the convex path (scipy-style 2/3) and the NLP path (Ipopt
ApplicationReturnStatus 2/4), so we compare MEANING, not the integer):
  INFEASIBLE / UNBOUNDED / OPTIMAL / OTHER
"""

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import pounce  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

TOL = 1e-7


# --------------------------------------------------------------------------
# Semantic bucketing of a pounce OptimizeResult, for BOTH paths.
# --------------------------------------------------------------------------
def bucket(res):
    """Map an OptimizeResult onto INFEASIBLE / UNBOUNDED / OPTIMAL / OTHER.

    Convex (routed) path: info['status'] is the raw certificate string
    ('primal_infeasible' / 'dual_infeasible' / 'optimal' / ...).
    NLP path: res.status is an Ipopt ApplicationReturnStatus
      0 Solve_Succeeded, 1 Solved_To_Acceptable_Level,
      2 Infeasible_Problem_Detected, 3 Search_Direction_Becomes_Too_Small,
      4 Diverging_Iterates (== unbounded), 5 User_Requested_Stop,
      6 Feasible_Point_Found, ...
    """
    info = getattr(res, "info", None) or {}
    raw = str(info.get("status", ""))
    if raw in ("primal_infeasible", "dual_infeasible", "optimal",
               "optimal_inaccurate", "iteration_limit", "numerical_failure"):
        # convex/conic routed path
        return {
            "primal_infeasible": "INFEASIBLE",
            "dual_infeasible": "UNBOUNDED",
            "optimal": "OPTIMAL",
            "optimal_inaccurate": "OPTIMAL",
        }.get(raw, "OTHER"), f"convex:{raw}"
    code = int(res.status)
    nlp = {0: "OPTIMAL", 1: "OPTIMAL", 2: "INFEASIBLE", 4: "UNBOUNDED"}
    return nlp.get(code, "OTHER"), f"nlp:code={code}"


def which_engine(res):
    info = getattr(res, "info", None) or {}
    raw = str(info.get("status", ""))
    return "convex" if raw and not raw.isdigit() and "_" in raw or raw in (
        "optimal",) else "nlp"


# --------------------------------------------------------------------------
# Case definitions.  Each is expressed in the SAME algebra for pounce
# (callables + LinearConstraint/dicts), for the analytic certificate, and for
# the external oracle.
# --------------------------------------------------------------------------
CASES = []


def lincon(A, lb, ub):
    from scipy.optimize import LinearConstraint

    return LinearConstraint(np.asarray(A, float), lb, ub)


# ---- 1. LP, primal infeasible -------------------------------------------
# min  x0 + x1
# s.t. x0 + x1 >= 3
#      x0 + x1 <= 1          <-- contradiction
#      x free
# Farkas: y = (1, 1) on rows (-(x0+x1) <= -3), (x0+x1 <= 1):
#   A_ub = [[-1,-1],[1,1]], b_ub = (-3, 1);  A_ub^T y = 0, b_ub^T y = -2 < 0.
CASES.append(dict(
    name="LP infeasible (contradictory parallel halfspaces)",
    structure="LP",
    truth="INFEASIBLE",
    n=2,
    fun=lambda x: x[0] + x[1],
    jac=lambda x: np.array([1.0, 1.0]),
    hess=lambda x: np.zeros((2, 2)),
    x0=np.array([0.0, 0.0]),
    bounds=None,
    cons=[lincon([[1.0, 1.0]], 3.0, np.inf),
          lincon([[1.0, 1.0]], -np.inf, 1.0)],
    farkas=dict(A_ub=[[-1.0, -1.0], [1.0, 1.0]], b_ub=[-3.0, 1.0],
                y=[1.0, 1.0]),
    lp=dict(c=[1.0, 1.0], A_ub=[[-1.0, -1.0], [1.0, 1.0]], b_ub=[-3.0, 1.0],
            bounds=[(None, None), (None, None)]),
))

# ---- 2. LP, unbounded (dual infeasible) ---------------------------------
# min  -x0 - x1
# s.t. x0 - x1 <= 1
#      x0, x1 >= 0
# Recession direction d = (1,1): A d = 0 <= 0, d >= 0, c^T d = -2 < 0.
CASES.append(dict(
    name="LP unbounded (recession ray along x0=x1)",
    structure="LP",
    truth="UNBOUNDED",
    n=2,
    fun=lambda x: -x[0] - x[1],
    jac=lambda x: np.array([-1.0, -1.0]),
    hess=lambda x: np.zeros((2, 2)),
    x0=np.array([0.5, 0.5]),
    bounds=[(0.0, None), (0.0, None)],
    cons=[lincon([[1.0, -1.0]], -np.inf, 1.0)],
    recession=dict(x_feas=[0.5, 0.5], d=[1.0, 1.0], c=[-1.0, -1.0],
                   A_ub=[[1.0, -1.0]], b_ub=[1.0], lb=[0.0, 0.0]),
    lp=dict(c=[-1.0, -1.0], A_ub=[[1.0, -1.0]], b_ub=[1.0],
            bounds=[(0.0, None), (0.0, None)]),
))

# ---- 3. Convex QP, primal infeasible ------------------------------------
# min  0.5*(x0^2 + x1^2)         (strictly convex, PD Hessian)
# s.t. x0 + x1 >= 2
#      x0 + x1 <= 1              <-- same Farkas certificate as case 1
CASES.append(dict(
    name="Convex QP infeasible (PD Hessian, empty polyhedron)",
    structure="QP",
    truth="INFEASIBLE",
    n=2,
    fun=lambda x: 0.5 * (x[0] ** 2 + x[1] ** 2),
    jac=lambda x: np.array([x[0], x[1]]),
    hess=lambda x: np.eye(2),
    x0=np.array([0.0, 0.0]),
    bounds=None,
    cons=[lincon([[1.0, 1.0]], 2.0, np.inf),
          lincon([[1.0, 1.0]], -np.inf, 1.0)],
    farkas=dict(A_ub=[[-1.0, -1.0], [1.0, 1.0]], b_ub=[-2.0, 1.0],
                y=[1.0, 1.0]),
    lp=dict(c=[0.0, 0.0], A_ub=[[-1.0, -1.0], [1.0, 1.0]], b_ub=[-2.0, 1.0],
            bounds=[(None, None), (None, None)]),
))

# ---- 4. Convex QP, unbounded (PSD-singular Hessian) ---------------------
# min  0.5*x0^2 - x1          (Hessian diag(1,0) is PSD but singular)
# s.t. x0 + x1 >= 0,   x free
# Recession direction d = (0,1): A d = -1 <= 0 in <= form, objective
# 0.5*x0^2 - (x1 + t) -> -inf.  Convex and unbounded below.
CASES.append(dict(
    name="Convex QP unbounded (singular PSD Hessian, linear ray)",
    structure="QP",
    truth="UNBOUNDED",
    n=2,
    fun=lambda x: 0.5 * x[0] ** 2 - x[1],
    jac=lambda x: np.array([x[0], -1.0]),
    hess=lambda x: np.array([[1.0, 0.0], [0.0, 0.0]]),
    x0=np.array([0.0, 0.0]),
    bounds=None,
    cons=[lincon([[1.0, 1.0]], 0.0, np.inf)],
    recession=dict(x_feas=[0.0, 0.0], d=[0.0, 1.0], c=[0.0, -1.0],
                   A_ub=[[-1.0, -1.0]], b_ub=[0.0], lb=[-np.inf, -np.inf]),
    lp=None,
))

# ---- 5. Convex QCQP / SOCP, primal infeasible ---------------------------
# min  x0
# s.t. x0^2 + x1^2 <= 1        (unit disk)
#      x0 >= 2                 (halfspace disjoint from the disk)
# Certificate: on the disk |x0| <= 1 < 2, so the feasible set is empty.
CASES.append(dict(
    name="Convex QCQP infeasible (disk vs disjoint halfspace)",
    structure="QCQP",
    truth="INFEASIBLE",
    n=2,
    fun=lambda x: x[0],
    jac=lambda x: np.array([1.0, 0.0]),
    hess=lambda x: np.zeros((2, 2)),
    x0=np.array([0.0, 0.0]),
    bounds=None,
    cons=[
        dict(type="ineq",
             fun=lambda x: np.array([1.0 - x[0] ** 2 - x[1] ** 2]),
             jac=lambda x: np.array([[-2.0 * x[0], -2.0 * x[1]]])),
        lincon([[1.0, 0.0]], 2.0, np.inf),
    ],
    disjoint=dict(kind="disk_vs_halfspace", radius=1.0, halfspace_at=2.0),
    lp=None,
))

# ---- 6. Convex QCQP, unbounded ------------------------------------------
# min  -x1
# s.t. x0^2 - x1 <= 0          (epigraph of a parabola; convex)
# Recession: (0, t) feasible for all t >= 0 from (0,0); objective -> -inf.
CASES.append(dict(
    name="Convex QCQP unbounded (parabola epigraph, vertical ray)",
    structure="QCQP",
    truth="UNBOUNDED",
    n=2,
    fun=lambda x: -x[1],
    jac=lambda x: np.array([0.0, -1.0]),
    hess=lambda x: np.zeros((2, 2)),
    x0=np.array([0.0, 0.0]),
    bounds=None,
    cons=[dict(type="ineq",
               fun=lambda x: np.array([x[1] - x[0] ** 2]),
               jac=lambda x: np.array([[-2.0 * x[0], 1.0]]))],
    ray=dict(x_feas=[0.0, 0.0], d=[0.0, 1.0]),
    lp=None,
))

# ---- 7. REVERSE TEST: infeasible NONLINEAR problem (auto must fall
#         through to NLP and still say INFEASIBLE) --------------------------
# min  x0
# s.t. exp(x0) + x1^4 <= -1     (LHS > 0 always -> empty feasible set)
CASES.append(dict(
    name="Nonconvex/nonlinear infeasible (auto falls through to NLP)",
    structure="NLP-fallthrough",
    truth="INFEASIBLE",
    n=2,
    fun=lambda x: x[0],
    jac=lambda x: np.array([1.0, 0.0]),
    hess=lambda x: np.zeros((2, 2)),
    x0=np.array([0.0, 0.0]),
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    cons=[dict(type="ineq",
               fun=lambda x: np.array([-1.0 - np.exp(x[0]) - x[1] ** 4]),
               jac=lambda x: np.array([[-np.exp(x[0]), -4.0 * x[1] ** 3]]))],
    always_positive=True,
    lp=None,
))


# ---- 8. THE AMBIGUITY TRAP: primal infeasible AND dual infeasible -------
# min  -x0 - x1
# s.t. x0 + x1 >= 3
#      x0 + x1 <= 1        <-- empty feasible set (Farkas y=(1,1))
# The objective ALSO has a dual-infeasibility direction d=(1,1) (c^T d < 0,
# A d = 0), so an HSDE can produce either certificate.  The correct verdict is
# INFEASIBLE: the feasible set is empty, so "unbounded below over the feasible
# region" is vacuously false.  Reporting UNBOUNDED here is the exact
# "merged infeasible_or_unbounded flattened to the wrong definite status" bug.
CASES.append(dict(
    name="LP primal-infeasible AND dual-infeasible (ambiguity trap)",
    structure="LP-ambig",
    truth="INFEASIBLE",
    n=2,
    fun=lambda x: -x[0] - x[1],
    jac=lambda x: np.array([-1.0, -1.0]),
    hess=lambda x: np.zeros((2, 2)),
    x0=np.array([0.0, 0.0]),
    bounds=None,
    cons=[lincon([[1.0, 1.0]], 3.0, np.inf),
          lincon([[1.0, 1.0]], -np.inf, 1.0)],
    farkas=dict(A_ub=[[-1.0, -1.0], [1.0, 1.0]], b_ub=[-3.0, 1.0],
                y=[1.0, 1.0]),
    lp=dict(c=[-1.0, -1.0], A_ub=[[-1.0, -1.0], [1.0, 1.0]], b_ub=[-3.0, 1.0],
            bounds=[(None, None), (None, None)]),
))

# ---- 9. Equality-form infeasible LP -------------------------------------
# min  x0 + x1 + x2
# s.t. x0 + x1 + x2 == 1
#      x0 + x1 + x2 == 2        <-- inconsistent
#      x >= 0
# Farkas (equality form): y = (1, -1),  A^T y = 0,  b^T y = -1 < 0.
CASES.append(dict(
    name="Equality-form LP infeasible (inconsistent Ax=b)",
    structure="LP-eq",
    truth="INFEASIBLE",
    n=3,
    fun=lambda x: x[0] + x[1] + x[2],
    jac=lambda x: np.ones(3),
    hess=lambda x: np.zeros((3, 3)),
    x0=np.zeros(3),
    bounds=[(0.0, None)] * 3,
    cons=[lincon([[1.0, 1.0, 1.0]], 1.0, 1.0),
          lincon([[1.0, 1.0, 1.0]], 2.0, 2.0)],
    farkas_eq=dict(A_eq=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], b_eq=[1.0, 2.0],
                   y=[1.0, -1.0]),
    lp=dict(c=[1.0, 1.0, 1.0], A_ub=[[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]],
            b_ub=[1.0, -2.0], bounds=[(0.0, None)] * 3),
))

# ---- 10. Bound-derived infeasibility (presolve should catch it) ---------
# min  x0 + x1
# s.t. 1 <= x0 <= 5,  1 <= x1 <= 5,  x0 + x1 <= 1.5
# min over the box of x0+x1 is 2 > 1.5 -> empty.
CASES.append(dict(
    name="Bound-derived LP infeasibility (box min exceeds row ub)",
    structure="LP-bnd",
    truth="INFEASIBLE",
    n=2,
    fun=lambda x: x[0] + x[1],
    jac=lambda x: np.ones(2),
    hess=lambda x: np.zeros((2, 2)),
    x0=np.array([1.0, 1.0]),
    bounds=[(1.0, 5.0), (1.0, 5.0)],
    cons=[lincon([[1.0, 1.0]], -np.inf, 1.5)],
    box_gap=dict(box_min=2.0, row_ub=1.5),
    lp=dict(c=[1.0, 1.0], A_ub=[[1.0, 1.0]], b_ub=[1.5],
            bounds=[(1.0, 5.0), (1.0, 5.0)]),
))

# ---- 11. Equality-constrained LP, unbounded -----------------------------
# min  -x0
# s.t. x0 - x1 == 0,   x free
# Ray d = (1,1): A d = 0, c^T d = -1 < 0.
CASES.append(dict(
    name="Equality-constrained LP unbounded (ray in null(A))",
    structure="LP-eq",
    truth="UNBOUNDED",
    n=2,
    fun=lambda x: -x[0],
    jac=lambda x: np.array([-1.0, 0.0]),
    hess=lambda x: np.zeros((2, 2)),
    x0=np.array([0.0, 0.0]),
    bounds=None,
    cons=[lincon([[1.0, -1.0]], 0.0, 0.0)],
    recession=dict(x_feas=[0.0, 0.0], d=[1.0, 1.0], c=[-1.0, 0.0],
                   A_ub=[[1.0, -1.0], [-1.0, 1.0]], b_ub=[0.0, 0.0],
                   lb=[-np.inf, -np.inf]),
    lp=dict(c=[-1.0, 0.0], A_ub=[[1.0, -1.0], [-1.0, 1.0]], b_ub=[0.0, 0.0],
            bounds=[(None, None), (None, None)]),
))


# --------------------------------------------------------------------------
# Verify the analytic certificates numerically (guards against MY algebra
# being wrong -- the #1 false-positive source).
# --------------------------------------------------------------------------
def check_certificate(case):
    if "farkas_eq" in case:
        f = case["farkas_eq"]
        A = np.asarray(f["A_eq"], float)
        b = np.asarray(f["b_eq"], float)
        y = np.asarray(f["y"], float)
        # x >= 0 form: A^T y <= 0 and b^T y > 0  (or the negation) certifies empty
        ok = np.abs(A.T @ y).max() < TOL and abs(b @ y) > TOL
        return ok, (f"Farkas(eq): |A^T y|={np.abs(A.T @ y).max():.1e}, "
                    f"b^T y={b @ y:.3f} != 0")
    if "box_gap" in case:
        g = case["box_gap"]
        return g["box_min"] > g["row_ub"] + TOL, \
            f"box min of row = {g['box_min']} > row ub {g['row_ub']} -> empty"
    if "farkas" in case:
        f = case["farkas"]
        A = np.asarray(f["A_ub"], float)
        b = np.asarray(f["b_ub"], float)
        y = np.asarray(f["y"], float)
        ok = (y >= -TOL).all() and np.abs(A.T @ y).max() < TOL and b @ y < -TOL
        return ok, f"Farkas: y>=0, |A^T y|={np.abs(A.T @ y).max():.1e}, b^T y={b @ y:.3f}<0"
    if "recession" in case:
        r = case["recession"]
        A = np.asarray(r["A_ub"], float)
        b = np.asarray(r["b_ub"], float)
        d = np.asarray(r["d"], float)
        c = np.asarray(r["c"], float)
        x = np.asarray(r["x_feas"], float)
        lb = np.asarray(r["lb"], float)
        feas = (A @ x <= b + TOL).all() and (x >= lb - TOL).all()
        rec = (A @ d <= TOL).all() and (d >= -TOL).all() if np.isfinite(lb).any() \
            else (A @ d <= TOL).all()
        return feas and rec and c @ d < -TOL, \
            f"recession: x0 feasible={feas}, A d<=0={rec}, c^T d={c @ d:.3f}<0"
    if case.get("disjoint"):
        d = case["disjoint"]
        return d["halfspace_at"] > d["radius"], \
            f"disk radius {d['radius']} < halfspace x0>={d['halfspace_at']} -> empty"
    if "ray" in case:
        r = case["ray"]
        x = np.asarray(r["x_feas"], float)
        d = np.asarray(r["d"], float)
        vals = [case["fun"](x + t * d) for t in (0, 10, 100, 1000)]
        cfeas = all(all(_eval_cons(case, x + t * d) >= -TOL)
                    for t in (0, 10, 100, 1000))
        return cfeas and vals[-1] < -100, \
            f"ray feasible={cfeas}, f along ray {vals[0]:.1f} -> {vals[-1]:.1f}"
    if case.get("always_positive"):
        rng = np.random.default_rng(0)
        pts = rng.uniform(-5, 5, size=(20000, 2))
        g = -1.0 - np.exp(pts[:, 0]) - pts[:, 1] ** 4
        return g.max() < -1.0, f"max g over 20000 box samples = {g.max():.3f} < 0"
    return None, "no analytic certificate"


def _eval_cons(case, x):
    out = []
    for c in case["cons"]:
        if isinstance(c, dict):
            out.extend(np.atleast_1d(c["fun"](x)))
    return np.asarray(out) if out else np.array([0.0])


# --------------------------------------------------------------------------
# External oracles
# --------------------------------------------------------------------------
def scipy_lp_oracle(case):
    if not case.get("lp"):
        return None
    d = case["lp"]
    r = linprog(d["c"], A_ub=d["A_ub"], b_ub=d["b_ub"], bounds=d["bounds"],
                method="highs")
    # highs status: 0 optimal, 2 infeasible, 3 unbounded
    return {0: "OPTIMAL", 2: "INFEASIBLE", 3: "UNBOUNDED"}.get(r.status, "OTHER")


def cvxpy_oracle(case):
    try:
        import cvxpy as cp
    except ImportError:
        return None
    x = cp.Variable(case["n"])
    st = case["structure"]
    if st == "LP-ambig":
        p = cp.Problem(cp.Minimize(-x[0] - x[1]),
                       [x[0] + x[1] >= 3, x[0] + x[1] <= 1])
    elif st == "LP-eq" and case["truth"] == "INFEASIBLE":
        p = cp.Problem(cp.Minimize(cp.sum(x)),
                       [cp.sum(x) == 1, cp.sum(x) == 2, x >= 0])
    elif st == "LP-eq":
        p = cp.Problem(cp.Minimize(-x[0]), [x[0] - x[1] == 0])
    elif st == "LP-bnd":
        p = cp.Problem(cp.Minimize(x[0] + x[1]),
                       [x >= 1, x <= 5, x[0] + x[1] <= 1.5])
    elif case["structure"] == "LP" and case["truth"] == "INFEASIBLE":
        p = cp.Problem(cp.Minimize(x[0] + x[1]),
                       [x[0] + x[1] >= 3, x[0] + x[1] <= 1])
    elif case["structure"] == "LP":
        p = cp.Problem(cp.Minimize(-x[0] - x[1]),
                       [x[0] - x[1] <= 1, x >= 0])
    elif case["structure"] == "QP" and case["truth"] == "INFEASIBLE":
        p = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x)),
                       [x[0] + x[1] >= 2, x[0] + x[1] <= 1])
    elif case["structure"] == "QP":
        p = cp.Problem(cp.Minimize(0.5 * cp.square(x[0]) - x[1]),
                       [x[0] + x[1] >= 0])
    elif case["structure"] == "QCQP" and case["truth"] == "INFEASIBLE":
        p = cp.Problem(cp.Minimize(x[0]),
                       [cp.sum_squares(x) <= 1, x[0] >= 2])
    elif case["structure"] == "QCQP":
        p = cp.Problem(cp.Minimize(-x[1]), [cp.square(x[0]) <= x[1]])
    else:
        return None
    try:
        p.solve(solver=cp.CLARABEL)
    except Exception as exc:  # pragma: no cover
        return f"ERROR({exc})"
    s = p.status
    if "infeasible" in s and "unbounded" in s:
        return "INF_OR_UNB"
    if "infeasible" in s:
        return "INFEASIBLE"
    if "unbounded" in s:
        return "UNBOUNDED"
    if "optimal" in s:
        return "OPTIMAL"
    return "OTHER"


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def run_pounce(case, selection):
    kw = dict(fun=case["fun"], x0=case["x0"], jac=case["jac"],
              hess=case["hess"], bounds=case["bounds"],
              constraints=case["cons"])
    t0 = time.perf_counter()
    try:
        res = pounce.minimize(**kw, options={"solver_selection": selection,
                                             "max_iter": 300})
    except Exception as exc:
        return None, f"EXC:{type(exc).__name__}: {exc}", time.perf_counter() - t0
    dt = time.perf_counter() - t0
    b, detail = bucket(res)
    return res, (b, detail), dt


rows = []
print("=" * 78)
print("AUTOROUTE STATUS-SURVIVAL PROBE (infeasible / unbounded certificates)")
print("=" * 78)

for i, case in enumerate(CASES, 1):
    print(f"\n--- [{i}] {case['name']}  ({case['structure']})")
    ok, msg = check_certificate(case)
    print(f"    analytic certificate: {msg}  -> valid={ok}")
    assert ok is not False, f"MY OWN CERTIFICATE FAILED for {case['name']}: {msg}"

    res_a, ba, ta = run_pounce(case, "auto")
    res_n, bn, tn = run_pounce(case, "nlp")
    sp = scipy_lp_oracle(case)
    cv = cvxpy_oracle(case)

    auto_b = ba[0] if isinstance(ba, tuple) else str(ba)
    auto_d = ba[1] if isinstance(ba, tuple) else str(ba)
    nlp_b = bn[0] if isinstance(bn, tuple) else str(bn)
    nlp_d = bn[1] if isinstance(bn, tuple) else str(bn)

    print(f"    truth (analytic)   : {case['truth']}")
    print(f"    pounce auto        : {auto_b:<12} [{auto_d}]  t={ta:.3f}s")
    print(f"    pounce nlp (forced): {nlp_b:<12} [{nlp_d}]  t={tn:.3f}s")
    print(f"    scipy linprog      : {sp}")
    print(f"    cvxpy CLARABEL     : {cv}")
    if res_a is not None:
        print(f"    auto message       : {getattr(res_a, 'message', '')!r}")
    if res_n is not None:
        print(f"    nlp  message       : {getattr(res_n, 'message', '')!r}")

    # classification
    agree = (auto_b == nlp_b)
    auto_right = (auto_b == case["truth"])
    nlp_right = (nlp_b == case["truth"])
    # Being unable to *prove* infeasible/unbounded (OTHER) is a limitation, not
    # a wrong answer.  Reporting the OPPOSITE definite status is a bug.
    auto_wrong = auto_b in ("INFEASIBLE", "UNBOUNDED", "OPTIMAL") and not auto_right
    nlp_wrong = nlp_b in ("INFEASIBLE", "UNBOUNDED", "OPTIMAL") and not nlp_right

    if auto_wrong:
        verdict = "SOLVER_BUG(auto reports definite wrong status)"
    elif not agree and (auto_wrong or nlp_wrong):
        verdict = "ROUTING_ERROR"
    elif not agree:
        verdict = "ROUTING_DISAGREE(one path weaker)"
    elif auto_right:
        verdict = "PASS"
    else:
        verdict = "SOLVER_LIMITATION(both paths inconclusive)"
    print(f"    => {verdict}")
    rows.append((i, case["name"], case["structure"], case["truth"], auto_b,
                 nlp_b, sp, cv, verdict, ta, tn))

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
hdr = f"{'#':<3}{'structure':<17}{'truth':<12}{'auto':<12}{'nlp':<12}{'scipy':<12}{'cvxpy':<12}verdict"
print(hdr)
for r in rows:
    print(f"{r[0]:<3}{r[2]:<17}{r[3]:<12}{r[4]:<12}{r[5]:<12}"
          f"{str(r[6]):<12}{str(r[7]):<12}{r[8]}")

bugs = [r for r in rows if r[8].startswith("SOLVER_BUG")]
routing = [r for r in rows if r[8].startswith("ROUTING")]
lims = [r for r in rows if r[8].startswith("SOLVER_LIMITATION")]

print()
if bugs:
    print(f"VERDICT: SOLVER_BUG ({len(bugs)} case(s) report a definite wrong status)")
elif routing:
    print(f"VERDICT: ROUTING_ERROR ({len(routing)} case(s) disagree across the boundary)")
elif lims:
    print(f"VERDICT: SOLVER_LIMITATION ({len(lims)} inconclusive)")
else:
    print("VERDICT: PASS")
