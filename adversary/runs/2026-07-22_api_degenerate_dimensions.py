"""Adversary cross-check: DEGENERATE DIMENSIONS across the public pounce API.

Family: api (contracts / option handling / input edge cases)
Class:  degenerate dimensions
Oracle: closed form + cvxpy (CLARABEL) + scipy.optimize.linprog

Each case is run in a subprocess-free but exception-guarded harness. A case
PASSes if either
  (a) pounce returns the mathematically correct answer, or
  (b) the input is genuinely invalid and pounce raises a clear, informative
      Python exception.
A case FAILs (SOLVER_BUG) on a Rust panic (pyo3 PanicException), a segfault,
a hang, or a silent wrong answer.
"""
import faulthandler
import signal
import sys
import time
import traceback

import numpy as np

faulthandler.enable()

import pounce
from pounce import solve_qp, solve_socp, minimize

try:
    import cvxpy as cp
    HAVE_CVXPY = True
except Exception:
    HAVE_CVXPY = False

from scipy.optimize import linprog

TOL = 1e-6
BUDGET_S = 10.0

RESULTS = []


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("case exceeded budget")


def case(name, fn):
    """Run one case. fn() -> (verdict_str, detail_str)."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(BUDGET_S))
    t0 = time.perf_counter()
    try:
        verdict, detail = fn()
    except Timeout:
        verdict, detail = "SOLVER_BUG", "HANG: exceeded %.0fs budget" % BUDGET_S
    except BaseException as e:  # noqa: BLE001 - want PanicException too
        tn = type(e).__name__
        if "Panic" in tn:
            verdict = "SOLVER_BUG"
            detail = "RUST PANIC (%s): %s" % (tn, e)
        else:
            verdict = "HARNESS_ERROR"
            detail = "%s: %s\n%s" % (tn, e, traceback.format_exc(limit=3))
    finally:
        signal.alarm(0)
    dt = time.perf_counter() - t0
    RESULTS.append((name, verdict, detail, dt))
    print("[%-9s] %-46s %6.3fs  %s" % (verdict, name, dt, detail.replace("\n", " | ")[:150]))


def clean_error(exc):
    """Is this a clear, informative Python-level exception (not a panic)?"""
    tn = type(exc).__name__
    if "Panic" in tn:
        return False, "RUST PANIC (%s): %s" % (tn, exc)
    msg = str(exc)
    informative = len(msg) > 8 and not msg.lower().startswith("none")
    return True, "%s(%s)%s" % (tn, msg[:160], "" if informative else "  [TERSE MESSAGE]")


def relerr(a, b):
    return abs(a - b) / max(1.0, abs(b))


# ---------------------------------------------------------------- cases -----

def c01_n1_unconstrained():
    """n=1: min 0.5 x^2 - x  ->  x*=1, f*=-0.5"""
    r = solve_qp(P=[[1.0]], c=[-1.0])
    ok = r.status == "optimal" and abs(r.x[0] - 1.0) < TOL and abs(r.obj + 0.5) < TOL
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s obj=%.12g (exact x=1, f=-0.5)" % (r.status, r.x, r.obj))


def c02_n1_with_bounds():
    """n=1 with an active bound: min 0.5x^2 - x, x <= 0.25 -> x=0.25, f=-0.21875"""
    r = solve_qp(P=[[1.0]], c=[-1.0], ub=[0.25])
    f = 0.5 * 0.25 ** 2 - 0.25
    ok = r.status == "optimal" and abs(r.x[0] - 0.25) < 1e-6 and abs(r.obj - f) < 1e-6
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s obj=%.12g (exact x=0.25 f=%.6g)" % (r.status, r.x, r.obj, f))


def c03_n1_equality_more_eq_than_vars():
    """n=1, m=3 identical consistent equalities: x=2 thrice. Over-determined."""
    A = np.ones((3, 1))
    b = np.array([2.0, 2.0, 2.0])
    r = solve_qp(P=[[1.0]], c=[0.0], A=A, b=b)
    ok = r.status == "optimal" and abs(r.x[0] - 2.0) < 1e-6
    return ("PASS" if ok else "SOLVER_LIMITATION",
            "status=%s x=%s obj=%.12g (exact x=2 f=2)" % (r.status, r.x, r.obj))


def c04_m0_constraints():
    """m=0: 3-var strictly convex QP, no constraints. x* = -P^-1 c."""
    P = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    c = np.array([-1.0, -2.0, -3.0])
    xs = np.linalg.solve(P, -c)
    fs = 0.5 * xs @ P @ xs + c @ xs
    r = solve_qp(P=P, c=c)
    ok = r.status == "optimal" and np.max(np.abs(r.x - xs)) < 1e-7 and relerr(r.obj, fs) < 1e-8
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s xerr=%.2e objerr=%.2e (closed form)"
            % (r.status, np.max(np.abs(r.x - xs)), relerr(r.obj, fs)))


def c05_zero_row_A():
    """A with shape (0, n) -- an equality block with no rows."""
    P = np.eye(3)
    c = np.array([-1.0, -2.0, -3.0])
    A = np.zeros((0, 3))
    b = np.zeros(0)
    try:
        r = solve_qp(P=P, c=c, A=A, b=b)
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected empty A: " + d
    xs = -c
    ok = r.status == "optimal" and np.max(np.abs(r.x - xs)) < 1e-7
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s (expect [1,2,3]) obj=%.12g" % (r.status, np.round(r.x, 9), r.obj))


def c06_zero_row_G():
    """G with shape (0, n) -- an inequality block with no rows."""
    P = np.eye(2)
    c = np.array([-1.0, -1.0])
    G = np.zeros((0, 2))
    h = np.zeros(0)
    try:
        r = solve_qp(P=P, c=c, G=G, h=h)
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected empty G: " + d
    ok = r.status == "optimal" and np.max(np.abs(r.x - np.array([1.0, 1.0]))) < 1e-7
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s (expect [1,1]) obj=%.12g" % (r.status, np.round(r.x, 9), r.obj))


def c07_zero_row_A_and_G():
    """Both A and G are (0,n) simultaneously."""
    P = np.eye(2)
    c = np.array([-3.0, 4.0])
    try:
        r = solve_qp(P=P, c=c, A=np.zeros((0, 2)), b=np.zeros(0),
                     G=np.zeros((0, 2)), h=np.zeros(0))
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected: " + d
    ok = r.status == "optimal" and np.max(np.abs(r.x - np.array([3.0, -4.0]))) < 1e-7
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s (expect [3,-4])" % (r.status, np.round(r.x, 9)))


def c08_P_none_lp():
    """P=None: pure LP.  min -x1-2x2 s.t. x1+x2<=1, x>=0 -> x=(0,1), f=-2."""
    c = np.array([-1.0, -2.0])
    G = np.array([[1.0, 1.0]])
    h = np.array([1.0])
    lb = np.zeros(2)
    r = solve_qp(P=None, c=c, G=G, h=h, lb=lb)
    ref = linprog(c, A_ub=G, b_ub=h, bounds=[(0, None)] * 2)
    ok = (r.status == "optimal" and relerr(r.obj, ref.fun) < 1e-7
          and np.max(np.abs(r.x - ref.x)) < 1e-6)
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s obj=%.12g scipy=%.12g x=%s" % (r.status, r.obj, ref.fun, np.round(r.x, 8)))


def c09_c_none_feasibility():
    """c=None with P given. The docstring says "``c`` is required and sets ``n``",
    so `c=None` is genuinely invalid input -> a clear error is the correct
    behaviour. (The signature default `c=None` is cosmetically misleading.)
    Also check the equivalent explicit-zeros form gives x=(1,1), f=1."""
    try:
        solve_qp(P=np.eye(2), c=None, A=np.array([[1.0, 1.0]]), b=np.array([2.0]))
        err = None
    except BaseException as e:
        clean, err = clean_error(e)
        if not clean:
            return "SOLVER_BUG", "panic on c=None: " + err
    r = solve_qp(P=np.eye(2), c=np.zeros(2), A=np.array([[1.0, 1.0]]), b=np.array([2.0]))
    ok = r.status == "optimal" and np.max(np.abs(r.x - 1.0)) < 1e-6 and relerr(r.obj, 1.0) < 1e-6
    return ("PASS" if ok else "SOLVER_BUG",
            "c=None -> %s ; c=zeros -> status=%s x=%s obj=%.12g (expect [1,1], 1.0)"
            % (err or "ACCEPTED(no error)", r.status, np.round(r.x, 9), r.obj))


def c10_c_all_zeros_feasibility():
    """c = all zeros, P=None: pure feasibility LP. 1<=x1<=2, x1+x2=3, x2>=0."""
    A = np.array([[1.0, 1.0]])
    b = np.array([3.0])
    lb = np.array([1.0, 0.0])
    ub = np.array([2.0, np.inf])
    r = solve_qp(P=None, c=np.zeros(2), A=A, b=b, lb=lb, ub=ub)
    if r.status != "optimal":
        return "SOLVER_LIMITATION", "status=%s on a feasible zero-objective LP" % r.status
    x = np.asarray(r.x)
    feas = (abs(x[0] + x[1] - 3.0) < 1e-6 and 1 - 1e-6 <= x[0] <= 2 + 1e-6 and x[1] >= -1e-6)
    ok = feas and abs(r.obj) < 1e-8
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s obj=%.3g feasible=%s" % (r.status, np.round(x, 8), r.obj, feas))


def c11_P_none_and_c_none():
    """Both P and c None -- no objective at all. Either solve feasibility or
    raise clearly."""
    try:
        r = solve_qp(P=None, c=None, A=np.array([[1.0, 1.0]]), b=np.array([2.0]))
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected P=c=None: " + d
    x = np.asarray(r.x)
    if r.status != "optimal":
        return "SOLVER_LIMITATION", "status=%s (no objective)" % r.status
    feas = abs(x.sum() - 2.0) < 1e-6
    return ("PASS" if feas else "SOLVER_BUG",
            "status=%s x=%s sum=%.12g (need 2) obj=%.3g" % (r.status, np.round(x, 8), x.sum(), r.obj))


def c12_lp_no_constraints_finite_bounds():
    """LP, zero constraints, finite box only: min -x1+x2 over [0,3]x[-2,5]
    -> x=(3,-2), f=-5."""
    c = np.array([-1.0, 1.0])
    r = solve_qp(P=None, c=c, lb=np.array([0.0, -2.0]), ub=np.array([3.0, 5.0]))
    ok = (r.status == "optimal" and np.max(np.abs(r.x - np.array([3.0, -2.0]))) < 1e-6
          and relerr(r.obj, -5.0) < 1e-7)
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s obj=%.12g (expect [3,-2], -5)" % (r.status, np.round(r.x, 8), r.obj))


def c13_lp_unbounded_no_constraints():
    """LP with no constraints and NO bounds -> genuinely unbounded.
    Must report it, not return garbage."""
    r = solve_qp(P=None, c=np.array([-1.0, 0.0]))
    st = str(r.status).lower()
    ok = ("unbound" in st or "infeas" in st or "dual" in st or not r.success)
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s success=%s x=%s obj=%.6g (problem is UNBOUNDED)"
            % (r.status, r.success, np.round(np.asarray(r.x), 4), r.obj))


def c14_more_eq_than_vars_consistent():
    """n=2, 4 equality rows, consistent (rank 2). x=(1,2) forced.
    min 0.5||x||^2 s.t. that system."""
    A = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0],
                  [2.0, -1.0]])
    b = np.array([1.0, 2.0, 3.0, 0.0])
    r = solve_qp(P=np.eye(2), c=np.zeros(2), A=A, b=b)
    if r.status != "optimal":
        return "SOLVER_LIMITATION", "status=%s on consistent over-determined system" % r.status
    ok = np.max(np.abs(r.x - np.array([1.0, 2.0]))) < 1e-6
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s (unique feasible point [1,2]) obj=%.12g" % (r.status, np.round(r.x, 9), r.obj))


def c15_more_eq_than_vars_inconsistent():
    """Same but INCONSISTENT (last row contradicts). Must report infeasible."""
    A = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])
    b = np.array([1.0, 2.0, 99.0])
    try:
        r = solve_qp(P=np.eye(2), c=np.zeros(2), A=A, b=b)
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "raised on infeasible: " + d
    st = str(r.status).lower()
    ok = "infeas" in st or not r.success
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s success=%s x=%s (system is INCONSISTENT)"
            % (r.status, r.success, np.round(np.asarray(r.x), 6)))


def c16_duplicate_identical_rows_A():
    """Rank-deficient A: row 0 duplicated 3x. min 0.5||x-e||^2 s.t. x1+x2=1 (x3)."""
    A = np.array([[1.0, 1.0, 0.0],
                  [1.0, 1.0, 0.0],
                  [1.0, 1.0, 0.0]])
    b = np.array([1.0, 1.0, 1.0])
    P = np.eye(3)
    c = -np.ones(3)
    r = solve_qp(P=P, c=c, A=A, b=b)
    if r.status != "optimal":
        return "SOLVER_LIMITATION", "status=%s on duplicated-row A" % r.status
    xs = np.array([0.5, 0.5, 1.0])
    ok = np.max(np.abs(r.x - xs)) < 1e-6
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s (exact [0.5,0.5,1]) obj=%.12g" % (r.status, np.round(r.x, 9), r.obj))


def c17_duplicate_identical_rows_G():
    """Rank-deficient active inequality set: same row 4x. min 0.5||x||^2 s.t.
    -x1-x2 <= -2 (i.e. x1+x2>=2) repeated -> x=(1,1)."""
    G = np.tile(np.array([[-1.0, -1.0]]), (4, 1))
    h = np.full(4, -2.0)
    r = solve_qp(P=np.eye(2), c=np.zeros(2), G=G, h=h)
    if r.status != "optimal":
        return "SOLVER_LIMITATION", "status=%s on duplicated-row G" % r.status
    ok = np.max(np.abs(r.x - 1.0)) < 1e-6
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s (exact [1,1]) obj=%.12g" % (r.status, np.round(r.x, 9), r.obj))


def c18_empty_bounds_arrays():
    """lb/ub given as length-0 arrays while n=2 -- a genuine shape mismatch.
    Must raise clearly (or silently ignore, which we flag)."""
    try:
        r = solve_qp(P=np.eye(2), c=np.array([-1.0, -1.0]),
                     lb=np.zeros(0), ub=np.zeros(0))
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected len-0 bounds for n=2: " + d
    return ("SILENT_ACCEPT", "accepted len-0 lb/ub for n=2: status=%s x=%s"
            % (r.status, np.round(np.asarray(r.x), 6)))


def c19_n0_empty_problem():
    """Fully empty problem: n=0. Either trivially optimal with obj=0 or a clear
    error. A panic is a bug."""
    try:
        r = solve_qp(P=np.zeros((0, 0)), c=np.zeros(0))
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected n=0: " + d
    ok = abs(r.obj) < 1e-12 and len(np.asarray(r.x)) == 0
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s obj=%.3g (n=0 should be trivially optimal, obj 0)"
            % (r.status, np.asarray(r.x), r.obj))


def c20_shape_mismatch_A_b():
    """A is (2,n) but b has length 3 -- invalid. Must be a clear error."""
    try:
        r = solve_qp(P=np.eye(2), c=np.zeros(2),
                     A=np.ones((2, 2)), b=np.ones(3))
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected A(2,2)/b(3): " + d
    return "SOLVER_BUG", "SILENTLY ACCEPTED mismatched A(2,2)/b(3): status=%s x=%s" % (
        r.status, np.asarray(r.x))


def c21_A_wrong_ncols():
    """A has 3 columns but P is 2x2 -- invalid."""
    try:
        r = solve_qp(P=np.eye(2), c=np.zeros(2), A=np.ones((1, 3)), b=np.ones(1))
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected A ncols=3 vs n=2: " + d
    return "SOLVER_BUG", "SILENTLY ACCEPTED A ncols mismatch: status=%s x=%s" % (
        r.status, np.asarray(r.x))


def c22_socp_zero_cones():
    """solve_socp with an empty cone list -- degenerates to a QP."""
    try:
        r = solve_socp(P=np.eye(2), c=np.array([-1.0, -1.0]), cones=[])
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected cones=[]: " + d
    ok = r.status == "optimal" and np.max(np.abs(np.asarray(r.x) - 1.0)) < 1e-6
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s (expect [1,1])" % (r.status, np.round(np.asarray(r.x), 8)))


def c23_socp_dim1_cone():
    """A degenerate SOC of dimension 1: ("soc",1) is just s>=0.
    min x s.t. x >= 1 encoded as G x + s = h with s in soc(1).
    Use G=[[-1]], h=[-1] -> s = -1 + x >= 0 -> x>=1. min x -> 1."""
    try:
        r = solve_socp(P=None, c=np.array([1.0]), G=np.array([[-1.0]]),
                       h=np.array([-1.0]), cones=[("soc", 1)])
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected soc dim 1: " + d
    ok = r.status == "optimal" and abs(np.asarray(r.x)[0] - 1.0) < 1e-6
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s obj=%.10g (expect x=1)" % (r.status, np.round(np.asarray(r.x), 8), r.obj))


def c24_socp_psd_dim1():
    """("psd",1) is a scalar nonnegativity constraint. min x s.t. x>=2."""
    try:
        r = solve_socp(P=None, c=np.array([1.0]), G=np.array([[-1.0]]),
                       h=np.array([-2.0]), cones=[("psd", 1)])
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected psd dim 1: " + d
    ok = r.status == "optimal" and abs(np.asarray(r.x)[0] - 2.0) < 1e-5
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s obj=%.10g (expect x=2)" % (r.status, np.round(np.asarray(r.x), 8), r.obj))


def c25_socp_cone_dim_mismatch():
    """cones declare more rows than G has -- invalid, must be a clear error."""
    try:
        r = solve_socp(P=None, c=np.array([1.0]), G=np.array([[-1.0]]),
                       h=np.array([-1.0]), cones=[("soc", 5)])
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected cone dim > G rows: " + d
    return "SOLVER_BUG", "SILENTLY ACCEPTED cone dim 5 with 1 G row: status=%s x=%s" % (
        r.status, np.asarray(r.x))


def c26_minimize_n1():
    """minimize() with a single variable: min (x-3)^2 -> x=3."""
    r = minimize(lambda x: float((x[0] - 3.0) ** 2), np.array([0.0]))
    ok = abs(np.asarray(r.x)[0] - 3.0) < 1e-5
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s fun=%.10g (expect 3)" % (getattr(r, "status", "?"),
                                                     np.round(np.asarray(r.x), 8), r.fun))


def c27_minimize_n1_bound_active():
    """minimize() n=1 with an active bound: min (x-3)^2, x<=1 -> x=1, f=4."""
    r = minimize(lambda x: float((x[0] - 3.0) ** 2), np.array([0.0]), bounds=[(-10.0, 1.0)])
    x = np.asarray(r.x)[0]
    ok = abs(x - 1.0) < 1e-5 and abs(r.fun - 4.0) < 1e-4
    return ("PASS" if ok else "SOLVER_BUG",
            "x=%.10g fun=%.10g (expect 1, 4)" % (x, r.fun))


def c28_minimize_no_constraints_empty_list():
    """constraints=[] (empty sequence) must behave like unconstrained."""
    r = minimize(lambda x: float((x[0] - 1) ** 2 + (x[1] + 2) ** 2),
                 np.array([0.0, 0.0]), constraints=[])
    x = np.asarray(r.x)
    ok = np.max(np.abs(x - np.array([1.0, -2.0]))) < 1e-4
    return ("PASS" if ok else "SOLVER_BUG",
            "x=%s (expect [1,-2]) fun=%.6g" % (np.round(x, 8), r.fun))


def c29_minimize_empty_x0():
    """minimize with x0 of length 0 -- degenerate; clear error or trivial success."""
    try:
        r = minimize(lambda x: 0.0, np.zeros(0))
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected empty x0: " + d
    return ("PASS", "accepted n=0: status=%s x=%s fun=%s"
            % (getattr(r, "status", "?"), np.asarray(r.x), r.fun))


def c30_qp_duplicate_rows_cvxpy_oracle():
    """Rank-deficient A + rank-deficient G together, cross-checked with cvxpy."""
    P = np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 1.5]])
    c = np.array([-1.0, -2.0, 0.5])
    A = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    b = np.array([1.0, 1.0])
    G = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    h = np.array([0.6, 0.6, 0.1])
    r = solve_qp(P=P, c=c, A=A, b=b, G=G, h=h)
    if not HAVE_CVXPY:
        return "SKIP", "cvxpy unavailable"
    x = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(P)) + c @ x),
                      [A @ x == b, G @ x <= h])
    prob.solve(solver=cp.CLARABEL)
    ok = r.status == "optimal" and relerr(r.obj, prob.value) < 1e-6 and \
        np.max(np.abs(np.asarray(r.x) - x.value)) < 1e-5
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s obj=%.12g cvxpy=%.12g xerr=%.2e"
            % (r.status, r.obj, prob.value, np.max(np.abs(np.asarray(r.x) - x.value))))


def c31_scalar_not_matrix_P():
    """P passed as a bare scalar/1-D for n=1 -- ergonomic edge. Clear behaviour?"""
    try:
        r = solve_qp(P=[[2.0]], c=[-4.0])
        base = (r.x[0], r.obj)
    except BaseException as e:
        return "HARNESS_ERROR", str(e)
    try:
        r2 = solve_qp(P=2.0, c=[-4.0])
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), \
            "scalar P rejected (baseline x=%.6g): %s" % (base[0], d)
    ok = abs(np.asarray(r2.x)[0] - base[0]) < 1e-9
    return ("PASS" if ok else "SOLVER_BUG",
            "scalar P accepted, x=%s vs matrix-P x=%.6g" % (np.asarray(r2.x), base[0]))


def c32_A_zero_rows_with_nonempty_G():
    """(0,n) A alongside a real G block -- mixed degenerate/normal."""
    P = np.eye(2)
    c = np.array([0.0, 0.0])
    G = np.array([[-1.0, 0.0], [0.0, -1.0]])
    h = np.array([-1.0, -1.0])   # x >= 1
    try:
        r = solve_qp(P=P, c=c, A=np.zeros((0, 2)), b=np.zeros(0), G=G, h=h)
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected: " + d
    ok = r.status == "optimal" and np.max(np.abs(np.asarray(r.x) - 1.0)) < 1e-6
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s (expect [1,1]) obj=%.10g" % (r.status, np.round(np.asarray(r.x), 8), r.obj))


def c33_equal_lb_ub_fixed_vars():
    """All variables fixed by lb==ub: feasible set is a point."""
    P = np.eye(2)
    c = np.array([-5.0, 7.0])
    lb = np.array([1.0, 2.0])
    ub = np.array([1.0, 2.0])
    r = solve_qp(P=P, c=c, lb=lb, ub=ub)
    fs = 0.5 * (1 + 4) + (-5 * 1 + 7 * 2)
    ok = r.status == "optimal" and np.max(np.abs(np.asarray(r.x) - lb)) < 1e-6 \
        and relerr(r.obj, fs) < 1e-6
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s obj=%.10g (expect [1,2], %.6g)"
            % (r.status, np.round(np.asarray(r.x), 8), r.obj, fs))


def c34_crossed_bounds():
    """lb > ub -- genuinely infeasible box. Clear infeasible/error required."""
    try:
        r = solve_qp(P=np.eye(1), c=[0.0], lb=[2.0], ub=[1.0])
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected lb>ub: " + d
    st = str(r.status).lower()
    ok = "infeas" in st or not r.success
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s success=%s x=%s (lb=2 > ub=1 is INFEASIBLE)"
            % (r.status, r.success, np.asarray(r.x)))


def c35_nan_in_input():
    """NaN in c -- garbage in. Must not silently return a 'solution'."""
    try:
        r = solve_qp(P=np.eye(2), c=np.array([np.nan, 1.0]))
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected NaN in c: " + d
    st = str(r.status).lower()
    xs = np.asarray(r.x)
    if st == "optimal" and np.all(np.isfinite(xs)):
        return "SOLVER_BUG", "claimed OPTIMAL x=%s for NaN input" % xs
    return "PASS", "status=%s x=%s (non-optimal / non-finite is acceptable)" % (r.status, xs)


def c36_n0_with_empty_A():
    """n=0 AND a (0,0) equality block."""
    try:
        r = solve_qp(P=np.zeros((0, 0)), c=np.zeros(0), A=np.zeros((0, 0)), b=np.zeros(0))
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected: " + d
    ok = abs(r.obj) < 1e-12 and len(np.asarray(r.x)) == 0
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s obj=%.3g" % (r.status, np.asarray(r.x), r.obj))


def c37_sparse_zero_row_A():
    """scipy.sparse (0,n) equality block."""
    import scipy.sparse as sp
    try:
        r = solve_qp(P=sp.eye(2, format="csc"), c=np.array([-1.0, -1.0]),
                     A=sp.csc_matrix((0, 2)), b=np.zeros(0))
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected sparse (0,n) A: " + d
    ok = r.status == "optimal" and np.max(np.abs(np.asarray(r.x) - 1.0)) < 1e-6
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s (expect [1,1])" % (r.status, np.round(np.asarray(r.x), 8)))


def c38_duplicate_rows_A_inconsistent():
    """Identical A rows with DIFFERENT rhs (x+y=1 and x+y=2) -> infeasible."""
    r = solve_qp(P=np.eye(2), c=np.zeros(2),
                 A=np.array([[1.0, 1.0], [1.0, 1.0]]), b=np.array([1.0, 2.0]))
    st = str(r.status).lower()
    ok = "infeas" in st or not r.success
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s success=%s x=%s (duplicate rows, contradictory rhs)"
            % (r.status, r.success, np.round(np.asarray(r.x), 8)))


def c39_batch_empty_list():
    """solve_qp_batch([]) -- zero problems."""
    from pounce import solve_qp_batch
    try:
        out = solve_qp_batch([])
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected empty batch: " + d
    ok = isinstance(out, (list, tuple)) and len(out) == 0
    return ("PASS" if ok else "SOLVER_BUG"), "solve_qp_batch([]) -> %r" % (out,)


def c40_multi_rhs_empty_cs():
    """solve_qp_multi_rhs with cs=[] -- zero right-hand sides."""
    from pounce import solve_qp_multi_rhs
    try:
        out = solve_qp_multi_rhs(P=np.eye(2), cs=[])
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected cs=[]: " + d
    ok = isinstance(out, (list, tuple)) and len(out) == 0
    return ("PASS" if ok else "SOLVER_BUG"), "multi_rhs(cs=[]) -> %r" % (out,)


def c41_c_as_2d_column():
    """c given as an (n,1) column instead of (n,) -- shape coercion edge."""
    try:
        r = solve_qp(P=np.eye(2), c=np.array([[-1.0], [-1.0]]))
    except BaseException as e:
        clean, d = clean_error(e)
        return ("PASS" if clean else "SOLVER_BUG"), "rejected (n,1) c: " + d
    ok = r.status == "optimal" and np.max(np.abs(np.asarray(r.x) - 1.0)) < 1e-7
    return ("PASS" if ok else "SOLVER_BUG",
            "accepted (n,1) c: status=%s x=%s (expect [1,1])"
            % (r.status, np.round(np.asarray(r.x), 8)))


def c42_all_infinite_bounds():
    """lb=-inf, ub=+inf everywhere -- a 'bounds' block that constrains nothing."""
    r = solve_qp(P=np.eye(2), c=np.array([-1.0, -1.0]),
                 lb=np.full(2, -np.inf), ub=np.full(2, np.inf))
    ok = r.status == "optimal" and np.max(np.abs(np.asarray(r.x) - 1.0)) < 1e-7
    return ("PASS" if ok else "SOLVER_BUG",
            "status=%s x=%s (expect [1,1])" % (r.status, np.round(np.asarray(r.x), 8)))


CASES = [
    ("n=1 unconstrained QP", c01_n1_unconstrained),
    ("n=1 QP, active upper bound", c02_n1_with_bounds),
    ("n=1, 3 duplicate equalities", c03_n1_equality_more_eq_than_vars),
    ("m=0 constraints (closed form)", c04_m0_constraints),
    ("A shape (0,n)", c05_zero_row_A),
    ("G shape (0,n)", c06_zero_row_G),
    ("A and G both (0,n)", c07_zero_row_A_and_G),
    ("P=None (pure LP) vs linprog", c08_P_none_lp),
    ("c=None (min norm)", c09_c_none_feasibility),
    ("c=zeros (feasibility LP)", c10_c_all_zeros_feasibility),
    ("P=None and c=None", c11_P_none_and_c_none),
    ("LP, no constraints, finite box", c12_lp_no_constraints_finite_bounds),
    ("LP, no constraints, no bounds (unbounded)", c13_lp_unbounded_no_constraints),
    ("4 eq rows / 2 vars, consistent", c14_more_eq_than_vars_consistent),
    ("3 eq rows / 2 vars, inconsistent", c15_more_eq_than_vars_inconsistent),
    ("duplicate identical rows in A", c16_duplicate_identical_rows_A),
    ("duplicate identical rows in G", c17_duplicate_identical_rows_G),
    ("len-0 lb/ub with n=2", c18_empty_bounds_arrays),
    ("n=0 empty problem", c19_n0_empty_problem),
    ("A(2,2) with b(3) mismatch", c20_shape_mismatch_A_b),
    ("A ncols=3 with n=2", c21_A_wrong_ncols),
    ("solve_socp cones=[]", c22_socp_zero_cones),
    ("solve_socp ('soc',1)", c23_socp_dim1_cone),
    ("solve_socp ('psd',1)", c24_socp_psd_dim1),
    ("solve_socp cone dim > G rows", c25_socp_cone_dim_mismatch),
    ("minimize() n=1", c26_minimize_n1),
    ("minimize() n=1 active bound", c27_minimize_n1_bound_active),
    ("minimize() constraints=[]", c28_minimize_no_constraints_empty_list),
    ("minimize() empty x0", c29_minimize_empty_x0),
    ("rank-def A+G vs cvxpy", c30_qp_duplicate_rows_cvxpy_oracle),
    ("scalar P for n=1", c31_scalar_not_matrix_P),
    ("A (0,n) with real G", c32_A_zero_rows_with_nonempty_G),
    ("lb == ub (all fixed)", c33_equal_lb_ub_fixed_vars),
    ("lb > ub (crossed)", c34_crossed_bounds),
    ("NaN in c", c35_nan_in_input),
    ("n=0 with (0,0) A block", c36_n0_with_empty_A),
    ("sparse (0,n) A", c37_sparse_zero_row_A),
    ("duplicate A rows, contradictory rhs", c38_duplicate_rows_A_inconsistent),
    ("solve_qp_batch([])", c39_batch_empty_list),
    ("solve_qp_multi_rhs(cs=[])", c40_multi_rhs_empty_cs),
    ("c as (n,1) column", c41_c_as_2d_column),
    ("lb=-inf / ub=+inf everywhere", c42_all_infinite_bounds),
]


def main():
    print("pounce %s  degenerate-dimension API sweep  (%d cases)"
          % (pounce.__version__, len(CASES)))
    print("=" * 100)
    for name, fn in CASES:
        case(name, fn)
    print("=" * 100)
    from collections import Counter
    tally = Counter(v for _, v, _, _ in RESULTS)
    for k in sorted(tally):
        print("%-16s %d" % (k, tally[k]))
    bad = [(n, v, d) for n, v, d, _ in RESULTS if v not in ("PASS", "SKIP")]
    if bad:
        print("\n--- NON-PASS ---")
        for n, v, d in bad:
            print("* [%s] %s\n    %s" % (v, n, d))
    hard = [b for b in bad if b[1] in ("SOLVER_BUG",)]
    print("\nVERDICT: %s" % ("SOLVER_BUG" if hard else
                             ("SOLVER_LIMITATION" if any(b[1] == "SOLVER_LIMITATION" for b in bad)
                              else "PASS")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
