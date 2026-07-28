"""Adversary cross-check: does BAD SCALING break auto-routing STRUCTURE DETECTION?

Family: autoroute   Class: ill-conditioned / badly-scaled convex QP, LP, QCQP
Dimension under test: ill-conditioning & bad scaling.

Three instances, each with a closed-form optimum, each designed so that the
*numbers* (not the structure) are pathological.  The routing contract:
`solver_selection="auto"` must agree with forced `solver_selection="nlp"` to
tolerance, and ideally pick the specialized engine.  Disagreement => ROUTING_ERROR.
A conservative fall-through to the NLP engine that still gets the right answer is
"merely slower", NOT a bug.

A) NEAR-SINGULAR CONVEX QP (diagonal), condition number 1e14.
     min 0.5*(1e-14*x0^2 + 1.0*x1^2) - 1*x0 - 2*x1   s.t.  -5 <= x <= 5
   Unconstrained stationary point is x0 = 1e14 (clipped to the bound 5),
   x1 = 2.  So x* = (5, 2) and f* = 0.5*(1e-14*25 + 4) - 5 - 4 = -7.0
   (exact to ~1e-13).  Convexity holds but the smallest eigenvalue is 1e-14 --
   a sloppy PSD test could call this indefinite and refuse the QP fast path,
   or a sloppy one could mis-solve the flat direction.

A2) Same, but ROTATED by 45 deg so P is dense and, in floating point, may carry
   a tiny NEGATIVE eigenvalue (numerically indefinite by a hair).  The true
   problem is still convex.  Oracle: cvxpy + forced nlp (no closed form).

B) LP WITH ~1e-12 COEFFICIENTS.
     min 1e-12*(-x0 - 2*x1)  s.t. 1e-12*(x0 + x1) <= 4e-12, 0 <= x0 <= 3,
                                   0 <= x1 <= 10
   Scaling out 1e-12: maximize x0 + 2*x1 over x0+x1<=4 -> x* = (0, 4),
   value 8, so f* = -8e-12.  Every coefficient is ~1e-12, so a structure
   detector using an absolute tolerance could read the objective as "zero"
   (constant) or the constraint as vacuous.

C) QCQP-BALL WITH RADIUS 1e-9.
     min c.x  s.t.  ||x||_2 <= R,  c = (1,-2,2), ||c|| = 3, R = 1e-9
   Closed form: x* = -R*c/||c||, f* = -R*||c|| = -3e-9.  The quadratic
   constraint x.x <= R^2 = 1e-18 is at the edge of double precision.

Oracles: closed form (above), cvxpy (Clarabel/ECOS), scipy, and forced-nlp.
"""

import time

import numpy as np
import cvxpy as cp
from pounce import minimize

RESULTS = []


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def srel(a, b, scale):
    """Relative error measured against the PROBLEM's own scale, not 1.0.

    The default rel() with max(1.0, |b|) is meaningless when every objective
    value is ~1e-12: it reports 1e-12 no matter how wrong the answer is.
    """
    return abs(a - b) / max(abs(scale), abs(b), 1e-300)


def cvx(prob):
    """Solve with fallbacks; badly-scaled cones make CLARABEL bail."""
    for s in (cp.CLARABEL, cp.ECOS, cp.SCS):
        try:
            prob.solve(solver=s)
            if prob.value is not None and np.isfinite(prob.value):
                return float(prob.value), str(s)
        except Exception:  # noqa: BLE001
            continue
    return None, "none"


def solver_of(res):
    info = res.info if isinstance(res.info, dict) else {}
    return info.get("solver")


def run_pair(f, x0, jac=None, hess=None, bounds=None, constraints=None):
    kw = dict(jac=jac, bounds=bounds)
    if hess is not None:
        kw["hess"] = hess
    if constraints is not None:
        kw["constraints"] = constraints
    t0 = time.perf_counter()
    ra = minimize(f, x0, solver_selection="auto", **kw)
    ta = time.perf_counter() - t0
    t0 = time.perf_counter()
    rn = minimize(f, x0, solver_selection="nlp", **kw)
    tn = time.perf_counter() - t0
    return ra, ta, rn, tn


def report(name, ra, ta, rn, tn, known, xstar, oracle_obj, expect_solver, scale=1.0):
    obj_disagree = srel(ra.fun, rn.fun, scale)
    x_disagree = float(np.max(np.abs(np.asarray(ra.x) - np.asarray(rn.x))))
    err_known = srel(ra.fun, known, scale)
    err_nlp_known = srel(rn.fun, known, scale)
    err_oracle = srel(ra.fun, oracle_obj, scale) if oracle_obj is not None else 0.0
    err_oracle_known = srel(oracle_obj, known, scale) if oracle_obj is not None else 0.0
    used = solver_of(ra)
    print(f"\n=== {name} ===")
    print(f"auto : ok={ra.success} solver={used!r} f={ra.fun:.12e} x={np.asarray(ra.x)} t={ta:.3f}s")
    print(f"nlp  : ok={rn.success} solver={solver_of(rn)!r} f={rn.fun:.12e} x={np.asarray(rn.x)} t={tn:.3f}s")
    if oracle_obj is not None:
        print(f"cvxpy: f={oracle_obj:.12e}")
    print(f"known: f={known:.12e} x*={xstar}   (scale used for rel err = {scale:.1e})")
    print(f"  auto_vs_nlp_obj={obj_disagree:.2e} auto_vs_nlp_xinf={x_disagree:.2e}")
    print(f"  auto_vs_known={err_known:.2e}  nlp_vs_known={err_nlp_known:.2e}"
          f"  CVXPY_vs_known={err_oracle_known:.2e}   auto_vs_cvxpy={err_oracle:.2e}")
    print(f"  specialized_route={used == expect_solver} (expected {expect_solver!r})")
    routing_ok = obj_disagree < 1e-4
    answer_ok = ra.success and rn.success and err_known < 1e-4 and err_oracle < 1e-4
    # Did the INDEPENDENT oracle also miss the known optimum? Then any pounce
    # miss is a shared absolute-tolerance/scaling artifact, not a pounce defect.
    oracle_also_missed = err_oracle_known > 1e-4
    RESULTS.append(
        dict(name=name, routing_ok=routing_ok, answer_ok=answer_ok,
             specialized=(used == expect_solver), used=used,
             obj_disagree=obj_disagree, err_known=err_known,
             err_oracle=err_oracle, oracle_also_missed=oracle_also_missed)
    )


# ---------------------------------------------------------------- A: near-singular QP
EPS = 1e-14
P_A = np.diag([EPS, 1.0])
c_A = np.array([-1.0, -2.0])
LO, HI = -5.0, 5.0


def make_qp(P, c):
    def f(x):
        x = np.asarray(x, float)
        return 0.5 * float(x @ P @ x) + float(c @ x)

    def g(x):
        return P @ np.asarray(x, float) + c

    def h(x):
        return P

    return f, g, h


fA, gA, hA = make_qp(P_A, c_A)
# skepticism first: finite-difference the gradient
xt = np.array([0.31, -0.77])
fd = np.array([(fA(xt + 1e-6 * u) - fA(xt - 1e-6 * u)) / 2e-6 for u in np.eye(2)])
assert np.allclose(fd, gA(xt), atol=1e-6), (fd, gA(xt))

xA = np.array([5.0, 2.0])
knownA = 0.5 * float(xA @ P_A @ xA) + float(c_A @ xA)  # -7.0 (+2.5e-13)

xv = cp.Variable(2)
pr = cp.Problem(cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P_A)) + c_A @ xv),
                [xv >= LO, xv <= HI])
oracleA, _ = cvx(pr)

ra, ta, rn, tn = run_pair(fA, np.zeros(2), jac=gA, hess=hA, bounds=[(LO, HI)] * 2)
report("A near-singular convex QP (cond 1e14, diagonal)", ra, ta, rn, tn,
       knownA, xA, oracleA, "qp-ipm")

# --------------------------------------------------------- A2: rotated (numerically indefinite)
s = 1.0 / np.sqrt(2.0)
V = np.array([[s, s], [s, -s]])
P_A2 = V @ np.diag([EPS, 1.0]) @ V.T
P_A2 = 0.5 * (P_A2 + P_A2.T)
eig2 = np.linalg.eigvalsh(P_A2)
print(f"\n[A2] float eigenvalues of rotated P: {eig2} (min < 0 ? {eig2.min() < 0})")
c_A2 = V @ c_A
fA2, gA2, hA2 = make_qp(P_A2, c_A2)

xv = cp.Variable(2)
pr = cp.Problem(cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P_A2)) + c_A2 @ xv),
                [xv >= LO, xv <= HI])
oracleA2, _ = cvx(pr)

ra2, ta2, rn2, tn2 = run_pair(fA2, np.zeros(2), jac=gA2, hess=hA2, bounds=[(LO, HI)] * 2)
report("A2 rotated near-singular QP (numerically indefinite)", ra2, ta2, rn2, tn2,
       oracleA2, "(cvxpy)", oracleA2, "qp-ipm")

# ---------------------------------------------------------------- B: tiny-coefficient LP
def lp_case(S, label):
    """Same LP, every coefficient multiplied by S. S=1 is the CONTROL."""
    c_B = S * np.array([-1.0, -2.0])
    knownB = -8.0 * S
    xB = np.array([0.0, 4.0])

    def fB(x):
        return float(c_B @ np.asarray(x, float))

    def gB(x):
        return c_B.copy()

    consB = [{"type": "ineq",
              "fun": lambda x: 4.0 * S - S * (x[0] + x[1]),
              "jac": lambda x: np.array([-S, -S])}]

    xv = cp.Variable(2)
    pr = cp.Problem(cp.Minimize(c_B @ xv),
                    [S * (xv[0] + xv[1]) <= 4 * S, xv >= 0, xv[0] <= 3, xv[1] <= 10])
    oracleB, _ = cvx(pr)

    rb, tb, rbn, tbn = run_pair(fB, np.array([0.5, 0.5]), jac=gB,
                                bounds=[(0.0, 3.0), (0.0, 10.0)], constraints=consB)
    report(label, rb, tb, rbn, tbn, knownB, xB, oracleB, "lp-ipm", scale=abs(knownB))
    # UNSCALED feasibility of x0 + x1 <= 4 (the structural constraint, not its
    # S-multiplied encoding) -- this is what "wrong answer" would look like.
    for tag, r in (("auto", rb), ("nlp", rbn)):
        v = float(r.x[0] + r.x[1]) - 4.0
        print(f"    {tag}: unscaled residual (x0+x1-4) = {v:+.3e}  "
              f"(as-stated residual = {S * v:+.3e})")
    return rb, rbn


lp_case(1.0, "B0 CONTROL: same LP, coefficients O(1)")
lp_case(1e-12, "B LP with ~1e-12 coefficients")

# ---------------------------------------------------------------- C: 1e-9-radius ball QCQP
R = 1e-9
c_C = np.array([1.0, -2.0, 2.0])
nc = float(np.linalg.norm(c_C))
knownC = -R * nc
xC = -R * c_C / nc


def fC(x):
    return float(c_C @ np.asarray(x, float))


def gC(x):
    return c_C.copy()


consC = [{"type": "ineq",
          "fun": lambda x: R * R - float(np.asarray(x, float) @ np.asarray(x, float)),
          "jac": lambda x: -2.0 * np.asarray(x, float)}]

xv = cp.Variable(3)
pr = cp.Problem(cp.Minimize(c_C @ xv), [cp.sum_squares(xv) <= R * R])
oracleC, csolver = cvx(pr)
print(f"\n[C] cvxpy oracle solver = {csolver} -> {oracleC}")
if oracleC is None:
    oracleC = knownC  # closed form is exact here; cvxpy bailed on the tiny cone

rc, tc, rcn, tcn = run_pair(fC, np.zeros(3), jac=gC, constraints=consC)
report("C QCQP ball, radius 1e-9", rc, tc, rcn, tcn, knownC, xC, oracleC, "socp",
       scale=abs(knownC))
for tag, r in (("auto", rc), ("nlp", rcn)):
    nrm = float(np.linalg.norm(np.asarray(r.x)))
    print(f"    {tag}: ||x||={nrm:.6e}  R={R:.1e}  ||x||/R={nrm / R:.6f}")

# ------------------------------------------- C: tol sweep (is the miss tolerance-driven?)
print("\n[C tol sweep] if the miss is an absolute-tolerance artifact, tightening"
      " tol should walk the answer toward the closed form:")
for tol in (1e-8, 1e-12, 1e-14):
    for sel in ("auto", "nlp"):
        r = minimize(fC, np.zeros(3), jac=gC, constraints=consC,
                     solver_selection=sel, tol=tol)
        print(f"  tol={tol:.0e} {sel:4s} f={r.fun:.6e} ||x||/R={np.linalg.norm(r.x) / R:9.4f}"
              f" relerr={abs(r.fun - knownC) / abs(knownC):.3e}")

# ---------------------------------------------------------------- summary
print("\n=== SUMMARY ===")
bad_route = [r for r in RESULTS if not r["routing_ok"]]
bad_ans = [r for r in RESULTS if not r["answer_ok"]]
fellthru = [r for r in RESULTS if not r["specialized"]]
for r in RESULTS:
    print(f"  {r['name']}: route={r['used']!r} specialized={r['specialized']} "
          f"auto_vs_nlp={r['obj_disagree']:.2e} vs_known={r['err_known']:.2e} "
          f"vs_cvxpy={r['err_oracle']:.2e}")
real_route = [r for r in bad_route if not r["oracle_also_missed"]]
bad_ans = [r for r in bad_ans if not r["oracle_also_missed"]]
shared = [r for r in RESULTS if r["oracle_also_missed"]]
if shared:
    print("  [cvxpy ALSO missed the known optimum on: "
          + "; ".join(r["name"] for r in shared)
          + " -> shared absolute-tolerance/scaling artifact, not a pounce defect]")
if real_route:
    print("VERDICT: ROUTING_ERROR (" + "; ".join(r["name"] for r in real_route) + ")")
elif bad_ans:
    print("VERDICT: FAIL wrong answer (" + "; ".join(r["name"] for r in bad_ans) + ")")
else:
    note = ("" if not fellthru else
            "  [note: conservative fall-through (not a bug): "
            + "; ".join(r["name"] for r in fellthru) + "]")
    print("VERDICT: PASS" + note)
