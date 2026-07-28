"""Adversary cross-check: auto-routing transparency when the optimum is a FACE.

Family: autoroute   Class: degenerate / constraint-qualification failure /
                            NON-UNIQUE optimal set
Dimension under test: degeneracy, CQ failure, non-unique optima.

The routing-transparency contract says `solver_selection="auto"` and forced
`solver_selection="nlp"` must AGREE.  When the optimal set is a whole face,
"agree" cannot mean "same x": a vertex-seeking simplex-ish path, an
interior-point path (which converges to the analytic center of the optimal
face), and a general NLP path legitimately land on DIFFERENT points of the SAME
face.  Both are correct.  So the grading here is:

  * ROUTING_ERROR  <=>  the OBJECTIVES disagree, or one of the returned points
    is not on the optimal face (infeasible, or feasible but suboptimal).
  * differing-but-both-on-the-face x  =>  PASS (logged as a transparency
    nuance: a naive x-comparison would falsely cry ROUTING_ERROR).

Each instance has an ANALYTIC characterization of the optimal face, so every
returned x is checked for membership directly, not by comparison.

------------------------------------------------------------------ instances

A) DEGENERATE LP, optimal EDGE.
     min -x0 - x1   s.t.  x0 + x1 <= 1,  0 <= x <= 1
   f* = -1.  Optimal face F_A = { x : x0 + x1 = 1, x0 in [0,1] } -- a whole
   1-D edge, every point of which is optimal.  Vertices (1,0) and (0,1);
   the analytic center is (0.5, 0.5).

A2) SAME LP + a DUPLICATED constraint  2*x0 + 2*x1 <= 2.
   Identical geometry, identical optimal face, but now LICQ FAILS at every
   point of the optimal face (two parallel active gradients).  MFCQ still
   holds; the multipliers are non-unique (a whole ray).  This is the CQ-failure
   stress on the router: a structure detector that de-duplicates rows, or an
   IPM that inverts an exactly rank-deficient KKT block, could diverge here.

B) SEMIDEFINITE-HESSIAN QP, optimal LINE SEGMENT.
     min 0.5*(x0-x1)^2 + (x0-x1)    s.t.  -5 <= x <= 5
   P = [[1,-1],[-1,1]] is PSD with rank 1 (eigenvalues 0 and 2) -- singular,
   so the unconstrained-in-the-null-direction problem has no unique minimizer.
   With s = x0 - x1: f = 0.5*s^2 + s, minimized at s = -1, f* = -0.5,
   INDEPENDENT of t = x0 + x1.  Optimal face
     F_B = { x : x0 - x1 = -1, x0 + x1 in [-9, 9] }
   (the box limits t: x0 = (t-1)/2 in [-5,5] and x1 = (t+1)/2 in [-5,5]
    => t in [-9, 9]).  A whole segment of optima, f* = -0.5 exactly.

C) SEMIDEFINITE-HESSIAN QP with an EQUALITY that does NOT pin the null space.
     min 0.5*(x0-x1)^2 + (x0-x1)   s.t.  x0 + x1 + x2 = 0,  -5 <= x <= 5
   The equality is satisfiable for any (x0,x1) on F_B's line via x2, so the
   optimal set is still 2-D-in-disguise: f* = -0.5, and
     F_C = { x : x0 - x1 = -1, x2 = -(x0+x1), all in [-5,5] }.
   Adds an equality block to the KKT so the route sees a different structure.

Oracles: closed form (derived above, exact), cvxpy (Clarabel/ECOS/SCS),
scipy.optimize.linprog for the LPs, and forced solver_selection="nlp".
"""

import time

import numpy as np
import cvxpy as cp
from scipy.optimize import linprog

from pounce import minimize

RESULTS = []
FTOL = 1e-6          # objective agreement tolerance
FACE_TOL = 1e-5      # tolerance for "x lies on the analytic optimal face"


def cvx(prob):
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


def run_pair(f, x0, **kw):
    t0 = time.perf_counter()
    ra = minimize(f, x0, solver_selection="auto", **kw)
    ta = time.perf_counter() - t0
    t0 = time.perf_counter()
    rn = minimize(f, x0, solver_selection="nlp", **kw)
    tn = time.perf_counter() - t0
    return ra, ta, rn, tn


def report(name, ra, ta, rn, tn, fstar, on_face, oracle_obj, expect_solver, f):
    """on_face(x) -> (bool, str) analytic membership test for the optimal face."""
    used = solver_of(ra)
    fa, fn = float(ra.fun), float(rn.fun)
    # recompute the objective from the returned x -- never trust the reported fun
    fa_re, fn_re = f(np.asarray(ra.x, float)), f(np.asarray(rn.x, float))
    obj_disagree = abs(fa - fn) / max(1.0, abs(fn))
    x_disagree = float(np.max(np.abs(np.asarray(ra.x) - np.asarray(rn.x))))
    err_known_a = abs(fa - fstar) / max(1.0, abs(fstar))
    err_known_n = abs(fn - fstar) / max(1.0, abs(fstar))
    err_oracle = (abs(fa - oracle_obj) / max(1.0, abs(oracle_obj))
                  if oracle_obj is not None else 0.0)
    ok_a, why_a = on_face(np.asarray(ra.x, float))
    ok_n, why_n = on_face(np.asarray(rn.x, float))

    print(f"\n=== {name} ===")
    print(f"auto : ok={ra.success} solver={used!r} f={fa:.12e} (recomputed {fa_re:.12e})")
    print(f"       x={np.asarray(ra.x)}  t={ta:.3f}s   ON_FACE={ok_a}  [{why_a}]")
    print(f"nlp  : ok={rn.success} solver={solver_of(rn)!r} f={fn:.12e} (recomputed {fn_re:.12e})")
    print(f"       x={np.asarray(rn.x)}  t={tn:.3f}s   ON_FACE={ok_n}  [{why_n}]")
    if oracle_obj is not None:
        print(f"cvxpy: f={oracle_obj:.12e}")
    print(f"known: f*={fstar:.12e}")
    print(f"  auto_vs_nlp_obj={obj_disagree:.2e}   auto_vs_nlp_xinf={x_disagree:.2e}"
          f"   <-- x gap is EXPECTED to be large (non-unique optimum)")
    print(f"  auto_vs_known={err_known_a:.2e}  nlp_vs_known={err_known_n:.2e}"
          f"  auto_vs_cvxpy={err_oracle:.2e}")
    print(f"  fun_matches_recomputed: auto={abs(fa - fa_re):.2e} nlp={abs(fn - fn_re):.2e}")
    print(f"  specialized_route={used == expect_solver} (expected {expect_solver!r})")

    RESULTS.append(dict(
        name=name, used=used, specialized=(used == expect_solver),
        obj_disagree=obj_disagree, x_disagree=x_disagree,
        err_known_a=err_known_a, err_known_n=err_known_n, err_oracle=err_oracle,
        face_a=ok_a, face_n=ok_n,
        self_consistent=(abs(fa - fa_re) < 1e-8 and abs(fn - fn_re) < 1e-8),
        success=(bool(ra.success) and bool(rn.success)),
        distinct_points=(x_disagree > 1e-4),
    ))


# =============================================================== A / A2: degenerate LP
def lp_case(dup, label):
    c = np.array([-1.0, -1.0])

    def f(x):
        return float(c @ np.asarray(x, float))

    def g(x):
        return c.copy()

    cons = [{"type": "ineq",
             "fun": lambda x: 1.0 - x[0] - x[1],
             "jac": lambda x: np.array([-1.0, -1.0])}]
    if dup:
        cons.append({"type": "ineq",
                     "fun": lambda x: 2.0 - 2.0 * x[0] - 2.0 * x[1],
                     "jac": lambda x: np.array([-2.0, -2.0])})

    fstar = -1.0

    def on_face(x):
        viol = [max(0.0, -x[0]), max(0.0, x[0] - 1.0),
                max(0.0, -x[1]), max(0.0, x[1] - 1.0),
                max(0.0, x[0] + x[1] - 1.0)]
        feas = max(viol)
        act = abs(x[0] + x[1] - 1.0)
        ok = feas <= FACE_TOL and act <= FACE_TOL
        return ok, f"feas_viol={feas:.2e} |x0+x1-1|={act:.2e}"

    xv = cp.Variable(2)
    con = [xv[0] + xv[1] <= 1, xv >= 0, xv <= 1]
    if dup:
        con.append(2 * xv[0] + 2 * xv[1] <= 2)
    oracle, csolv = cvx(cp.Problem(cp.Minimize(c @ xv), con))

    A_ub = [[1.0, 1.0]] + ([[2.0, 2.0]] if dup else [])
    b_ub = [1.0] + ([2.0] if dup else [])
    lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1), (0, 1)])
    print(f"\n[{label}] cvxpy({csolv})={oracle}  linprog f={lp.fun} x={lp.x}")

    ra, ta, rn, tn = run_pair(f, np.array([0.1, 0.1]), jac=g,
                              bounds=[(0.0, 1.0)] * 2, constraints=cons)
    report(label, ra, ta, rn, tn, fstar, on_face, oracle, "lp-ipm", f)


lp_case(False, "A degenerate LP, optimal EDGE (unique face, non-unique x)")
lp_case(True, "A2 same LP + duplicated row (LICQ FAILS on the whole face)")

# =============================================================== B: PSD-singular QP
P_B = np.array([[1.0, -1.0], [-1.0, 1.0]])
c_B = np.array([1.0, -1.0])
eigB = np.linalg.eigvalsh(P_B)
print(f"\n[B] eigenvalues of P = {eigB}  (rank {np.linalg.matrix_rank(P_B)}"
      f" of 2 -> singular Hessian, null direction (1,1))")


def fB(x):
    x = np.asarray(x, float)
    return 0.5 * float(x @ P_B @ x) + float(c_B @ x)


def gB(x):
    return P_B @ np.asarray(x, float) + c_B


def hB(x):
    return P_B


# skepticism first: finite-difference the gradient of the SINGULAR QP
xt = np.array([0.37, -0.81])
fd = np.array([(fB(xt + 1e-6 * u) - fB(xt - 1e-6 * u)) / 2e-6 for u in np.eye(2)])
assert np.allclose(fd, gB(xt), atol=1e-6), (fd, gB(xt))

FSTAR_B = -0.5


def on_face_B(x):
    viol = max(max(0.0, abs(x[0]) - 5.0), max(0.0, abs(x[1]) - 5.0))
    s = abs((x[0] - x[1]) - (-1.0))
    return (viol <= FACE_TOL and s <= FACE_TOL), f"box_viol={viol:.2e} |s+1|={s:.2e}"


xv = cp.Variable(2)
oracleB, csolv = cvx(cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P_B)) + c_B @ xv),
    [xv >= -5, xv <= 5]))
print(f"[B] cvxpy({csolv}) f={oracleB}  x={xv.value}")

raB, taB, rnB, tnB = run_pair(fB, np.array([0.3, -0.2]), jac=gB, hess=hB,
                              bounds=[(-5.0, 5.0)] * 2)
report("B PSD-singular QP, optimal LINE SEGMENT", raB, taB, rnB, tnB,
       FSTAR_B, on_face_B, oracleB, "qp-ipm", fB)

# =============================================================== C: + equality
def fC(x):
    x = np.asarray(x, float)
    return 0.5 * (x[0] - x[1]) ** 2 + (x[0] - x[1])


def gC(x):
    s = float(x[0] - x[1])
    return np.array([s + 1.0, -(s + 1.0), 0.0])


def hC(x):
    return np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


xt3 = np.array([0.4, -0.9, 1.3])
fd3 = np.array([(fC(xt3 + 1e-6 * u) - fC(xt3 - 1e-6 * u)) / 2e-6 for u in np.eye(3)])
assert np.allclose(fd3, gC(xt3), atol=1e-6), (fd3, gC(xt3))

consC = [{"type": "eq",
          "fun": lambda x: float(x[0] + x[1] + x[2]),
          "jac": lambda x: np.array([1.0, 1.0, 1.0])}]


def on_face_C(x):
    viol = max(max(0.0, abs(xi) - 5.0) for xi in x)
    viol = max(viol, abs(x[0] + x[1] + x[2]))
    s = abs((x[0] - x[1]) - (-1.0))
    return (viol <= FACE_TOL and s <= FACE_TOL), f"viol={viol:.2e} |s+1|={s:.2e}"


xv = cp.Variable(3)
oracleC, csolv = cvx(cp.Problem(
    cp.Minimize(0.5 * cp.square(xv[0] - xv[1]) + (xv[0] - xv[1])),
    [cp.sum(xv) == 0, xv >= -5, xv <= 5]))
print(f"\n[C] cvxpy({csolv}) f={oracleC}  x={xv.value}")

raC, taC, rnC, tnC = run_pair(fC, np.array([0.2, -0.1, 0.05]), jac=gC, hess=hC,
                              bounds=[(-5.0, 5.0)] * 3, constraints=consC)
report("C PSD-singular QP + equality, optimal SEGMENT", raC, taC, rnC, tnC,
       -0.5, on_face_C, oracleC, "qp-ipm", fC)

# =============================================================== summary
print("\n=== SUMMARY ===")
bad_obj, bad_face, bad_self, distinct, fellthru = [], [], [], [], []
for r in RESULTS:
    print(f"  {r['name']}")
    print(f"    route={r['used']!r} specialized={r['specialized']} success={r['success']}"
          f" auto_vs_nlp_obj={r['obj_disagree']:.2e} x_gap={r['x_disagree']:.2e}"
          f" face(auto,nlp)=({r['face_a']},{r['face_n']})"
          f" vs_known=({r['err_known_a']:.1e},{r['err_known_n']:.1e})")
    if r["obj_disagree"] > FTOL or r["err_known_a"] > FTOL or r["err_known_n"] > FTOL \
            or r["err_oracle"] > FTOL:
        bad_obj.append(r)
    if not (r["face_a"] and r["face_n"]):
        bad_face.append(r)
    if not r["self_consistent"]:
        bad_self.append(r)
    if r["distinct_points"]:
        distinct.append(r)
    if not r["specialized"]:
        fellthru.append(r)

if distinct:
    print("  [transparency nuance: auto and nlp returned DIFFERENT points of the "
          "SAME optimal face on: " + "; ".join(r["name"] for r in distinct)
          + " -- both verified on the analytic face, so this is CORRECT, and a "
            "naive x-comparison would have falsely cried ROUTING_ERROR]")
if fellthru:
    print("  [conservative fall-through (NOT a bug, merely slower): "
          + "; ".join(f"{r['name']} -> {r['used']!r}" for r in fellthru) + "]")

if bad_obj or bad_face:
    parts = []
    if bad_obj:
        parts.append("objective disagreement: " + "; ".join(r["name"] for r in bad_obj))
    if bad_face:
        parts.append("returned x OFF the optimal face: "
                     + "; ".join(r["name"] for r in bad_face))
    print("VERDICT: ROUTING_ERROR (" + " | ".join(parts) + ")")
elif bad_self:
    print("VERDICT: FAIL reported fun != f(returned x) ("
          + "; ".join(r["name"] for r in bad_self) + ")")
else:
    print("VERDICT: PASS")
