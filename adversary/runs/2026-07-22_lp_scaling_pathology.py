"""Adversary cross-check: badly-scaled LP (one row x1e8, one column x1e-8)

Family: lp   Class: ill-conditioned / mismatched row-column scaling, known closed-form vertex

Source: constructed instance in the style of the classic badly-scaled NETLIB
LPs (pilot / scagr / bnl) whose difficulty is *pure scaling*, not degeneracy.
Construction follows the standard "diagonal equilibration" pathology discussed
in Nocedal & Wright, "Numerical Optimization" 2e, S13.5 (Presolve/Scaling) and
Gill, Murray & Wright, "Practical Optimization" S8.7: an LP with a unique
nondegenerate optimal vertex is left-multiplied by a row-scaling R and its
variables substituted x = S u, which leaves the optimal objective and the
geometry invariant but multiplies cond(A) by ~1e16.

BASE PROBLEM (well-scaled, exact rational optimum by construction)

    maximize   8 x1 + 3 x2 + 10 x3
    s.t.       x1 +  x2 +   x3 <= 4      (row 1)
              2 x1 +  x2         <= 4    (row 2)
               x1        + 3 x3 <= 4     (row 3)
               x >= 0

  A = [[1,1,1],[2,1,0],[1,0,3]] is nonsingular (det = -4) and by construction
  x* = (1, 2, 1) satisfies A x* = b = (4,4,4) with all three rows ACTIVE and
  all three variables strictly positive.  The objective was built as
  c_max = A^T lambda with lambda = (1, 2, 3) > 0, so lambda is the unique dual,
  strict complementarity holds, and x* is the unique nondegenerate optimum.

    KNOWN OPTIMAL:  z*_max = 8(1) + 3(2) + 10(1) = 24
                    x*      = (1, 2, 1)
                    duals   = (1, 2, 3)      (for A x <= b)

  pounce minimizes, so c = -c_max and obj* = -24.  With pounce's convention
  c + G^T z - z_lb = 0, the returned z must equal lambda = (1,2,3).

SCALED VARIANTS (same LP, invariant optimum -- this is the whole point)

  (a) row-scaled:  row 1 multiplied by R11 = 1e8.
      primal x* unchanged;  duals become lambda / R  = (1e-8, 2, 3).
  (b) row+col scaled: additionally x1 = 1e-8 * u1, i.e. column 1 of A and
      c1 multiplied by S11 = 1e-8.
      u* = (1e8, 2, 1);  objective still -24;  row duals still (1e-8, 2, 3).

  A "true" answer that changes between (base), (a) and (b) is a scaling bug.

ORACLES
  1. EXACT vertex enumeration in Fractions over the *exact float data pounce
     receives* (C(6,3) = 20 basis candidates) -- ground truth, no floating point.
  2. scipy.optimize.linprog (HiGHS), which reports row duals in the
     Lagrangian sign convention (marginals <= 0 for <= rows in a min problem).
  3. cvxpy / CLARABEL, an independent IPM, with its constraint dual variables.
"""
import itertools
import time
from fractions import Fraction

import numpy as np

# ----------------------------------------------------------------- base data
A = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 0.0], [1.0, 0.0, 3.0]])
b = np.array([4.0, 4.0, 4.0])
c_max = np.array([8.0, 3.0, 10.0])
KNOWN_X = np.array([1.0, 2.0, 1.0])
KNOWN_OBJ_MIN = -24.0
KNOWN_DUALS = np.array([1.0, 2.0, 3.0])


def make_variant(row_scale, col_scale):
    """Return (c, G, h, known_x, known_duals) for the scaled LP."""
    R = np.asarray(row_scale, dtype=float)
    S = np.asarray(col_scale, dtype=float)
    G = (R[:, None] * A) * S[None, :]
    h = R * b
    c = -c_max * S
    known_u = KNOWN_X / S
    known_z = KNOWN_DUALS / R
    return c, G, h, known_u, known_z


# --------------------------------------------------- exact vertex enumeration
def exact_lp_min(c, G, h):
    """Exact optimal (obj, x) of  min c.x  s.t.  Gx <= h, x >= 0  by enumerating
    all n-subsets of the m+n hyperplanes.  Fractions => zero roundoff."""
    n = len(c)
    m = len(h)
    Gf = [[Fraction(v) for v in row] for row in G]
    hf = [Fraction(v) for v in h]
    # append -x_j <= 0
    for j in range(n):
        Gf.append([Fraction(-1) if k == j else Fraction(0) for k in range(n)])
        hf.append(Fraction(0))
    cf = [Fraction(v) for v in c]
    M = m + n
    best = None
    for S in itertools.combinations(range(M), n):
        # solve the n x n system exactly by Gauss-Jordan
        aug = [[Gf[i][j] for j in range(n)] + [hf[i]] for i in S]
        singular = False
        for col in range(n):
            piv = next((r for r in range(col, n) if aug[r][col] != 0), None)
            if piv is None:
                singular = True
                break
            aug[col], aug[piv] = aug[piv], aug[col]
            pv = aug[col][col]
            aug[col] = [v / pv for v in aug[col]]
            for r in range(n):
                if r != col and aug[r][col] != 0:
                    f = aug[r][col]
                    aug[r] = [a - f * bb for a, bb in zip(aug[r], aug[col])]
        if singular:
            continue
        x = [aug[r][n] for r in range(n)]
        if any(sum(Gf[i][j] * x[j] for j in range(n)) > hf[i] for i in range(M)):
            continue
        obj = sum(cf[j] * x[j] for j in range(n))
        if best is None or obj < best[0]:
            best = (obj, x)
    return best


def rel(a, b_):
    return abs(a - b_) / max(1.0, abs(b_))


# --------------------------------------------------------------------- solve
import pounce
from scipy.optimize import linprog
import cvxpy as cp

VARIANTS = [
    ("base            ", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
    ("row x1e8        ", [1e8, 1.0, 1.0], [1.0, 1.0, 1.0]),
    ("row1e8 + col1e-8", [1e8, 1.0, 1.0], [1e-8, 1.0, 1.0]),
    ("row1e8 + col1e-8 (2 rows)", [1e8, 1.0, 1e-6], [1e-8, 1.0, 1.0]),
]

rows = []
worst = 0.0
fails = []

for name, rs, cs in VARIANTS:
    c, G, h, known_u, known_z = make_variant(rs, cs)
    lb = np.zeros(3)

    # exact ground truth on the *actual float data*
    t0 = time.perf_counter()
    exact_obj, exact_x = exact_lp_min(c, G, h)
    t_exact = time.perf_counter() - t0
    exact_obj_f = float(exact_obj)
    exact_x_f = np.array([float(v) for v in exact_x])

    # pounce
    t0 = time.perf_counter()
    rp = pounce.solve_qp(P=None, c=c, G=G, h=h, lb=lb)
    t_p = time.perf_counter() - t0
    x_p = np.asarray(rp.x, dtype=float)
    z_p = np.asarray(rp.z, dtype=float)
    obj_p = float(rp.obj)

    # scipy HiGHS
    t0 = time.perf_counter()
    ls = linprog(c, A_ub=G, b_ub=h, bounds=[(0, None)] * 3, method="highs")
    t_s = time.perf_counter() - t0
    z_scipy = -np.asarray(ls.ineqlin.marginals, dtype=float)  # -> pounce sign

    # cvxpy / CLARABEL
    xv = cp.Variable(3)
    con_ub = G @ xv <= h
    prob = cp.Problem(cp.Minimize(c @ xv), [con_ub, xv >= 0])
    t0 = time.perf_counter()
    try:
        prob.solve(solver=cp.CLARABEL)
        obj_cvx = float(prob.value)
        z_cvx = np.asarray(con_ub.dual_value, dtype=float)
        st_cvx = prob.status
    except Exception as exc:
        obj_cvx, z_cvx, st_cvx = float("nan"), np.full(3, np.nan), f"ERR {type(exc).__name__}"
    t_c = time.perf_counter() - t0

    # relative errors (scale-free: compare u* componentwise relatively)
    e_obj_exact = rel(obj_p, exact_obj_f)
    e_obj_known = rel(obj_p, KNOWN_OBJ_MIN)
    e_obj_scipy = rel(obj_p, float(ls.fun))
    e_obj_cvx = rel(obj_p, obj_cvx)
    denom_x = np.maximum(np.abs(known_u), 1.0)
    e_x = float(np.max(np.abs(x_p - known_u) / denom_x))
    denom_z = np.maximum(np.abs(known_z), 1.0)
    e_z_known = float(np.max(np.abs(z_p - known_z) / denom_z))
    e_z_scipy = float(np.max(np.abs(z_p - z_scipy) / denom_z))
    e_z_cvx = float(np.max(np.abs(z_p - z_cvx) / denom_z))

    # KKT stationarity residual for the point pounce reports (scaled by |c|)
    stat = c + G.T @ z_p - np.asarray(rp.z_lb, dtype=float)
    e_stat = float(np.max(np.abs(stat)) / max(1.0, float(np.max(np.abs(c)))))

    print(f"=== variant: {name.strip()}  (R={rs}, S={cs}) ===")
    print(f"  exact vertex enum : obj={exact_obj_f:.12e}  x={exact_x_f}  ({t_exact:.3f}s)")
    print(f"  pounce            : status={rp.status} obj={obj_p:.12e} iters={rp.iters} t={t_p:.4f}s")
    print(f"    x = {x_p}")
    print(f"    z = {z_p}          (expected {known_z})")
    print(f"    z_lb = {np.asarray(rp.z_lb, dtype=float)}   kkt_error={rp.kkt_error:.3e}")
    print(f"  scipy HiGHS       : status={ls.status} obj={float(ls.fun):.12e} t={t_s:.4f}s")
    print(f"    x = {ls.x}")
    print(f"    z = {z_scipy}")
    print(f"  cvxpy CLARABEL    : status={st_cvx} obj={obj_cvx:.12e} t={t_c:.4f}s")
    print(f"    z = {z_cvx}")
    # feasibility audit in UNSCALED units (row i divided back by R_i) -- the
    # only fair way to compare violations across row scalings.
    Rv = np.asarray(rs, dtype=float)
    for who, xx in (("pounce", x_p), ("scipy", np.asarray(ls.x, dtype=float))):
        viol = float(np.max(np.maximum((G @ xx - h) / Rv, 0.0)))
        print(f"    unscaled max row violation [{who}] = {viol:.3e}")
    print(f"  rel_err obj vs exact={e_obj_exact:.2e} vs known(-24)={e_obj_known:.2e} "
          f"vs scipy={e_obj_scipy:.2e} vs clarabel={e_obj_cvx:.2e}")
    print(f"  rel_err x vs known={e_x:.2e}")
    print(f"  rel_err z vs known={e_z_known:.2e} vs scipy={e_z_scipy:.2e} vs clarabel={e_z_cvx:.2e}")
    print(f"  scaled stationarity residual = {e_stat:.2e}")
    print()

    bad = []
    if rp.status != "optimal":
        bad.append(f"status={rp.status}")
    if e_obj_exact > 1e-4:
        bad.append(f"obj_err={e_obj_exact:.2e}")
    if e_x > 1e-4:
        bad.append(f"x_err={e_x:.2e}")
    if e_z_known > 1e-4:
        bad.append(f"dual_err={e_z_known:.2e}")
    if bad:
        fails.append((name.strip(), ", ".join(bad)))
    worst = max(worst, e_obj_exact, e_x, e_z_known)
    rows.append((name.strip(), rp.status, obj_p, e_obj_exact, e_x, e_z_known, rp.iters, t_p, t_s, t_c))

# ------------------------------------------------------- invariance summary
print("| variant | status | obj | obj_err | x_err | dual_err | iters | pounce | scipy | clarabel |")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]:.10f} | {r[3]:.1e} | {r[4]:.1e} | {r[5]:.1e} | "
          f"{r[6]} | {r[7]:.4f}s | {r[8]:.4f}s | {r[9]:.4f}s |")

objs = [r[2] for r in rows]
print(f"\nobjective spread across scalings = {max(objs) - min(objs):.3e} "
      f"(must be ~0: scaling is objective-invariant)")
print(f"worst relative error over all variants and quantities = {worst:.2e}")

if not fails:
    print("VERDICT: PASS")
else:
    print("VERDICT: FAIL " + "; ".join(f"[{n}: {w}]" for n, w in fails))
