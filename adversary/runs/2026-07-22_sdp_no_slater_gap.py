"""Adversary cross-check: pathological SDP with NO strictly feasible point
                          (Slater fails) and a NONZERO duality gap.
Family: sdp   Class: ill-conditioned / weakly feasible (no Slater point)

Base problem (the textbook nonzero-duality-gap SDP):

    minimize    x1
    subject to  F(x) = [[ 0 , x1,   0   ],
                        [ x1, x2,   0   ],
                        [ 0 , 0 , x1 + 1]]  >= 0   (3x3 PSD)

SOURCE: L. Vandenberghe and S. Boyd, "Semidefinite Programming", SIAM Review
    38(1):49-95, 1996, Section 4 (Duality) -- the standard example of an SDP
    with a positive duality gap (p* = 0, d* = -1).  The same instance recurs in
    Waki/Muramatsu-style catalogues of weakly feasible / pathological SDPs.

KNOWN OPTIMAL (analytic, exact):
    A PSD matrix with a zero diagonal entry must have that whole row/column
    zero, so the leading 2x2 block [[0, x1],[x1, x2]] >= 0  <=>  x1 = 0, x2 >= 0.
    With x1 = 0 the (3,3) entry is 1 > 0.  Hence
        feasible set = { (0, x2) : x2 >= 0 },   p* = 0  (attained).
    F(x) is SINGULAR at EVERY feasible point (rank <= 2 of 3), so there is no
    Slater point and no central path; the dual optimum is d* = -1 (gap = 1).

TWO VARIANTS ARE RUN:
  (A) bounded  -- the same problem plus x2 <= 1.  This changes nothing
      analytically (p* = 0, feasible set {(0,x2): 0<=x2<=1}, still no Slater
      point, still gap 1) but makes the instance NUMERICALLY well posed, so the
      cvxpy oracles can actually solve it.  This is the graded test.
  (B) unbounded -- the raw V&B instance.  Reported for information only: in
      floating point the PSD constraint det = -x1^2 >= 0 can be "satisfied" to
      machine precision with |x1| ~ sqrt(eps * x2), so as x2 -> inf the problem
      is numerically unbounded below.  No solver can be graded on it (SCS
      returns -14.9, CLARABEL raises SolverError) -- see the report.

svec layout (pounce): lower triangle, COLUMN-MAJOR, off-diagonals * sqrt(2).
    3x3 -> svec(M) = [M00, s2*M10, s2*M20, M11, s2*M21, M22].
    Re-derived and asserted numerically below against <X,Y> = svec(X).svec(Y).
"""
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

s2 = np.sqrt(2.0)


def svec3(M):
    """lower triangle, column-major, off-diag * sqrt(2)."""
    return np.array([M[0, 0], s2 * M[1, 0], s2 * M[2, 0],
                     M[1, 1], s2 * M[2, 1], M[2, 2]])


_rng = np.random.default_rng(0)
for _ in range(5):
    A_ = _rng.standard_normal((3, 3)); A_ = A_ + A_.T
    B_ = _rng.standard_normal((3, 3)); B_ = B_ + B_.T
    assert abs(svec3(A_) @ svec3(B_) - np.trace(A_ @ B_)) < 1e-12, "svec layout wrong"

KNOWN_OPTIMAL = 0.0          # p*  (both variants)
KNOWN_DUAL = -1.0            # d*  -> duality gap 1

# ------------------------------------------------------------------ encoding
# v = (x1, x2);  s = h - G v must equal svec(F(x)) block by block.
# svec(F) rows: [F00, s2*F10, s2*F20, F11, s2*F21, F22]
#             = [ 0 , s2*x1 ,   0   , x2 ,   0   , x1+1 ]
G_psd = np.array([
    [0.0,  0.0],   # F00    = 0
    [-s2,  0.0],   # s2*F10 = s2*x1
    [0.0,  0.0],   # s2*F20 = 0
    [0.0, -1.0],   # F11    = x2
    [0.0,  0.0],   # s2*F21 = 0
    [-1.0, 0.0],   # F22    = x1 + 1
])
h_psd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
c = np.array([1.0, 0.0])

# encoding sanity check against the explicit matrix
xt = np.array([0.3, 1.7])
Ft = np.array([[0.0, xt[0], 0.0], [xt[0], xt[1], 0.0], [0.0, 0.0, xt[0] + 1.0]])
assert np.allclose(h_psd - G_psd @ xt, svec3(Ft)), "affine map encoding wrong"

# bound x2 <= 1 as a 1x1 PSD cone:  s = 1 - x2 >= 0
G_bnd = np.array([[0.0, 1.0]])
h_bnd = np.array([1.0])

VARIANTS = {
    "A_bounded":   (np.vstack([G_psd, G_bnd]), np.concatenate([h_psd, h_bnd]),
                    [("psd", 3), ("psd", 1)], True),
    "B_unbounded": (G_psd, h_psd, [("psd", 3)], False),
}

import pounce   # noqa: E402
import cvxpy as cp  # noqa: E402


def run_pounce(G, h, cones):
    t0 = time.perf_counter()
    r = pounce.solve_socp(c=c, G=G, h=h, cones=cones)
    t = time.perf_counter() - t0
    x = np.asarray(r.x, float) if r.x is not None else np.full(2, np.nan)
    return r.status, float(c @ x), x, t, getattr(r, "iters", None)


def run_cvxpy(solver, bounded):
    x = cp.Variable(2)
    z = np.zeros((1, 1))
    F = cp.bmat([[z, cp.reshape(x[0], (1, 1), order="C"), z],
                 [cp.reshape(x[0], (1, 1), order="C"),
                  cp.reshape(x[1], (1, 1), order="C"), z],
                 [z, z, cp.reshape(x[0] + 1.0, (1, 1), order="C")]])
    cons = [F >> 0] + ([x[1] <= 1] if bounded else [])
    prob = cp.Problem(cp.Minimize(x[0]), cons)
    t0 = time.perf_counter()
    try:
        prob.solve(solver=solver)
    except Exception as exc:                      # noqa: BLE001
        return "ERROR:%s" % type(exc).__name__, float("nan"), None, time.perf_counter() - t0
    t = time.perf_counter() - t0
    val = prob.value if prob.value is not None else float("nan")
    return prob.status, float(val), (None if x.value is None else np.asarray(x.value, float)), t


def eig_of(x):
    F = np.array([[0.0, x[0], 0.0], [x[0], x[1], 0.0], [0.0, 0.0, x[0] + 1.0]])
    return np.linalg.eigvalsh(F)


print("=== problem ===")
print("Vandenberghe & Boyd 1996 SIAM Rev. 38(1) Sec.4 -- nonzero-duality-gap SDP")
print(f"analytic p*={KNOWN_OPTIMAL}  d*={KNOWN_DUAL}  (gap=1; F(x) singular at every feasible x)")

summary = {}
for name, (G, h, cones, bounded) in VARIANTS.items():
    print(f"\n--- variant {name} (bounded={bounded}) ---")
    st, obj, xp, tp, iters = run_pounce(G, h, cones)
    ev = eig_of(xp)
    print(f"pounce   : status={st} obj={obj:.6e} t={tp:.4f}s iters={iters} x={xp}")
    print(f"           eig(F)={ev}  min_eig={np.min(ev):.3e}  "
          f"spread={np.max(np.abs(ev)) / max(np.min(np.abs(ev)), 1e-300):.3e}")
    orc = {}
    for sname, solver in (("SCS", cp.SCS), ("CLARABEL", cp.CLARABEL)):
        ost, oval, ox, ot = run_cvxpy(solver, bounded)
        orc[sname] = (ost, oval, ot)
        print(f"{sname:9s}: status={ost} obj={oval!r} t={ot:.4f}s x={ox}")
    good = [v for (s_, v, _) in orc.values() if isinstance(s_, str) and "optimal" in s_ and np.isfinite(v)]
    agree = len(good) == 2 and abs(good[0] - good[1]) < 1e-3
    summary[name] = dict(status=st, obj=obj, t=tp, eig=ev, oracles=orc,
                         oracles_agree=agree, iters=iters)
    print(f"oracles_agree(<1e-3)={agree}  oracle_vals={good}")

# ------------------------------------------------------------------- grading
A = summary["A_bounded"]
abs_err = abs(A["obj"] - KNOWN_OPTIMAL)
psd_viol = max(0.0, -float(np.min(A["eig"])))
print("\n=== graded variant: A_bounded ===")
print(f"abs_err_vs_known(p*=0) = {abs_err:.3e}")
print(f"pounce_psd_violation   = {psd_viol:.3e}")
print(f"oracles_agree          = {A['oracles_agree']}")

ok = ("optimal" in str(A["status"]).lower() and abs_err < 1e-4
      and psd_viol < 1e-6 and A["oracles_agree"])
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={A['status']}, abs_err={abs_err:.3e}, psd_viol={psd_viol:.3e})")
