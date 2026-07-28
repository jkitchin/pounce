"""Adversary cross-check: DEGENERATE SDPs -- strict-complementarity failure and
                          a non-unique optimal FACE.
Family: sdp   Class: degeneracy / constraint-qualification / non-unique optima

Two hand-constructed 3x3 SDPs in standard primal form

    min  <C,X>   s.t.  <A_i,X> = b_i,   X >= 0   (PSD)

with analytically derivable optima.  Both are STRICTLY feasible (Slater holds
for primal and dual, so there is no duality gap) -- the difficulty is purely
DEGENERACY, which is exactly the regime interior-point methods lose their
superlinear convergence in (Alizadeh, Haeberly & Overton, "Complementarity and
nondegeneracy in semidefinite programming", Math. Prog. 77 (1997) 111-128;
see also Todd, "Semidefinite optimization", Acta Numerica 10 (2001) 515-560,
Sec. 4, on the role of strict complementarity for IPM convergence).

--------------------------------------------------------------------------
VARIANT A -- STRICT COMPLEMENTARITY FAILS  (rank X* + rank S* = 2 < 3 = n)

    C  = 3*E11 + E33 = diag(3,0,1)
    A1 = E11,                          b1 = 1        ->  X11 = 1
    A2 = E22 + (E13 + E31),            b2 = 0        ->  X22 + 2*X13 = 0

  Primal:  X psd, X11 = 1, X22 = -2*X13.  X22 >= 0 forces X13 <= 0, and the
  {1,3} minor gives X33 >= X13^2 >= 0.  So <C,X> = 3 + X33 >= 3, with equality
  iff X33 = 0 => X13 = 0 => X22 = 0 => (psd) row/col 2 vanishes.
      p* = 3,   X* = diag(1,0,0)  (UNIQUE),  rank X* = 1.

  Dual:  max b'y  s.t.  S = C - y1*A1 - y2*A2 >= 0,
      S = [[3-y1, 0, -y2], [0, -y2, 0], [-y2, 0, 1]],  objective = y1.
  PSD needs y2 <= 0 and (3-y1) - y2^2 >= 0, i.e. y1 <= 3 - y2^2.
      d* = 3 at y* = (3, 0)  (UNIQUE),  S* = diag(0,0,1),  rank S* = 1.

  rank X* + rank S* = 1 + 1 = 2 < 3 = n  ==>  STRICT COMPLEMENTARITY FAILS,
  yet strong duality holds (p* = d* = 3) and both problems have Slater points
  (primal: X=[[1,0,-.1],[0,.2,0],[-.1,0,1]] > 0; dual: y=(-2,-1) gives S > 0).
  The e2 direction lies in the kernel of BOTH X* and S*: the central path
  approaches the optimum tangentially and mu -> 0 only sublinearly there.

--------------------------------------------------------------------------
VARIANT B -- NON-UNIQUE OPTIMAL SET (a whole 2-dimensional face of optima)

    C  = I + E33 = diag(1,1,2),   A1 = I, b1 = 1   ->  trace X = 1

      <C,X> = trace(X) + X33 = 1 + X33  >= 1.
      p* = 1;  optimal face = { X >= 0 : trace X = 1, X33 = 0 }
             = 2x2 psd matrices of trace 1 embedded in the leading block,
               a 2-DIMENSIONAL face (a spectrahedral disk), NOT a point.
  Dual: S = C - y*I = diag(1-y,1-y,2-y) >= 0  =>  y <= 1, d* = 1,
        S* = diag(0,0,1), rank 1; max rank over optimal X* is 2, so
        1 + 2 = 3 = n: strict complementarity HOLDS here, the pathology is
        pure non-uniqueness.  X is graded on OBJECTIVE + PSD FEASIBILITY only.

--------------------------------------------------------------------------
POUNCE ENCODING.  pounce.solve_socp takes  min c'v  s.t.  s = h - G v in K.
There are no equality constraints, so each X is parametrized by free variables
that satisfy <A_i,X> = b_i identically, and svec(X) = h - G v is the psd block.
Constant terms in the objective are added back by hand.

svec layout (re-derived and asserted numerically below):
    lower triangle, COLUMN-MAJOR, off-diagonals * sqrt(2):
    svec(M) = [M00, s2*M10, s2*M20, M11, s2*M21, M22]  for n = 3.
"""
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

s2 = np.sqrt(2.0)


def svec3(M):
    return np.array([M[0, 0], s2 * M[1, 0], s2 * M[2, 0],
                     M[1, 1], s2 * M[2, 1], M[2, 2]])


def smat3(v):
    return np.array([[v[0], v[1] / s2, v[2] / s2],
                     [v[1] / s2, v[3], v[4] / s2],
                     [v[2] / s2, v[4] / s2, v[5]]])


# --- svec layout check: <X,Y> = svec(X).svec(Y), and smat(svec(M)) == M ---
_rng = np.random.default_rng(0)
for _ in range(5):
    A_ = _rng.standard_normal((3, 3)); A_ = A_ + A_.T
    B_ = _rng.standard_normal((3, 3)); B_ = B_ + B_.T
    assert abs(svec3(A_) @ svec3(B_) - np.trace(A_ @ B_)) < 1e-12, "svec layout wrong"
    assert np.allclose(smat3(svec3(A_)), A_), "smat/svec mismatch"

import pounce      # noqa: E402
import cvxpy as cp  # noqa: E402

# =========================================================================
# VARIANT A: X = [[1, a, c], [a, -2c, d], [c, d, t]];  free v = (a, c, d, t)
#            objective 3*X11 + X33 = 3 + t
# svec(X) = [1, s2*a, s2*c, -2c, s2*d, t] = h - G v
# =========================================================================
A_h = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
A_G = np.array([
    [0.0, 0.0, 0.0, 0.0],    # X00    = 1
    [-s2, 0.0, 0.0, 0.0],    # s2*X10 = s2*a
    [0.0, -s2, 0.0, 0.0],    # s2*X20 = s2*c
    [0.0, 2.0, 0.0, 0.0],    # X11    = -2c
    [0.0, 0.0, -s2, 0.0],    # s2*X21 = s2*d
    [0.0, 0.0, 0.0, -1.0],   # X22    = t
])
A_c = np.array([0.0, 0.0, 0.0, 1.0])
A_const = 3.0


def A_mat(v):
    a, c, d, t = v
    return np.array([[1.0, a, c], [a, -2.0 * c, d], [c, d, t]])


_vt = np.array([0.11, -0.23, 0.37, 0.9])
assert np.allclose(A_h - A_G @ _vt, svec3(A_mat(_vt))), "variant A encoding wrong"
_Ct = np.diag([3.0, 0.0, 1.0])
assert abs(np.trace(_Ct @ A_mat(_vt)) - (A_c @ _vt + A_const)) < 1e-12, "variant A objective wrong"

# =========================================================================
# VARIANT B: trace X = 1 -> X11 = 1 - u - t.
#            X = [[1-u-t, a, c], [a, u, d], [c, d, t]];  v = (u, t, a, c, d)
#            objective trace(X) + X33 = 1 + t
# svec(X) = [1-u-t, s2*a, s2*c, u, s2*d, t] = h - G v
# =========================================================================
B_h = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
B_G = np.array([
    [1.0, 1.0, 0.0, 0.0, 0.0],   # X00    = 1 - u - t
    [0.0, 0.0, -s2, 0.0, 0.0],   # s2*X10 = s2*a
    [0.0, 0.0, 0.0, -s2, 0.0],   # s2*X20 = s2*c
    [-1.0, 0.0, 0.0, 0.0, 0.0],  # X11    = u
    [0.0, 0.0, 0.0, 0.0, -s2],   # s2*X21 = s2*d
    [0.0, -1.0, 0.0, 0.0, 0.0],  # X22    = t
])
B_c = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
B_const = 1.0


def B_mat(v):
    u, t, a, c, d = v
    return np.array([[1.0 - u - t, a, c], [a, u, d], [c, d, t]])


_vt = np.array([0.3, 0.2, 0.05, -0.07, 0.11])
assert np.allclose(B_h - B_G @ _vt, svec3(B_mat(_vt))), "variant B encoding wrong"
_Ct = np.diag([1.0, 1.0, 2.0])
assert abs(np.trace(_Ct @ B_mat(_vt)) - (B_c @ _vt + B_const)) < 1e-12, "variant B objective wrong"

VARIANTS = {
    "A_strict_comp_fail": dict(
        G=A_G, h=A_h, c=A_c, const=A_const, mat=A_mat, C=np.diag([3.0, 0.0, 1.0]),
        pstar=3.0, nfree=4,
        cvx=lambda X: ([cp.trace(np.diag([1.0, 0.0, 0.0]) @ X) == 1,
                        X[1, 1] + 2 * X[0, 2] == 0],
                       cp.trace(np.diag([3.0, 0.0, 1.0]) @ X)),
    ),
    "B_nonunique_face": dict(
        G=B_G, h=B_h, c=B_c, const=B_const, mat=B_mat, C=np.diag([1.0, 1.0, 2.0]),
        pstar=1.0, nfree=5,
        cvx=lambda X: ([cp.trace(X) == 1],
                       cp.trace(np.diag([1.0, 1.0, 2.0]) @ X)),
    ),
}


def num_rank(M, tol=1e-6):
    ev = np.linalg.eigvalsh(M)
    return int(np.sum(np.abs(ev) > tol * max(1.0, np.max(np.abs(ev)))))


results = {}
for name, V in VARIANTS.items():
    print(f"\n================ variant {name} ================")
    print(f"analytic p* = {V['pstar']}")

    t0 = time.perf_counter()
    r = pounce.solve_socp(c=V["c"], G=V["G"], h=V["h"], cones=[("psd", 3)])
    tp = time.perf_counter() - t0
    v = np.asarray(r.x, float) if r.x is not None else np.full(V["nfree"], np.nan)
    Xp = V["mat"](v)
    obj_p = float(np.trace(V["C"] @ Xp))
    ev = np.linalg.eigvalsh(Xp)
    print(f"pounce   : status={r.status} obj={obj_p:.10e} t={tp:.4f}s "
          f"iters={getattr(r, 'iters', None)}")
    print(f"           v={v}")
    print(f"           eig(X)={ev}  min_eig={ev.min():.3e}  rank~{num_rank(Xp)}")
    print(f"           X=\n{np.array2string(Xp, precision=6)}")

    oracle = {}
    for sname, solver in (("SCS", cp.SCS), ("CLARABEL", cp.CLARABEL)):
        X = cp.Variable((3, 3), symmetric=True)
        cons, obj = V["cvx"](X)
        prob = cp.Problem(cp.Minimize(obj), cons + [X >> 0])
        t0 = time.perf_counter()
        try:
            prob.solve(solver=solver)
            st, val, to = prob.status, float(prob.value), time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            st, val, to = f"ERROR:{type(exc).__name__}", float("nan"), time.perf_counter() - t0
        oracle[sname] = (st, val, to)
        print(f"{sname:9s}: status={st} obj={val:.10e} t={to:.4f}s")

    vals = [val for (st, val, _) in oracle.values()
            if isinstance(st, str) and "optimal" in st and np.isfinite(val)]
    agree = len(vals) == 2 and abs(vals[0] - vals[1]) < 1e-5
    err_known = abs(obj_p - V["pstar"])
    err_oracle = (abs(obj_p - float(np.mean(vals))) if vals else float("nan"))
    psd_viol = max(0.0, -float(ev.min()))
    ok = ("optimal" in str(r.status).lower() and err_known < 1e-4
          and psd_viol < 1e-6 and agree and err_oracle < 1e-4)
    results[name] = dict(status=r.status, obj=obj_p, t=tp, err_known=err_known,
                         err_oracle=err_oracle, psd_viol=psd_viol, agree=agree,
                         ok=ok, rank=num_rank(Xp), oracle=oracle,
                         iters=getattr(r, "iters", None))
    print(f"oracles_agree(<1e-5) = {agree}   oracle_vals={vals}")
    print(f"abs_err_vs_known = {err_known:.3e}   abs_err_vs_oracle = {err_oracle:.3e}   "
          f"psd_violation = {psd_viol:.3e}")

print("\n=== summary ===")
for name, R in results.items():
    print(f"{name:22s} status={R['status']:10s} obj={R['obj']:.8f} "
          f"err_known={R['err_known']:.2e} psd_viol={R['psd_viol']:.2e} "
          f"t={R['t']:.4f}s ok={R['ok']}")

all_ok = all(R["ok"] for R in results.values())
if all_ok:
    print("VERDICT: PASS")
else:
    bad = [n for n, R in results.items() if not R["ok"]]
    print(f"VERDICT: FAIL (variants={bad})")
