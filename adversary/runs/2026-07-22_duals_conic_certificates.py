"""Adversary cross-check: CONIC DUAL CERTIFICATE correctness (soc / psd / exp / pow)

Family: socp / sdp / exp / power     Class: dual multipliers & mathematical invariants
Dimension: duals, multipliers, sensitivity and mathematical invariants

For one well-posed problem per cone type, each with a hand-derived optimum AND a
hand-derived dual, we verify the certificate returned by `pounce.solve_socp`:

  (a) DUAL CONE MEMBERSHIP  z_i in K_i^*   (soc/psd/nonneg self-dual; exp/pow NOT)
  (b) ZERO DUALITY GAP      p* == d*
  (c) COMPLEMENTARY SLACKNESS  <s_i, z_i> = 0,  s = h - Gx
  (d) DUAL FEASIBILITY      P x + c + A'y + G'z = 0
  (e) cvxpy cross-check of the primal optimum and (where the mapping is
      unambiguous) the dual values.

--------------------------------------------------------------------------
CONVENTIONS (read read-only from pounce source/docs, not assumed)
--------------------------------------------------------------------------
pounce standard form (docs/src/convex-solver.md, python/pounce/qp.py:515):

    min  1/2 x'P x + c'x   s.t.  A x = b,   s = h - G x  in  K

Stationarity as computed by pounce itself
(crates/pounce-convex/src/qp.rs::kkt_residuals_inner):

    P x + c + A'y + G'z  (- z_lb + z_ub) = 0

So the Lagrangian is L = 1/2 x'Px + c'x + y'(Ax-b) + z'(Gx-h), hence

    dual objective  d(y,z) = -1/2 x'P x - b'y - h'z ,  z in K^*
    p* - d* = x'Px + c'x + b'y + h'z = <s, z>          (gap == complementarity)

Cone definitions (python/pounce/qp.py docstring for solve_socp):
    soc(d)  : { s : s0 >= ||s_{1:}||_2 }                      SELF-DUAL
    psd(n)  : svec (lower triangle, column-major, off-diag * sqrt2), smat(s) psd
                                                              SELF-DUAL
    exp     : { (a,b,c) : b*exp(a/b) <= c, b > 0 } (closure)  NOT self-dual
    pow(al) : { (a,b,c) : |a| <= b^al c^(1-al), b,c >= 0 }    NOT self-dual

--------------------------------------------------------------------------
DUAL CONES -- derived here, then checked against MOSEK
--------------------------------------------------------------------------
Definition used: K^* = { z : <z, s> >= 0 for all s in K }.
(This is the sign convention forced by the Lagrangian above: dual feasibility
needs z'(Gx-h) = -z's <= 0 on the cone.)

EXPONENTIAL CONE.  With K_exp = cl{(a,b,c) : b>0, b e^{a/b} <= c}:
  <z,s> = u a + v b + w c >= 0 for all such (a,b,c).
  c unbounded above  =>  w >= 0.  Minimizing over c sets c = b e^{a/b}.
  Positive homogeneity in (a,b) lets us fix b = 1:
        g(a) = u a + v + w e^a >= 0  for all a in R.
  w > 0: g' = u + w e^a = 0 needs u < 0 (u > 0 diverges as a -> -inf).
         a* = ln(-u/w),  g(a*) = u ln(-u/w) + v - u >= 0.
         Put p = -u > 0:  v + p - p ln(p/w) >= 0  <=>  w >= p e^{-1} e^{-v/p}.
  u = 0: inf_a g = v (a -> -inf)  =>  v >= 0, w >= 0.
  Therefore

    K_exp^* = cl{ (u,v,w) : u < 0,  w >= -u * e^{-1} * exp(v/u) }
              union { (0,v,w) : v >= 0, w >= 0 }.

  CHECK vs MOSEK Modeling Cookbook v3.3.0 (Sec. 5.3 / Sec. 4.4, "Exponential
  cone" and its dual):  MOSEK writes K_exp = {x : x1 >= x2 exp(x3/x2), x2>0}
  and K_exp^* = {u : u1 >= -u3 e^{-1} exp(u2/u3), u3 < 0} u {(u1,u2,0): u1,u2>=0}.
  Mapping MOSEK (x1,x2,x3) = pounce (c,b,a), so (u1,u2,u3) = (w,v,u):
  "u1 >= -u3 e^{-1} exp(u2/u3)"  ==  "w >= -u e^{-1} exp(v/u)".  IDENTICAL.

  Hand sanity point: s = (0,1,1) is on dK_exp (1*e^0 = 1 <= 1).
  z = (-1,-1,1): u=-1<0, -u e^{-1} exp(v/u) = 1*e^{-1}*e^{1} = 1 <= w = 1, on
  dK_exp^*.  <s,z> = 0*(-1) + 1*(-1) + 1*(1) = 0.  A complementary pair.

POWER CONE.  With K_pow(al) = {(a,b,c) : |a| <= b^al c^(1-al), b,c >= 0}:
  worst case a = -sign(u) b^al c^(1-al); b,c free upward with a=0 => v,w >= 0.
  min over the cone of v b + w c subject to b^al c^(1-al) = 1 is, by the
  weighted AM-GM (b = al k/v, c = (1-al) k/w gives v b + w c = k and the
  constraint gives k = (v/al)^al (w/(1-al))^(1-al)), so

    K_pow(al)^* = { (u,v,w) : |u| <= (v/al)^al * (w/(1-al))^(1-al), v,w >= 0 }.

  CHECK vs MOSEK Cookbook Sec. 4.1.2:  MOSEK's P_3^{al,1-al} =
  {x : x1^al x2^{1-al} >= |x3|, x1,x2>=0} has dual
  {u : (u1/al)^al (u2/(1-al))^{1-al} >= |u3|, u1,u2>=0}.
  Mapping MOSEK (x1,x2,x3) = pounce (b,c,a) => (u1,u2,u3) = (v,w,u).  IDENTICAL.

  Hand sanity point: al=1/3, s = (4,1,8) on the boundary (1^{1/3} 8^{2/3} = 4).
  z = (-1, 4/3, 1/3): (v/al)^al (w/(1-al))^{1-al} = (4)^{1/3} (1/2)^{2/3}
  = 2^{2/3} 2^{-2/3} = 1 = |u|, on the boundary.
  <s,z> = 4*(-1) + 1*(4/3) + 8*(1/3) = -4 + 4/3 + 8/3 = 0.  Complementary pair.
"""

import math
import time

import numpy as np

SQ2 = math.sqrt(2.0)
TOL = 1e-6


# ---------------------------------------------------------------- cone helpers
def svec(M):
    """Lower triangle, column by column, off-diagonals * sqrt(2)."""
    n = M.shape[0]
    out = []
    for j in range(n):
        for i in range(j, n):
            out.append(M[i, j] if i == j else SQ2 * M[i, j])
    return np.array(out, float)


def smat(v):
    n = int((math.isqrt(8 * len(v) + 1) - 1) // 2)
    M = np.zeros((n, n))
    k = 0
    for j in range(n):
        for i in range(j, n):
            if i == j:
                M[i, i] = v[k]
            else:
                M[i, j] = M[j, i] = v[k] / SQ2
            k += 1
    return M


def viol_soc(s):
    return max(0.0, np.linalg.norm(s[1:]) - s[0])


def viol_psd(s):
    return max(0.0, -np.linalg.eigvalsh(smat(s))[0])


def viol_nonneg(s):
    return max(0.0, -np.min(s))


def viol_exp_primal(s):
    """K_exp = cl{(a,b,c): b>0, b e^{a/b} <= c}."""
    a, b, c = s
    if b > 1e-12:
        return max(0.0, b * math.exp(a / b) - c)
    return max(0.0, -b) + max(0.0, -c) + max(0.0, a)


def viol_exp_dual(z):
    """K_exp^* (derived + MOSEK-checked above)."""
    u, v, w = z
    if u < -1e-12:
        return max(0.0, -u * math.exp(-1.0) * math.exp(v / u) - w)
    return abs(min(0.0, u)) + max(0.0, -v) + max(0.0, -w)


def viol_pow_primal(s, al):
    a, b, c = s
    if b < -1e-12 or c < -1e-12:
        return max(0.0, -b) + max(0.0, -c)
    b = max(b, 0.0)
    c = max(c, 0.0)
    return max(0.0, abs(a) - (b ** al) * (c ** (1.0 - al)))


def viol_pow_dual(z, al):
    u, v, w = z
    if v < -1e-12 or w < -1e-12:
        return max(0.0, -v) + max(0.0, -w)
    v = max(v, 0.0)
    w = max(w, 0.0)
    return max(0.0, abs(u) - ((v / al) ** al) * ((w / (1.0 - al)) ** (1.0 - al)))


# ------------------------------------------ self-test of the dual-cone formulas
def selftest():
    """Be more skeptical of my derivation than of pounce: brute-force verify
    K^* = {z : <z,s> >= 0 for all s in K} by sampling K densely, and verify that
    points just OUTSIDE my claimed K^* really do admit a violating s in K."""
    rng = np.random.default_rng(0)
    problems = []

    # --- exp ---
    S = []
    for _ in range(60000):
        b = rng.uniform(0.02, 5.0)
        a = b * rng.uniform(-12.0, 12.0)  # keep a/b bounded so exp is finite
        c = b * math.exp(a / b) * (1.0 + abs(rng.normal()))
        S.append((a, b, c))
    S = np.array(S)
    assert max(viol_exp_primal(s) for s in S[:2000]) < 1e-9

    # in-dual-cone points must have <z,s> >= 0 for every sampled s
    inside = []
    for _ in range(2000):
        u = -rng.uniform(0.05, 5.0)
        v = (-u) * rng.uniform(-12.0, 12.0)
        w = -u * math.exp(-1.0) * math.exp(v / u) * (1.0 + abs(rng.normal()))
        inside.append((u, v, w))
    inside = np.array(inside)
    ip = S @ inside.T
    problems.append(("exp K* inside min <z,s>", ip.min()))

    # points just outside (shrink w below the boundary) must be refuted
    # Points just outside (shrink w below the boundary) must be REFUTED by some
    # s in K.  Random sampling of K misses the refuting s (it can sit at very
    # large |a/b|), so use the analytic minimizer: at b=1 the inner product
    # u a + v + w e^a is minimized at a* = ln(-u/w), s* = (a*, 1, e^{a*}).
    outside = inside.copy()
    outside[:, 2] *= 0.5
    ok_out = np.array([viol_exp_dual(z) > 1e-9 for z in outside])
    refuted = 0
    n_out = 0
    for u, v, w in outside[ok_out]:
        n_out += 1
        astar = math.log(-u / w)
        sstar = np.array([astar, 1.0, math.exp(astar)])
        assert viol_exp_primal(sstar) < 1e-9
        if float(np.dot(sstar, (u, v, w))) < -1e-9:
            refuted += 1
    problems.append(("exp K* outside frac refuted", refuted / max(n_out, 1)))

    # --- pow ---
    al = 1.0 / 3.0
    Sp = []
    for _ in range(60000):
        b = abs(rng.normal()) * 3
        c = abs(rng.normal()) * 3
        a = (b ** al) * (c ** (1 - al)) * rng.uniform(-1, 1)
        Sp.append((a, b, c))
    Sp = np.array(Sp)
    assert max(viol_pow_primal(s, al) for s in Sp[:2000]) < 1e-9

    ins = []
    for _ in range(2000):
        v = abs(rng.normal()) * 3
        w = abs(rng.normal()) * 3
        bnd = ((v / al) ** al) * ((w / (1 - al)) ** (1 - al))
        ins.append((bnd * rng.uniform(-1, 1), v, w))
    ins = np.array(ins)
    problems.append(("pow K* inside min <z,s>", (Sp @ ins.T).min()))

    # analytic refuting s for a point outside K_pow(al)^*:
    # b = al k / v, c = (1-al) k / w with k = 1, a = -sign(u) b^al c^(1-al).
    outs = ins.copy()
    outs[:, 0] = np.sign(outs[:, 0] + 1e-12) * (
        ((outs[:, 1] / al) ** al) * ((outs[:, 2] / (1 - al)) ** (1 - al)) * 1.8
    )
    refuted = 0
    n_out = 0
    for u, v, w in outs:
        if viol_pow_dual((u, v, w), al) <= 1e-9:
            continue
        n_out += 1
        bb, cc_ = al / v, (1 - al) / w
        aa = -math.copysign((bb ** al) * (cc_ ** (1 - al)), u)
        sstar = np.array([aa, bb, cc_])
        assert viol_pow_primal(sstar, al) < 1e-9
        if float(np.dot(sstar, (u, v, w))) < -1e-9:
            refuted += 1
    problems.append(("pow K* outside frac refuted", refuted / max(n_out, 1)))

    # --- hand-computed complementary pairs from the docstring ---
    problems.append(("exp hand pair viol_p", viol_exp_primal((0, 1, 1))))
    problems.append(("exp hand pair viol_d", viol_exp_dual((-1, -1, 1))))
    problems.append(("exp hand pair <s,z>", float(np.dot((0, 1, 1), (-1, -1, 1)))))
    problems.append(("pow hand pair viol_p", viol_pow_primal((4, 1, 8), al)))
    problems.append(("pow hand pair viol_d", viol_pow_dual((-1, 4 / 3, 1 / 3), al)))
    problems.append(
        ("pow hand pair <s,z>", float(np.dot((4, 1, 8), (-1, 4 / 3, 1 / 3))))
    )

    print("=== self-test of dual-cone derivations ===")
    ok = True
    for name, val in problems:
        print(f"  {name:34s} = {val: .3e}")
    if problems[0][1] < -1e-9 or problems[2][1] < -1e-9:
        ok = False
    if problems[1][1] < 0.999 or problems[3][1] < 0.999:
        ok = False
    for name, val in problems[4:]:
        if abs(val) > 1e-12:
            ok = False
    print(f"  self-test: {'OK' if ok else 'FAILED'}")
    return ok


# ------------------------------------------------------------------- the cases
def build_soc():
    """min -x0 - x1  s.t.  ||x||_2 <= 1.   p* = -sqrt(2) at (1,1)/sqrt(2).

    s = (1, x0, x1) in soc(3):  G = [[0,0],[-1,0],[0,-1]], h = (1,0,0).
    Dual: c + G'z = 0 => z1 = z2 = -1; gap/self-duality => z0 = sqrt(2).
    """
    c = np.array([-1.0, -1.0])
    G = np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([1.0, 0.0, 0.0])
    return dict(
        name="soc: min -x0-x1 s.t. ||x||<=1",
        c=c, G=G, h=h, A=None, b=None,
        cones=[("soc", 3)],
        blocks=[("soc", 3, None)],
        p_star=-math.sqrt(2.0),
        x_star=np.array([1, 1]) / math.sqrt(2.0),
        z_star=np.array([math.sqrt(2.0), -1.0, -1.0]),
        y_star=np.zeros(0),
    )


def build_psd():
    """min <C,X>  s.t.  tr X = 1, X psd,  C = [[2,1],[1,3]].
    p* = lambda_min(C) = (5 - sqrt5)/2.

    x := svec(X) directly; G = -I, h = 0 so s = x.
    c = svec(C); A = svec(I)' = [1,0,1], b = 1.
    Dual: c + A'y + G'z = 0 => z = svec(C + y I); psd => y >= -lam_min;
    d* = -y maximized at y = -lam_min.
    """
    C = np.array([[2.0, 1.0], [1.0, 3.0]])
    lam = np.linalg.eigvalsh(C)[0]
    c = svec(C)
    G = -np.eye(3)
    h = np.zeros(3)
    A = svec(np.eye(2)).reshape(1, 3)
    # note svec(I) = (1,0,1) -> A x = X11 + X22 = tr X.  correct.
    b = np.array([1.0])
    w = np.linalg.eigh(C)[1][:, 0]
    return dict(
        name="psd: min <C,X> s.t. tr X = 1",
        c=c, G=G, h=h, A=A, b=b,
        cones=[("psd", 2)],
        blocks=[("psd", 3, None)],
        p_star=lam,
        x_star=svec(np.outer(w, w)),
        z_star=svec(C - lam * np.eye(2)),
        y_star=np.array([-lam]),
    )


def build_exp():
    """min e^u + e^{-u}  (= min x + 1/x).  p* = 2 at u = 0.

    Vars (u,t1,t2); c = (0,1,1).
    Block1 s = (u, 1, t1) in Kexp; Block2 s = (-u, 1, t2) in Kexp.
    Dual derived by hand: z = (-1,-1,1, -1,-1,1); d* = -h'z = 2.
    """
    c = np.array([0.0, 1.0, 1.0])
    G = np.zeros((6, 3))
    G[0, 0] = -1.0
    G[2, 1] = -1.0
    G[3, 0] = 1.0
    G[5, 2] = -1.0
    h = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    return dict(
        name="exp: min e^u + e^-u  (GP, x + 1/x)",
        c=c, G=G, h=h, A=None, b=None,
        cones=[("exp", 3), ("exp", 3)],
        blocks=[("exp", 3, None), ("exp", 3, None)],
        p_star=2.0,
        x_star=np.array([0.0, 1.0, 1.0]),
        z_star=np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0]),
        y_star=np.zeros(0),
    )


def build_pow():
    """min -a  s.t.  b = 1, c = 8, (a,b,c) in Kpow(1/3).
    p* = -(1^{1/3} 8^{2/3}) = -4.

    x = (a,b,c); G = -I, h = 0 so s = x.
    Dual (hand): z = (-1, 4/3, 1/3), y = (4/3, 1/3), d* = -b'y = -4.
    """
    al = 1.0 / 3.0
    c = np.array([-1.0, 0.0, 0.0])
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    b = np.array([1.0, 8.0])
    G = -np.eye(3)
    h = np.zeros(3)
    return dict(
        name="pow: max b^(1/3) c^(2/3) with b=1, c=8",
        c=c, G=G, h=h, A=A, b=b,
        cones=[("pow", al)],
        blocks=[("pow", 3, al)],
        p_star=-4.0,
        x_star=np.array([4.0, 1.0, 8.0]),
        z_star=np.array([-1.0, 4.0 / 3.0, 1.0 / 3.0]),
        y_star=np.array([4.0 / 3.0, 1.0 / 3.0]),
    )


def block_viol(kind, s, z, al):
    if kind == "soc":
        return viol_soc(s), viol_soc(z)
    if kind == "psd":
        return viol_psd(s), viol_psd(z)
    if kind == "nonneg":
        return viol_nonneg(s), viol_nonneg(z)
    if kind == "exp":
        return viol_exp_primal(s), viol_exp_dual(z)
    if kind == "pow":
        return viol_pow_primal(s, al), viol_pow_dual(z, al)
    raise ValueError(kind)


# -------------------------------------------------------------- cvxpy cross-ck
def cvxpy_check(case):
    import cvxpy as cp

    n = len(case["c"])
    x = cp.Variable(n)
    cons = []
    if case["A"] is not None:
        eq = cp.Constraint  # noqa
        con_eq = case["A"] @ x == case["b"]
        cons.append(con_eq)
    else:
        con_eq = None
    s = case["h"] - case["G"] @ x
    off = 0
    cone_cons = []
    for kind, d, al in case["blocks"]:
        blk = s[off:off + d]
        if kind == "soc":
            cc = cp.SOC(blk[0], blk[1:])
        elif kind == "psd":
            # rebuild the symmetric matrix from svec
            nn = int((math.isqrt(8 * d + 1) - 1) // 2)
            rows = []
            idx = {}
            k = 0
            for j in range(nn):
                for i in range(j, nn):
                    idx[(i, j)] = k
                    k += 1
            for i in range(nn):
                row = []
                for j in range(nn):
                    ii, jj = max(i, j), min(i, j)
                    e = blk[idx[(ii, jj)]]
                    row.append(e if i == j else e / SQ2)
                rows.append(cp.hstack(row))
            cc = cp.bmat([[r] for r in rows]) >> 0
        elif kind == "exp":
            cc = cp.constraints.ExpCone(blk[0], blk[1], blk[2])
        elif kind == "pow":
            # cvxpy PowCone3D(xx, yy, zz, al): xx^al yy^(1-al) >= |zz|
            cc = cp.constraints.PowCone3D(blk[1], blk[2], blk[0], al)
        cone_cons.append(cc)
        cons.append(cc)
        off += d
    prob = cp.Problem(cp.Minimize(case["c"] @ x), cons)
    t0 = time.perf_counter()
    prob.solve(solver=cp.CLARABEL)
    t = time.perf_counter() - t0
    duals = []
    for cc in cone_cons:
        try:
            dv = cc.dual_value
            if isinstance(dv, (list, tuple)):  # cp.SOC returns [t_dual, X_dual]
                dv = np.concatenate([np.atleast_1d(np.asarray(p, float).ravel())
                                     for p in dv])
            duals.append(np.atleast_1d(np.asarray(dv, float).ravel()))
        except Exception:
            duals.append(None)
    y = None
    if con_eq is not None:
        try:
            y = np.atleast_1d(np.asarray(con_eq.dual_value, float).ravel())
        except Exception:
            y = None
    return prob.value, np.asarray(x.value, float), duals, y, t


# ------------------------------------------------------------------------ main
def run(case):
    from pounce import solve_socp

    print()
    print("=" * 74)
    print(f"CASE: {case['name']}")
    print("=" * 74)
    t0 = time.perf_counter()
    r = solve_socp(
        c=case["c"], A=case["A"], b=case["b"], G=case["G"], h=case["h"],
        cones=case["cones"], tol=1e-10,
    )
    t_p = time.perf_counter() - t0
    x = np.asarray(r.x, float)
    z = np.asarray(r.z, float)
    y = np.asarray(r.y, float)
    G, h, c = case["G"], case["h"], case["c"]
    A, bvec = case["A"], case["b"]
    s = h - G @ x

    print(f"status={r.status} iters={r.iters} t={t_p:.4f}s")
    print(f"obj = {r.obj:.12e}   p* (analytic) = {case['p_star']:.12e}")
    print(f"x   = {np.array2string(x, precision=8)}")
    print(f"x*  = {np.array2string(case['x_star'], precision=8)}")
    print(f"z   = {np.array2string(z, precision=8)}")
    print(f"z*  = {np.array2string(case['z_star'], precision=8)}")
    if y.size:
        print(f"y   = {np.array2string(y, precision=8)}")
        print(f"y*  = {np.array2string(case['y_star'], precision=8)}")

    fails = []
    scale = max(1.0, np.linalg.norm(z, np.inf), np.linalg.norm(x, np.inf))

    # objective vs analytic
    obj_err = abs(r.obj - case["p_star"]) / max(1.0, abs(case["p_star"]))
    print(f"\n  [0] primal objective rel err vs analytic  = {obj_err:.3e}")
    if obj_err > 1e-6:
        fails.append("primal objective")

    # (d) dual feasibility / stationarity
    stat = c + G.T @ z + (A.T @ y if A is not None else 0.0)
    stat_r = float(np.max(np.abs(stat))) / scale
    print(f"  [d] stationarity ||c + A'y + G'z||_inf     = {stat_r:.3e}")
    if stat_r > 1e-7:
        fails.append("dual feasibility (stationarity)")

    # (a) cone membership, primal and dual, per block
    off = 0
    tot_comp = 0.0
    for i, (kind, d, al) in enumerate(case["blocks"]):
        sb, zb = s[off:off + d], z[off:off + d]
        vp, vd = block_viol(kind, sb, zb, al)
        comp = float(np.dot(sb, zb))
        tot_comp += comp
        print(
            f"  [a/c] block {i} ({kind:6s}) primal viol={vp:.3e}  "
            f"DUAL viol={vd:.3e}  <s,z>={comp:+.3e}"
        )
        if kind in ("exp", "pow"):
            # DISCRIMINATING CHECK: the whole point of (a) is that K_exp/K_pow
            # are NOT self-dual.  Show that the returned dual is genuinely
            # OUTSIDE the primal cone, so a self-dual test would have rejected
            # it and this test is actually sensitive to the distinction.
            wrong = (viol_exp_primal(zb) if kind == "exp"
                     else viol_pow_primal(zb, al))
            print(f"        (non-self-dual discriminator: z in K^* ok, but "
                  f"z in K would violate by {wrong:.3e})")
            if wrong < 1e-3:
                fails.append(f"block {i} dual-cone test not discriminating")
        if vp > 1e-7 * scale:
            fails.append(f"block {i} primal cone membership")
        if vd > 1e-7 * scale:
            fails.append(f"block {i} DUAL cone membership ({kind})")
        if abs(comp) > 1e-6 * scale:
            fails.append(f"block {i} complementary slackness")
        off += d

    # (b) duality gap
    d_star = -(bvec @ y if A is not None else 0.0) - h @ z
    gap = abs(r.obj - d_star) / max(1.0, abs(r.obj))
    print(f"  [b] dual objective d* = -b'y - h'z         = {d_star:.12e}")
    print(f"  [b] relative duality gap                   = {gap:.3e}")
    print(f"      (identity check: p*-d* == <s,z> = {tot_comp:+.3e})")
    if gap > 1e-7:
        fails.append("duality gap")

    # dual vs analytic dual
    # Dual VALUE accuracy is an accuracy note, not a certificate invariant: it
    # is graded against cvxpy's own dual error on the same problem below.
    zerr = float(np.max(np.abs(z - case["z_star"]))) / scale
    print(f"      ||z - z*_analytic||_inf / scale        = {zerr:.3e}")
    if zerr > 1e-4:
        fails.append("dual value vs analytic (>1e-4)")
    if y.size:
        yerr = float(np.max(np.abs(y - case["y_star"])))
        print(f"      ||y - y*_analytic||_inf               = {yerr:.3e}")
        if yerr > 1e-4:
            fails.append("eq multiplier vs analytic (>1e-4)")

    # (e) cvxpy
    try:
        pv, xv, cvd, cvy, t_o = cvxpy_check(case)
        print(f"\n  [e] cvxpy(CLARABEL) obj = {pv:.12e}  t={t_o:.4f}s")
        print(f"      |obj_pounce - obj_cvxpy| rel = "
              f"{abs(r.obj - pv) / max(1.0, abs(pv)):.3e}")
        print(f"      ||x_pounce - x_cvxpy||_inf   = "
              f"{np.max(np.abs(x - xv)):.3e}")
        # map each cvxpy cone dual into pounce's block ordering/scaling so the
        # two certificates can be compared entry by entry.
        off = 0
        zc = []
        for i, ((kind, d, al), dv) in enumerate(zip(case["blocks"], cvd)):
            zb = z[off:off + d]
            if dv is None:
                print(f"      cvxpy dual blk{i} ({kind}): unavailable")
                zc.append(None)
            else:
                if kind == "psd" and dv.size != d:
                    nn = int(round(math.sqrt(dv.size)))
                    m = svec(dv.reshape(nn, nn))
                elif kind == "pow":
                    # PowCone3D(b, c, a, al) -> dual ordered (v, w, u);
                    # pounce order is (u, v, w).
                    m = np.array([dv[2], dv[0], dv[1]])
                else:
                    m = dv
                zc.append(m)
                print(f"      cvxpy dual blk{i} ({kind}) mapped = "
                      f"{np.array2string(m, precision=8)}")
                print(f"        pounce z blk{i}                 = "
                      f"{np.array2string(zb, precision=8)}")
                if m.size == d:
                    zs = case["z_star"][off:off + d]
                    ep = np.max(np.abs(zb - zs))
                    ec = np.max(np.abs(m - zs))
                    print(f"        |z-z*| pounce={ep:.3e}  cvxpy={ec:.3e}"
                          f"  (pounce/cvxpy = {ep / max(ec, 1e-300):.2f}x)")
            off += d
        if cvy is not None and y.size:
            print(f"      cvxpy eq dual = {np.array2string(cvy, precision=6)}  "
                  f"pounce y = {np.array2string(y, precision=6)}")
        if abs(r.obj - pv) / max(1.0, abs(pv)) > 1e-6:
            fails.append("cvxpy objective disagreement")
    except Exception as e:  # pragma: no cover
        print(f"  [e] cvxpy check unavailable: {type(e).__name__}: {e}")

    print(f"\n  RESULT: {'PASS' if not fails else 'FAIL -> ' + '; '.join(fails)}")
    return fails, t_p


if __name__ == "__main__":
    ok = selftest()
    if not ok:
        print("VERDICT: FORMULATION_ERROR (dual-cone self-test failed)")
        raise SystemExit(1)

    allfails = {}
    for build in (build_soc, build_psd, build_exp, build_pow):
        case = build()
        f, t = run(case)
        allfails[case["name"]] = f

    # tolerance sweep: characterise the `optimal_inaccurate` status seen on the
    # non-symmetric (exp/pow) HSDE path at tol=1e-10.
    print()
    print("=" * 74)
    print("TOLERANCE SWEEP (status / dual error vs analytic)")
    from pounce import solve_socp as _ss
    for build in (build_soc, build_psd, build_exp, build_pow):
        case = build()
        row = []
        for tl in (None, 1e-8, 1e-9, 1e-10, 1e-12):
            rr = _ss(c=case["c"], A=case["A"], b=case["b"], G=case["G"],
                     h=case["h"], cones=case["cones"], tol=tl)
            ze = float(np.max(np.abs(np.asarray(rr.z) - case["z_star"])))
            row.append(f"tol={tl}: {rr.status}/{rr.iters}it/dz={ze:.1e}")
        print(f"  {case['name'][:26]:28s} " + " | ".join(row))

    print()
    print("=" * 74)
    bad = {k: v for k, v in allfails.items() if v}
    for k, v in allfails.items():
        print(f"  {'PASS' if not v else 'FAIL'}  {k}" + ("" if not v else f"  :: {v}"))
    print("VERDICT: PASS" if not bad else f"VERDICT: FAIL {bad}")
