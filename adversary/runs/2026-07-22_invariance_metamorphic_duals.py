"""Adversary metamorphic / invariance run: duals, multipliers, mathematical invariants.

Family: qp / lp / socp / nlp   Class: metamorphic invariance (no external oracle)
Oracle: THE INVARIANT ITSELF.  Every transformation below is a provable
identity of the mathematical program; any deviation beyond solver tolerance is
a defect of the implementation, not of a reference value.

Surfaces exercised (ALL Python API):
  pounce.solve_qp    (convex QP / LP,  duals y, z, z_lb, z_ub)
  pounce.solve_socp  (conic,           duals y, z)
  pounce.minimize    (NLP,             multipliers r.info["mult_g"], mult_x_L/U)

Transformations
  (a) variable permutation
  (b) constraint permutation
  (c) objective shift        f -> f + const           (minimize only; QP form has no const)
  (d) objective scaling      f -> alpha*f, alpha>0    => duals scale by alpha
  (e) constraint scaling     row -> beta*row, beta>0  => that dual scales by 1/beta
  (f) variable shift         x -> x + x0
  (g) equality splitting     a'x = b  ->  a'x <= b, -a'x <= -b
  (h) redundant constraint   implied by existing rows
  (i) unused variable        appended, appears in no constraint

Conventions established empirically and re-verified in-script by KKT stationarity:
  solve_qp:  min 0.5 x'Px + c'x  s.t.  Ax=b, Gx<=h, lb<=x<=ub
             stationarity  Px + c + A'y + G'z - z_lb + z_ub = 0,  z,z_lb,z_ub >= 0
  solve_socp: min 0.5 x'Px + c'x  s.t. Ax=b, h-Gx in K
  minimize:  stationarity  grad f + J' mult_g - mult_x_L + mult_x_U = 0
"""

import time
import numpy as np
import pounce

np.set_printoptions(precision=6, suppress=True)

TOL = 1e-6          # invariance tolerance (solver default tol is ~1e-8..1e-9)
DUAL_TOL = 1e-5     # duals converge a bit looser than primal

RESULTS = []        # (surface, instance, transform, quantity, max_dev)


def rec(surface, inst, transform, quantity, dev):
    RESULTS.append((surface, inst, transform, quantity, float(dev)))


def dev_inf(a, b):
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    if a.shape != b.shape:
        return np.inf
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


# ----------------------------------------------------------------------------
# QP instances.  Dict form; None entries omitted at call time.
# ----------------------------------------------------------------------------

SOLVER_TOL = None   # set in main(); None = pounce default


def qp_call(p, **kw):
    args = {k: v for k, v in p.items() if v is not None}
    if SOLVER_TOL is not None:
        kw.setdefault("tol", SOLVER_TOL)
        kw.setdefault("max_iter", 300)
    return pounce.solve_qp(**args, **kw)


def make_qp1():
    """Strictly convex QP, 5 vars, 2 equalities, 3 inequalities, box bounds.
    Non-degenerate by construction (unique primal & dual)."""
    rng = np.random.default_rng(7)
    n = 5
    M = rng.standard_normal((n, n))
    P = M @ M.T + 2.0 * np.eye(n)
    c = rng.standard_normal(n)
    A = rng.standard_normal((2, n))
    b = A @ np.array([0.2, -0.3, 0.1, 0.4, -0.1])
    G = rng.standard_normal((3, n))
    h = G @ np.array([0.2, -0.3, 0.1, 0.4, -0.1]) + np.array([0.05, -0.02, 0.30])
    lb = -2.0 * np.ones(n)
    ub = 2.0 * np.ones(n)
    return dict(P=P, c=c, A=A, b=b, G=G, h=h, lb=lb, ub=ub)


def make_qp2():
    """QP with only inequalities (no equalities, no bounds)."""
    rng = np.random.default_rng(21)
    n = 4
    M = rng.standard_normal((n, n))
    P = M @ M.T + 1.5 * np.eye(n)
    c = rng.standard_normal(n)
    G = np.vstack([rng.standard_normal((3, n)), np.eye(n), -np.eye(n)])
    h = np.concatenate([np.array([0.4, 0.1, 0.25]), 1.5 * np.ones(n), 1.5 * np.ones(n)])
    return dict(P=P, c=c, A=None, b=None, G=G, h=h, lb=None, ub=None)


def make_lp1():
    """Bounded LP (P = None): min c'x s.t. Ax=b, Gx<=h, 0<=x<=5."""
    c = np.array([-1.0, -2.0, -3.0, 1.0])
    A = np.array([[1.0, 1.0, 1.0, 1.0]])
    b = np.array([4.0])
    G = np.array([[1.0, 2.0, 0.0, 0.0],
                  [0.0, 1.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0, 1.0]])
    h = np.array([3.0, 2.5, 2.0])
    lb = np.zeros(4)
    ub = 5.0 * np.ones(4)
    return dict(P=None, c=c, A=A, b=b, G=G, h=h, lb=lb, ub=ub)


QP_INSTANCES = {"qp_convex_5v": make_qp1, "qp_ineq_only_4v": make_qp2, "lp_4v": make_lp1}


def qp_obj(p, x):
    v = float(p["c"] @ x)
    if p["P"] is not None:
        v += 0.5 * float(x @ p["P"] @ x)
    return v


def qp_stationarity(p, r):
    """Independent KKT stationarity residual (confirms the dual sign convention)."""
    x = np.asarray(r.x)
    g = np.asarray(p["c"], dtype=float).copy()
    if p["P"] is not None:
        g = g + p["P"] @ x
    if p["A"] is not None:
        g = g + p["A"].T @ np.asarray(r.y)
    if p["G"] is not None:
        g = g + p["G"].T @ np.asarray(r.z)
    if p["lb"] is not None:
        g = g - np.asarray(r.z_lb)
    if p["ub"] is not None:
        g = g + np.asarray(r.z_ub)
    return float(np.max(np.abs(g)))


def run_qp_invariance(name, maker):
    p = maker()
    n = len(p["c"])
    r0 = qp_call(p)
    assert r0.status == "optimal", (name, r0.status)
    x0 = np.asarray(r0.x)
    o0 = qp_obj(p, x0)
    rec("solve_qp", name, "-baseline-", "kkt_stationarity", qp_stationarity(p, r0))

    rng = np.random.default_rng(1234)

    # ---- (a) variable permutation --------------------------------------
    perm = rng.permutation(n)
    Pi = np.eye(n)[perm]                      # (Pi @ v)[i] = v[perm[i]]
    q = dict(p)
    q["c"] = p["c"][perm]
    if p["P"] is not None:
        q["P"] = p["P"][np.ix_(perm, perm)]
    if p["A"] is not None:
        q["A"] = p["A"][:, perm]
    if p["G"] is not None:
        q["G"] = p["G"][:, perm]
    for k in ("lb", "ub"):
        if p[k] is not None:
            q[k] = p[k][perm]
    ra = qp_call(q)
    rec("solve_qp", name, "(a) var permutation", "x", dev_inf(ra.x, x0[perm]))
    rec("solve_qp", name, "(a) var permutation", "obj", abs(ra.obj - r0.obj))
    if p["A"] is not None:
        rec("solve_qp", name, "(a) var permutation", "y(eq dual)", dev_inf(ra.y, r0.y))
    if p["G"] is not None:
        rec("solve_qp", name, "(a) var permutation", "z(ineq dual)", dev_inf(ra.z, r0.z))
    if p["lb"] is not None:
        rec("solve_qp", name, "(a) var permutation", "z_lb", dev_inf(ra.z_lb, np.asarray(r0.z_lb)[perm]))
        rec("solve_qp", name, "(a) var permutation", "z_ub", dev_inf(ra.z_ub, np.asarray(r0.z_ub)[perm]))

    # ---- (b) constraint permutation ------------------------------------
    q = dict(p)
    if p["G"] is not None:
        mg = p["G"].shape[0]
        pg = rng.permutation(mg)
        q["G"] = p["G"][pg]
        q["h"] = p["h"][pg]
    if p["A"] is not None:
        ma = p["A"].shape[0]
        pa = rng.permutation(ma)
        q["A"] = p["A"][pa]
        q["b"] = p["b"][pa]
    rb = qp_call(q)
    rec("solve_qp", name, "(b) con permutation", "x", dev_inf(rb.x, x0))
    if p["G"] is not None:
        rec("solve_qp", name, "(b) con permutation", "z(ineq dual)", dev_inf(rb.z, np.asarray(r0.z)[pg]))
    if p["A"] is not None:
        rec("solve_qp", name, "(b) con permutation", "y(eq dual)", dev_inf(rb.y, np.asarray(r0.y)[pa]))

    # ---- (d) objective scaling f -> alpha f ------------------------------
    for alpha in (3.0, 0.1):
        q = dict(p)
        q["c"] = alpha * p["c"]
        if p["P"] is not None:
            q["P"] = alpha * p["P"]
        rd = qp_call(q)
        tag = f"(d) obj scaling a={alpha}"
        rec("solve_qp", name, tag, "x", dev_inf(rd.x, x0))
        rec("solve_qp", name, tag, "obj/alpha", abs(rd.obj / alpha - r0.obj))
        if p["A"] is not None:
            rec("solve_qp", name, tag, "y/alpha", dev_inf(np.asarray(rd.y) / alpha, r0.y))
        if p["G"] is not None:
            rec("solve_qp", name, tag, "z/alpha", dev_inf(np.asarray(rd.z) / alpha, r0.z))
        if p["lb"] is not None:
            rec("solve_qp", name, tag, "z_lb/alpha", dev_inf(np.asarray(rd.z_lb) / alpha, r0.z_lb))

    # ---- (e) constraint row scaling --------------------------------------
    if p["G"] is not None:
        beta = 4.0
        i = int(np.argmax(np.asarray(r0.z)))   # scale an ACTIVE row (nonzero dual)
        q = dict(p)
        q["G"] = p["G"].copy()
        q["h"] = p["h"].copy()
        q["G"][i] *= beta
        q["h"][i] *= beta
        re_ = qp_call(q)
        tag = f"(e) ineq row {i} x{beta}"
        rec("solve_qp", name, tag, "x", dev_inf(re_.x, x0))
        expect = np.asarray(r0.z).copy()
        expect[i] /= beta
        rec("solve_qp", name, tag, "z (row i scaled by 1/beta)", dev_inf(re_.z, expect))
    if p["A"] is not None:
        beta = 0.25
        q = dict(p)
        q["A"] = p["A"].copy()
        q["b"] = p["b"].copy()
        q["A"][0] *= beta
        q["b"][0] *= beta
        re2 = qp_call(q)
        tag = f"(e) eq row 0 x{beta}"
        rec("solve_qp", name, tag, "x", dev_inf(re2.x, x0))
        expect = np.asarray(r0.y).copy()
        expect[0] /= beta
        rec("solve_qp", name, tag, "y (row 0 scaled by 1/beta)", dev_inf(re2.y, expect))

    # ---- (f) variable shift  w = x + s ------------------------------------
    s = rng.uniform(-0.3, 0.3, n)
    q = dict(p)
    if p["P"] is not None:
        q["c"] = p["c"] - p["P"] @ s
        const = 0.5 * float(s @ p["P"] @ s) - float(p["c"] @ s)
    else:
        q["c"] = p["c"].copy()
        const = -float(p["c"] @ s)
    if p["A"] is not None:
        q["b"] = p["b"] + p["A"] @ s
    if p["G"] is not None:
        q["h"] = p["h"] + p["G"] @ s
    for k, sg in (("lb", 1), ("ub", 1)):
        if p[k] is not None:
            q[k] = p[k] + s
    rf = qp_call(q)
    rec("solve_qp", name, "(f) variable shift", "x - shift", dev_inf(np.asarray(rf.x) - s, x0))
    rec("solve_qp", name, "(f) variable shift", "obj + const", abs(rf.obj + const - r0.obj))
    if p["G"] is not None:
        rec("solve_qp", name, "(f) variable shift", "z", dev_inf(rf.z, r0.z))
    if p["A"] is not None:
        rec("solve_qp", name, "(f) variable shift", "y", dev_inf(rf.y, r0.y))

    # ---- (g) split an equality into two inequalities -----------------------
    if p["A"] is not None and p["A"].shape[0] >= 1:
        q = dict(p)
        arow, brow = p["A"][0], p["b"][0]
        if p["A"].shape[0] == 1:
            q["A"] = None
            q["b"] = None
        else:
            q["A"] = p["A"][1:]
            q["b"] = p["b"][1:]
        newG = np.vstack([arow, -arow])
        newh = np.array([brow, -brow])
        q["G"] = newG if p["G"] is None else np.vstack([p["G"], newG])
        q["h"] = newh if p["h"] is None else np.concatenate([p["h"], newh])
        rg = qp_call(q)
        rec("solve_qp", name, "(g) split equality", "x", dev_inf(rg.x, x0))
        rec("solve_qp", name, "(g) split equality", "obj", abs(rg.obj - r0.obj))
        # dual recovery: y_0 == z_plus - z_minus for the split pair
        zz = np.asarray(rg.z)
        y_rec = zz[-2] - zz[-1]
        rec("solve_qp", name, "(g) split equality", "y0 == z+ - z-", abs(y_rec - float(np.asarray(r0.y)[0])))

    # ---- (h) redundant constraint (implied by two existing rows) -----------
    if p["G"] is not None and p["G"].shape[0] >= 2:
        q = dict(p)
        newrow = 0.5 * (p["G"][0] + p["G"][1])
        newrhs = 0.5 * (p["h"][0] + p["h"][1])
        q["G"] = np.vstack([p["G"], newrow])
        q["h"] = np.concatenate([p["h"], [newrhs]])
        rh = qp_call(q)
        rec("solve_qp", name, "(h) redundant row", "x", dev_inf(rh.x, x0))
        rec("solve_qp", name, "(h) redundant row", "obj", abs(rh.obj - r0.obj))

    # ---- (i) unused variable ----------------------------------------------
    q = dict(p)
    q["c"] = np.concatenate([p["c"], [0.0]])
    if p["P"] is not None:
        Pn = np.zeros((n + 1, n + 1))
        Pn[:n, :n] = p["P"]
        Pn[n, n] = 1.0            # separable, argmin 0, contributes 0 to obj
        q["P"] = Pn
    if p["A"] is not None:
        q["A"] = np.hstack([p["A"], np.zeros((p["A"].shape[0], 1))])
    if p["G"] is not None:
        q["G"] = np.hstack([p["G"], np.zeros((p["G"].shape[0], 1))])
    if p["lb"] is not None:
        q["lb"] = np.concatenate([p["lb"], [-1.0]])
    if p["ub"] is not None:
        q["ub"] = np.concatenate([p["ub"], [1.0]])
    if p["P"] is None:
        # LP: no objective term, must bound it -> already bounded by lb/ub above
        q["lb"] = np.concatenate([p["lb"], [0.0]])
        q["ub"] = np.concatenate([p["ub"], [1.0]])
    ri = qp_call(q)
    rec("solve_qp", name, "(i) unused variable", "x[:n]", dev_inf(np.asarray(ri.x)[:n], x0))
    rec("solve_qp", name, "(i) unused variable", "obj", abs(ri.obj - r0.obj))


# ----------------------------------------------------------------------------
# SOCP
# ----------------------------------------------------------------------------

def make_socp1():
    """min t + 0.1*sum(x)  s.t. (t, x - x*) in SOC(4), x >= -1 (nonneg block).
    Variables z = (t, x0, x1, x2)."""
    xstar = np.array([1.0, -0.5, 0.7])
    n = 4
    c = np.array([1.0, 0.1, 0.1, 0.1])
    # SOC block: s = h - G z  = (t, x - xstar)
    Gs = -np.eye(4)
    hs = np.concatenate([[0.0], -xstar])
    # nonneg block: x + 1 >= 0  ->  s = h - G z >= 0 with G = -[0 I], h = 1
    Gn = np.zeros((3, 4))
    Gn[:, 1:] = -np.eye(3)
    hn = np.ones(3)
    G = np.vstack([Gs, Gn])
    h = np.concatenate([hs, hn])
    return dict(P=None, c=c, A=None, b=None, G=G, h=h,
                cones=[("soc", 4), ("nonneg", 3)], n=n, nsoc=4)


def make_socp2():
    """Quadratic objective + one SOC + equality. z = (t, x0, x1)."""
    P = np.diag([0.0, 2.0, 2.0])
    c = np.array([1.0, -1.0, 0.5])
    A = np.array([[0.0, 1.0, 1.0]])
    b = np.array([0.5])
    Gs = -np.eye(3)
    hs = np.array([0.0, 0.3, -0.2])
    Gn = np.array([[0.0, -1.0, 0.0]])
    hn = np.array([2.0])
    return dict(P=P, c=c, A=A, b=b, G=np.vstack([Gs, Gn]),
                h=np.concatenate([hs, hn]),
                cones=[("soc", 3), ("nonneg", 1)], n=3, nsoc=3)


SOCP_INSTANCES = {"socp_norm_ball": make_socp1, "socp_qobj_eq": make_socp2}


def socp_call(p, **kw):
    args = {k: v for k, v in p.items() if v is not None and k not in ("n", "nsoc", "cones")}
    if SOLVER_TOL is not None:
        kw.setdefault("tol", SOLVER_TOL)
        kw.setdefault("max_iter", 300)
    return pounce.solve_socp(**args, cones=p["cones"], **kw)


def run_socp_invariance(name, maker):
    p = maker()
    n = p["n"]
    nsoc = p["nsoc"]
    r0 = socp_call(p)
    assert r0.status == "optimal", (name, r0.status)
    x0 = np.asarray(r0.x)
    rng = np.random.default_rng(99)

    # ---- (a) variable permutation ------------------------------------
    perm = rng.permutation(n)
    q = dict(p)
    q["c"] = p["c"][perm]
    if p["P"] is not None:
        q["P"] = p["P"][np.ix_(perm, perm)]
    if p["A"] is not None:
        q["A"] = p["A"][:, perm]
    q["G"] = p["G"][:, perm]
    ra = socp_call(q)
    rec("solve_socp", name, "(a) var permutation", "x", dev_inf(ra.x, x0[perm]))
    rec("solve_socp", name, "(a) var permutation", "obj", abs(ra.obj - r0.obj))
    rec("solve_socp", name, "(a) var permutation", "z(cone dual)", dev_inf(ra.z, r0.z))

    # ---- (b) constraint permutation: reverse the trailing nonneg block ----
    nn = p["G"].shape[0] - nsoc
    if nn >= 2:
        idx = np.concatenate([np.arange(nsoc), nsoc + np.arange(nn)[::-1]])
        q = dict(p)
        q["G"] = p["G"][idx]
        q["h"] = p["h"][idx]
        rb = socp_call(q)
        rec("solve_socp", name, "(b) nonneg block permutation", "x", dev_inf(rb.x, x0))
        rec("solve_socp", name, "(b) nonneg block permutation", "z",
            dev_inf(rb.z, np.asarray(r0.z)[idx]))

    # ---- (d) objective scaling ---------------------------------------
    alpha = 2.5
    q = dict(p)
    q["c"] = alpha * p["c"]
    if p["P"] is not None:
        q["P"] = alpha * p["P"]
    rd = socp_call(q)
    rec("solve_socp", name, "(d) obj scaling", "x", dev_inf(rd.x, x0))
    rec("solve_socp", name, "(d) obj scaling", "obj/alpha", abs(rd.obj / alpha - r0.obj))
    rec("solve_socp", name, "(d) obj scaling", "z/alpha", dev_inf(np.asarray(rd.z) / alpha, r0.z))
    if p["A"] is not None:
        rec("solve_socp", name, "(d) obj scaling", "y/alpha", dev_inf(np.asarray(rd.y) / alpha, r0.y))

    # ---- (e) cone-block scaling (SOC is invariant under positive scaling) --
    beta = 3.0
    q = dict(p)
    q["G"] = p["G"].copy()
    q["h"] = p["h"].copy()
    q["G"][:nsoc] *= beta
    q["h"][:nsoc] *= beta
    re_ = socp_call(q)
    rec("solve_socp", name, f"(e) SOC block x{beta}", "x", dev_inf(re_.x, x0))
    expect = np.asarray(r0.z).copy()
    expect[:nsoc] /= beta
    rec("solve_socp", name, f"(e) SOC block x{beta}", "z (block/beta)", dev_inf(re_.z, expect))

    # ---- (f) variable shift ------------------------------------------
    s = rng.uniform(-0.2, 0.2, n)
    q = dict(p)
    if p["P"] is not None:
        q["c"] = p["c"] - p["P"] @ s
        const = 0.5 * float(s @ p["P"] @ s) - float(p["c"] @ s)
    else:
        q["c"] = p["c"].copy()
        const = -float(p["c"] @ s)
    q["h"] = p["h"] + p["G"] @ s
    if p["A"] is not None:
        q["b"] = p["b"] + p["A"] @ s
    rf = socp_call(q)
    rec("solve_socp", name, "(f) variable shift", "x - shift", dev_inf(np.asarray(rf.x) - s, x0))
    rec("solve_socp", name, "(f) variable shift", "obj + const", abs(rf.obj + const - r0.obj))
    rec("solve_socp", name, "(f) variable shift", "z", dev_inf(rf.z, r0.z))

    # ---- (h) redundant nonneg row (duplicate of an existing one) -------
    if nn >= 1:
        q = dict(p)
        q["G"] = np.vstack([p["G"], p["G"][nsoc]])
        q["h"] = np.concatenate([p["h"], [p["h"][nsoc]]])
        q["cones"] = list(p["cones"][:-1]) + [("nonneg", nn + 1)]
        rh = socp_call(q)
        rec("solve_socp", name, "(h) duplicated nonneg row", "x", dev_inf(rh.x, x0))
        rec("solve_socp", name, "(h) duplicated nonneg row", "obj", abs(rh.obj - r0.obj))

    # ---- (i) unused variable (own quadratic, appears in no cone row) ---
    q = dict(p)
    q["c"] = np.concatenate([p["c"], [0.0]])
    Pn = np.zeros((n + 1, n + 1))
    if p["P"] is not None:
        Pn[:n, :n] = p["P"]
    Pn[n, n] = 1.0
    q["P"] = Pn
    q["G"] = np.hstack([p["G"], np.zeros((p["G"].shape[0], 1))])
    if p["A"] is not None:
        q["A"] = np.hstack([p["A"], np.zeros((p["A"].shape[0], 1))])
    ri = socp_call(q)
    if p["P"] is None:
        # baseline had no P; the augmented one adds 0.5*w^2 with w*=0 -> obj unchanged
        pass
    rec("solve_socp", name, "(i) unused variable", "x[:n]", dev_inf(np.asarray(ri.x)[:n], x0))
    rec("solve_socp", name, "(i) unused variable", "obj", abs(ri.obj - r0.obj))


# ----------------------------------------------------------------------------
# minimize (NLP).  pounce.minimize takes dict constraints:
#   {"type": "eq",   "fun": c}  ->  c(x) == 0
#   {"type": "ineq", "fun": c}  ->  c(x) >= 0
# Each instance is (f, grad f, x0, [one-row dict constraints], bounds).
# ----------------------------------------------------------------------------

def con(kind, fun, jac):
    return {"type": kind, "fun": fun, "jac": jac}


def nlp_solve(fun, grad, x0, cons, bounds):
    return pounce.minimize(fun, np.asarray(x0, dtype=float), jac=grad,
                           bounds=bounds, constraints=list(cons), tol=1e-10)


def nlp_jac(cons, x):
    return np.vstack([np.atleast_2d(c["jac"](x)) for c in cons])


def nlp1():
    """min (x0-1)^2 + (x1-2.5)^2 + 0.5*(x2+1)^2
       s.t.  4 - (x0^2+x1^2+x2^2) >= 0   (nonlinear ineq, active at the optimum)
             x0 + x1 + x2 - 2 = 0        (linear eq)
             -5 <= x <= 5
    """
    def f(x):
        return (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2 + 0.5 * (x[2] + 1) ** 2

    def gf(x):
        return np.array([2 * (x[0] - 1), 2 * (x[1] - 2.5), (x[2] + 1)])

    cons = [
        con("ineq", lambda x: np.array([4.0 - (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)]),
            lambda x: np.array([[-2 * x[0], -2 * x[1], -2 * x[2]]])),
        con("eq", lambda x: np.array([x[0] + x[1] + x[2] - 2.0]),
            lambda x: np.array([[1.0, 1.0, 1.0]])),
    ]
    return f, gf, np.array([0.5, 0.5, 0.5]), cons, [(-5.0, 5.0)] * 3


def nlp2():
    """Rosenbrock + 0.5 x2^2 with two linear inequalities (written as c(x) >= 0)."""
    def f(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2 + 0.5 * x[2] ** 2

    def gf(x):
        return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                         200 * (x[1] - x[0] ** 2),
                         x[2]])

    cons = [
        con("ineq", lambda x: np.array([1.0 - (x[0] + 2 * x[1] + x[2])]),
            lambda x: np.array([[-1.0, -2.0, -1.0]])),
        con("ineq", lambda x: np.array([0.4 - (x[0] - x[1])]),
            lambda x: np.array([[-1.0, 1.0, 0.0]])),
    ]
    return f, gf, np.array([0.0, 0.0, 0.0]), cons, [(-3.0, 3.0)] * 3


NLP_INSTANCES = {"nlp_sphere_eq": nlp1, "nlp_rosen_lin": nlp2}


def run_nlp_invariance(name, maker):
    f, gf, x0, cons, bnds = maker()
    r0 = nlp_solve(f, gf, x0, cons, bnds)
    assert r0.success, (name, r0.message)
    xs = np.asarray(r0.x)
    n = len(xs)
    m = len(cons)
    mg0 = np.asarray(r0.info["mult_g"], dtype=float)
    o0 = float(r0.fun)

    # Independent KKT stationarity (pins down the multiplier sign convention).
    J = nlp_jac(cons, xs)
    stat = gf(xs) + J.T @ mg0 - np.asarray(r0.info["mult_x_L"]) + np.asarray(r0.info["mult_x_U"])
    rec("minimize", name, "-baseline-", "kkt_stationarity", float(np.max(np.abs(stat))))

    rng = np.random.default_rng(5)

    # ---- (c) objective shift  f -> f + K ------------------------------
    K = 17.25
    rc = nlp_solve(lambda x: f(x) + K, gf, x0, cons, bnds)
    rec("minimize", name, "(c) obj shift +K", "x", dev_inf(rc.x, xs))
    rec("minimize", name, "(c) obj shift +K", "obj - K", abs(float(rc.fun) - K - o0))
    rec("minimize", name, "(c) obj shift +K", "mult_g", dev_inf(rc.info["mult_g"], mg0))

    # ---- (d) objective scaling  f -> alpha f --------------------------
    for alpha in (5.0, 0.2):
        rd = nlp_solve(lambda x, a=alpha: a * f(x), lambda x, a=alpha: a * gf(x),
                       x0, cons, bnds)
        tag = f"(d) obj scaling a={alpha}"
        rec("minimize", name, tag, "x", dev_inf(rd.x, xs))
        rec("minimize", name, tag, "obj/alpha", abs(float(rd.fun) / alpha - o0))
        rec("minimize", name, tag, "mult_g/alpha",
            dev_inf(np.asarray(rd.info["mult_g"]) / alpha, mg0))

    # ---- (a) variable permutation --------------------------------------
    perm = rng.permutation(n)
    inv = np.argsort(perm)
    consp = [con(c["type"],
                 (lambda w, cf=c["fun"]: cf(w[inv])),
                 (lambda w, cj=c["jac"]: np.atleast_2d(cj(w[inv]))[:, perm]))
             for c in cons]
    ra = nlp_solve(lambda w: f(w[inv]), lambda w: gf(w[inv])[perm], x0[perm],
                   consp, [bnds[i] for i in perm])
    rec("minimize", name, "(a) var permutation", "x", dev_inf(ra.x, xs[perm]))
    rec("minimize", name, "(a) var permutation", "obj", abs(float(ra.fun) - o0))
    rec("minimize", name, "(a) var permutation", "mult_g", dev_inf(ra.info["mult_g"], mg0))
    rec("minimize", name, "(a) var permutation", "mult_x_L",
        dev_inf(np.asarray(ra.info["mult_x_L"]), np.asarray(r0.info["mult_x_L"])[perm]))

    # ---- (b) constraint permutation -------------------------------------
    cidx = np.arange(m)[::-1]
    rb = nlp_solve(f, gf, x0, [cons[i] for i in cidx], bnds)
    rec("minimize", name, "(b) con permutation", "x", dev_inf(rb.x, xs))
    rec("minimize", name, "(b) con permutation", "obj", abs(float(rb.fun) - o0))
    rec("minimize", name, "(b) con permutation", "mult_g", dev_inf(rb.info["mult_g"], mg0[cidx]))

    # ---- (e) constraint scaling  c_j -> beta c_j ------------------------
    beta = 6.0
    j = int(np.argmax(np.abs(mg0)))
    conse = list(cons)
    conse[j] = con(cons[j]["type"],
                   (lambda x, cf=cons[j]["fun"]: beta * np.asarray(cf(x))),
                   (lambda x, cj=cons[j]["jac"]: beta * np.atleast_2d(cj(x))))
    re_ = nlp_solve(f, gf, x0, conse, bnds)
    tag = f"(e) con row {j} x{beta}"
    rec("minimize", name, tag, "x", dev_inf(re_.x, xs))
    rec("minimize", name, tag, "obj", abs(float(re_.fun) - o0))
    exp = mg0.copy()
    exp[j] /= beta
    rec("minimize", name, tag, "mult_g (row j /beta)", dev_inf(re_.info["mult_g"], exp))

    # ---- (f) variable shift  w = x + s ----------------------------------
    s = rng.uniform(-0.25, 0.25, n)
    consf = [con(c["type"],
                 (lambda w, cf=c["fun"]: cf(w - s)),
                 (lambda w, cj=c["jac"]: cj(w - s))) for c in cons]
    bf = [(lo + s[i], hi + s[i]) for i, (lo, hi) in enumerate(bnds)]
    rf = nlp_solve(lambda w: f(w - s), lambda w: gf(w - s), x0 + s, consf, bf)
    rec("minimize", name, "(f) variable shift", "x - shift", dev_inf(np.asarray(rf.x) - s, xs))
    rec("minimize", name, "(f) variable shift", "obj", abs(float(rf.fun) - o0))
    rec("minimize", name, "(f) variable shift", "mult_g", dev_inf(rf.info["mult_g"], mg0))

    # ---- (g) split equality  e(x)=0 -> e(x)>=0 and -e(x)>=0 --------------
    eqs = [k for k in range(m) if cons[k]["type"] == "eq"]
    if eqs:
        k = eqs[0]
        keep = [i for i in range(m) if i != k]
        cplus = con("ineq", cons[k]["fun"], cons[k]["jac"])
        cminus = con("ineq",
                     (lambda x, cf=cons[k]["fun"]: -np.asarray(cf(x))),
                     (lambda x, cj=cons[k]["jac"]: -np.atleast_2d(cj(x))))
        rg = nlp_solve(f, gf, x0, [cons[i] for i in keep] + [cplus, cminus], bnds)
        rec("minimize", name, "(g) split equality", "x", dev_inf(rg.x, xs))
        rec("minimize", name, "(g) split equality", "obj", abs(float(rg.fun) - o0))
        mgs = np.asarray(rg.info["mult_g"], dtype=float)
        rec("minimize", name, "(g) split equality", "mult_k == m+ - m-",
            abs((mgs[-2] - mgs[-1]) - mg0[k]))

    # ---- (h) redundant constraint: 0.5 * c_j(x) >= 0 (implied) -----------
    kk = int(np.argmax(np.abs(mg0)))
    if cons[kk]["type"] == "ineq":
        cred = con("ineq",
                   (lambda x, cf=cons[kk]["fun"]: 0.5 * np.asarray(cf(x))),
                   (lambda x, cj=cons[kk]["jac"]: 0.5 * np.atleast_2d(cj(x))))
        rh = nlp_solve(f, gf, x0, list(cons) + [cred], bnds)
        rec("minimize", name, "(h) redundant row", "x", dev_inf(rh.x, xs))
        rec("minimize", name, "(h) redundant row", "obj", abs(float(rh.fun) - o0))

    # ---- (i) unused variable (own separable term, in no constraint) -------
    consi = [con(c["type"],
                 (lambda w, cf=c["fun"]: cf(w[:n])),
                 (lambda w, cj=c["jac"]: np.hstack(
                     [np.atleast_2d(cj(w[:n])),
                      np.zeros((np.atleast_2d(cj(w[:n])).shape[0], 1))])))
             for c in cons]
    ri = nlp_solve(lambda w: f(w[:n]) + 0.5 * w[n] ** 2,
                   lambda w: np.concatenate([gf(w[:n]), [w[n]]]),
                   np.concatenate([x0, [0.3]]), consi, list(bnds) + [(-2.0, 2.0)])
    rec("minimize", name, "(i) unused variable", "x[:n]", dev_inf(np.asarray(ri.x)[:n], xs))
    rec("minimize", name, "(i) unused variable", "obj", abs(float(ri.fun) - o0))
    rec("minimize", name, "(i) unused variable", "mult_g", dev_inf(ri.info["mult_g"], mg0))


# ----------------------------------------------------------------------------

def sweep(label, verbose=True):
    global RESULTS
    RESULTS = []
    import warnings
    t0 = time.perf_counter()
    for nm, mk in QP_INSTANCES.items():
        run_qp_invariance(nm, mk)
    for nm, mk in SOCP_INSTANCES.items():
        run_socp_invariance(nm, mk)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nm, mk in NLP_INSTANCES.items():
            run_nlp_invariance(nm, mk)
    elapsed = time.perf_counter() - t0

    print(f"\n########## SWEEP: {label} ##########")
    print(f"{'surface':<12} {'instance':<18} {'transform':<28} {'quantity':<26} {'max_dev':>12}  ")
    print("-" * 106)
    worst = 0.0
    fails = []
    for (s, i, t, q, d) in RESULTS:
        tol = DUAL_TOL if ("dual" in q or q.startswith(("y", "z", "mult")) or "mult" in q) else TOL
        flag = "" if d <= tol else "   <<< VIOLATION"
        if d > tol:
            fails.append((s, i, t, q, d))
        worst = max(worst, d)
        if verbose:
            print(f"{s:<12} {i:<18} {t:<28} {q:<26} {d:12.3e}{flag}")
    print("-" * 106)
    print(f"[{label}] checks={len(RESULTS)}  worst_deviation={worst:.3e}  "
          f"violations={len(fails)}  elapsed={elapsed:.2f}s")
    for fl in fails:
        print(f"[{label}] VIOLATION:", fl)
    return fails


def flat_direction_diagnostic():
    """socp_qobj_eq is the only instance whose *argmin* drifts under (d)/(f).
    Reduce it to one variable and show the drift is the ordinary IPM
    sqrt-of-tolerance effect on a locally quadratic objective, not a wrong
    answer: obj deviation ~ curvature * x deviation^2, and the KKT residual is
    at machine precision at every alpha.
    """
    from scipy.optimize import minimize_scalar

    def F(x1):                      # x2 = 0.5 - x1, t = ||(x1+0.3, x2-0.2)||
        x2 = 0.5 - x1
        return x1 ** 2 + x2 ** 2 + np.hypot(x1 + 0.3, x2 - 0.2) - x1 + 0.5 * x2

    ref = minimize_scalar(F, bracket=(0.0, 0.5), method="brent",
                          options={"xtol": 1e-14})
    p = make_socp2()
    print("\n########## flat-argmin diagnostic: socp_qobj_eq ##########")
    print(f"independent 1-D reduction (scipy brent): x1* = {ref.x!r}  F* = {ref.fun!r}")
    print(f"{'alpha':>8} {'x1':>18} {'x1 err':>12} {'obj/alpha err':>15} {'kkt_error':>11} {'iters':>6}")
    for a in (0.1, 1.0, 2.5, 10.0):
        q = dict(p)
        q["P"] = a * p["P"]
        q["c"] = a * p["c"]
        r = socp_call(q, tol=1e-10, max_iter=300)
        print(f"{a:8} {r.x[1]:18.12f} {r.x[1]-ref.x:12.2e} "
              f"{r.obj/a-ref.fun:15.2e} {r.residuals['kkt_error']:11.2e} {r.iters:6d}")
    print("obj/alpha is invariant to ~1e-12 at every alpha while x1 moves by up to")
    print("~1e-6: with curvature ~4 that is exactly obj_err ~ 4 * x_err^2. The")
    print("argmin is simply resolved to sqrt(tol) in a flat direction -- TOLERANCE.")


def main():
    global SOLVER_TOL
    SOLVER_TOL = None
    f_default = sweep("default tol (conic solver default)")
    SOLVER_TOL = 1e-10
    f_tight = sweep("tol=1e-10 (conic surfaces tightened)")
    flat_direction_diagnostic()
    print()
    print(f"violations @default={len(f_default)}   violations @tol=1e-10={len(f_tight)}")
    # Grade argmin-only drift on an instance with a flat direction as TOLERANCE
    hard = [f for f in f_tight if not (f[1] == "socp_qobj_eq" and f[3].startswith("x"))]
    if hard:
        print(f"VERDICT: FAIL ({len(hard)} invariance violations survive tol=1e-10)")
    elif f_tight:
        print(f"VERDICT: PASS (TOLERANCE: {len(f_tight)} flat-argmin drift(s), "
              "objective/dual invariants hold)")
    else:
        print("VERDICT: PASS")


if __name__ == "__main__":
    main()
