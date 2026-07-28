"""Adversary cross-check: Lagrange multiplier / dual correctness on Hock-Schittkowski problems.

Family: nlp   Class: duals / multipliers / sensitivity invariants
Dimension: duals, multipliers, sensitivity and mathematical invariants

Problems (none previously in adversary/log.org):
  HS21  -- convex QP, 1 linear inequality (inactive), ACTIVE lower bound.  f* = -99.96
  HS43  -- Rosen-Suzuki, 3 nonlinear inequalities (2 active), no bounds.   f* = -44
           analytic multipliers lam = (1, 0, 2)  [derived in report]
  HS53  -- convex QP, 3 linear EQUALITIES, bounds inactive.                f* = 176/43
  HS73  -- cattle feed: linear obj, 1 equality + 2 inequalities, ACTIVE
           lower bound x2 = 0.                                             f* = 29.894378

Source: W. Hock and K. Schittkowski, "Test Examples for Nonlinear Programming
Codes", Lecture Notes in Economics and Mathematical Systems 187, Springer 1981
(problems 21, 43, 53, 73).

Oracles, in order of authority:
  (1) finite-difference dobj/db by perturbing each constraint rhs (and each
      active bound) and re-solving -- CONVENTION FREE ground truth;
  (2) analytic / published KKT multipliers;
  (3) Ipopt on the identical Pyomo model.

Checks performed per problem:
  (a) KKT stationarity identity in pounce's own documented (cyipopt) convention
      grad f + J^T lam - z_L + z_U = 0
  (b) magnitudes vs analytic multipliers
  (c) sign vs documented convention
  (d) finite-difference dobj/db  (primary)
  (e) bound multipliers for ACTIVE bounds
  (f) Ipopt comparison on the same Pyomo model; a UNIFORM sign flip vs Ipopt is
      the already-known .sol dual-negation issue.  A NON-UNIFORM flip (e.g.
      equalities flipped but inequalities not) would be a distinct, worse bug.
"""

import math
import time

import numpy as np
import sympy as sp

import pounce

np.set_printoptions(precision=6, suppress=True)

# --------------------------------------------------------------------------
# Problem definitions.  Each `expr` builder takes (x, sq) where sq is the sqrt
# function of the target backend, so the SAME source builds the sympy model
# (for exact derivatives), the numpy model, and the Pyomo model.
# --------------------------------------------------------------------------


def hs21(x, sq):
    f = 0.01 * x[0] ** 2 + x[1] ** 2 - 100.0
    # g >= 0 form
    g = [10.0 * x[0] - x[1] - 10.0]
    return f, g


def hs43(x, sq):
    f = (
        x[0] ** 2 + x[1] ** 2 + 2 * x[2] ** 2 + x[3] ** 2
        - 5 * x[0] - 5 * x[1] - 21 * x[2] + 7 * x[3]
    )
    g = [
        8 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - x[3] ** 2 - x[0] + x[1] - x[2] + x[3],
        10 - x[0] ** 2 - 2 * x[1] ** 2 - x[2] ** 2 - 2 * x[3] ** 2 + x[0] + x[3],
        5 - 2 * x[0] ** 2 - x[1] ** 2 - x[2] ** 2 - 2 * x[0] + x[1] + x[3],
    ]
    return f, g


def hs53(x, sq):
    f = (
        (x[0] - x[1]) ** 2
        + (x[1] + x[2] - 2) ** 2
        + (x[3] - 1) ** 2
        + (x[4] - 1) ** 2
    )
    g = [
        x[0] + 3 * x[1],
        x[2] + x[3] - 2 * x[4],
        x[1] - x[4],
    ]
    return f, g


def hs73(x, sq):
    f = 24.55 * x[0] + 26.75 * x[1] + 39.0 * x[2] + 40.50 * x[3]
    g = [
        2.3 * x[0] + 5.6 * x[1] + 11.1 * x[2] + 1.3 * x[3] - 5.0,
        12 * x[0] + 11.9 * x[1] + 41.8 * x[2] + 52.1 * x[3]
        - 21.0
        - 1.645 * sq(0.28 * x[0] ** 2 + 0.19 * x[1] ** 2 + 20.5 * x[2] ** 2 + 0.62 * x[3] ** 2),
        x[0] + x[1] + x[2] + x[3] - 1.0,
    ]
    return f, g


BIG = 2.0e19

PROBLEMS = [
    dict(
        name="HS21",
        build=hs21,
        n=2,
        # constraint rows: (cl, cu); ">= 0" -> cl=0, cu=+inf
        cl=[0.0],
        cu=[BIG],
        lb=[2.0, -50.0],
        ub=[50.0, 50.0],
        x0=[-1.0, -1.0],
        fstar=-99.96,
        xstar=[2.0, 0.0],
        # analytic: only active thing is x1 >= 2, grad f = (0.04, 0)
        lam_analytic=[0.0],
        zL_analytic=[0.04, 0.0],
        zU_analytic=[0.0, 0.0],
    ),
    dict(
        name="HS43",
        build=hs43,
        n=4,
        cl=[0.0, 0.0, 0.0],
        cu=[BIG, BIG, BIG],
        lb=[-BIG] * 4,
        ub=[BIG] * 4,
        x0=[0.0, 0.0, 0.0, 0.0],
        fstar=-44.0,
        xstar=[0.0, 1.0, 2.0, -1.0],
        # solving grad f = mu1*grad c1 + mu3*grad c3 gives mu = (1, 0, 2)
        lam_analytic=[1.0, 0.0, 2.0],
        zL_analytic=[0.0] * 4,
        zU_analytic=[0.0] * 4,
    ),
    dict(
        name="HS53",
        build=hs53,
        n=5,
        cl=[0.0, 0.0, 0.0],
        cu=[0.0, 0.0, 0.0],
        lb=[-10.0] * 5,
        ub=[10.0] * 5,
        x0=[2.0, 2.0, 2.0, 2.0, 2.0],
        fstar=176.0 / 43.0,
        xstar=[-33.0 / 43.0, 11.0 / 43.0, 27.0 / 43.0, -5.0 / 43.0, 11.0 / 43.0],
        lam_analytic=None,  # solved numerically from the KKT system below
        zL_analytic=[0.0] * 5,
        zU_analytic=[0.0] * 5,
    ),
    dict(
        name="HS73",
        build=hs73,
        n=4,
        cl=[0.0, 0.0, 0.0],
        cu=[BIG, BIG, 0.0],
        lb=[0.0] * 4,
        ub=[BIG] * 4,
        x0=[1.0, 1.0, 1.0, 1.0],
        fstar=29.894378,
        xstar=[0.6355216, 0.0, 0.3127019, 0.05177655],
        lam_analytic=None,
        zL_analytic=None,
        zU_analytic=[0.0] * 4,
    ),
]


# --------------------------------------------------------------------------
# Symbolic derivative machinery (exact grad / jac / hess of the Lagrangian).
# --------------------------------------------------------------------------


class SymProblem:
    """cyipopt-style problem object with exact sympy-generated derivatives."""

    def __init__(self, build, n, m):
        self.n, self.m = n, m
        xs = sp.symbols(f"x0:{n}", real=True)
        f, g = build(list(xs), sp.sqrt)
        self.f_l = sp.lambdify(xs, f, "numpy")
        grad = [sp.diff(f, v) for v in xs]
        self.grad_l = sp.lambdify(xs, grad, "numpy")
        self.g_l = sp.lambdify(xs, g, "numpy") if m else None
        jac = [[sp.diff(gi, v) for v in xs] for gi in g]
        self.jac_l = sp.lambdify(xs, jac, "numpy") if m else None
        of = sp.Symbol("obj_factor", real=True)
        ls = sp.symbols(f"lam0:{max(m,1)}", real=True)
        lag = of * f + sum(ls[i] * g[i] for i in range(m))
        H = [[sp.diff(lag, a, b) for b in xs] for a in xs]
        self.hess_l = sp.lambdify((xs, of, ls), H, "numpy")

    def objective(self, x):
        return float(self.f_l(*x))

    def gradient(self, x):
        return np.asarray(self.grad_l(*x), dtype=float).reshape(self.n)

    def constraints(self, x):
        return np.asarray(self.g_l(*x), dtype=float).reshape(self.m)

    def jacobian_dense(self, x):
        return np.asarray(self.jac_l(*x), dtype=float).reshape(self.m, self.n)

    def jacobian(self, x):
        return self.jacobian_dense(x).ravel()

    def jacobianstructure(self):
        return (
            np.repeat(np.arange(self.m), self.n),
            np.tile(np.arange(self.n), self.m),
        )

    def hessianstructure(self):
        r, c = np.tril_indices(self.n)
        return (r, c)

    def hessian(self, x, lagrange, obj_factor):
        lam = np.asarray(lagrange, dtype=float)
        if self.m == 0:
            lam = np.zeros(1)
        H = np.asarray(self.hess_l(list(x), float(obj_factor), list(lam)), dtype=float)
        H = H.reshape(self.n, self.n)
        r, c = np.tril_indices(self.n)
        return H[r, c]


def fd_check_derivatives(sp_obj, x, tag):
    """Guard against the #1 false positive: a wrong hand-built derivative."""
    x = np.asarray(x, float)
    h = 1e-6
    gnum = np.zeros(sp_obj.n)
    for i in range(sp_obj.n):
        e = np.zeros(sp_obj.n)
        e[i] = h
        gnum[i] = (sp_obj.objective(x + e) - sp_obj.objective(x - e)) / (2 * h)
    gerr = np.max(np.abs(gnum - sp_obj.gradient(x)))
    jerr = 0.0
    if sp_obj.m:
        Jnum = np.zeros((sp_obj.m, sp_obj.n))
        for i in range(sp_obj.n):
            e = np.zeros(sp_obj.n)
            e[i] = h
            Jnum[:, i] = (sp_obj.constraints(x + e) - sp_obj.constraints(x - e)) / (2 * h)
        jerr = np.max(np.abs(Jnum - sp_obj.jacobian_dense(x)))
    print(f"  [{tag}] derivative FD check: grad_err={gerr:.2e} jac_err={jerr:.2e}")
    return gerr, jerr


# --------------------------------------------------------------------------
# pounce solve helper
# --------------------------------------------------------------------------


def solve_pounce(P, cl=None, cu=None, lb=None, ub=None, tol=1e-10):
    n = P["n"]
    m = len(P["cl"])
    obj = SymProblem(P["build"], n, m)
    prob = pounce.Problem(
        n=n,
        m=m,
        problem_obj=obj,
        lb=list(lb if lb is not None else P["lb"]),
        ub=list(ub if ub is not None else P["ub"]),
        cl=list(cl if cl is not None else P["cl"]),
        cu=list(cu if cu is not None else P["cu"]),
    )
    prob.add_option("tol", tol)
    prob.add_option("print_level", 0)
    prob.add_option("nlp_scaling_method", "none")
    x, info = prob.solve(x0=np.asarray(P["x0"], float))
    return np.asarray(x, float), info, obj


# --------------------------------------------------------------------------
# Analytic multiplier recovery: least-squares solve of the stationarity system
# restricted to the active set (independent of pounce).
# --------------------------------------------------------------------------


def analytic_multipliers(obj, x, cl, cu, lb, ub, atol=1e-6):
    """Return (lam, zL, zU) satisfying grad f + J^T lam - zL + zU = 0 on the
    active set, computed WITHOUT pounce, in the documented cyipopt convention."""
    n, m = obj.n, obj.m
    gx = obj.constraints(x) if m else np.zeros(0)
    J = obj.jacobian_dense(x) if m else np.zeros((0, n))
    cols, kinds = [], []
    for i in range(m):
        eq = abs(cu[i] - cl[i]) < 1e-12
        lo_act = cl[i] > -1e18 and abs(gx[i] - cl[i]) < atol
        hi_act = cu[i] < 1e18 and abs(gx[i] - cu[i]) < atol
        if eq or lo_act or hi_act:
            cols.append(J[i])
            kinds.append(("c", i))
    for j in range(n):
        if lb[j] > -1e18 and abs(x[j] - lb[j]) < atol:
            e = np.zeros(n)
            e[j] = -1.0  # -zL
            cols.append(e)
            kinds.append(("zL", j))
        if ub[j] < 1e18 and abs(x[j] - ub[j]) < atol:
            e = np.zeros(n)
            e[j] = 1.0  # +zU
            cols.append(e)
            kinds.append(("zU", j))
    lam = np.zeros(m)
    zL = np.zeros(n)
    zU = np.zeros(n)
    if cols:
        A = np.array(cols).T
        sol, *_ = np.linalg.lstsq(A, -obj.gradient(x), rcond=None)
        for v, (k, idx) in zip(sol, kinds):
            if k == "c":
                lam[idx] = v
            elif k == "zL":
                zL[idx] = v
            else:
                zU[idx] = v
    return lam, zL, zU


# --------------------------------------------------------------------------
# (d) finite-difference dobj/db : the convention-free oracle
# --------------------------------------------------------------------------


def fd_sensitivities(P, delta=1e-5):
    """Central-difference dobj/db for each constraint rhs and each finite bound."""
    m = len(P["cl"])
    n = P["n"]
    dcon = np.full(m, np.nan)
    dlb = np.full(n, np.nan)
    dub = np.full(n, np.nan)

    for i in range(m):
        eq = abs(P["cu"][i] - P["cl"][i]) < 1e-12
        vals = []
        ok = True
        for s in (+1, -1):
            cl = list(P["cl"])
            cu = list(P["cu"])
            if eq:
                cl[i] += s * delta
                cu[i] += s * delta
            elif P["cl"][i] > -1e18:
                cl[i] += s * delta
            else:
                cu[i] += s * delta
            try:
                _, info, _ = solve_pounce(P, cl=cl, cu=cu)
            except Exception:
                ok = False
                break
            if info["status"] not in (0, 1):
                ok = False
                break
            vals.append(info["obj_val"])
        if ok and len(vals) == 2:
            dcon[i] = (vals[0] - vals[1]) / (2 * delta)

    for j in range(n):
        for which, store, base in (("lb", dlb, P["lb"]), ("ub", dub, P["ub"])):
            if abs(base[j]) > 1e18:
                continue
            vals = []
            ok = True
            for s in (+1, -1):
                lb = list(P["lb"])
                ub = list(P["ub"])
                if which == "lb":
                    lb[j] += s * delta
                else:
                    ub[j] += s * delta
                try:
                    _, info, _ = solve_pounce(P, lb=lb, ub=ub)
                except Exception:
                    ok = False
                    break
                if info["status"] not in (0, 1):
                    ok = False
                    break
                vals.append(info["obj_val"])
            if ok and len(vals) == 2:
                store[j] = (vals[0] - vals[1]) / (2 * delta)
    return dcon, dlb, dub


# --------------------------------------------------------------------------
# (f) Ipopt + pounce via Pyomo on the identical model (.sol dual path)
# --------------------------------------------------------------------------


def pyomo_duals(P):
    import pyomo.environ as pyo

    out = {}
    for solver_name in ("ipopt", "pounce"):
        mdl = pyo.ConcreteModel()
        n = P["n"]
        mdl.I = pyo.RangeSet(0, n - 1)

        def _b(mm, j):
            lo = P["lb"][j]
            hi = P["ub"][j]
            return (None if lo < -1e18 else lo, None if hi > 1e18 else hi)

        mdl.x = pyo.Var(mdl.I, bounds=_b, initialize=lambda mm, j: P["x0"][j])
        xs = [mdl.x[j] for j in range(n)]
        f, g = P["build"](xs, pyo.sqrt)
        mdl.obj = pyo.Objective(expr=f, sense=pyo.minimize)
        mdl.cons = pyo.ConstraintList()
        for i, gi in enumerate(g):
            lo, hi = P["cl"][i], P["cu"][i]
            if abs(hi - lo) < 1e-12:
                mdl.cons.add(gi == lo)
            elif lo > -1e18 and hi < 1e18:
                mdl.cons.add(pyo.inequality(lo, gi, hi))
            elif lo > -1e18:
                mdl.cons.add(gi >= lo)
            else:
                mdl.cons.add(gi <= hi)
        mdl.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        try:
            opt = pyo.SolverFactory(solver_name)
            res = opt.solve(mdl, tee=False)
            duals = [mdl.dual.get(mdl.cons[i + 1], float("nan")) for i in range(len(g))]
            out[solver_name] = dict(
                obj=float(pyo.value(mdl.obj)),
                x=[float(pyo.value(mdl.x[j])) for j in range(n)],
                duals=np.array(duals, float),
                status=str(res.solver.termination_condition),
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            out[solver_name] = dict(error=repr(exc)[:200])
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

FINDINGS = []
SIGN_TABLE = []  # (problem, kind, index, lam_pounce, dobj_db, ratio)


def report(P):
    name = P["name"]
    print("=" * 78)
    print(f"### {name}")
    n, m = P["n"], len(P["cl"])

    t0 = time.perf_counter()
    x, info, obj = solve_pounce(P)
    t_p = time.perf_counter() - t0

    fd_check_derivatives(obj, x, name)

    lam = np.asarray(info["mult_g"], float)
    zL = np.asarray(info["mult_x_L"], float)
    zU = np.asarray(info["mult_x_U"], float)

    print(f"  status={info['status']} ({info.get('status_msg','')}) "
          f"obj={info['obj_val']:.10f} known f*={P['fstar']:.10f} "
          f"rel_err={abs(info['obj_val']-P['fstar'])/max(1,abs(P['fstar'])):.2e} "
          f"t={t_p:.3f}s")
    print(f"  x        = {x}")
    print(f"  x* (H&S) = {np.array(P['xstar'])}")
    print(f"  mult_g   = {lam}")
    print(f"  mult_x_L = {zL}")
    print(f"  mult_x_U = {zU}")

    # ---- (a) KKT stationarity in the documented cyipopt convention ---------
    J = obj.jacobian_dense(x) if m else np.zeros((0, n))
    stat = obj.gradient(x) + J.T @ lam - zL + zU
    print(f"  (a) stationarity ||grad f + J^T lam - zL + zU||_inf = "
          f"{np.max(np.abs(stat)):.3e}")
    stat_flip = obj.gradient(x) - J.T @ lam - zL + zU
    print(f"      (with lam negated, for reference)               = "
          f"{np.max(np.abs(stat_flip)):.3e}")
    if np.max(np.abs(stat)) > 1e-6 and np.max(np.abs(stat_flip)) > 1e-6:
        FINDINGS.append(f"{name}: KKT stationarity fails in BOTH sign conventions")

    # complementarity
    gx = obj.constraints(x) if m else np.zeros(0)
    compl = 0.0
    for i in range(m):
        eq = abs(P["cu"][i] - P["cl"][i]) < 1e-12
        if eq:
            continue
        slack = min(
            abs(gx[i] - P["cl"][i]) if P["cl"][i] > -1e18 else np.inf,
            abs(gx[i] - P["cu"][i]) if P["cu"][i] < 1e18 else np.inf,
        )
        compl = max(compl, abs(lam[i]) * slack)
    print(f"      complementarity max |lam_i| * slack_i = {compl:.3e}")

    # ---- (b)/(c) analytic multipliers -------------------------------------
    a_lam, a_zL, a_zU = analytic_multipliers(obj, x, P["cl"], P["cu"], P["lb"], P["ub"])
    print(f"  (b) analytic lam (cyipopt convention) = {a_lam}")
    print(f"      analytic zL = {a_zL}")
    print(f"      analytic zU = {a_zU}")
    if P["lam_analytic"] is not None:
        pub = np.array(P["lam_analytic"], float)
        print(f"      published |lam| (H&S / derived)      = {pub}")
        err = np.max(np.abs(np.abs(a_lam) - pub))
        print(f"      |analytic| vs published max err      = {err:.2e}")
        if err > 1e-6:
            FINDINGS.append(f"{name}: analytic-vs-published multiplier mismatch {err:.2e}")
    mag_err = np.max(np.abs(np.abs(lam) - np.abs(a_lam))) if m else 0.0
    print(f"  (b) MAGNITUDE err |pounce lam| vs |analytic lam| = {mag_err:.2e}")
    if mag_err > 1e-5:
        FINDINGS.append(f"{name}: multiplier MAGNITUDE error {mag_err:.2e}")
    zmag = max(
        np.max(np.abs(zL - np.abs(a_zL))) if n else 0.0,
        np.max(np.abs(zU - np.abs(a_zU))) if n else 0.0,
    )
    print(f"  (e) bound-multiplier magnitude err (active bounds) = {zmag:.2e}")
    if zmag > 1e-5:
        FINDINGS.append(f"{name}: bound multiplier magnitude error {zmag:.2e}")

    # ---- (d) finite-difference dobj/db, convention free --------------------
    dcon, dlb, dub = fd_sensitivities(P)
    print("  (d) finite-difference sensitivities vs reported multipliers:")
    for i in range(m):
        eq = abs(P["cu"][i] - P["cl"][i]) < 1e-12
        kind = "eq" if eq else "ineq"
        r = dcon[i] / lam[i] if abs(lam[i]) > 1e-8 else float("nan")
        print(f"      con[{i}] ({kind:4s}) dobj/db={dcon[i]: .8f}  "
              f"lam={lam[i]: .8f}  ratio={r: .6f}")
        if abs(lam[i]) > 1e-8 and not math.isnan(dcon[i]):
            SIGN_TABLE.append((P["name"], kind, i, lam[i], dcon[i], r))
        elif abs(lam[i]) <= 1e-8 and not math.isnan(dcon[i]) and abs(dcon[i]) > 1e-5:
            FINDINGS.append(
                f"{name}: con[{i}] reported lam=0 but dobj/db={dcon[i]:.3e}")
    for j in range(n):
        if zL[j] > 1e-8 and not math.isnan(dlb[j]):
            r = dlb[j] / zL[j]
            print(f"      lb[{j}]  (bnd ) dobj/db={dlb[j]: .8f}  "
                  f"zL={zL[j]: .8f}  ratio={r: .6f}")
            SIGN_TABLE.append((P["name"], "boundL", j, zL[j], dlb[j], r))
        if zU[j] > 1e-8 and not math.isnan(dub[j]):
            r = dub[j] / zU[j]
            print(f"      ub[{j}]  (bnd ) dobj/db={dub[j]: .8f}  "
                  f"zU={zU[j]: .8f}  ratio={r: .6f}")
            SIGN_TABLE.append((P["name"], "boundU", j, zU[j], dub[j], r))

    # ---- (f) Ipopt (and pounce's own .sol path) via Pyomo ------------------
    py = pyomo_duals(P)
    print("  (f) Pyomo / .sol dual path:")
    for k, v in py.items():
        if "error" in v:
            print(f"      {k}: ERROR {v['error']}")
        else:
            print(f"      {k}: status={v['status']} obj={v['obj']:.8f} duals={v['duals']}")
    if "ipopt" in py and "duals" in py["ipopt"] and "pounce" in py and "duals" in py["pounce"]:
        di, dp = py["ipopt"]["duals"], py["pounce"]["duals"]
        ratios = []
        for i in range(m):
            if abs(di[i]) > 1e-7:
                ratios.append((i, dp[i] / di[i]))
        print(f"      pounce/ipopt dual ratios per row: "
              + ", ".join(f"c{i}={r:+.4f}" for i, r in ratios))
        mag = max((abs(abs(dp[i]) - abs(di[i])) for i in range(m)), default=0.0)
        print(f"      max |.|magnitude difference pounce vs ipopt = {mag:.2e}")
        signs = {round(np.sign(r)) for _, r in ratios if abs(abs(r) - 1) < 0.05}
        if len(signs) > 1:
            FINDINGS.append(
                f"{name}: NON-UNIFORM dual sign vs Ipopt across constraint rows "
                f"(ratios {ratios})")
        if mag > 1e-4:
            FINDINGS.append(f"{name}: Pyomo dual MAGNITUDE differs from Ipopt by {mag:.2e}")
    return dict(name=name, obj=info["obj_val"], t=t_p, status=info["status"], py=py)


def main():
    rows = [report(P) for P in PROBLEMS]

    print("=" * 78)
    print("SIGN-CONVENTION CONSISTENCY (ratio dobj/db divided by reported multiplier)")
    print("  a correct, self-consistent implementation gives the SAME constant")
    print("  (+1 or -1) for every row and every kind.")
    print(f"  {'problem':8s} {'kind':7s} {'idx':>3s} {'mult':>14s} {'dobj/db':>14s} {'ratio':>9s}")
    ratios_by_kind = {}
    for nm, kind, i, mult, d, r in SIGN_TABLE:
        print(f"  {nm:8s} {kind:7s} {i:3d} {mult:14.8f} {d:14.8f} {r:9.4f}")
        ratios_by_kind.setdefault(kind, []).append(r)
    # The documented (cyipopt/Ipopt) Lagrangian is
    #     L = f + lam^T g - z_L (x - x_L) + z_U (x - x_U)
    # so the EXPECTED finite-difference ratios are:
    #     general-constraint rows (eq AND ineq alike): dobj/db  / lam = -1
    #     lower-bound rows:                            dobj/dxL / zL  = +1
    #     upper-bound rows:                            dobj/dxU / zU  = -1
    # A correct implementation is uniform WITHIN each of these groups; the
    # eq-vs-ineq comparison is the one that would expose a real sign bug.
    EXPECTED = {"eq": -1, "ineq": -1, "boundL": +1, "boundU": -1}
    all_r = [r for _, _, _, _, _, r in SIGN_TABLE]
    con_r = [r for _, k, _, _, _, r in SIGN_TABLE if k in ("eq", "ineq")]
    if con_r:
        con_signs = {round(np.sign(r)) for r in con_r}
        print(f"  general-constraint rows (eq + ineq): distinct signs {sorted(con_signs)}")
        if len(con_signs) > 1:
            FINDINGS.append(
                "NON-UNIFORM sign between equality and inequality multipliers "
                f"vs finite-difference dobj/db: {ratios_by_kind}")
    for kind, rs in ratios_by_kind.items():
        exp = EXPECTED[kind]
        bad = [r for r in rs if abs(r - exp) > 2e-3]
        print(f"  kind={kind:7s} expected ratio {exp:+d}  observed "
              f"[{min(rs):+.6f}, {max(rs):+.6f}]  -> "
              + ("OK" if not bad else f"MISMATCH {bad}"))
        if bad:
            FINDINGS.append(
                f"{kind} rows: dobj/db / multiplier = {bad}, expected {exp:+d} "
                "under the documented cyipopt convention")
    if all_r and not all(abs(abs(r) - 1.0) < 2e-3 for r in all_r):
        FINDINGS.append(f"|dobj/db / multiplier| != 1 for some row: {all_r}")

    print("=" * 78)
    print("FINDINGS:")
    if FINDINGS:
        for f in FINDINGS:
            print("  - " + f)
    else:
        print("  (none)")
    ok = not FINDINGS
    print("VERDICT: PASS" if ok else "VERDICT: FAIL")


if __name__ == "__main__":
    main()
