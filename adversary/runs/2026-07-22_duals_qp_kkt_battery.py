"""Adversary cross-check: convex QP DUAL correctness battery (Python API).

Family: qp    Class: duals / multipliers / sensitivity / KKT invariants
Surface under test: PYTHON API `pounce.qp.solve_qp` -> QpResult(y, z, z_lb, z_ub).
                    (NOT the .sol/AMPL writer.)

Documented convention (read-only, from pounce source/docs):
  crates/pounce-convex/src/crossover.rs:29-30
      "`y` (free), inequality dual `z >= 0`, and bound duals `z_lb, z_ub >= 0`;
       stationarity `Px + c + A'y + G'z - z_lb + z_ub = 0`."
  docs/src/differentiable-solves.md: OptNet / Amos & Kolter (2017) convention.
  python/pounce/qp.py:104-109: y = equality multipliers; z, z_lb, z_ub >= 0.

Implied Lagrangian:
  L = 1/2 x'Px + c'x + y'(Ax-b) + z'(Gx-h) + z_lb'(lb-x) + z_ub'(x-ub)
Implied shadow prices:
  dobj*/db = -y     dobj*/dh = -z     dobj*/dlb = +z_lb     dobj*/dub = -z_ub

Checks per case:
 (a) stationarity  Px + c + A'y + G'z - z_lb + z_ub = 0
 (b) dual feasibility z, z_lb, z_ub >= 0
 (c) complementarity z_i*(h-Gx)_i = 0, z_lb_i*(x-lb)_i = 0, z_ub_i*(ub-x)_i = 0
 (d) strong duality: primal obj == dual obj
        g = -1/2 x'Px - b'y - h'z + lb'z_lb - ub'z_ub
 (e) finite-difference shadow prices dobj*/d(rhs) by re-solve (convention-free)
 (f) cvxpy duals (CLARABEL), mapped to pounce's convention
 (g) analytic multipliers where closed form exists
"""

import time

import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=140)

from pounce.qp import solve_qp

TOL_KKT = 1e-6
TOL_FD = 5e-5
TOL_ORACLE = 1e-6

CASES = []


def case(name, **kw):
    kw["name"] = name
    CASES.append(kw)


# ---------------------------------------------------------------- case bank --
# 1. Equality-only. Closed-form KKT: [[P, A'],[A, 0]] [x; y] = [-c; b].
case(
    "eq_only_closed_form",
    P=np.array([[2.0, 0.0, 0.0], [0.0, 4.0, 1.0], [0.0, 1.0, 6.0]]),
    c=np.array([-1.0, -2.0, 3.0]),
    A=np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]]),
    b=np.array([1.0, 0.5]),
)

# 2. Inequality-only. Nocedal & Wright, "Numerical Optimization" 2e, Example
#    16.4 (p. 475): min (x1-1)^2 + (x2-5/2)^2 s.t.
#      x1-2x2+2 >= 0, -x1-2x2+6 >= 0, -x1+2x2+2 >= 0, x1 >= 0, x2 >= 0.
#    Published: x* = (1.4, 1.7), q* = 0.8. At x* only c1 is active
#    (c1=0, c2=1.2, c3=4, c4=1.4, c5=1.7), and grad q = (0.8, -1.6) = 0.8*grad c1,
#    so lambda* = (0.8, 0, 0, 0, 0). (N&W's "{3,5}" is the algorithm's *starting*
#    working set in that example, not the solution's active set.)
#    In <= form the rows below are exactly c1..c5 negated, so z == lambda*.
case(
    "ineq_only_NW_ex16_4",
    P=np.diag([2.0, 2.0]),
    c=np.array([-2.0, -5.0]),
    G=np.array([[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0], [-1.0, 0.0], [0.0, -1.0]]),
    h=np.array([2.0, 6.0, 2.0, 0.0, 0.0]),
    analytic_x=np.array([1.4, 1.7]),
    analytic_z=np.array([0.8, 0.0, 0.0, 0.0, 0.0]),
    obj_const=7.25,  # q(x) = 1/2 x'Px + c'x + 7.25 ; q* = 0.8
)


# 2b. Non-degenerate LP (unique dual) — FD must match exactly here.
case(
    "lp_nondegenerate",
    P=np.zeros((2, 2)),
    c=np.array([-1.0, -2.0]),
    G=np.array([[1.0, 1.0], [3.0, 1.0]]),
    h=np.array([2.0, 4.0]),
    lb=np.array([0.0, 0.0]),
)

# 3. Bounds only, mixed active lower/upper/interior. Analytic:
#    unconstrained min of diag(2)x + c is x = -c/2 = [1.5, 2.0, -1.0]; box [0,1]
#    -> x = [1,1,0]; z_ub = -(Px+c) at upper, z_lb = (Px+c) at lower.
case(
    "bounds_only_analytic",
    P=np.diag([2.0, 2.0, 2.0]),
    c=np.array([-3.0, -4.0, 2.0]),
    lb=np.array([0.0, 0.0, 0.0]),
    ub=np.array([1.0, 1.0, 1.0]),
)

# 4. Mixed: equality + inequality + active bounds simultaneously.
case(
    "mixed_eq_ineq_bounds",
    P=np.array([[4.0, 1.0, 0.0, 0.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 2.0, 0.5], [0.0, 0.0, 0.5, 3.0]]),
    c=np.array([-1.0, -3.0, 1.0, -2.0]),
    A=np.array([[1.0, 1.0, 1.0, 1.0]]),
    b=np.array([2.0]),
    G=np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
    h=np.array([0.3, 1.2]),
    lb=np.array([-1.0, -1.0, -1.0, 0.4]),
    ub=np.array([0.9, 5.0, 5.0, 5.0]),
)

# 5. Degenerate: an inequality row duplicated (dual not unique) + a redundant
#    active bound coinciding with an active inequality.
case(
    "degenerate_duplicate_rows",
    P=np.diag([2.0, 2.0]),
    c=np.array([-4.0, -4.0]),
    G=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
    h=np.array([1.0, 1.0, 1.0]),
    lb=np.array([0.0, 0.0]),
    ub=np.array([1.0, 10.0]),
)

# 6. Least-squares with equality (P from A'A, well-conditioned).
_M = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
_d = np.array([1.0, 2.0, -1.0, 0.5])
case(
    "lsq_with_equality",
    P=_M.T @ _M,
    c=-_M.T @ _d,
    A=np.array([[1.0, 0.0, -1.0]]),
    b=np.array([0.25]),
    G=np.array([[0.0, 1.0, 0.0]]),
    h=np.array([0.1]),
)

# 7. LP (P = 0): duals must still satisfy the same identity.
case(
    "lp_degenerate_vertex",
    P=np.zeros((2, 2)),
    c=np.array([-1.0, -1.0]),
    G=np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
    h=np.array([1.0, 1.5, 1.5]),
    lb=np.array([0.0, 0.0]),
)


# ------------------------------------------------------------------ helpers --
def unpack(cs):
    n = len(cs["c"])
    P = cs.get("P")
    P = np.zeros((n, n)) if P is None else np.asarray(P, float)
    c = np.asarray(cs["c"], float)
    A = cs.get("A")
    b = cs.get("b")
    G = cs.get("G")
    h = cs.get("h")
    lb = cs.get("lb")
    ub = cs.get("ub")
    A = None if A is None else np.asarray(A, float)
    b = None if b is None else np.asarray(b, float)
    G = None if G is None else np.asarray(G, float)
    h = None if h is None else np.asarray(h, float)
    lb = None if lb is None else np.asarray(lb, float)
    ub = None if ub is None else np.asarray(ub, float)
    return n, P, c, A, b, G, h, lb, ub


def run_pounce(P, c, A, b, G, h, lb, ub, **extra):
    kw = dict(P=P, c=c, **extra)
    if A is not None:
        kw.update(A=A, b=b)
    if G is not None:
        kw.update(G=G, h=h)
    if lb is not None:
        kw.update(lb=lb)
    if ub is not None:
        kw.update(ub=ub)
    return solve_qp(**kw)


def obj_of(P, c, x):
    return float(0.5 * x @ P @ x + c @ x)


# ------------------------------------------------------------------- driver --
FAILURES = []
ROWS = []


def check(case_name, tag, ok, detail=""):
    ROWS.append((case_name, tag, "ok" if ok else "FAIL", detail))
    if not ok:
        FAILURES.append(f"{case_name}/{tag}: {detail}")
    return ok


for cs in CASES:
    name = cs["name"]
    n, P, c, A, b, G, h, lb, ub = unpack(cs)
    t0 = time.perf_counter()
    r = run_pounce(P, c, A, b, G, h, lb, ub)
    t_p = time.perf_counter() - t0

    print("\n" + "=" * 78)
    print(f"CASE {name}   n={n} m_eq={0 if A is None else A.shape[0]} "
          f"m_ineq={0 if G is None else G.shape[0]} "
          f"bounds={'yes' if (lb is not None or ub is not None) else 'no'}")
    print("=" * 78)
    print(f"pounce: status={r.status} obj={r.obj:.12e} iters={r.iters} t={t_p:.4f}s")

    if not check(name, "status", r.status == "optimal", f"status={r.status}"):
        continue

    x = np.asarray(r.x, float)
    y = np.asarray(r.y, float).reshape(-1)
    z = np.asarray(r.z, float).reshape(-1)
    zl = np.asarray(r.z_lb, float).reshape(-1)
    zu = np.asarray(r.z_ub, float).reshape(-1)
    print(f"  x    = {x}")
    print(f"  y    = {y}")
    print(f"  z    = {z}")
    print(f"  z_lb = {zl}")
    print(f"  z_ub = {zu}")

    # --- (a) stationarity, documented sign -----------------------------------
    stat = P @ x + c
    if A is not None:
        stat = stat + A.T @ y
    if G is not None:
        stat = stat + G.T @ z
    stat = stat - zl + zu
    s_doc = float(np.max(np.abs(stat)))
    # alternate sign hypotheses, to report which one actually holds
    alts = {}
    base = P @ x + c
    for sy in (+1, -1):
        for sz in (+1, -1):
            for sb in (+1, -1):
                v = base.copy()
                if A is not None:
                    v = v + sy * A.T @ y
                if G is not None:
                    v = v + sz * G.T @ z
                v = v + sb * (-zl + zu)
                alts[(sy, sz, sb)] = float(np.max(np.abs(v)))
    holding = [k for k, v in alts.items() if v < 1e-6]
    print(f"  (a) stationarity |Px+c+A'y+G'z-z_lb+z_ub|_inf = {s_doc:.3e}")
    print(f"      sign combos (sy,sz,sb) satisfying identity <1e-6: {holding}")
    check(name, "stationarity", s_doc < TOL_KKT, f"{s_doc:.3e}")

    # --- (b) dual feasibility -------------------------------------------------
    dmin = min([z.min() if z.size else 0.0, zl.min() if zl.size else 0.0,
                zu.min() if zu.size else 0.0])
    print(f"  (b) dual feasibility min(z,z_lb,z_ub) = {dmin:.3e}")
    check(name, "dual_feas", dmin > -TOL_KKT, f"min={dmin:.3e}")

    # --- (c) complementarity --------------------------------------------------
    comp = 0.0
    if G is not None:
        s = h - G @ x
        comp = max(comp, float(np.max(np.abs(z * s))) if z.size else 0.0)
        print(f"      slack h-Gx = {s}")
    if lb is not None:
        comp = max(comp, float(np.max(np.abs(zl * (x - lb)))))
    if ub is not None:
        comp = max(comp, float(np.max(np.abs(zu * (ub - x)))))
    # bound multipliers must vanish where there is no bound
    if lb is None:
        comp = max(comp, float(np.max(np.abs(zl))))
    if ub is None:
        comp = max(comp, float(np.max(np.abs(zu))))
    print(f"  (c) complementarity max = {comp:.3e}")
    check(name, "complementarity", comp < 1e-6, f"{comp:.3e}")

    # --- (d) strong duality ---------------------------------------------------
    g = -0.5 * x @ P @ x
    if A is not None:
        g -= b @ y
    if G is not None:
        g -= h @ z
    if lb is not None:
        g += lb @ zl
    if ub is not None:
        g -= ub @ zu
    gap = abs(r.obj - float(g))
    print(f"  (d) strong duality primal={r.obj:.12e} dual={float(g):.12e} gap={gap:.3e}")
    check(name, "duality_gap", gap < 1e-6 * max(1.0, abs(r.obj)), f"gap={gap:.3e}")

    # --- (e) finite-difference shadow prices ---------------------------------
    eps = 1e-5

    def resolve(**over):
        kw = dict(P=P, c=c, A=A, b=b, G=G, h=h, lb=lb, ub=ub)
        kw.update(over)
        rr = run_pounce(**kw)
        assert rr.status == "optimal", rr.status
        return rr.obj

    # The value function p*(theta) is convex in theta = (b, h, lb, ub) with the
    # predicted gradient g = (-y, -z, +z_lb, -z_ub). When p* is differentiable
    # at theta the one-sided derivatives D+ and D- coincide and must equal g.
    # At a kink (non-unique dual, e.g. a degenerate LP vertex) they differ and
    # the correct, convention-free test is the SUBGRADIENT BRACKET
    #     D- <= g <= D+
    # which is what "the returned dual is a valid shadow price" actually means.
    def one_sided(key, i, base_vec):
        vp, vm = base_vec.copy(), base_vec.copy()
        vp[i] += eps
        vm[i] -= eps
        f0 = resolve()
        dplus = (resolve(**{key: vp}) - f0) / eps
        dminus = (f0 - resolve(**{key: vm})) / eps
        return dminus, dplus

    fd_report = []
    if A is not None:
        for i in range(len(b)):
            fd_report.append(("dobj/db[%d]" % i, one_sided("b", i, b), -y[i]))
    if G is not None:
        for i in range(len(h)):
            fd_report.append(("dobj/dh[%d]" % i, one_sided("h", i, h), -z[i]))
    if lb is not None:
        for i in range(n):
            fd_report.append(("dobj/dlb[%d]" % i, one_sided("lb", i, lb), +zl[i]))
    if ub is not None:
        for i in range(n):
            fd_report.append(("dobj/dub[%d]" % i, one_sided("ub", i, ub), -zu[i]))

    worst_smooth = 0.0
    worst_bracket = 0.0
    n_kinks = 0
    print("  (e) shadow prices: one-sided FD D-/D+ vs predicted g:")
    for lab, (dm, dp), pred in fd_report:
        kink = abs(dp - dm) > 1e-3
        lo, hi = min(dm, dp), max(dm, dp)
        slack = max(lo - pred, pred - hi, 0.0)  # 0 if bracketed
        if kink:
            n_kinks += 1
            worst_bracket = max(worst_bracket, slack)
            tagtxt = f"KINK (non-unique dual)  bracket_viol={slack:.2e}"
        else:
            e = abs(0.5 * (dm + dp) - pred)
            worst_smooth = max(worst_smooth, e)
            tagtxt = f"err={e:.2e}"
        flag = ""
        if (kink and slack > 1e-4) or ((not kink) and abs(0.5 * (dm + dp) - pred) > TOL_FD):
            flag = "   <-- MISMATCH"
        print(f"      {lab:14s} D-={dm:+.8f} D+={dp:+.8f} g={pred:+.8f}  {tagtxt}{flag}")
    print(f"      ({n_kinks} kinked / non-unique-dual coordinates)")
    check(name, "fd_shadow_price_smooth", worst_smooth < TOL_FD, f"worst={worst_smooth:.2e}")
    check(name, "fd_subgradient_bracket", worst_bracket < 1e-4, f"worst={worst_bracket:.2e}")

    # --- (g) analytic multipliers, where published ---------------------------
    if "analytic_z" in cs:
        az = np.asarray(cs["analytic_z"], float)
        ax = np.asarray(cs["analytic_x"], float)
        ex = float(np.max(np.abs(x - ax)))
        ez = float(np.max(np.abs(z - az)))
        print(f"  (g) analytic x*={ax} z*={az}")
        print(f"      x_err={ex:.2e}  z_err={ez:.2e}  "
              f"q*={r.obj + cs.get('obj_const', 0.0):.10f}")
        check(name, "analytic_x", ex < 1e-6, f"{ex:.2e}")
        check(name, "analytic_z", ez < 1e-6, f"{ez:.2e}")

    # --- (f) cvxpy cross-check ------------------------------------------------
    try:
        import cvxpy as cp

        xv = cp.Variable(n)
        cons, tags = [], []
        if A is not None:
            ce = A @ xv == b
            cons.append(ce)
            tags.append(("eq", ce))
        if G is not None:
            ci = G @ xv <= h
            cons.append(ci)
            tags.append(("ineq", ci))
        if lb is not None:
            cl = xv >= lb
            cons.append(cl)
            tags.append(("lb", cl))
        if ub is not None:
            cu = xv <= ub
            cons.append(cu)
            tags.append(("ub", cu))
        obj = cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P)) + c @ xv)
        prob = cp.Problem(obj, cons)
        t0 = time.perf_counter()
        prob.solve(solver=cp.CLARABEL)
        t_o = time.perf_counter() - t0
        xo = np.asarray(xv.value, float)
        print(f"  (f) cvxpy: status={prob.status} obj={prob.value:.12e} t={t_o:.4f}s")
        print(f"      obj_err={abs(prob.value - r.obj):.2e}  "
              f"x_inf_err={np.max(np.abs(xo - x)):.2e}")
        check(name, "cvxpy_obj", abs(prob.value - r.obj) < 1e-6 * max(1, abs(r.obj)),
              f"{abs(prob.value - r.obj):.2e}")
        for tag, con in tags:
            dv = np.asarray(con.dual_value, float).reshape(-1)
            mine = {"eq": y, "ineq": z, "lb": zl, "ub": zu}[tag]
            # cvxpy's x >= lb dual has the same sign as z_lb under
            # L = f + ... + z_lb'(lb - x); cvxpy reports it >= 0 too.
            e = float(np.max(np.abs(dv - mine))) if dv.size else 0.0
            print(f"      cvxpy dual[{tag}] = {dv}   pounce = {mine}   diff={e:.2e}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (f) cvxpy unavailable/failed: {exc}")

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
for row in ROWS:
    print(f"  {row[0]:26s} {row[1]:18s} {row[2]:5s} {row[3]}")
print()
if FAILURES:
    print("FAILURES:")
    for f in FAILURES:
        print("  -", f)
    print(f"VERDICT: FAIL ({len(FAILURES)} checks)")
else:
    print("All KKT / duality / shadow-price checks passed.")
    print("VERDICT: PASS")
