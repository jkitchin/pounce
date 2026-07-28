"""Adversary cross-check: convex QP status edge cases with a SINGULAR PSD Hessian.

Family: qp    Class: status reporting (unbounded / bounded near-miss / infeasible)
Source: analytic. Standard convex-QP boundedness theorem (Frank & Wolfe 1956;
        Nocedal & Wright, "Numerical Optimization" 2e, Ch. 16; Boyd &
        Vandenberghe 4.4): for min 1/2 x'Px + c'x over a polyhedron F with
        P >= 0, the problem is bounded below IFF every recession direction
        d of F with P d = 0 satisfies c'd >= 0.  When bounded below, the
        infimum is ATTAINED (Frank-Wolfe theorem) -- no "finite but not
        attained" trap here.

Shared skeleton (n = 3):

    P = [[ 2, -2, 0],
         [-2,  2, 0],
         [ 0,  0, 0]]                 eigenvalues {4, 0, 0}  => PSD, singular
    null(P) = span{ (1,1,0), (0,0,1) }

    F = { x : x >= 0 }                recession cone = nonneg orthant
    null(P) INTERSECT rec(F) = cone{ (1,1,0), (0,0,1) }

    So bounded-below  <=>  c1 + c2 >= 0  AND  c3 >= 0.

Cases (a) and (b) differ ONLY in the SIGN OF c3.  Confusing them is the finding.

  (a) c = (-1, 2, -1):  c3 = -1 < 0, d = (0,0,1) is a recession direction in
      null(P) with c'd = -1 < 0  =>  UNBOUNDED BELOW.
      Witness ray: x(t) = (0.5, 0, t), f = -0.25 - t -> -inf.

  (b) c = (-1, 2, +1):  c1+c2 = 1 >= 0, c3 = 1 >= 0  =>  BOUNDED.
      With u = x1 - x2:  f = u^2 - u + x2 + x3 over x >= 0, so x2 = x3 = 0,
      u = x1 = 1/2.  UNIQUE optimum x* = (0.5, 0, 0), f* = -0.25 EXACTLY.

  (b0) c = (-1, 1, 0):  c1+c2 = 0, c3 = 0 -- the null-space component of c is
      EXACTLY ORTHOGONAL to every recession direction in null(P).  The sharpest
      near-miss: still BOUNDED, f* = -0.25, but the optimal SET is unbounded
      ({x1 - x2 = 1/2, x3 >= 0} intersect x >= 0).

  (c1) same P, c=(-1,2,1), plus contradictory (rank-deficient) equalities
       x1+x2 = 1 and x1+x2 = 2   =>  INFEASIBLE.
  (c2) same P, c, plus FULL-ROW-RANK equalities x1+x2 = 1, x1+x2+x3 = 0 with
       x >= 0  =>  forces x3 = -1 < 0  =>  INFEASIBLE.
  (d)  same P, c=(-1,2,1), no constraints, but BOUNDS lb=(0,0,1), ub=(10,10,0):
       x3 in [1, 0] is empty  =>  INFEASIBLE.  Must not be silently swapped
       into [0,1] and must not crash.

Oracles: (1) the analytic argument above (primary, exact);
         (2) cvxpy CLARABEL and OSQP.
"""

import time

import numpy as np

np.set_printoptions(precision=6, suppress=True)

P = np.array([[2.0, -2.0, 0.0], [-2.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
# x >= 0 written as G x <= h so the sign convention is unambiguous.
G = -np.eye(3)
h = np.zeros(3)

C_UNBOUNDED = np.array([-1.0, 2.0, -1.0])
C_BOUNDED = np.array([-1.0, 2.0, 1.0])
C_ORTHO = np.array([-1.0, 1.0, 0.0])

UNBOUNDED_STATUSES = {"unbounded", "dual_infeasible", "infeasible_or_unbounded",
                      "primal_unbounded", "unbounded_below"}
INFEASIBLE_STATUSES = {"infeasible", "primal_infeasible", "infeasible_or_unbounded",
                       "primal_infeasible_inaccurate"}


def sanity_check_ground_truth():
    """Re-derive the analytic facts numerically before accusing anybody."""
    w = np.linalg.eigvalsh(P)
    assert w.min() > -1e-12, f"P is not PSD: {w}"
    assert abs(w.min()) < 1e-12, f"P is not singular: {w}"
    # null(P) basis
    _, s, vt = np.linalg.svd(P)
    null_basis = vt[s < 1e-10]
    # the two claimed generators must lie in null(P)
    for d in (np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])):
        assert np.linalg.norm(P @ d) < 1e-12, f"{d} not in null(P)"
        assert (d >= 0).all(), f"{d} not a recession direction of x>=0"
    print(f"[gt] eig(P) = {w}  (PSD, rank {np.linalg.matrix_rank(P)}), "
          f"dim null(P) = {null_basis.shape[0]}")
    print(f"[gt] (a) c3 = {C_UNBOUNDED[2]:+.1f} < 0 along d=(0,0,1) in null(P) "
          f"& rec(F)  => UNBOUNDED")
    print(f"[gt] (b) c1+c2 = {C_BOUNDED[0] + C_BOUNDED[1]:+.1f} >= 0, "
          f"c3 = {C_BOUNDED[2]:+.1f} >= 0  => BOUNDED, f* = -0.25")
    print(f"[gt] (b0) c1+c2 = {C_ORTHO[0] + C_ORTHO[1]:+.1f}, "
          f"c3 = {C_ORTHO[2]:+.1f}  => BOUNDED (exactly orthogonal), f* = -0.25")
    # numeric witness that (a) really dives
    for t in (0.0, 1e3, 1e6):
        x = np.array([0.5, 0.0, t])
        f = 0.5 * x @ P @ x + C_UNBOUNDED @ x
        assert (x >= 0).all()
        print(f"[gt] (a) witness t={t:>9.0e}: feasible, f = {f:.6e}")
    # numeric witness that (b) does NOT dive along the same ray
    for t in (0.0, 1e3, 1e6):
        x = np.array([0.5, 0.0, t])
        f = 0.5 * x @ P @ x + C_BOUNDED @ x
        print(f"[gt] (b) same ray t={t:>9.0e}: f = {f:.6e}  (rises)")
    # brute-force confirm f* = -0.25 for (b) on a grid
    g = np.linspace(0.0, 3.0, 301)
    X1, X2 = np.meshgrid(g, g)
    F = (X1 - X2) ** 2 - X1 + 2 * X2
    print(f"[gt] (b) grid min over x1,x2 in [0,3], x3=0: {F.min():.10f} "
          f"(analytic -0.25)")
    print()


def run_pounce(name, **kw):
    from pounce import solve_qp
    t0 = time.perf_counter()
    try:
        r = solve_qp(**kw)
        dt = time.perf_counter() - t0
        status = r.status
        obj = r.obj
        x = np.asarray(r.x) if r.x is not None else None
        print(f"[pounce {name}] status={status!r} obj={obj!r} "
              f"success={r.success} t={dt:.4f}s")
        if x is not None:
            print(f"[pounce {name}] x = {x}")
        return dict(status=status, obj=obj, x=x, t=dt, exc=None, res=r)
    except Exception as e:  # noqa: BLE001 - a raise is itself a legitimate answer
        dt = time.perf_counter() - t0
        print(f"[pounce {name}] RAISED {type(e).__name__}: {e}  t={dt:.4f}s")
        return dict(status=f"raised:{type(e).__name__}", obj=None, x=None,
                    t=dt, exc=e, res=None)


def run_cvxpy(name, c, eq=None, beq=None, lb=None, ub=None, nonneg=True):
    import cvxpy as cp
    out = {}
    for solver in ("CLARABEL", "OSQP"):
        x = cp.Variable(3)
        cons = []
        if nonneg:
            cons.append(x >= 0)
        if eq is not None:
            cons.append(eq @ x == beq)
        if lb is not None:
            cons.append(x >= lb)
        if ub is not None:
            cons.append(x <= ub)
        prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(P)) + c @ x),
                          cons)
        t0 = time.perf_counter()
        try:
            prob.solve(solver=getattr(cp, solver))
            st, val = prob.status, prob.value
        except Exception as e:  # noqa: BLE001
            st, val = f"raised:{type(e).__name__}", None
        dt = time.perf_counter() - t0
        print(f"[cvxpy/{solver} {name}] status={st!r} obj={val!r} t={dt:.4f}s")
        out[solver] = (st, val)
    return out


def classify(name, truth, p, cvx, expect_obj=None):
    st = str(p["status"]).lower()
    if truth == "unbounded":
        ok = st in UNBOUNDED_STATUSES or st.startswith("raised:")
        detail = "reported unbounded/dual-infeasible" if ok else \
                 f"CONFIDENT WRONG STATUS {p['status']!r} obj={p['obj']!r}"
    elif truth == "infeasible":
        ok = st in INFEASIBLE_STATUSES or st.startswith("raised:")
        detail = "reported infeasible" if ok else \
                 f"CONFIDENT WRONG STATUS {p['status']!r} obj={p['obj']!r}"
    else:  # bounded
        ok = st == "optimal" and p["obj"] is not None and \
            abs(p["obj"] - expect_obj) < 1e-6
        if st in UNBOUNDED_STATUSES or st in INFEASIBLE_STATUSES:
            detail = f"CONFIDENT WRONG STATUS {p['status']!r} on a provably " \
                     f"bounded QP (f* = {expect_obj})"
        elif st == "optimal":
            detail = (f"obj err vs analytic {abs(p['obj'] - expect_obj):.3e}"
                      if p["obj"] is not None else "optimal with obj=None")
        else:
            detail = f"non-optimal status {p['status']!r}"
    print(f"[verdict {name}] truth={truth} -> {'OK' if ok else 'FINDING'}: {detail}")
    return ok, detail


def main():
    sanity_check_ground_truth()
    results = {}

    print("### (a) singular PSD Hessian, UNBOUNDED below (c3 = -1)")
    p = run_pounce("a", P=P, c=C_UNBOUNDED, G=G, h=h)
    cvx = run_cvxpy("a", C_UNBOUNDED)
    results["a_unbounded"] = classify("a", "unbounded", p, cvx)
    print()

    print("### (b) SAME QP, c3 flipped to +1: provably BOUNDED, f* = -0.25")
    p = run_pounce("b", P=P, c=C_BOUNDED, G=G, h=h)
    cvx = run_cvxpy("b", C_BOUNDED)
    results["b_bounded"] = classify("b", "bounded", p, cvx, expect_obj=-0.25)
    print()

    print("### (b0) c null-space part EXACTLY orthogonal to rec(F): "
          "BOUNDED, f* = -0.25, optimal set unbounded")
    p = run_pounce("b0", P=P, c=C_ORTHO, G=G, h=h)
    cvx = run_cvxpy("b0", C_ORTHO)
    results["b0_ortho"] = classify("b0", "bounded", p, cvx, expect_obj=-0.25)
    print()

    print("### (c1) contradictory RANK-DEFICIENT equalities x1+x2=1, x1+x2=2")
    Aeq = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    beq = np.array([1.0, 2.0])
    p = run_pounce("c1", P=P, c=C_BOUNDED, A=Aeq, b=beq, G=G, h=h)
    cvx = run_cvxpy("c1", C_BOUNDED, eq=Aeq, beq=beq)
    results["c1_infeasible_dup"] = classify("c1", "infeasible", p, cvx)
    print()

    print("### (c2) full-rank equalities x1+x2=1, x1+x2+x3=0 with x>=0 "
          "(forces x3=-1)")
    Aeq2 = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    beq2 = np.array([1.0, 0.0])
    p = run_pounce("c2", P=P, c=C_BOUNDED, A=Aeq2, b=beq2, G=G, h=h)
    cvx = run_cvxpy("c2", C_BOUNDED, eq=Aeq2, beq=beq2)
    results["c2_infeasible_rank_full"] = classify("c2", "infeasible", p, cvx)
    print()

    print("### (d) empty feasible set from BOUNDS ONLY: lb=(0,0,1) > ub=(10,10,0)")
    lb = np.array([0.0, 0.0, 1.0])
    ub = np.array([10.0, 10.0, 0.0])
    p = run_pounce("d", P=P, c=C_BOUNDED, lb=lb, ub=ub)
    cvx = run_cvxpy("d", C_BOUNDED, lb=lb, ub=ub, nonneg=False)
    ok, detail = classify("d", "infeasible", p, cvx)
    # extra trap: did it silently swap the bounds and solve [0,1]?
    if p["x"] is not None and str(p["status"]).lower() == "optimal":
        x3 = p["x"][2]
        print(f"[verdict d] returned x3 = {x3:.6f}; lb3=1 ub3=0 -> "
              f"{'SILENT BOUND SWAP' if -1e-9 <= x3 <= 1 + 1e-9 else 'nonsense'}")
    results["d_infeasible_bounds"] = (ok, detail)
    print()

    print("=" * 70)
    n_fail = sum(1 for ok, _ in results.values() if not ok)
    for k, (ok, detail) in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k:28s} {detail}")
    print("=" * 70)
    print("VERDICT: PASS" if n_fail == 0 else f"VERDICT: FAIL ({n_fail} case(s))")


if __name__ == "__main__":
    main()
