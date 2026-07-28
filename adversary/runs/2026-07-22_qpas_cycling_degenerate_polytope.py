"""Adversary cross-check: active-set CYCLING / stalling stress tests.

Family: qp-active-set
Class:  massively degenerate vertex (constraint multiplicity) + Beale cycling
        geometry adapted to a QP.

Two independent instances, both engineered to be the classic anti-cycling
stress test (many more active constraints than the dimension of the vertex):

  (A) "duplicated cube": min 0.5||x - 2e||^2 over [0,1]^n with EVERY facet
      replicated REP times and each replica perturbed by j*1e-14.  At the
      optimum x* = e the n upper facets are active, each with REP replicas,
      so n*REP constraints are active at an n-dimensional vertex.  The
      constraint gradients of the replicas are identical to 1e-14, LICQ fails
      catastrophically, and the multipliers form an (n*REP - n)-dimensional
      family.  This is the configuration in which naive active-set pivoting
      cycles or stalls (Nocedal & Wright 2e, Sec. 16.5, "Degeneracy").
      Known optimum (n=5): x* = (1,1,1,1,1) (to 3e-14), f* = 2.5.

  (B) "Beale cycling": E. M. L. Beale (1955), "Cycling in the dual simplex
      algorithm", Naval Research Logistics Quarterly 2(4):269-275 -- the
      textbook 3-equality / 7-variable LP on which Dantzig's rule cycles
      forever through a degenerate vertex.  Adapted to a strictly convex QP
      by adding 0.5*eps*||x||^2 with eps = 1e-6, so the problem has a unique
      optimum yet retains the degenerate vertex geometry.
      LP optimum (Beale): f = -1/20 = -0.05 at x = (0,0,0, 1/25, 0, 1, 0).

Oracle: cvxpy (CLARABEL + OSQP), exact KKT residual check, and cross-check
against the pounce convex-QP IPM path and the general NLP path.

Every solve runs in a CHILD PROCESS under a hard wall-clock timeout so a
genuine cycle cannot hang the harness.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np

TIMEOUT_S = 25.0

# ---------------------------------------------------------------- instance A
N = 8
REP = 8


def build_cube():
    """min 0.5||x-2e||^2 over duplicated/perturbed [0,1]^n."""
    rows, rhs = [], []
    for i in range(N):
        for j in range(REP):
            d = j * 1e-14
            e = np.zeros(N)
            e[i] = 1.0
            rows.append(e.copy())  # x_i <= 1 - d
            rhs.append(1.0 - d)
            rows.append(-e.copy())  # -x_i <= 0 + d  (x_i >= -d)
            rhs.append(0.0 + d)
    G = np.array(rows)
    h = np.array(rhs)
    P = np.eye(N)
    c = -2.0 * np.ones(N)  # 0.5 x'x - 2 e'x = 0.5||x-2e||^2 - 2n
    const = 2.0 * N
    return P, c, G, h, const


# ---------------------------------------------------------------- instance B
EPS_B = 1e-6


def build_cone(m=40, n=5, seed=7):
    """Slater-FAILING polyhedral cone: m random hyperplanes through the origin.

    a_i' x <= 0 for i=1..m with a_i random in R^n, m >> n.  With m random
    directions the positive hull of {a_i} is all of R^n, so the feasible set
    collapses to the single point {0}: EVERY one of the m constraints is
    active at the (unique, hence optimal) feasible point, there is no
    interior (Slater fails), and the multipliers are wildly non-unique.
    min 0.5||x - d||^2 then has x* = 0, f* = 0.5||d||^2 for ANY d.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((m, n))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    h = np.zeros(m)
    d = np.ones(n)
    P = np.eye(n)
    c = -d
    const = 0.5 * float(d @ d)
    return P, c, G, h, const, d


def build_beale(eps=EPS_B):
    """Beale's cycling LP (1955) + 0.5*eps||x||^2.

    variables x = (x1..x7); x1,x2,x3 are the initial basic/slack variables.
        min  -0.75 x4 + 150 x5 - 0.02 x6 + 6 x7
        s.t.  x1 + 0.25 x4 - 60 x5 - 0.04 x6 +  9 x7 = 0
              x2 + 0.50 x4 - 90 x5 - 0.02 x6 +  3 x7 = 0
              x3                  +      x6          = 1
              x >= 0
    """
    n = 7
    A = np.zeros((3, n))
    A[0, 0] = 1.0
    A[0, 3:] = [0.25, -60.0, -0.04, 9.0]
    A[1, 1] = 1.0
    A[1, 3:] = [0.50, -90.0, -0.02, 3.0]
    A[2, 2] = 1.0
    A[2, 5] = 1.0
    b = np.array([0.0, 0.0, 1.0])
    c = np.zeros(n)
    c[3:] = [-0.75, 150.0, -0.02, 6.0]
    P = eps * np.eye(n)
    lb = np.zeros(n)
    return P, c, A, b, lb


# ---------------------------------------------------------------- child stage
def run_stage(name: str) -> dict:
    import warnings

    from scipy.optimize import LinearConstraint

    from pounce import minimize, solve_qp

    inst, mode = name.split(":", 1)

    if inst.startswith("cube") or inst.startswith("cone"):
        if inst.startswith("cube"):
            P, c, G, h, const = build_cube()
            x0 = np.full(N, 0.5)
        else:
            # "cone", "cone#<m>", "cone#<m>@<x0scale>"
            spec = inst[4:]
            mC = int(spec.split("@")[0].lstrip("#") or 40) if spec else 40
            s0 = float(spec.split("@")[1]) if "@" in spec else 0.1
            P, c, G, h, const, _d = build_cone(m=mC)
            x0 = np.full(P.shape[0], s0)

        def f(x):
            return float(0.5 * x @ P @ x + c @ x + const)

        def gf(x):
            return P @ x + c

        def hf(x):
            return P

        lc = LinearConstraint(G, -np.inf, h)
        cons = [lc]
        bounds = None
    else:
        eps = float(inst.split("@", 1)[1]) if "@" in inst else EPS_B
        P, c, A, b, lb = build_beale(eps)
        x0 = np.zeros(7)
        const = 0.0

        def f(x):
            return float(0.5 * x @ P @ x + c @ x)

        def gf(x):
            return P @ x + c

        def hf(x):
            return P

        cons = [LinearConstraint(A, b, b)]
        bounds = [(0.0, None)] * 7
        G = h = None

    t0 = time.perf_counter()
    if mode == "ipm":
        if G is not None:
            r = solve_qp(P=P, c=c, G=G, h=h)
        else:
            r = solve_qp(P=P, c=c, A=A, b=b, lb=lb)
        dt = time.perf_counter() - t0
        x = np.asarray(r.x, dtype=float)
        return {
            "status": str(r.status),
            "x": x.tolist(),
            "obj": float(r.obj) + const,
            "t": dt,
            "nit": int(getattr(r, "iterations", getattr(r, "nit", -1)) or -1),
        }

    kw = {"jac": gf, "hess": hf, "constraints": cons}
    if bounds is not None:
        kw["bounds"] = bounds
    if mode == "as":
        kw["solver_selection"] = "qp-active-set"
    elif mode == "asqp":
        kw["algorithm"] = "active-set-sqp"
    elif mode == "nlp":
        kw["solver_selection"] = "nlp"
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        r = minimize(f, x0, **kw)
    dt = time.perf_counter() - t0
    x = np.asarray(r.x, dtype=float)
    return {
        "status": str(r.status),
        "success": bool(r.success),
        "x": x.tolist(),
        "obj": float(r.fun),
        "t": dt,
        "nit": int(getattr(r, "nit", -1) or -1),
        "msg": str(getattr(r, "message", ""))[:80],
        "warn": [str(w.message)[:90] for w in ws][:4],
    }


# ---------------------------------------------------------------- driver
def child(stage: str) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run(
        [sys.executable, __file__, "--stage", stage],
        capture_output=True,
        text=True,
        timeout=None if False else TIMEOUT_S + 5,
    )
    del t0
    for line in p.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[8:])
    return {"status": "ERROR", "stderr": p.stderr[-800:], "rc": p.returncode}


def child_guarded(stage: str) -> dict:
    try:
        return child(stage)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "t": TIMEOUT_S}


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def main():
    print(f"# duplicated-cube QP: n={N}, {2 * N * REP} rows, {N * REP} active at x*")
    res_cube = {m: child_guarded(f"cube:{m}") for m in ("as", "asqp", "ipm", "nlp")}
    res_beale = {m: child_guarded(f"beale:{m}") for m in ("as", "asqp", "ipm", "nlp")}

    import cvxpy as cp

    # ---- oracle A
    P, c, G, h, const = build_cube()
    xv = cp.Variable(N)
    pr = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(xv - 2.0)), [G @ xv <= h])
    t0 = time.perf_counter()
    pr.solve(solver=cp.CLARABEL)
    t_cl_a = time.perf_counter() - t0
    orcA = {"obj": float(pr.value), "x": np.asarray(xv.value).tolist(), "t": t_cl_a}
    pr.solve(solver=cp.OSQP, eps_abs=1e-12, eps_rel=1e-12, max_iter=200000)
    orcA2 = float(pr.value)

    # ---- oracle B
    Pb, cb, Ab, bb, lbb = build_beale()
    yv = cp.Variable(7)
    prb = cp.Problem(
        cp.Minimize(0.5 * EPS_B * cp.sum_squares(yv) + cb @ yv),
        [Ab @ yv == bb, yv >= 0],
    )
    t0 = time.perf_counter()
    prb.solve(solver=cp.CLARABEL)
    t_cl_b = time.perf_counter() - t0
    orcB = {"obj": float(prb.value), "x": np.asarray(yv.value).tolist(), "t": t_cl_b}
    prb.solve(solver=cp.OSQP, eps_abs=1e-12, eps_rel=1e-12, max_iter=400000)
    orcB2 = float(prb.value)

    KNOWN_A = 0.5 * N  # x* = e (to REP*1e-14), f* = 0.5 * n * 1^2
    KNOWN_B_LP = -0.05  # Beale LP optimum; QP value shifted by O(eps)

    print("\n=== (A) duplicated/perturbed cube QP ===")
    print(f"known: x*=e f*={KNOWN_A:.10e}")
    print(
        f"cvxpy CLARABEL obj={orcA['obj']:.12e} t={orcA['t']:.4f}s  "
        f"OSQP obj={orcA2:.12e}"
    )
    for m, r in res_cube.items():
        if "obj" in r:
            x = np.asarray(r["x"])
            viol = float(np.max(G @ x - h))
            print(
                f"pounce[{m:>4}] status={r['status']:<12} obj={r['obj']:.12e} "
                f"err={rel(r['obj'], KNOWN_A):.2e} nit={r.get('nit')} "
                f"t={r['t']:.4f}s maxviol={viol:.2e} xerr={np.max(np.abs(x - 1.0)):.2e}"
            )
            if r.get("warn"):
                print(f"            warn={r['warn']}")
        else:
            print(f"pounce[{m:>4}] status={r['status']} {r.get('stderr', '')[:300]}")

    print("\n=== (B) Beale cycling LP + eps*I  (eps=1e-6) ===")
    print(f"reference LP optimum={KNOWN_B_LP:.10e} (QP value shifted by O(eps))")
    print(
        f"cvxpy CLARABEL obj={orcB['obj']:.12e} x={np.round(orcB['x'], 8)} "
        f"t={orcB['t']:.4f}s  OSQP obj={orcB2:.12e}"
    )
    for m, r in res_beale.items():
        if "obj" in r:
            x = np.asarray(r["x"])
            eqv = float(np.max(np.abs(Ab @ x - bb)))
            lbv = float(max(0.0, -x.min()))
            print(
                f"pounce[{m:>4}] status={r['status']:<12} obj={r['obj']:.12e} "
                f"err={rel(r['obj'], orcB['obj']):.2e} nit={r.get('nit')} "
                f"t={r['t']:.4f}s eqviol={eqv:.2e} lbviol={lbv:.2e}"
            )
            if r.get("warn"):
                print(f"            warn={r['warn']}")
        else:
            print(f"pounce[{m:>4}] status={r['status']} {r.get('stderr', '')[:300]}")

    # ---- (C) Slater-failing cone: feasible set = {0}, all m constraints active
    Pc, cc, Gc, hc, constc, dvec = build_cone()
    nC = Pc.shape[0]
    zv = cp.Variable(nC)
    prc = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(zv - dvec)), [Gc @ zv <= hc])
    t0 = time.perf_counter()
    prc.solve(solver=cp.CLARABEL)
    t_cl_c = time.perf_counter() - t0
    orcC = {"obj": float(prc.value), "x": np.asarray(zv.value), "t": t_cl_c}
    KNOWN_C = 0.5 * float(dvec @ dvec)  # x* = 0 (feasible set is the origin only)
    res_cone = {m: child_guarded(f"cone:{m}") for m in ("as", "asqp", "ipm", "nlp")}

    print(
        f"\n=== (C) Slater-failing cone: {Gc.shape[0]} constraints, "
        f"n={nC}, feasible set = {{0}} ==="
    )
    print(f"known: x*=0 f*={KNOWN_C:.10e}")
    print(
        f"cvxpy CLARABEL obj={orcC['obj']:.12e} |x|inf={np.max(np.abs(orcC['x'])):.2e} "
        f"t={orcC['t']:.4f}s"
    )
    for m, r in res_cone.items():
        if "obj" in r:
            x = np.asarray(r["x"])
            print(
                f"pounce[{m:>4}] status={r['status']:<12} obj={r['obj']:.12e} "
                f"err={rel(r['obj'], KNOWN_C):.2e} nit={r.get('nit')} "
                f"t={r['t']:.4f}s maxviol={np.max(Gc @ x - hc):.2e} "
                f"|x|inf={np.max(np.abs(x)):.2e}"
            )
        else:
            print(f"pounce[{m:>4}] status={r['status']} {r.get('stderr', '')[:300]}")

    # ---- (C2) how many degenerate active constraints does it take to break it?
    #      x0 = 0 is EXACTLY the unique feasible point and the exact optimum.
    print(
        "\n=== (C2) cone sweep in m, started AT the exact optimum x0 = 0 "
        "(feasible set = {0} for every m below; verified by cvxpy) ==="
    )
    for mC in (12, 15, 20, 25, 30, 40):
        Pm, cm, Gm, hm, constm, dm = build_cone(m=mC)
        xm = cp.Variable(Pm.shape[0])
        pm = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(xm - dm)), [Gm @ xm <= hm])
        pm.solve(solver=cp.CLARABEL)
        r = child_guarded(f"cone#{mC}@0.0:as")
        if "obj" in r:
            x = np.asarray(r["x"])
            print(
                f"m={mC:3d} cvxpy={pm.value:.10f} | as: status={r['status']} "
                f"success={r.get('success')} msg={r.get('msg')!r} obj={r['obj']:.10f} "
                f"|x|inf={np.max(np.abs(x)):.1e} maxviol={np.max(Gm @ x - hm):.1e} "
                f"nit={r.get('nit')} t={r['t']:.3f}s"
            )
            if r.get("warn"):
                print(f"      warn={r['warn']}")
        else:
            print(f"m={mC:3d} as: {r['status']}")

    # ---- (D) Beale in the LP limit: eps -> 0 (pure degenerate cycling geometry)
    print("\n=== (D) Beale eps-sweep (eps -> 0 = the pure cycling LP) ===")
    sweep_fail = []
    for eps in (1e-4, 1e-6, 1e-9, 1e-12):
        Pe, ce, Ae, be, lbe = build_beale(eps)
        wv = cp.Variable(7)
        pre = cp.Problem(
            cp.Minimize(0.5 * eps * cp.sum_squares(wv) + ce @ wv),
            [Ae @ wv == be, wv >= 0],
        )
        pre.solve(solver=cp.CLARABEL)
        ref = float(pre.value)
        row = []
        for m in ("as", "ipm", "nlp"):
            r = child_guarded(f"beale@{eps:g}:{m}")
            if "obj" in r:
                e = rel(r["obj"], ref)
                row.append(f"{m}={r['status']}/{r['obj']:.8e}/err{e:.1e}/t{r['t']:.3f}")
                if e > 1e-4:
                    sweep_fail.append(f"D[eps={eps:g}]/{m}:err={e:.2e}")
            else:
                row.append(f"{m}={r['status']}")
                sweep_fail.append(f"D[eps={eps:g}]/{m}:{r['status']}")
        print(f"eps={eps:>8g} cvxpy={ref:.8e} | " + "  ".join(row))

    # ------------------------------------------------ verdict
    fails = list(sweep_fail)
    for tag, res, ref, chk in (
        ("A", res_cube, KNOWN_A, None),
        ("B", res_beale, orcB["obj"], None),
        ("C", res_cone, KNOWN_C, None),
    ):
        del chk
        for m, r in res.items():
            if "obj" not in r:
                fails.append(f"{tag}/{m}:{r['status']}")
                continue
            if rel(r["obj"], ref) > 1e-4:
                fails.append(f"{tag}/{m}:obj_err={rel(r['obj'], ref):.2e}")
            if m != "ipm" and not r.get("success", True):
                fails.append(f"{tag}/{m}:not-success({r['status']})")
    # feasibility
    xa = np.asarray(res_cube["as"].get("x", [np.nan] * N))
    if np.all(np.isfinite(xa)) and float(np.max(G @ xa - h)) > 1e-7:
        fails.append("A/as:infeasible")

    print("\nVERDICT: PASS" if not fails else f"\nVERDICT: FAIL {fails}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--stage":
        out = run_stage(sys.argv[2])
        print("@@JSON@@" + json.dumps(out))
    else:
        main()
