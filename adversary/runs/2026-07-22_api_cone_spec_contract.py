"""Adversary cross-check: THE CONE SPECIFICATION CONTRACT of solve_socp(cones=[...])

Family: socp / api-contract   Class: option handling & input edge cases
Source: pounce solve_socp docstring (cone spec contract) + Boyd & Vandenberghe,
        *Convex Optimization*, S4.4 (SOCP standard form); projection onto the
        nonnegative orthant has a closed-form optimum.

Contract under test:
  (a) sum(cone dims) must equal rows(G)/len(h)  -> clear error, no truncation / OOB
  (b) ("soc", 0) and ("soc", 1) edge dims
  (c) ("exp", d != 3), ("pow", alpha) with alpha in {0, 1, -0.5, 1.5, NaN}
  (d) ("psd", n) with rows != n(n+1)/2
  (e) empty cone list with nonempty G
  (f) unknown cone type string ("banana", 3)
  (g) cone ORDER invariance (permuted cone list + permuted G/h rows)
  (h) 200 x ("soc", 2) vs one equivalent 400-row nonneg block

For malformed specs the contract is: CLEAR EXCEPTION, no silent garbage, no panic.
"""

import math
import time
import traceback

import numpy as np

from pounce import solve_socp

RESULTS = []  # (case_id, label, outcome, detail)

CLEAN_EXC = (ValueError, TypeError, IndexError, KeyError, OverflowError, RuntimeError)


def _is_panic(exc):
    """Rust panic surfaced through pyo3 -> pyo3_runtime.PanicException."""
    name = type(exc).__name__
    mod = type(exc).__module__ or ""
    return "Panic" in name or "panic" in str(exc).lower() or mod.startswith("pyo3_runtime")


def expect_error(case_id, label, fn):
    """A malformed spec must raise a clean exception. Panic or silent success = BUG."""
    t0 = time.perf_counter()
    try:
        r = fn()
    except BaseException as exc:  # noqa: BLE001 - we want to see panics too
        dt = time.perf_counter() - t0
        if _is_panic(exc):
            RESULTS.append((case_id, label, "BUG",
                            f"PANIC {type(exc).__name__}: {str(exc)[:200]}"))
            print(f"[{case_id}] {label}: *** PANIC *** {type(exc).__name__}: {str(exc)[:300]}")
        elif isinstance(exc, CLEAN_EXC):
            RESULTS.append((case_id, label, "PASS",
                            f"{type(exc).__name__}: {str(exc)[:160]}"))
            print(f"[{case_id}] {label}: clean {type(exc).__name__}: {str(exc)[:200]}  ({dt:.3f}s)")
        else:
            RESULTS.append((case_id, label, "SUSPECT",
                            f"unusual exception {type(exc).__name__}: {str(exc)[:160]}"))
            print(f"[{case_id}] {label}: unusual {type(exc).__name__}: {str(exc)[:200]}")
        return None
    dt = time.perf_counter() - t0
    obj = getattr(r, "obj", None)
    st = getattr(r, "status", None)
    RESULTS.append((case_id, label, "NO_ERROR",
                    f"returned status={st} obj={obj} x={np.asarray(getattr(r, 'x', []))[:6]}"))
    print(f"[{case_id}] {label}: NO ERROR -> status={st} obj={obj} "
          f"x={np.asarray(getattr(r, 'x', []))[:6]}  ({dt:.3f}s)")
    return r


def expect_value(case_id, label, fn, expected, tol=1e-6):
    t0 = time.perf_counter()
    try:
        r = fn()
    except BaseException as exc:  # noqa: BLE001
        kind = "BUG" if _is_panic(exc) else "ERROR"
        RESULTS.append((case_id, label, kind, f"{type(exc).__name__}: {str(exc)[:160]}"))
        print(f"[{case_id}] {label}: raised {type(exc).__name__}: {str(exc)[:200]}")
        return None
    dt = time.perf_counter() - t0
    obj = float(r.obj)
    err = abs(obj - expected) / max(1.0, abs(expected))
    ok = (r.status == "optimal") and err < tol
    RESULTS.append((case_id, label, "PASS" if ok else "FAIL",
                    f"status={r.status} obj={obj:.12e} expected={expected:.12e} rel_err={err:.2e}"))
    print(f"[{case_id}] {label}: status={r.status} obj={obj:.12e} "
          f"expected={expected:.12e} rel_err={err:.2e} t={dt:.3f}s "
          f"{'OK' if ok else '<<< MISMATCH'}")
    return r


# ---------------------------------------------------------------------------
# Well-posed reference problem (used by (g), and as the base for (a))
#
#   variables z = (x1, x2, t)
#   minimize  t
#   s.t.      (t, x1 + 1, x2 - 2) in SOC(3)      i.e. t >= ||x - (-1, 2)||
#             x1 >= 0, x2 >= 0                    nonneg(2)
#
#   => projection of (-1, 2) onto the nonneg orthant is (0, 2), distance 1.
#   Known optimum: t* = 1, x* = (0, 2).
# ---------------------------------------------------------------------------
KNOWN_OPT_G = 1.0
C_G = np.array([0.0, 0.0, 1.0])

# SOC block rows (3), then nonneg block rows (2)
G_soc = np.array([
    [0.0, 0.0, -1.0],   # s = t
    [-1.0, 0.0, 0.0],   # s = x1 + 1
    [0.0, -1.0, 0.0],   # s = x2 - 2
])
h_soc = np.array([0.0, 1.0, -2.0])
G_nn = np.array([
    [-1.0, 0.0, 0.0],   # s = x1
    [0.0, -1.0, 0.0],   # s = x2
])
h_nn = np.array([0.0, 0.0])

G_base = np.vstack([G_soc, G_nn])
h_base = np.concatenate([h_soc, h_nn])
CONES_BASE = [("soc", 3), ("nonneg", 2)]

# permuted: nonneg block first, then soc block
G_perm = np.vstack([G_nn, G_soc])
h_perm = np.concatenate([h_nn, h_soc])
CONES_PERM = [("nonneg", 2), ("soc", 3)]


def main():
    print("=" * 78)
    print("CONE SPECIFICATION CONTRACT of solve_socp(cones=[...])")
    print("=" * 78)

    # ---------------- sanity: the base problem itself ----------------
    print("\n--- sanity: base well-posed problem ---")
    r0 = expect_value("S0", "base [soc3, nonneg2]",
                      lambda: solve_socp(c=C_G, G=G_base, h=h_base, cones=CONES_BASE),
                      KNOWN_OPT_G)
    if r0 is not None:
        print(f"     x = {np.asarray(r0.x)}  (expected x1=0, x2=2, t=1)")

    # ---------------- (a) dimension sum mismatch ----------------
    print("\n--- (a) cone dims do not sum to rows(G) ---")
    expect_error("a1", "dims sum SHORT (soc3+nonneg1, 5 rows)",
                 lambda: solve_socp(c=C_G, G=G_base, h=h_base,
                                    cones=[("soc", 3), ("nonneg", 1)]))
    expect_error("a2", "dims sum LONG (soc3+nonneg7, 5 rows)",
                 lambda: solve_socp(c=C_G, G=G_base, h=h_base,
                                    cones=[("soc", 3), ("nonneg", 7)]))
    expect_error("a3", "dims sum WAY LONG (soc 100000, 5 rows) OOB read?",
                 lambda: solve_socp(c=C_G, G=G_base, h=h_base,
                                    cones=[("soc", 100000)]))
    expect_error("a4", "negative dim (soc -3)",
                 lambda: solve_socp(c=C_G, G=G_base, h=h_base,
                                    cones=[("soc", -3), ("nonneg", 8)]))
    expect_error("a5", "h shorter than G rows",
                 lambda: solve_socp(c=C_G, G=G_base, h=h_base[:4], cones=CONES_BASE))

    # ---------------- (b) zero-dim and 1-dim SOC ----------------
    print("\n--- (b) ('soc', 0) and ('soc', 1) ---")
    # ("soc", 0) padding a otherwise-valid spec: rows still sum to 5
    expect_value("b1", "('soc',0) prefix + soc3 + nonneg2",
                 lambda: solve_socp(c=C_G, G=G_base, h=h_base,
                                    cones=[("soc", 0)] + CONES_BASE),
                 KNOWN_OPT_G)
    # ("soc", 0) alone with zero rows -> degenerate but arguably fine
    expect_error("b2", "('soc',0) alone with 0-row G",
                 lambda: solve_socp(c=[1.0], G=np.zeros((0, 1)), h=np.zeros(0),
                                    cones=[("soc", 0)]))
    # ("soc", 1): s0 >= ||()|| = 0, i.e. a nonnegativity constraint.
    #   min x  s.t.  x - 1 >= 0  ->  x* = 1
    # 0-dim blocks contribute 0 rows, so the row-sum check PASSES and the
    # spec reaches the cone constructors. nonneg(0) is fine; soc/psd panic.
    for cid, kind in [("b1n", "nonneg"), ("b1s", "soc"), ("b1p", "psd")]:
        expect_value(cid, f"('{kind}',0) + soc3 + nonneg2 (rows sum = 5, valid)",
                     lambda k=kind: solve_socp(c=C_G, G=G_base, h=h_base,
                                               cones=[(k, 0)] + CONES_BASE),
                     KNOWN_OPT_G)
    # negative dims saturate to 0 on the Rust f64->usize cast: same panics
    for cid, kind in [("b2n", "nonneg"), ("b2s", "soc"), ("b2p", "psd")]:
        expect_value(cid, f"('{kind}',-3) saturates to 0 (rows sum = 5, valid)",
                     lambda k=kind: solve_socp(c=C_G, G=G_base, h=h_base,
                                               cones=[(k, -3)] + CONES_BASE),
                     KNOWN_OPT_G)
    # fractional dim rounding to 0 hits the same path
    expect_value("b2f", "('soc',0.4) rounds to 0 (rows sum = 5, valid)",
                 lambda: solve_socp(c=C_G, G=G_base, h=h_base,
                                    cones=[("soc", 0.4)] + CONES_BASE),
                 KNOWN_OPT_G)

    expect_value("b3", "('soc',1) == nonneg: min x s.t. x>=1",
                 lambda: solve_socp(c=[1.0], G=np.array([[-1.0]]), h=np.array([-1.0]),
                                    cones=[("soc", 1)]),
                 1.0)
    expect_value("b3ref", "('nonneg',1) reference: min x s.t. x>=1",
                 lambda: solve_socp(c=[1.0], G=np.array([[-1.0]]), h=np.array([-1.0]),
                                    cones=[("nonneg", 1)]),
                 1.0)
    # ("soc",1) as a *binding* constraint inside a bigger problem:
    #   min -x  s.t. (2 - x) in SOC(1) -> x <= 2 -> obj = -2
    expect_value("b4", "('soc',1) binding: min -x s.t. x<=2",
                 lambda: solve_socp(c=[-1.0], G=np.array([[1.0]]), h=np.array([2.0]),
                                    cones=[("soc", 1)]),
                 -2.0)

    # ---------------- (c) exp / pow malformed ----------------
    print("\n--- (c) ('exp', d!=3) and ('pow', alpha) out of range ---")
    # A valid 3-row exp problem for shape reference:
    #   min t s.t. (u, 1, t) in Kexp with u fixed = 0 -> t >= 1
    Gexp = np.zeros((3, 2))
    Gexp[0, 0] = -1.0     # s0 = u
    Gexp[2, 1] = -1.0     # s2 = t
    hexp = np.array([0.0, 1.0, 0.0])
    cexp = np.array([0.0, 1.0])
    # NOTE: u must be PINNED, else u -> -inf and t -> 0 (my first version of
    # this check left u free and was a FORMULATION_ERROR, not a solver bug).
    expect_value("c0", "sanity ('exp',3): min t, (u,1,t) in Kexp, u==1 -> t=e",
                 lambda: solve_socp(c=cexp, A=[[1.0, 0.0]], b=[1.0],
                                    G=Gexp, h=hexp, cones=[("exp", 3)]),
                 math.e, tol=1e-5)
    expect_error("c1", "('exp',2) with 2-row G",
                 lambda: solve_socp(c=cexp, G=Gexp[:2], h=hexp[:2], cones=[("exp", 2)]))
    expect_error("c2", "('exp',4) with 4-row G",
                 lambda: solve_socp(c=cexp, G=np.vstack([Gexp, Gexp[:1]]),
                                    h=np.concatenate([hexp, hexp[:1]]),
                                    cones=[("exp", 4)]))
    expect_error("c3", "('exp',0)",
                 lambda: solve_socp(c=cexp, G=Gexp, h=hexp,
                                    cones=[("exp", 0), ("exp", 3)]))

    # power cone: 3 rows, second element is the EXPONENT alpha in (0,1)
    Gpow = np.zeros((3, 3))
    Gpow[0, 0] = -1.0
    Gpow[1, 1] = -1.0
    Gpow[2, 2] = -1.0
    hpow = np.array([0.0, 0.0, 0.0])
    cpow = np.array([-1.0, 1.0, 1.0])
    # NOTE: pin y,z away from the cone apex; the apex-only version was
    # degenerate (optimal_inaccurate at obj~5e-8) -- again my error.
    #   max x s.t. |x| <= y^0.5 z^0.5, y=4, z=1  ->  x* = 2, obj = -2
    expect_value("c4", "sanity ('pow',0.5): max x, y=4 z=1 -> -2",
                 lambda: solve_socp(c=[-1.0, 0.0, 0.0],
                                    A=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], b=[4.0, 1.0],
                                    G=Gpow, h=hpow, cones=[("pow", 0.5)]),
                 -2.0, tol=1e-5)
    for cid, alpha in [("c5", 0.0), ("c6", 1.0), ("c7", -0.5), ("c8", 1.5),
                       ("c9", float("nan")), ("c10", float("inf"))]:
        expect_error(cid, f"('pow', {alpha})",
                     lambda a=alpha: solve_socp(c=[0.0, 1.0, 1.0], G=Gpow, h=hpow,
                                                cones=[("pow", a)]))

    # ---------------- (d) psd with wrong row count ----------------
    print("\n--- (d) ('psd', n) with rows != n(n+1)/2 ---")
    # sanity: n=2 -> 3 rows of svec. min t s.t. [[t,1],[1,1]] >= 0 -> t >= 1
    # svec lower triangle column-major: (X00, sqrt2*X10, X11)
    # s = h - G z, z = (t,)
    Gpsd = np.array([[-1.0], [0.0], [0.0]])
    hpsd = np.array([0.0, math.sqrt(2.0) * 1.0, 1.0])
    expect_value("d0", "sanity ('psd',2): min t, [[t,1],[1,1]]>=0",
                 lambda: solve_socp(c=[1.0], G=Gpsd, h=hpsd, cones=[("psd", 2)]),
                 1.0, tol=1e-5)
    expect_error("d1", "('psd',2) with 4 rows",
                 lambda: solve_socp(c=[1.0], G=np.vstack([Gpsd, [[0.0]]]),
                                    h=np.concatenate([hpsd, [1.0]]),
                                    cones=[("psd", 2)]))
    expect_error("d2", "('psd',3) with 3 rows (needs 6)",
                 lambda: solve_socp(c=[1.0], G=Gpsd, h=hpsd, cones=[("psd", 3)]))
    expect_error("d3", "('psd',0)",
                 lambda: solve_socp(c=[1.0], G=Gpsd, h=hpsd,
                                    cones=[("psd", 0), ("psd", 2)]))
    expect_error("d4", "('psd',-2)",
                 lambda: solve_socp(c=[1.0], G=Gpsd, h=hpsd, cones=[("psd", -2)]))

    # ---------------- (e) empty cone list with nonempty G ----------------
    print("\n--- (e) empty cone list with nonempty G ---")
    expect_error("e1", "cones=[] with 5-row G",
                 lambda: solve_socp(c=C_G, G=G_base, h=h_base, cones=[]))

    # ---------------- (f) unknown cone type ----------------
    print("\n--- (f) unknown cone type ---")
    expect_error("f1", "('banana', 3)",
                 lambda: solve_socp(c=C_G, G=G_base, h=h_base,
                                    cones=[("banana", 3), ("nonneg", 2)]))
    # parse_cones() lowercases the kind, so 'SOC' is accepted BY DESIGN
    # (documented in crates/pounce-py/src/qp.rs). Expect the right answer.
    expect_value("f2", "('SOC',3) case-insensitive by design",
                 lambda: solve_socp(c=C_G, G=G_base, h=h_base,
                                    cones=[("SOC", 3), ("nonneg", 2)]),
                 KNOWN_OPT_G)
    expect_error("f3", "malformed tuple ('soc',)",
                 lambda: solve_socp(c=C_G, G=G_base, h=h_base,
                                    cones=[("soc",), ("nonneg", 2)]))
    expect_error("f4", "3-tuple ('soc',3,9)",
                 lambda: solve_socp(c=C_G, G=G_base, h=h_base,
                                    cones=[("soc", 3, 9), ("nonneg", 2)]))
    expect_error("f5", "None in cone list",
                 lambda: solve_socp(c=C_G, G=G_base, h=h_base,
                                    cones=[None, ("nonneg", 2)]))

    # ---------------- (g) cone ORDER invariance ----------------
    print("\n--- (g) cone order invariance (known optimum t*=1, x*=(0,2)) ---")
    rg1 = expect_value("g1", "order [soc3, nonneg2]",
                       lambda: solve_socp(c=C_G, G=G_base, h=h_base, cones=CONES_BASE),
                       KNOWN_OPT_G)
    rg2 = expect_value("g2", "order [nonneg2, soc3] (rows permuted)",
                       lambda: solve_socp(c=C_G, G=G_perm, h=h_perm, cones=CONES_PERM),
                       KNOWN_OPT_G)
    # split the nonneg block into two 1-D blocks straddling the soc block
    G_split = np.vstack([G_nn[:1], G_soc, G_nn[1:]])
    h_split = np.concatenate([h_nn[:1], h_soc, h_nn[1:]])
    rg3 = expect_value("g3", "order [nonneg1, soc3, nonneg1]",
                       lambda: solve_socp(c=C_G, G=G_split, h=h_split,
                                          cones=[("nonneg", 1), ("soc", 3), ("nonneg", 1)]),
                       KNOWN_OPT_G)
    xs = [np.asarray(r.x) for r in (rg1, rg2, rg3) if r is not None]
    if len(xs) == 3:
        d12 = float(np.max(np.abs(xs[0] - xs[1])))
        d13 = float(np.max(np.abs(xs[0] - xs[2])))
        print(f"     x order-invariance: |g1-g2|_inf={d12:.3e}  |g1-g3|_inf={d13:.3e}")
        ok = max(d12, d13) < 1e-6
        RESULTS.append(("g*", "order invariance of x", "PASS" if ok else "FAIL",
                        f"|g1-g2|={d12:.3e} |g1-g3|={d13:.3e}"))

    # ---------------- (h) many tiny cones vs one big block ----------------
    #   min sum_i t_i   s.t.  t_i >= |x_i|,  sum_i x_i = 10,   n = 200
    #   Optimum: sum t_i = 10 (L1 norm of any x with sum 10 is >= 10).
    #   Form A: 200 x ("soc", 2) blocks (t_i, x_i)
    #   Form B: one ("nonneg", 400) block  t_i - x_i >= 0, t_i + x_i >= 0
    print("\n--- (h) 200 x ('soc',2) vs one 400-row nonneg block (known opt = 10) ---")
    n = 200
    nv = 2 * n                      # z = (t_0..t_{n-1}, x_0..x_{n-1})
    c_h = np.concatenate([np.ones(n), np.zeros(n)])
    A_h = np.zeros((1, nv))
    A_h[0, n:] = 1.0
    b_h = np.array([10.0])

    # Form A: rows 2i = t_i, 2i+1 = x_i  (s = h - Gz, h = 0)
    G_a = np.zeros((2 * n, nv))
    for i in range(n):
        G_a[2 * i, i] = -1.0
        G_a[2 * i + 1, n + i] = -1.0
    h_a = np.zeros(2 * n)
    cones_a = [("soc", 2)] * n

    # Form B: rows 2i = t_i - x_i, 2i+1 = t_i + x_i
    G_b = np.zeros((2 * n, nv))
    for i in range(n):
        G_b[2 * i, i] = -1.0
        G_b[2 * i, n + i] = 1.0
        G_b[2 * i + 1, i] = -1.0
        G_b[2 * i + 1, n + i] = -1.0
    h_b = np.zeros(2 * n)
    cones_b = [("nonneg", 2 * n)]

    ra = expect_value("h1", f"{n} x ('soc',2)",
                      lambda: solve_socp(c=c_h, A=A_h, b=b_h, G=G_a, h=h_a, cones=cones_a),
                      10.0, tol=1e-6)
    rb = expect_value("h2", "one ('nonneg',400)",
                      lambda: solve_socp(c=c_h, A=A_h, b=b_h, G=G_b, h=h_b, cones=cones_b),
                      10.0, tol=1e-6)
    if ra is not None and rb is not None:
        print(f"     obj_soc={ra.obj:.12e}  obj_nonneg={rb.obj:.12e}  "
              f"diff={abs(ra.obj - rb.obj):.3e}")

    # cvxpy oracle for (g) and (h)
    print("\n--- cvxpy oracle ---")
    try:
        import cvxpy as cp
        xv = cp.Variable(2)
        tv = cp.Variable()
        pr = cp.Problem(cp.Minimize(tv),
                        [cp.norm(xv - np.array([-1.0, 2.0]), 2) <= tv, xv >= 0])
        pr.solve(solver=cp.CLARABEL)
        print(f"     (g) cvxpy obj={pr.value:.12e} x={xv.value}")
        RESULTS.append(("g-oracle", "cvxpy (g)", "INFO", f"obj={pr.value:.10e}"))

        xh = cp.Variable(n)
        pr2 = cp.Problem(cp.Minimize(cp.norm1(xh)), [cp.sum(xh) == 10])
        pr2.solve(solver=cp.CLARABEL)
        print(f"     (h) cvxpy obj={pr2.value:.12e}")
        RESULTS.append(("h-oracle", "cvxpy (h)", "INFO", f"obj={pr2.value:.10e}"))
    except Exception as exc:  # noqa: BLE001
        print(f"     cvxpy unavailable/failed: {exc}")

    # ---------------- summary ----------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    bad = []
    for cid, label, outcome, detail in RESULTS:
        print(f"{outcome:9s} [{cid:9s}] {label}: {detail}")
        if outcome in ("BUG", "FAIL", "NO_ERROR", "SUSPECT", "ERROR"):
            bad.append((cid, label, outcome, detail))

    print()
    if not bad:
        print("VERDICT: PASS")
    else:
        print(f"VERDICT: REVIEW ({len(bad)} non-clean outcomes)")
        for cid, label, outcome, detail in bad:
            print(f"  !! [{cid}] {outcome}: {label} -> {detail}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print("VERDICT: HARNESS_ERROR")
