"""Adversary cross-check: polynomials with a CONTINUUM of global minimizers.

Family: sos   Class: degenerate / non-unique optima (infinite minimizer set)

Primary instance
    f1(x,y) = (x^2 + y^2 - 1)^2
    Global minimum 0, attained on the ENTIRE unit circle x^2+y^2 = 1.
    f1 is a sum of squares (it is literally one square), so the order-2
    Lasserre relaxation is exact and the bound must be 0.
    The moment matrix at the optimum has rank = infinity in the limit (the
    optimal measure can be any probability measure supported on the circle),
    so the flat-extension rank test cannot terminate with a finite atomic
    decomposition. Reporting is_exact=False / 0 minimizers is CORRECT here.

Variants (same continuum structure, different stressors)
    f2(x,y) = (x^2 + y^2 - 1)^2 + 3       -> min 3 on the unit circle
                                             (constant shift; catches a bound
                                              that ignores the constant term)
    f3(x,y) = (x*y - 1)^2                 -> min 0 on the NON-COMPACT branch
                                             hyperbola xy = 1 (unbounded
                                             minimizer set; no compact
                                             Archimedean structure)

Source: standard degenerate SOS test cases. (x^2+y^2-1)^2 appears as the
canonical "positive dimensional real variety" example in Nie, Demmel &
Sturmfels, "Minimizing polynomials via sum of squares over the gradient
ideal", Math. Program. 106 (2006) 587-606, Sec. 5; and in Henrion & Lasserre,
GloptiPoly documentation, as a case where the rank/flat-extension test does
not certify finite minimizers.

CONTRACT UNDER TEST (per adversary spec):
  * PASS iff the returned LOWER BOUND is <= true global min (valid) and
    tight to tolerance at the tested order.
  * lower_bound > true_min + tol  ==>  SOLVER_BUG (invalid lower bound).
  * is_exact=False / 0 minimizers on a continuum  ==>  EXPECTED, not a bug.
  * is_exact=True with a returned point that does NOT attain the bound
    ==>  extraction bug (already known for Himmelblau; recorded, not re-filed).
"""

import time

import numpy as np
from scipy.optimize import minimize as spmin

# ---------------------------------------------------------------------------
# problems: (name, poly-dict, callable, known global min, minimizer-set descr)
# ---------------------------------------------------------------------------

# (x^2+y^2-1)^2 = x^4 + 2x^2y^2 + y^4 - 2x^2 - 2y^2 + 1
P1 = {(4, 0): 1.0, (2, 2): 2.0, (0, 4): 1.0, (2, 0): -2.0, (0, 2): -2.0, (0, 0): 1.0}
P2 = dict(P1)
P2[(0, 0)] = 1.0 + 3.0
# (xy-1)^2 = x^2y^2 - 2xy + 1
P3 = {(2, 2): 1.0, (1, 1): -2.0, (0, 0): 1.0}


def f1(x, y):
    return (x * x + y * y - 1.0) ** 2


def f2(x, y):
    return f1(x, y) + 3.0


def f3(x, y):
    return (x * y - 1.0) ** 2


CASES = [
    ("circle", P1, f1, 0.0, "unit circle x^2+y^2=1 (compact continuum)"),
    ("circle_shift3", P2, f2, 3.0, "unit circle, objective shifted by +3"),
    ("hyperbola", P3, f3, 0.0, "xy=1 (non-compact continuum)"),
]


def poly_eval(poly, x, y):
    return sum(c * x**a * y**b for (a, b), c in poly.items())


# ---------------------------------------------------------------------------
# 0. formulation sanity: dict expansion == closed form
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
XY = rng.uniform(-3, 3, size=(3000, 2))
for name, poly, fn, _, _ in CASES:
    err = float(np.max(np.abs(poly_eval(poly, XY[:, 0], XY[:, 1]) - fn(XY[:, 0], XY[:, 1]))))
    print(f"expansion_check[{name}]_max_abs_err={err:.3e}")
    assert err < 1e-9, f"expansion wrong for {name} -> FORMULATION_ERROR"

# ---------------------------------------------------------------------------
# 1. oracles: dense grid + scipy multistart (refutation search)
# ---------------------------------------------------------------------------
oracle = {}
g = np.linspace(-4.0, 4.0, 1601)
GX, GY = np.meshgrid(g, g)
t0 = time.perf_counter()
for name, poly, fn, known, _ in CASES:
    grid_min = float(np.min(fn(GX, GY)))
    best, best_x = np.inf, None
    for s in rng.uniform(-4, 4, size=(200, 2)):
        r = spmin(lambda z: fn(z[0], z[1]), s, method="BFGS", tol=1e-14)
        if r.fun < best:
            best, best_x = float(r.fun), r.x
    oracle[name] = (grid_min, best, best_x)
    print(
        f"=== oracle[{name}] === grid_min={grid_min:.10e} multistart_min={best:.10e} "
        f"at {np.array2string(best_x, precision=6)} known={known:.10e}"
    )
t_oracle = time.perf_counter() - t0
print(f"oracle_total_time={t_oracle:.4f}s")

# ---------------------------------------------------------------------------
# 2. pounce
# ---------------------------------------------------------------------------
from pounce.sos import sos_minimize  # noqa: E402

TOLB = 1e-5  # slack allowed above the true min before calling the bound invalid
rows = []
for name, poly, fn, known, setdesc in CASES:
    grid_min, ms_min, _ = oracle[name]
    true_min = min(known, grid_min, ms_min)
    for order in (2, 3):
        t0 = time.perf_counter()
        try:
            r = sos_minimize(poly, order=order)
            dt = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            dt = time.perf_counter() - t0
            print(f"=== pounce[{name}] order={order} === EXCEPTION {type(exc).__name__}: {exc}")
            rows.append((name, order, "exception", np.nan, None, 0, dt, "EXCEPTION"))
            continue

        lb = float(r.lower_bound) if r.lower_bound is not None else float("nan")
        print(f"=== pounce[{name}] order={order} ===")
        print(
            f"status={r.status} lower_bound={lb:.10e} is_exact={r.is_exact} "
            f"num_minimizers={r.num_minimizers} t={dt:.4f}s"
        )
        mobjs = []
        for m in r.minimizers:
            fm = float(fn(m[0], m[1]))
            mobjs.append(fm)
            print(
                f"   minimizer={np.array2string(np.asarray(m), precision=6)} "
                f"f={fm:.6e}  |bound-f|={abs(fm - lb):.3e}"
            )

        # classify this (case, order)
        if r.status != "optimal" or not np.isfinite(lb):
            cls = "NONOPTIMAL"
        elif lb > true_min + TOLB:
            cls = "INVALID_BOUND"
        else:
            tight = abs(lb - true_min) < 1e-5
            if r.is_exact and r.num_minimizers > 0:
                bad = [o for o in mobjs if abs(o - true_min) > 1e-4]
                cls = ("EXACT_OK" if (tight and not bad) else "EXACT_MISMATCH")
            else:
                # continuum: not-exact is the CORRECT report
                cls = "VALID_TIGHT_NOTEXACT" if tight else "VALID_LOOSE"
        rows.append((name, order, r.status, lb, r.is_exact, r.num_minimizers, dt, cls))
        print(f"   true_min={true_min:.10e}  class={cls}")

print()
print("=== summary ===")
print("| case | order | status | lower_bound | is_exact | n_min | t(s) | class |")
for name, order, st, lb, ex, nm, dt, cls in rows:
    print(f"| {name} | {order} | {st} | {lb:.6e} | {ex} | {nm} | {dt:.4f} | {cls} |")

classes = [c for *_, c in rows]
if "INVALID_BOUND" in classes:
    print("VERDICT: FAIL (INVALID LOWER BOUND -> SOLVER_BUG)")
elif "EXACT_MISMATCH" in classes:
    print("VERDICT: FAIL (minimizer extraction: is_exact with non-attaining point)")
elif "NONOPTIMAL" in classes or "EXCEPTION" in classes:
    print(f"VERDICT: FAIL ({[c for c in classes if c in ('NONOPTIMAL', 'EXCEPTION')]})")
elif all(c in ("VALID_TIGHT_NOTEXACT", "EXACT_OK") for c in classes):
    print("VERDICT: PASS")
else:
    print(f"VERDICT: INCONCLUSIVE ({classes})")
