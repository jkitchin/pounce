#!/usr/bin/env python
"""SOS adversary test: Goldstein-Price function (unconstrained polynomial).

f(x,y) = [1 + (x+y+1)^2 (19 - 14x + 3x^2 - 14y + 6xy + 3y^2)]
       * [30 + (2x-3y)^2 (18 - 32x + 12x^2 + 48y - 36xy + 27y^2)]

Degree 8 polynomial in (x,y). KNOWN global minimum f* = 3.0 at (x,y)=(0,-1).

SOURCE: Goldstein & Price (1971); Dixon-Szego / Molga-Smutnicki standard
global-optimization test set.

Goldstein-Price is NOT globally coercive in the usual unconstrained sense the
way a sum of even powers is, but its leading degree-8 form is positive
definite, so it IS bounded below and -> +inf along every ray; unconstrained
SOS minimization is valid. We expand it with sympy to obtain the exact
exponent->coefficient dictionary that pounce consumes.

pounce returns a certified LOWER BOUND. Valid requires lower_bound <= f*.
A bound exceeding f* is a SOLVER_BUG.
"""
import time
import numpy as np
import sympy as sp
from scipy.optimize import minimize
import pounce

KNOWN_MIN = 3.0
KNOWN_ARGMIN = (0.0, -1.0)


def build_obj():
    x, y = sp.symbols("x y", real=True)
    a = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    b = 30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    expr = sp.expand(a * b)
    poly = sp.Poly(expr, x, y)
    obj = {}
    for monom, coeff in poly.terms():
        obj[tuple(int(e) for e in monom)] = float(coeff)
    return expr, obj, (x, y)


def main():
    expr, OBJ, (sx, sy) = build_obj()
    fnum = sp.lambdify((sx, sy), expr, "numpy")

    def f(x, y):
        return fnum(x, y)

    # sanity: known argmin reproduces known min
    assert abs(f(*KNOWN_ARGMIN) - KNOWN_MIN) < 1e-9, f(*KNOWN_ARGMIN)
    deg = sp.Poly(expr, sx, sy).total_degree()
    print("=== SOS Goldstein-Price ===")
    print(f"polynomial total degree = {deg}")
    print(f"KNOWN global min f* = {KNOWN_MIN:.10f} at {KNOWN_ARGMIN}")

    results = {}
    # degree 8 -> minimum relaxation order is 4 (half-degree). order kwarg is the
    # number of extra levels above the minimum, so try 0 and 1.
    for order in (0, 1):
        t0 = time.time()
        r = pounce.sos_minimize(OBJ, n_vars=2, order=order)
        dt = time.time() - t0
        results[order] = (r, dt)
        print(
            f"order={order:>2}  lower_bound={r.lower_bound:+.8f}  "
            f"is_exact={r.is_exact}  status={r.status}  "
            f"num_min={r.num_minimizers}  time={dt:.3f}s"
        )

    best_order = max(results, key=lambda o: results[o][0].lower_bound)
    r, dt = results[best_order]
    lb = r.lower_bound
    print(f"\nBest (tightest) lower bound: {lb:+.8f} at order={best_order}")
    gap = KNOWN_MIN - lb
    print(f"Gap (f* - lower_bound) = {gap:+.3e}  (should be >= 0 for valid bound)")

    # -------- ORACLE 1: dense grid refutation --------
    gx = np.linspace(-2.0, 2.0, 1601)
    gy = np.linspace(-2.0, 2.0, 1601)
    GX, GY = np.meshgrid(gx, gy)
    FV = f(GX, GY)
    grid_min = float(FV.min())
    idx = np.unravel_index(np.argmin(FV), FV.shape)
    print(f"\nGrid min over [-2,2]^2 = {grid_min:.8f} "
          f"at (x,y)=({GX[idx]:.4f},{GY[idx]:.4f})")

    # -------- ORACLE 2: scipy multistart refutation --------
    rng = np.random.default_rng(0)
    best_ms = np.inf
    best_pt = None
    for _ in range(400):
        x0 = rng.uniform([-2, -2], [2, 2])
        res = minimize(lambda p: float(f(p[0], p[1])), x0, method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 5000})
        if res.fun < best_ms:
            best_ms = float(res.fun)
            best_pt = res.x
    print(f"Multistart best = {best_ms:.10f} at "
          f"(x,y)=({best_pt[0]:.6f},{best_pt[1]:.6f})")

    refutation_min = min(grid_min, best_ms, KNOWN_MIN)
    print(f"\nRefutation min (best point found / known) = {refutation_min:.10f}")

    if r.is_exact and r.minimizers:
        for m in r.minimizers:
            fm = float(f(m[0], m[1]))
            print(f"recovered minimizer {np.round(m,6)} -> f={fm:.8f}")

    # -------- VERDICT --------
    tol = 1e-3
    if not np.isfinite(lb):
        # SDP failed to converge (iteration_limit -> NaN). A NaN/absent bound
        # does NOT exceed the true global minimum, so this is NOT an invalid
        # bound / SOLVER_BUG. It is a SOLVER_LIMITATION (poorly-scaled SDP:
        # |coeffs| up to ~2.4e4). Classify accordingly.
        print("\nNo finite lower bound returned (status=iteration_limit).")
        print("This is a SOLVER_LIMITATION (no certificate), not an invalid "
              "bound. No bound exceeds f*, so NOT a correctness bug.")
        verdict = "FAIL"  # script-level FAIL: pounce produced no usable bound
    else:
        bound_invalid = lb > refutation_min + 1e-4
        matches = abs(lb - KNOWN_MIN) <= tol
        if bound_invalid:
            verdict = "FAIL"  # SOLVER_BUG: bound exceeds true global min
            print("\nBOUND IS INVALID: lower_bound exceeds best known point!")
        elif matches:
            verdict = "PASS"
        else:
            print(f"\nNOTE: valid bound but gap {gap:.3e} > tol {tol} "
                  f"(loose relaxation, not a correctness bug)")
            verdict = "PASS" if gap >= -1e-4 else "FAIL"

    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
