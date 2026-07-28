#!/usr/bin/env python
"""SOS adversary test: six-hump camel function (unconstrained polynomial).

f(x,y) = (4 - 2.1 x^2 + x^4/3) x^2 + x y + (-4 + 4 y^2) y^2
       = 4 x^2 - 2.1 x^4 + (1/3) x^6 + x y - 4 y^2 + 4 y^4

Degree 6 (x^6 term). KNOWN global minimum f* = -1.0316284534898774
attained at two symmetric points
  (x,y) = (+0.0898420131, -0.7126564030)
  (x,y) = (-0.0898420131, +0.7126564030)

SOURCE: Molga & Smutnicki, "Test functions for optimization needs" (2005);
also Dixon-Szego standard global-optimization test set.

The leading terms x^6/3 and 4 y^4 make f coercive (-> +inf as |x|,|y| -> inf),
so the function is bounded below and unconstrained SOS minimization applies.

pounce returns a certified LOWER BOUND. A valid lower bound must satisfy
  lower_bound <= f* (the true global minimum).
A bound that EXCEEDS f* is a SOLVER_BUG (invalid lower bound).
"""
import time
import numpy as np
from scipy.optimize import minimize
import pounce

KNOWN_MIN = -1.0316284534898774
KNOWN_ARGMINS = [
    (0.0898420131, -0.7126564030),
    (-0.0898420131, 0.7126564030),
]

# Exponent-tuple (i,j) -> coefficient for x^i y^j
OBJ = {
    (2, 0): 4.0,
    (4, 0): -2.1,
    (6, 0): 1.0 / 3.0,
    (1, 1): 1.0,
    (0, 2): -4.0,
    (0, 4): 4.0,
}


def f(x, y):
    return (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2


def main():
    # sanity: known argmins reproduce known min
    for (x, y) in KNOWN_ARGMINS:
        assert abs(f(x, y) - KNOWN_MIN) < 1e-6, (x, y, f(x, y))

    print("=== SOS six-hump camel ===")
    print(f"KNOWN global min f* = {KNOWN_MIN:.10f}")

    results = {}
    for order in (0, 1, 2):
        t0 = time.time()
        r = pounce.sos_minimize(OBJ, n_vars=2, order=order)
        dt = time.time() - t0
        results[order] = (r, dt)
        print(
            f"order={order:>2}  lower_bound={r.lower_bound:+.8f}  "
            f"is_exact={r.is_exact}  status={r.status}  "
            f"num_min={r.num_minimizers}  time={dt:.3f}s"
        )

    # pick the tightest (highest) lower bound across orders for verdict
    best_order = max(results, key=lambda o: results[o][0].lower_bound)
    r, dt = results[best_order]
    lb = r.lower_bound
    print(f"\nBest (tightest) lower bound: {lb:+.8f} at order={best_order}")
    gap = KNOWN_MIN - lb
    print(f"Gap (f* - lower_bound) = {gap:+.3e}  (should be >= 0 for valid bound)")

    # -------- ORACLE 1: dense grid refutation --------
    # search box generously containing the known minimizers
    gx = np.linspace(-3.0, 3.0, 1201)
    gy = np.linspace(-2.0, 2.0, 801)
    GX, GY = np.meshgrid(gx, gy)
    FV = f(GX, GY)
    grid_min = FV.min()
    idx = np.unravel_index(np.argmin(FV), FV.shape)
    print(f"\nGrid min over [-3,3]x[-2,2] = {grid_min:.8f} "
          f"at (x,y)=({GX[idx]:.4f},{GY[idx]:.4f})")

    # -------- ORACLE 2: scipy multistart refutation --------
    rng = np.random.default_rng(0)
    best_ms = np.inf
    best_pt = None
    for _ in range(400):
        x0 = rng.uniform([-3, -2], [3, 2])
        res = minimize(lambda p: f(p[0], p[1]), x0, method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 5000})
        if res.fun < best_ms:
            best_ms = res.fun
            best_pt = res.x
    print(f"Multistart best = {best_ms:.10f} at "
          f"(x,y)=({best_pt[0]:.6f},{best_pt[1]:.6f})")

    refutation_min = min(grid_min, best_ms, KNOWN_MIN)
    print(f"\nRefutation min (best point found / known) = {refutation_min:.10f}")

    # If recovered minimizers exist, check their objective
    if r.is_exact and r.minimizers:
        for m in r.minimizers:
            fm = f(m[0], m[1])
            print(f"recovered minimizer {np.round(m,6)} -> f={fm:.8f}")

    # -------- VERDICT --------
    # Valid lower bound: lb must not exceed any point we can find.
    bound_invalid = lb > refutation_min + 1e-6
    tol = 1e-4
    matches = abs(lb - KNOWN_MIN) <= tol

    if bound_invalid:
        verdict = "FAIL"  # SOLVER_BUG: bound exceeds true global min
        print("\nBOUND IS INVALID: lower_bound exceeds best known point!")
    elif matches:
        verdict = "PASS"
    else:
        # valid but loose -> still PASS for correctness; note looseness
        print(f"\nNOTE: valid bound but gap {gap:.3e} > tol {tol} "
              f"(loose relaxation, not a correctness bug)")
        verdict = "PASS" if gap >= -1e-6 else "FAIL"

    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
