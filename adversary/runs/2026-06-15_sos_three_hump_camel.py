#!/usr/bin/env python
"""SOS adversary test: three-hump camel function (unconstrained polynomial).

f(x,y) = 2 x^2 - 1.05 x^4 + x^6/6 + x y + y^2

Degree 6 (x^6 term). KNOWN global minimum f* = 0 attained at (x,y) = (0,0).

SOURCE: Molga & Smutnicki, "Test functions for optimization needs" (2005);
also the standard global-optimization test-function collection
(e.g. Jamil & Yang, "A literature survey of benchmark functions for global
optimization problems", 2013, function "Three-Hump Camel").
Known global minimum f(0,0) = 0.

The leading terms x^6/6 and y^2 make f coercive (-> +inf as |x|,|y| -> inf),
so the function is bounded below and unconstrained SOS minimization applies.

pounce returns a certified LOWER BOUND on the global minimum. A valid lower
bound must satisfy  lower_bound <= f* (the true global minimum). A bound that
EXCEEDS f* is a SOLVER_BUG (invalid lower bound). A loose (too-low) bound at
small relaxation order is EXPECTED, not a bug.
"""
import time
import numpy as np
from scipy.optimize import minimize
import pounce

KNOWN_MIN = 0.0
KNOWN_ARGMIN = (0.0, 0.0)

# Exponent-tuple (i,j) -> coefficient for x^i y^j
OBJ = {
    (2, 0): 2.0,
    (4, 0): -1.05,
    (6, 0): 1.0 / 6.0,
    (1, 1): 1.0,
    (0, 2): 1.0,
}


def f(x, y):
    return 2.0 * x**2 - 1.05 * x**4 + x**6 / 6.0 + x * y + y**2


def main():
    # sanity: known argmin reproduces known min
    assert abs(f(*KNOWN_ARGMIN) - KNOWN_MIN) < 1e-12, f(*KNOWN_ARGMIN)

    print("=== SOS three-hump camel ===")
    print(f"KNOWN global min f* = {KNOWN_MIN:.10f} at {KNOWN_ARGMIN}")

    results = {}
    for order in (0, 1, 2, 3):
        t0 = time.time()
        r = pounce.sos_minimize(OBJ, n_vars=2, order=order)
        dt = time.time() - t0
        results[order] = (r, dt)
        print(
            f"order={order:>2}  lower_bound={r.lower_bound:+.8f}  "
            f"is_exact={r.is_exact}  status={r.status}  "
            f"num_min={r.num_minimizers}  success={r.success}  time={dt:.3f}s"
        )

    # pick the tightest (highest) lower bound across orders for verdict
    best_order = max(results, key=lambda o: results[o][0].lower_bound)
    r, dt = results[best_order]
    lb = r.lower_bound
    t_pounce = dt
    print(f"\nBest (tightest) lower bound: {lb:+.8f} at order={best_order}")
    gap = KNOWN_MIN - lb
    print(f"Gap (f* - lower_bound) = {gap:+.3e}  (should be >= 0 for valid bound)")

    # -------- ORACLE 1: dense grid refutation --------
    t_or0 = time.time()
    gx = np.linspace(-3.0, 3.0, 1601)
    gy = np.linspace(-3.0, 3.0, 1601)
    GX, GY = np.meshgrid(gx, gy)
    FV = f(GX, GY)
    grid_min = float(FV.min())
    idx = np.unravel_index(np.argmin(FV), FV.shape)
    print(f"\nGrid min over [-3,3]^2 = {grid_min:.8f} "
          f"at (x,y)=({GX[idx]:.4f},{GY[idx]:.4f})")

    # -------- ORACLE 2: scipy multistart refutation --------
    rng = np.random.default_rng(0)
    best_ms = np.inf
    best_pt = None
    for _ in range(400):
        x0 = rng.uniform([-3, -3], [3, 3])
        res = minimize(lambda p: f(p[0], p[1]), x0, method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 5000})
        if res.fun < best_ms:
            best_ms = float(res.fun)
            best_pt = res.x
    t_oracle = time.time() - t_or0
    print(f"Multistart best = {best_ms:.10f} at "
          f"(x,y)=({best_pt[0]:.6f},{best_pt[1]:.6f})")

    refutation_min = min(grid_min, best_ms, KNOWN_MIN)
    print(f"\nRefutation min (best point found / known) = {refutation_min:.10f}")

    # -------- VERDICT --------
    # Valid lower bound: lb must not exceed any point we can find.
    bound_invalid = lb > refutation_min + 1e-6
    tol = 1e-4
    matches = abs(lb - KNOWN_MIN) <= tol

    print(f"\n[timing] pounce={t_pounce:.4f}s oracle={t_oracle:.4f}s")

    if bound_invalid:
        print("\nBOUND IS INVALID: lower_bound exceeds best known point!")
        print(f"VERDICT: FAIL (invalid lower bound {lb:+.8f} exceeds "
              f"true global min {refutation_min:+.8f})")
    elif matches:
        print(f"VERDICT: PASS (lower_bound {lb:+.8f} matches known global min "
              f"{KNOWN_MIN:.8f} to tol {tol}, order={best_order})")
    else:
        if gap >= -1e-6:
            print(f"\nNOTE: valid bound but gap {gap:.3e} > tol {tol} "
                  f"(loose relaxation, not a correctness bug)")
            print(f"VERDICT: PASS (valid lower bound {lb:+.8f} <= true min, "
                  f"loose at order={best_order})")
        else:
            print(f"VERDICT: FAIL (bound {lb:+.8f} below refutation but "
                  f"inconsistent gap)")


if __name__ == "__main__":
    main()
