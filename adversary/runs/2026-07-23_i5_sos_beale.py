"""SOS adversary test: Beale function (degree-6 sum-of-squares polynomial).
Family: sos   Class: global polynomial minimization (Lasserre lower bound)

f(x,y) = (1.5 - x + x y)^2 + (2.25 - x + x y^2)^2 + (2.625 - x + x y^3)^2

A polynomial of total degree 8 (the (x y^3)^2 term) that is manifestly a SUM
OF THREE SQUARES,
hence globally >= 0, with KNOWN global minimum f* = 0 attained at (x,y)=(3, 0.5)
(each squared term vanishes there). Because f is an exact SOS, the Lasserre
lower bound should be exactly 0 (tight at the minimum relaxation order).
SOURCE: Beale, E.M.L. (1958); standard global-optimization benchmark
(Molga & Smutnicki 2005, "Test functions for optimization needs").

pounce.sos_minimize returns a certified LOWER BOUND: valid iff lower_bound<=f*=0.
A bound EXCEEDING 0 is a SOLVER_BUG. Refuted with dense grid + multistart.

Distinct from logged sos tests (camel family, Goldstein-Price, Motzkin,
Himmelblau, Styblinski-Tang, quartic): a rational-root SOS with a UNIQUE minimizer
at f*=0 and non-homogeneous degree-6 structure.
"""
import time
import numpy as np
from scipy.optimize import minimize
import pounce

KNOWN_MIN = 0.0
XSTAR = (3.0, 0.5)


def f(x, y):
    return ((1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2
            + (2.625 - x + x * y ** 3) ** 2)


# Expand to a monomial dict {(i,j): coeff}.  Build symbolically with sympy.
import sympy as sp
xs, ys = sp.symbols("x y")
expr = sp.expand((sp.Rational(3, 2) - xs + xs * ys) ** 2
                 + (sp.Rational(9, 4) - xs + xs * ys ** 2) ** 2
                 + (sp.Rational(21, 8) - xs + xs * ys ** 3) ** 2)
poly = sp.Poly(expr, xs, ys)
OBJ = {tuple(int(e) for e in mono): float(cf) for mono, cf in poly.terms()}


def main():
    # sanity: dict matches f, and f(x*)=0
    for _ in range(5):
        x, y = np.random.uniform(-1, 4, 2)
        v = sum(cf * x ** i * y ** j for (i, j), cf in OBJ.items())
        assert abs(v - f(x, y)) < 1e-8, (x, y, v, f(x, y))
    assert abs(f(*XSTAR) - KNOWN_MIN) < 1e-12

    print("=== SOS Beale ===")
    print(f"KNOWN global min f* = {KNOWN_MIN} at {XSTAR}; poly degree = {poly.total_degree()}")

    results = {}
    for order in (0, 1, 2, 3):
        t0 = time.time()
        r = pounce.sos_minimize(OBJ, n_vars=2, order=order)
        dt = time.time() - t0
        results[order] = (r, dt)
        print(f"order={order:>2} lower_bound={r.lower_bound:+.8e} is_exact={r.is_exact} "
              f"status={r.status} num_min={r.num_minimizers} time={dt:.3f}s")

    best_order = max(results, key=lambda o: results[o][0].lower_bound)
    r, dt = results[best_order]
    lb = r.lower_bound
    gap = KNOWN_MIN - lb
    print(f"\nBest lower bound = {lb:+.8e} at order={best_order}; gap (f*-lb) = {gap:+.3e}")

    # refutation: grid + multistart
    gx = np.linspace(0, 4.5, 900)
    gy = np.linspace(-1, 1.5, 700)
    GX, GY = np.meshgrid(gx, gy)
    grid_min = float(f(GX, GY).min())
    rng = np.random.default_rng(0)
    best_ms = np.inf; best_pt = None
    for _ in range(200):
        x0 = rng.uniform([-1, -1], [5, 2])
        res = minimize(lambda p: f(p[0], p[1]), x0, method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 5000})
        if res.fun < best_ms:
            best_ms = res.fun; best_pt = res.x
    print(f"grid min = {grid_min:.6e}; multistart best = {best_ms:.3e} at "
          f"({best_pt[0]:.5f},{best_pt[1]:.5f})")

    refute_min = min(grid_min, best_ms, KNOWN_MIN)
    if r.is_exact and r.minimizers:
        for m in r.minimizers:
            print(f"recovered minimizer {np.round(m,6)} -> f={f(m[0], m[1]):.3e}")

    bound_invalid = lb > refute_min + 1e-6
    matches = abs(lb - KNOWN_MIN) <= 1e-4
    if bound_invalid:
        print("\nBOUND INVALID: lower_bound exceeds best known point (SOLVER_BUG)")
        verdict = "FAIL"
    elif matches:
        verdict = "PASS"
    else:
        print(f"\nNOTE valid but loose bound, gap={gap:.3e}")
        verdict = "PASS" if gap >= -1e-6 else "FAIL"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
