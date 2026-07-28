"""SOS adversary test: 2-D Styblinski-Tang function (coercive quartic).
Family: sos   Class: global polynomial minimization (Lasserre lower bound)

f(x,y) = 0.5*(x^4 - 16 x^2 + 5 x) + 0.5*(y^4 - 16 y^2 + 5 y)

Separable degree-4 polynomial, coercive (x^4, y^4 -> +inf), so bounded below.
KNOWN global minimum:  each 1-D term g(t)=0.5(t^4-16t^2+5t) has its global min
at t* = -2.903534, g(t*) = -39.166166... so
  f* = 2 * (-39.16616570...) = -78.33233141   at (x,y)=(-2.903534, -2.903534).
SOURCE: Styblinski, M.A.; Tang, T.-S., "Experiments in nonconvex optimization:
stochastic approximation with function smoothing and simulated annealing,"
Neural Networks 3(4):467-483 (1990). Standard global-optimization benchmark.

pounce.sos_minimize returns a certified LOWER BOUND. A valid bound satisfies
lower_bound <= f*. A bound EXCEEDING the true global min is a SOLVER_BUG.
Refuted with a dense grid + scipy multistart.

Distinct from logged sos entries (six-hump/three-hump camel, Goldstein-Price,
Motzkin, Himmelblau, quartic, Lasserre-constrained, circle-continuum): this is
a separable double-well quartic with a NEGATIVE, non-trivial global minimum.
"""
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize
import pounce

# exact 1-D minimizer/min via high-precision scalar solve
_g = lambda t: 0.5 * (t ** 4 - 16 * t ** 2 + 5 * t)
_r = minimize_scalar(_g, bounds=(-5, 0), method="bounded",
                     options={"xatol": 1e-12})
T_STAR = _r.x
KNOWN_MIN = 2.0 * _g(T_STAR)

OBJ = {
    (4, 0): 0.5, (2, 0): -8.0, (1, 0): 2.5,
    (0, 4): 0.5, (0, 2): -8.0, (0, 1): 2.5,
}


def f(x, y):
    return 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x) + 0.5 * (y ** 4 - 16 * y ** 2 + 5 * y)


def main():
    # sanity: dict encodes the same polynomial
    for _ in range(5):
        x, y = np.random.uniform(-3, 3, 2)
        v = sum(cf * x ** i * y ** j for (i, j), cf in OBJ.items())
        assert abs(v - f(x, y)) < 1e-9, (x, y, v, f(x, y))
    assert abs(f(T_STAR, T_STAR) - KNOWN_MIN) < 1e-9

    print("=== SOS Styblinski-Tang 2D ===")
    print(f"KNOWN global min f* = {KNOWN_MIN:.10f} at t*={T_STAR:.8f}")

    results = {}
    for order in (0, 1, 2):
        t0 = time.time()
        r = pounce.sos_minimize(OBJ, n_vars=2, order=order)
        dt = time.time() - t0
        results[order] = (r, dt)
        print(f"order={order:>2} lower_bound={r.lower_bound:+.8f} is_exact={r.is_exact} "
              f"status={r.status} num_min={r.num_minimizers} time={dt:.3f}s")

    best_order = max(results, key=lambda o: results[o][0].lower_bound)
    r, dt = results[best_order]
    lb = r.lower_bound
    gap = KNOWN_MIN - lb
    print(f"\nBest lower bound = {lb:+.8f} at order={best_order}; gap (f*-lb) = {gap:+.3e}")

    # ---- refutation: dense grid ----
    gx = np.linspace(-4, 4, 1601)
    GX, GY = np.meshgrid(gx, gx)
    FV = f(GX, GY)
    grid_min = float(FV.min())
    idx = np.unravel_index(np.argmin(FV), FV.shape)
    print(f"grid min over [-4,4]^2 = {grid_min:.8f} at ({GX[idx]:.4f},{GY[idx]:.4f})")

    # ---- refutation: multistart ----
    rng = np.random.default_rng(0)
    best_ms = np.inf; best_pt = None
    for _ in range(200):
        x0 = rng.uniform(-4, 4, 2)
        res = minimize(lambda p: f(p[0], p[1]), x0, method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000})
        if res.fun < best_ms:
            best_ms = res.fun; best_pt = res.x
    print(f"multistart best = {best_ms:.10f} at ({best_pt[0]:.6f},{best_pt[1]:.6f})")

    refute_min = min(grid_min, best_ms, KNOWN_MIN)
    if r.is_exact and r.minimizers:
        for m in r.minimizers:
            print(f"recovered minimizer {np.round(m,6)} -> f={f(m[0], m[1]):.8f}")

    bound_invalid = lb > refute_min + 1e-6
    matches = abs(lb - KNOWN_MIN) <= 1e-4
    if bound_invalid:
        print("\nBOUND INVALID: lower_bound exceeds best known point (SOLVER_BUG)")
        verdict = "FAIL"
    elif matches:
        verdict = "PASS"
    else:
        print(f"\nNOTE valid but loose bound, gap={gap:.3e} (not a correctness bug)")
        verdict = "PASS" if gap >= -1e-6 else "FAIL"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
