"""Adversary cross-check: Himmelblau's function (global polynomial minimization)
Family: sos   Class: unconstrained global poly min, MULTIPLE global minimizers
Source: D. M. Himmelblau, "Applied Nonlinear Programming", McGraw-Hill 1972.
        f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
Known global minimum: 0.0, attained at FOUR points:
        (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
"""

import time

import numpy as np
from scipy.optimize import minimize as spmin

KNOWN_OPTIMAL = 0.0

# Expanded: x^4 + y^4 + 2 x^2 y + 2 x y^2 - 21 x^2 - 13 y^2 - 14 x - 22 y + 170
POLY = {
    (4, 0): 1.0,
    (0, 4): 1.0,
    (2, 1): 2.0,
    (1, 2): 2.0,
    (2, 0): -21.0,
    (0, 2): -13.0,
    (1, 0): -14.0,
    (0, 1): -22.0,
    (0, 0): 170.0,
}


def f_expanded(x, y):
    return sum(c * x**a * y**b for (a, b), c in POLY.items())


def f_orig(x, y):
    return (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2


# --- sanity: expansion is correct -------------------------------------------
rng = np.random.default_rng(0)
XY = rng.uniform(-4, 4, size=(2000, 2))
exp_err = float(np.max(np.abs(f_expanded(XY[:, 0], XY[:, 1]) - f_orig(XY[:, 0], XY[:, 1]))))
print(f"expansion_max_abs_err={exp_err:.3e}")
assert exp_err < 1e-8, "polynomial expansion is wrong -> FORMULATION_ERROR"

# --- oracle 1: dense grid ----------------------------------------------------
g = np.linspace(-6, 6, 1201)
GX, GY = np.meshgrid(g, g)
grid_min = float(np.min(f_orig(GX, GY)))

# --- oracle 2: scipy multistart ---------------------------------------------
t0 = time.perf_counter()
best = np.inf
best_x = None
for s in rng.uniform(-6, 6, size=(400, 2)):
    r = spmin(lambda z: f_orig(z[0], z[1]), s, method="BFGS", tol=1e-14)
    if r.fun < best:
        best, best_x = float(r.fun), r.x
t_oracle = time.perf_counter() - t0
print(f"=== oracle ===\ngrid_min={grid_min:.10e}  multistart_min={best:.10e} at {best_x}  t={t_oracle:.4f}s")

TRUE_MIN = min(grid_min, best, KNOWN_OPTIMAL)

# --- pounce ------------------------------------------------------------------
from pounce.sos import sos_minimize  # noqa: E402

results = []
for order in (2, 3):
    t0 = time.perf_counter()
    r = sos_minimize(POLY, order=order)
    dt = time.perf_counter() - t0
    results.append((order, r, dt))
    print(f"=== pounce order={order} ===")
    print(
        f"status={r.status} lower_bound={r.lower_bound:.10e} "
        f"is_exact={r.is_exact} num_minimizers={r.num_minimizers} t={dt:.4f}s"
    )
    for m in r.minimizers:
        print(f"   minimizer={np.array2string(np.asarray(m), precision=6)} "
              f"f={f_orig(m[0], m[1]):.6e}")

print(f"known_optimal={KNOWN_OPTIMAL:.10e}")

# --- classify ----------------------------------------------------------------
TOL = 1e-6
invalid = []
verdicts = []
for order, r, dt in results:
    if r.status != "optimal" or not np.isfinite(r.lower_bound):
        verdicts.append((order, "NONOPTIMAL"))
        continue
    # INVALID LOWER BOUND: bound strictly above the true global minimum
    if r.lower_bound > TRUE_MIN + 1e-6:
        invalid.append((order, r.lower_bound))
        verdicts.append((order, "INVALID_BOUND"))
        continue
    tight = abs(r.lower_bound - KNOWN_OPTIMAL) < 1e-5
    if r.is_exact:
        mobj = [float(f_orig(m[0], m[1])) for m in r.minimizers]
        bad = [o for o in mobj if abs(o - r.lower_bound) > 1e-4]
        verdicts.append((order, "EXACT_OK" if (tight and not bad) else "EXACT_MISMATCH"))
    else:
        verdicts.append((order, "VALID_LOOSE" if not tight else "VALID_TIGHT"))

print("per_order_verdicts=" + str(verdicts))
print(f"true_min_used={TRUE_MIN:.10e}")

if invalid:
    print(f"VERDICT: FAIL (INVALID LOWER BOUND at orders {invalid})")
elif any(v in ("EXACT_MISMATCH", "NONOPTIMAL") for _, v in verdicts):
    print(f"VERDICT: FAIL ({verdicts})")
elif any(v in ("EXACT_OK", "VALID_TIGHT") for _, v in verdicts):
    print("VERDICT: PASS")
else:
    print("VERDICT: INCONCLUSIVE (bound valid but never tight)")
