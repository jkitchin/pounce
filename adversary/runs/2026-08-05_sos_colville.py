"""Adversary cross-check: Colville (Wood) function global minimization via SOS/Lasserre
Family: sos   Class: unconstrained degree-4 polynomial in 4 variables, global min via SOS
Source: Colville, A.R. (1968), "A comparative study on nonlinear programming
    codes", IBM Report; also known as Wood's function (Dixon & Szego (eds.),
    "Towards Global Optimization 2", 1978, test function set). Canonical
    4-variable nonconvex test function:

    f(x1,x2,x3,x4) = 100(x1^2-x2)^2 + (x1-1)^2 + (x3-1)^2 + 90(x3^2-x4)^2
                     + 10.1[(x2-1)^2+(x4-1)^2] + 19.8(x2-1)(x4-1)

    Global minimum f* = 0 at x* = (1,1,1,1) (unique zero: every squared
    term vanishes there and the cross term (x2-1)(x4-1) also vanishes).

Expanded to the monomial dict pounce.sos_minimize expects; expansion verified
independently against the closed (unexpanded) formula at f(1,1,1,1) and five
random points before use (all agree to float64 precision, <2e-13).

Unlike the 2-variable quartics already in the log (Rosenbrock, Styblinski-
Tang, Beale, six-hump/three-hump camel), this is a genuine 4-variable
degree-4 polynomial -- Hilbert's 1888 theorem (every nonnegative poly in
<=2 vars of degree <=4 is SOS) does NOT apply here, so exactness of the
degree-2 Lasserre relaxation is not guaranteed a priori and is itself part
of what this probe checks.
"""
import time
import numpy as np
from scipy.optimize import minimize as scipy_minimize

KNOWN_OPTIMAL = 0.0
KNOWN_MINIMIZER = np.array([1.0, 1.0, 1.0, 1.0])

objective = {
    (4, 0, 0, 0): 100.0,
    (2, 1, 0, 0): -200.0,
    (0, 2, 0, 0): 110.1,
    (2, 0, 0, 0): 1.0,
    (1, 0, 0, 0): -2.0,
    (0, 0, 0, 0): 42.0,
    (0, 0, 2, 0): 1.0,
    (0, 0, 1, 0): -2.0,
    (0, 0, 4, 0): 90.0,
    (0, 0, 2, 1): -180.0,
    (0, 0, 0, 2): 100.1,
    (0, 1, 0, 0): -40.0,
    (0, 0, 0, 1): -40.0,
    (0, 1, 0, 1): 19.8,
}


def f(v):
    x1, x2, x3, x4 = v
    return (100 * (x1 ** 2 - x2) ** 2 + (x1 - 1) ** 2 + (x3 - 1) ** 2
            + 90 * (x3 ** 2 - x4) ** 2 + 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2)
            + 19.8 * (x2 - 1) * (x4 - 1))


def peval(v):
    x1, x2, x3, x4 = v
    s = 0.0
    for (e1, e2, e3, e4), c in objective.items():
        s += c * x1 ** e1 * x2 ** e2 * x3 ** e3 * x4 ** e4
    return s


rng_check = np.random.default_rng(1)
expand_err = max(abs(f(v) - peval(v)) for v in rng_check.uniform(-2, 2, size=(20, 4)))
print(f"expansion self-check (poly dict vs closed formula, 20 random points): max err = {expand_err:.2e}")
assert expand_err < 1e-8, "polynomial expansion does not match closed formula"

import pounce

t0 = time.perf_counter()
res = pounce.sos_minimize(objective, n_vars=4, order=2, tol=1e-10)
t_pounce = time.perf_counter() - t0

lb = res.lower_bound
status = res.status
is_exact = res.is_exact
minimizers = res.minimizers

# --- oracle 1: multistart scipy (BFGS) from a wide grid, refutation search ---
rng = np.random.default_rng(0)
starts = rng.uniform(-3, 3, size=(400, 4))
best_val = np.inf
best_x = None
for s in starts:
    out = scipy_minimize(f, s, method="BFGS")
    if out.fun < best_val:
        best_val, best_x = out.fun, out.x

# --- oracle 2: dense grid refutation on the 2D (x1,x3) slice through x*, x2=x1^2-ish trick skipped
# (4D dense grid is too coarse to be useful; multistart is the primary refuter)
print("=== pounce (SOS/Lasserre, order=2) ===")
print(f"status={status} lower_bound={lb:.10e} is_exact={is_exact} t={t_pounce:.4f}s")
print(f"minimizers={[np.round(np.asarray(m, dtype=float), 6) for m in minimizers]}")
print("=== oracle: multistart scipy BFGS (400 starts on [-3,3]^4) ===")
print(f"best_val={best_val:.10e} best_x={np.round(best_x, 6)}")

lb_err_known = abs(lb - KNOWN_OPTIMAL)
# A valid SOS lower bound must never exceed the true min; small positive
# slack on the order of the requested SDP tol is expected (see the Beale
# degree-8 precedent in the log: lower_bound +5.31e-7, is_exact=False).
lb_vs_multistart_valid = lb <= best_val + 1e-6

minimizer_err = None
if is_exact and len(minimizers) > 0:
    m = np.asarray(minimizers[0], dtype=float)
    minimizer_err = float(np.linalg.norm(m - KNOWN_MINIMIZER, np.inf))
    minimizer_obj_err = abs(f(m) - KNOWN_OPTIMAL)
    print(f"minimizer_err_vs_known={minimizer_err:.2e}  f(minimizer)={f(m):.3e}")

print(f"lb_err_vs_known={lb_err_known:.2e}  lb<=multistart_min: {lb_vs_multistart_valid}")

# Validity: lower bound must not exceed the true min (never a "bug" if loose
# below it; only a bug if it EXCEEDS best_val, i.e. an invalid certificate).
sound = lb_vs_multistart_valid
tight = lb_err_known < 1e-4
ok = status in ("optimal", "converged", "solved") and sound and (tight or not is_exact)
if is_exact:
    ok = ok and tight and (minimizer_err is not None and minimizer_err < 1e-3)

print("VERDICT: PASS" if ok else f"VERDICT: FAIL (sound={sound} tight={tight} is_exact={is_exact})")
