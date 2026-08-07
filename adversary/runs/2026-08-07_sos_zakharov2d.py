"""Adversary cross-check: Zakharov function (n=2), global polynomial minimum
Family: sos   Class: unconstrained quartic, Lasserre order 2 (should be exact:
               Hilbert 1888 -- every nonnegative poly in <=2 vars of degree <=4
               is a sum of squares, so the order-2 relaxation is exact here)
Source: Zakharov function, standard global-optimization benchmark (Molga &
        Smutnicki, "Test functions for optimization needs", 2005, Sec 2.19).
        f(x) = sum x_i^2 + (sum 0.5*i*x_i)^2 + (sum 0.5*i*x_i)^4
        For n=2: f(x1,x2) = x1^2+x2^2 + (0.5x1+x2)^2 + (0.5x1+x2)^4
Known optimal: 0.0, unique minimizer at (0,0) -- f is a sum of even powers of
        x1,x2 and (0.5x1+x2), so f(x)=0 forces x1=x2=0.
"""
import time
import numpy as np
from scipy.optimize import minimize as scipy_minimize

KNOWN_OPTIMAL = 0.0
KNOWN_X = np.array([0.0, 0.0])


def f_np(x):
    x1, x2 = x
    u = 0.5 * x1 + x2
    return x1**2 + x2**2 + u**2 + u**4


# --- pounce: sos_minimize with exponent-tuple polynomial dict ---
# f = x1^2 + x2^2 + (0.5x1+x2)^2 + (0.5x1+x2)^4
#   (0.5x1+x2)^2 = 0.25x1^2 + x1x2 + x2^2
#   (0.5x1+x2)^4 = 0.0625x1^4 + 0.5x1^3x2 + 1.5x1^2x2^2 + 2x1x2^3 + x2^4
from pounce import sos_minimize

objective = {
    (2, 0): 1.0 + 0.25,       # x1^2 (own) + from square term
    (0, 2): 1.0 + 1.0,        # x2^2 (own) + from square term
    (1, 1): 1.0,              # x1 x2, from square term
    (4, 0): 0.0625,
    (3, 1): 0.5,
    (2, 2): 1.5,
    (1, 3): 2.0,
    (0, 4): 1.0,
}

t0 = time.perf_counter()
res = sos_minimize(objective, n_vars=2, order=2)
t_pounce = time.perf_counter() - t0

# --- oracle 1: scipy multistart local minimization (independent numerical refutation) ---
best = None
rng = np.random.default_rng(0)
t0 = time.perf_counter()
for _ in range(200):
    x0 = rng.uniform(-5, 5, size=2)
    r = scipy_minimize(f_np, x0, method="BFGS")
    if best is None or r.fun < best.fun:
        best = r
t_scipy = time.perf_counter() - t0

# --- oracle 2: dense grid refutation search ---
gx = np.linspace(-3, 3, 601)
gy = np.linspace(-3, 3, 601)
GX, GY = np.meshgrid(gx, gy)
U = 0.5 * GX + GY
GF = GX**2 + GY**2 + U**2 + U**4
grid_min = float(GF.min())

pounce_bound = res.lower_bound
print("=== pounce (sos_minimize) ===")
print(f"status={res.status} lower_bound={pounce_bound:.10e} is_exact={res.is_exact} "
      f"num_minimizers={res.num_minimizers} t={t_pounce:.4f}s minimizers={res.minimizers}")
print("=== oracle (scipy multistart BFGS) ===")
print(f"best_obj={best.fun:.10e} x={best.x} t={t_scipy:.4f}s")
print("=== oracle (dense grid refutation) ===")
print(f"grid_min={grid_min:.6e} (grid cannot go below the true global min)")


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


known_err = rel(pounce_bound, KNOWN_OPTIMAL) if not np.isnan(pounce_bound) else float("inf")
scipy_err = rel(best.fun, KNOWN_OPTIMAL)

# validity check: a sound lower bound must not EXCEED the scipy/grid witnesses
bound_valid = pounce_bound <= min(best.fun, grid_min) + 1e-6

recovered_ok = True
if res.is_exact and res.num_minimizers > 0:
    for m in res.minimizers:
        recovered_ok &= rel(f_np(m), pounce_bound) < 1e-4

print(f"known_optimal={KNOWN_OPTIMAL} rel_err_vs_known={known_err:.2e} scipy_err_vs_known={scipy_err:.2e}")
print(f"bound_valid(<=witnesses)={bound_valid} recovered_minimizer_consistent={recovered_ok}")

ok = (
    res.status == "optimal"
    and bound_valid
    and known_err < 1e-4
    and recovered_ok
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={res.status}, known_err={known_err:.2e}, bound_valid={bound_valid})")
