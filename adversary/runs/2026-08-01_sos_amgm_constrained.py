"""Adversary cross-check: AM-GM-constrained polynomial minimization
Family: sos   Class: equality-constrained polynomial program (Lasserre).
    Fresh polynomial for sos -- prior sos runs used Goldstein-Price, plain
    quartic, six/three-hump camel, Motzkin, Styblinski-Tang, Himmelblau,
    Beale, a generic Lasserre-constrained instance, circle-continuum,
    duality-diagnostics, and extraction-regression probes; none used the
    AM-GM equality-constrained instance below.
Source: AM-GM inequality (elementary; see e.g. Hardy, Littlewood, Polya,
    "Inequalities", or any calculus/optimization text): for x,y,z with
    xyz = 1,  (x^2+y^2+z^2)/3 >= (x^2 y^2 z^2)^(1/3) = (xyz)^(2/3) = 1,
    so x^2+y^2+z^2 >= 3, with equality iff x^2=y^2=z^2=1 and xyz=1 (i.e.
    an even number of the three are negative): (1,1,1), (1,-1,-1),
    (-1,1,-1), (-1,-1,1) -- four global minimizers, objective value 3.
Known optimal: 3 (exact, by AM-GM).
"""
import time
import itertools
import numpy as np

KNOWN_OPTIMAL = 3.0
KNOWN_MINIMIZERS = [
    (1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0),
]

# objective: x^2 + y^2 + z^2
objective = {(2, 0, 0): 1.0, (0, 2, 0): 1.0, (0, 0, 2): 1.0}
# equality: xyz - 1 = 0
equalities = [{(1, 1, 1): 1.0, (0, 0, 0): -1.0}]
# redundant compactness (Archimedean) constraint R^2 - (x^2+y^2+z^2) >= 0:
# the feasible set {xyz=1} is unbounded, and Putinar/Lasserre convergence
# guarantees require a compact (Archimedean) feasible set. R=5 is far outside
# any candidate optimum (known optimal region has |x|,|y|,|z|=1) so it does
# not change the true optimal value, only the relaxation's convergence.
R2 = 25.0
inequalities = [{(0, 0, 0): R2, (2, 0, 0): -1.0, (0, 2, 0): -1.0, (0, 0, 2): -1.0}]

from pounce.sos import sos_minimize

t0 = time.perf_counter()
r = None
used_order = None
for order in (2, 3, 4):
    r = sos_minimize(objective, inequalities=inequalities, equalities=equalities,
                      n_vars=3, order=order)
    used_order = order
    print(f"  [order={order}] status={r.status} lower_bound={r.lower_bound:.6e} is_exact={r.is_exact}")
    if r.status == "optimal" and r.is_exact:
        break
t_pounce = time.perf_counter() - t0

print(f"status={r.status} order={used_order} lower_bound={r.lower_bound:.10e} "
      f"is_exact={r.is_exact} num_minimizers={r.num_minimizers} t={t_pounce:.4f}s")
for m in r.minimizers:
    print(f"  minimizer: {m}  poly_val={m[0]**2+m[1]**2+m[2]**2:.6f}  xyz={m[0]*m[1]*m[2]:.6f}")

# --- independent oracle #1: dense grid + local refinement (multistart) ---
best = None
for x0 in np.linspace(-2.5, 2.5, 21):
    for y0 in np.linspace(-2.5, 2.5, 21):
        if abs(x0) < 1e-6 or abs(y0) < 1e-6:
            continue
        z0 = 1.0 / (x0 * y0)
        val = x0 ** 2 + y0 ** 2 + z0 ** 2
        if best is None or val < best[0]:
            best = (val, x0, y0, z0)
grid_val = best[0]

from scipy.optimize import minimize as scipy_minimize


def fun(v):
    x, y, z = v
    return x ** 2 + y ** 2 + z ** 2


def con(v):
    x, y, z = v
    return x * y * z - 1.0


best_ms = None
rng_starts = [(1.5, 1.3, 0.4), (-1.2, -0.8, 1.0), (0.5, 2.0, 1.0), (1, 1, 1), (-1, -1, 1)]
for x0 in rng_starts:
    res = scipy_minimize(fun, x0, constraints=[{"type": "eq", "fun": con}], method="SLSQP")
    if res.success and (best_ms is None or res.fun < best_ms.fun):
        best_ms = res
ms_val = float(best_ms.fun)
ms_x = best_ms.x

def rel(a, ref):
    return abs(a - ref) / max(1.0, abs(ref))

lb_err = rel(r.lower_bound, KNOWN_OPTIMAL)
grid_err = rel(grid_val, KNOWN_OPTIMAL)
ms_err = rel(ms_val, KNOWN_OPTIMAL)
# refute: no candidate point should beat the claimed lower bound (that would
# be an INVALID lower bound -- a real SOLVER_BUG)
valid_lb = (r.lower_bound <= grid_val + 1e-6) and (r.lower_bound <= ms_val + 1e-6)

print(f"oracle: dense-grid (coarse) best={grid_val:.6f} at "
      f"x0,y0,z0=({best[1]:.3f},{best[2]:.3f},{best[3]:.3f})")
print(f"oracle: scipy SLSQP multistart best={ms_val:.10e} x={ms_x}")
print(f"known (AM-GM exact)={KNOWN_OPTIMAL:.10e}")
print(f"lb_err_vs_known={lb_err:.2e} grid_err={grid_err:.2e} ms_err={ms_err:.2e} "
      f"valid_lower_bound={valid_lb}")

ok = r.status == "optimal" and valid_lb and lb_err < 1e-4 and (not r.is_exact or ms_err < 1e-4)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={r.status}, lb_err={lb_err:.2e}, valid_lb={valid_lb}, "
      f"ms_err={ms_err:.2e})")
