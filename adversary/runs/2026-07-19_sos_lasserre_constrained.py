"""Adversary cross-check: constrained polynomial program (Lasserre Example 5)
Family: sos   Class: constrained global polynomial min (quartic inequalities)
Source: J. B. Lasserre, "Global Optimization with Polynomials and the Problem of
        Moments", SIAM J. Optim. 11(3):796-817, 2001, Example 5 (Sec. 5).
        Originally Floudas & Pardalos, "A Collection of Test Problems for
        Constrained Global Optimization Algorithms", Problem 2.5.

    min  -x1 - x2
    s.t.  x2 <= 2 x1^4 - 8 x1^3 + 8 x1^2 + 2
          x2 <= 4 x1^4 - 32 x1^3 + 88 x1^2 - 96 x1 + 36
          0 <= x1 <= 3,  0 <= x2 <= 4

Known global optimum: -5.5080 at (x1, x2) = (2.3295, 3.1783).
"""

import time

import numpy as np
from scipy.optimize import minimize as spmin

KNOWN_OPTIMAL = -5.5080

# objective: -x1 - x2
OBJ = {(1, 0): -1.0, (0, 1): -1.0}

# inequalities g(x) >= 0
G1 = {(4, 0): 2.0, (3, 0): -8.0, (2, 0): 8.0, (0, 0): 2.0, (0, 1): -1.0}
G2 = {(4, 0): 4.0, (3, 0): -32.0, (2, 0): 88.0, (1, 0): -96.0, (0, 0): 36.0, (0, 1): -1.0}
G3 = {(1, 0): 1.0}                 # x1 >= 0
G4 = {(0, 0): 3.0, (1, 0): -1.0}   # 3 - x1 >= 0
G5 = {(0, 1): 1.0}                 # x2 >= 0
G6 = {(0, 0): 4.0, (0, 1): -1.0}   # 4 - x2 >= 0
INEQ = [G1, G2, G3, G4, G5, G6]


def polyval(P, x1, x2):
    return sum(c * x1**a * x2**b for (a, b), c in P.items())


def obj(z):
    return -z[0] - z[1]


def g_all(z):
    return np.array([polyval(P, z[0], z[1]) for P in INEQ])


# --- oracle 1: dense feasible grid -------------------------------------------
n = 3001
gx = np.linspace(0.0, 3.0, n)
gy = np.linspace(0.0, 4.0, n)
GX, GY = np.meshgrid(gx, gy)
feas = (polyval(G1, GX, GY) >= 0) & (polyval(G2, GX, GY) >= 0)
vals = np.where(feas, -GX - GY, np.inf)
k = int(np.argmin(vals))
grid_min = float(vals.flat[k])
grid_pt = (float(GX.flat[k]), float(GY.flat[k]))

# --- oracle 2: scipy SLSQP multistart ----------------------------------------
rng = np.random.default_rng(1)
cons = [{"type": "ineq", "fun": (lambda z, P=P: polyval(P, z[0], z[1]))} for P in INEQ]
t0 = time.perf_counter()
best, best_x = np.inf, None
for s in rng.uniform([0, 0], [3, 4], size=(300, 2)):
    r = spmin(obj, s, method="SLSQP", constraints=cons,
              bounds=[(0, 3), (0, 4)], options={"maxiter": 400, "ftol": 1e-14})
    if r.success and g_all(r.x).min() >= -1e-9 and r.fun < best:
        best, best_x = float(r.fun), r.x
t_oracle = time.perf_counter() - t0

print("=== oracle ===")
print(f"grid_min={grid_min:.10e} at {grid_pt}")
print(f"multistart_min={best:.10e} at {best_x}  t={t_oracle:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e}")

TRUE_MIN = min(grid_min, best)   # best certified-feasible upper bound on the global min

# --- pounce -------------------------------------------------------------------
from pounce.sos import sos_minimize  # noqa: E402

rows = []
for order in (2, 3, 4):
    t0 = time.perf_counter()
    r = sos_minimize(OBJ, inequalities=INEQ, order=order)
    dt = time.perf_counter() - t0
    rows.append((order, r, dt))
    print(f"=== pounce order={order} ===")
    print(f"status={r.status} lower_bound={r.lower_bound:.10e} is_exact={r.is_exact} "
          f"num_minimizers={r.num_minimizers} t={dt:.4f}s")
    for m in r.minimizers:
        print(f"   minimizer={np.array2string(np.asarray(m), precision=6)} "
              f"obj={obj(m):.6e} min_g={g_all(m).min():.3e}")

# --- classify -----------------------------------------------------------------
SLACK = 1e-5           # SDP solver tolerance allowance
verdicts, invalid = [], []
for order, r, dt in rows:
    if r.status != "optimal" or not np.isfinite(r.lower_bound):
        verdicts.append((order, "NONOPTIMAL"))
        continue
    if r.lower_bound > TRUE_MIN + SLACK:
        invalid.append((order, r.lower_bound))
        verdicts.append((order, "INVALID_BOUND"))
        continue
    tight = abs(r.lower_bound - TRUE_MIN) < 1e-4
    if r.is_exact:
        bad = [m for m in r.minimizers
               if abs(obj(m) - r.lower_bound) > 1e-4 or g_all(m).min() < -1e-6]
        verdicts.append((order, "EXACT_OK" if (tight and not bad) else "EXACT_MISMATCH"))
    else:
        verdicts.append((order, "VALID_TIGHT" if tight else "VALID_LOOSE"))

print(f"true_min_used={TRUE_MIN:.10e}")
print("per_order_verdicts=" + str(verdicts))
for order, r, dt in rows:
    if np.isfinite(r.lower_bound):
        print(f"gap_order{order}={TRUE_MIN - r.lower_bound:.6e} (>=0 means VALID lower bound)")

if invalid:
    print(f"VERDICT: FAIL (INVALID LOWER BOUND at {invalid})")
elif any(v in ("EXACT_MISMATCH", "NONOPTIMAL") for _, v in verdicts):
    print(f"VERDICT: FAIL ({verdicts})")
elif any(v in ("EXACT_OK", "VALID_TIGHT") for _, v in verdicts):
    print("VERDICT: PASS")
else:
    print("VERDICT: INCONCLUSIVE (valid but loose at all tested orders)")
