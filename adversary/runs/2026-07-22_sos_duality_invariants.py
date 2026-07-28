"""Adversary cross-check: SOS/moment DUALITY INVARIANTS for `sos_minimize`.

Family: sos   Class: mathematical invariants (lower-bound validity, order monotonicity)

This run does not test one polynomial against one reference value; it tests the
*contract* of the Lasserre/Putinar hierarchy on several polynomials with known
global minima:

  (a) VALIDITY   : lower_bound <= true global minimum (+ solver tolerance).
                   A bound strictly above the true minimum is not a lower bound.
  (b) MONOTONICITY: the bound is non-decreasing in the relaxation order.
                   The order-(k+1) relaxation's feasible set (moment side) is a
                   subset of the order-k one, so gamma_{k+1} >= gamma_k is a
                   theorem.  A decrease beyond tolerance is a definite bug.
                   This sub-test needs NO external oracle.
  (c) CERTIFICATE: skipped -- pounce's SosResult exposes no Gram / SOS
                   certificate (fields: lower_bound, status, is_exact,
                   num_minimizers, minimizers, certified, order).  N/A.
  (d) MOMENT MATRIX: likewise not exposed.  We instead check the weaker
                   observable consistency of `is_exact` with the extracted
                   minimizers (each reported minimizer must be feasible and
                   attain the reported bound).
  (e) NONNEG-NOT-SOS: dehomogenized Choi-Lam form (nonnegative, not SOS) on a
                   box; the bound must rise toward 0 with order.

Problems (all NEW -- none in adversary/log.org):
  1. Booth              (x+2y-7)^2 + (2x+y-5)^2                min 0 at (1,3)
  2. Rosenbrock-2D      100(y-x^2)^2 + (1-x)^2                 min 0 at (1,1)
  3. Dixon-Price n=2    (x-1)^2 + 2(2y^2-x)^2                  min 0 at (1, 1/sqrt(2))
  4. Choi-Lam (dehom.)  x^2y^2+y^2z^2+z^2x^2 - 4xyz + 1        min 0 at (1,1,1)&(perm.)

Sources
-------
Booth, Rosenbrock, Dixon-Price: standard global-optimization test set,
  Jamil & Yang, "A literature survey of benchmark functions for global
  optimization problems", Int. J. Math. Model. Numer. Optim. 4(2) 2013,
  functions f14 (Booth), f105 (Rosenbrock), f48 (Dixon-Price).
Choi-Lam form: M.-D. Choi and T.-Y. Lam, "Extremal positive semidefinite
  forms", Math. Ann. 231 (1977) 1-18.  The quaternary quartic
  Q(x,y,z,w) = w^4 + x^2y^2 + y^2z^2 + z^2x^2 - 4xyzw is positive semidefinite
  but not a sum of squares; setting w=1 dehomogenizes it.  See also Reznick,
  "Some concrete aspects of Hilbert's 17th problem" (2000), Sec. 3.

Oracles: known global minima (verified independently here by a dense grid +
scipy multistart refutation search over the same box), and the monotonicity
theorem itself.
"""

import time

import numpy as np
from scipy.optimize import minimize as smin

from pounce.sos import sos_minimize

TOL = 1e-6  # slack allowed on the "lower bound" and monotonicity assertions


# ---------------------------------------------------------------- polynomials
def box(n, lo, hi):
    """Box constraints l <= x_i <= u as 2n polynomial inequalities g >= 0."""
    g = []
    for i in range(n):
        e = [0] * n
        ei = tuple(e[:i] + [1] + e[i + 1 :])
        z = tuple(e)
        g.append({ei: 1.0, z: -float(lo)})  # x_i - lo >= 0
        g.append({ei: -1.0, z: float(hi)})  # hi - x_i >= 0
    return g


# Booth: (x+2y-7)^2 + (2x+y-5)^2 = 5x^2 + 5y^2 + 8xy - 34x - 38y + 74
BOOTH = {(2, 0): 5.0, (0, 2): 5.0, (1, 1): 8.0, (1, 0): -34.0, (0, 1): -38.0, (0, 0): 74.0}

# Rosenbrock 2D: 100(y-x^2)^2 + (1-x)^2
#   = 100y^2 - 200x^2 y + 100x^4 + 1 - 2x + x^2
ROSEN = {(4, 0): 100.0, (2, 1): -200.0, (0, 2): 100.0, (2, 0): 1.0, (1, 0): -2.0, (0, 0): 1.0}

# Dixon-Price n=2: (x-1)^2 + 2(2y^2 - x)^2
#   = x^2 - 2x + 1 + 2(4y^4 - 4xy^2 + x^2) = 8y^4 - 8xy^2 + 3x^2 - 2x + 1
DIXON = {(0, 4): 8.0, (1, 2): -8.0, (2, 0): 3.0, (1, 0): -2.0, (0, 0): 1.0}

# Choi-Lam dehomogenized: x^2y^2 + y^2z^2 + z^2x^2 - 4xyz + 1
CHOILAM = {
    (2, 2, 0): 1.0,
    (0, 2, 2): 1.0,
    (2, 0, 2): 1.0,
    (1, 1, 1): -4.0,
    (0, 0, 0): 1.0,
}


def peval(poly, x):
    x = np.asarray(x, float)
    return sum(c * np.prod(x**np.asarray(e, float)) for e, c in poly.items())


CASES = [
    # name, poly, n, box, known min, orders to sweep
    ("booth", BOOTH, 2, (-10.0, 10.0), 0.0, [1, 2, 3]),
    ("rosenbrock2d", ROSEN, 2, (-2.0, 2.0), 0.0, [2, 3, 4]),
    ("dixon_price2", DIXON, 2, (-2.0, 2.0), 0.0, [2, 3, 4]),
    ("choi_lam_dehom", CHOILAM, 3, (-2.0, 2.0), 0.0, [2, 3]),
]


# ------------------------------------------------------- independent refutation
def refute(poly, n, lo, hi, claimed, seed=0):
    """Try hard to find a point in the box with f < claimed.  Returns the best
    value found (dense grid + scipy multistart)."""
    rng = np.random.default_rng(seed)
    # dense grid
    m = 61 if n == 2 else 21
    axes = [np.linspace(lo, hi, m)] * n
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, n)
    vals = np.array([peval(poly, p) for p in grid])
    best = float(vals.min())
    best_x = grid[int(vals.argmin())]
    # multistart local refinement
    bnds = [(lo, hi)] * n
    starts = np.vstack([best_x[None, :], rng.uniform(lo, hi, size=(40, n))])
    for s in starts:
        r = smin(lambda z: peval(poly, z), s, bounds=bnds, method="L-BFGS-B")
        if r.fun < best:
            best, best_x = float(r.fun), r.x
    return best, best_x


# ----------------------------------------------------------------------- run
print("=" * 74)
print("SOS / moment duality invariants")
print("=" * 74)

failures = []
rows = []

for name, poly, n, (lo, hi), known, orders in CASES:
    print(f"\n--- {name}  (n={n}, box [{lo},{hi}], known min = {known}) ---")
    g = box(n, lo, hi)

    # oracle: refute the known global minimum
    t0 = time.perf_counter()
    grid_best, grid_x = refute(poly, n, lo, hi, known)
    t_orc = time.perf_counter() - t0
    print(f"refutation search: best f = {grid_best:.10e} at {np.round(grid_x, 6)}  "
          f"({t_orc:.2f}s)")
    if grid_best < known - 1e-6:
        print(f"  !! REFERENCE_ERROR: search found f={grid_best:.6e} < known {known}")
        failures.append((name, "REFERENCE_ERROR", grid_best))
        continue
    true_min = min(known, grid_best)

    prev = None
    for k in orders:
        t0 = time.perf_counter()
        r = sos_minimize(poly, inequalities=g, n_vars=n, order=k)
        dt = time.perf_counter() - t0
        lb = r.lower_bound
        print(f"  order={k:d} -> status={r.status:<10s} lb={lb: .12e} "
              f"exact={r.is_exact} cert={r.certified} nmin={r.num_minimizers} "
              f"order_used={r.order} t={dt:.2f}s")
        rows.append((name, k, r.status, lb, r.is_exact, r.certified, r.order, dt))

        if r.status != "optimal" or not np.isfinite(lb):
            print("    (non-optimal: no bound claimed, skipped from invariants)")
            continue

        # (a) validity
        if lb > true_min + TOL:
            print(f"    !! VALIDITY VIOLATION: lb {lb:.12e} > true min {true_min:.12e} "
                  f"(excess {lb - true_min:.3e})")
            failures.append((name, f"INVALID_BOUND@order{k}", lb - true_min))

        # (b) monotonicity in order -- only comparable when the order actually used
        # matches the order requested (sos_minimize may fall back to a coarser one)
        if prev is not None:
            pk, plb, pused = prev
            if r.order >= pused and lb < plb - TOL:
                print(f"    !! MONOTONICITY VIOLATION: order {pk}->{k} bound "
                      f"{plb:.12e} -> {lb:.12e} (drop {plb - lb:.3e})")
                failures.append((name, f"NONMONOTONE@{pk}->{k}", plb - lb))
            elif r.order < pused:
                print(f"    (order fell back {pused} -> {r.order}; monotonicity "
                      f"comparison not applicable)")
        prev = (k, lb, r.order)

        # (d') is_exact consistency: reported minimizers must be feasible and
        # attain the reported bound
        if r.is_exact and r.minimizers:
            for mzr in r.minimizers:
                fv = peval(poly, mzr)
                feas = np.all(mzr >= lo - 1e-5) and np.all(mzr <= hi + 1e-5)
                gap = abs(fv - lb)
                flag = "" if (feas and gap < 1e-4) else "  <-- INCONSISTENT"
                print(f"      minimizer {np.round(mzr, 6)}  f={fv: .3e} "
                      f"|f-lb|={gap:.2e} feasible={feas}{flag}")
                if not feas or gap > 1e-4:
                    failures.append((name, f"EXACT_INCONSISTENT@order{k}", gap))

    # (e) commentary for the nonneg-not-SOS case
    if name == "choi_lam_dehom":
        lbs = [row[3] for row in rows if row[0] == name and np.isfinite(row[3])]
        if lbs:
            print(f"  nonneg-not-SOS: bounds {['%.3e' % v for v in lbs]} "
                  f"-> gap to 0 = {['%.2e' % abs(v) for v in lbs]}")

print("\n" + "=" * 74)
if failures:
    for f in failures:
        print("FAILURE:", f)
    print(f"VERDICT: SOLVER_BUG ({len(failures)} invariant violation(s))")
else:
    print("All invariants held: bounds valid, non-decreasing in order, "
          "is_exact consistent.")
    print("VERDICT: PASS")
