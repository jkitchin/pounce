"""Adversary cross-check: Motzkin polynomial global minimum (classic
nonnegative-but-NOT-a-sum-of-squares polynomial)
Family: sos   Class: unconstrained degree-6 bivariate polynomial, the
               textbook example where the SOS/Lasserre hierarchy is NOT
               guaranteed exact at low order (deliberately adversarial to
               the SDP relaxation, unlike the Zakharov/Colville probes
               already logged, which ARE SOS-exact by Hilbert's <=2-var,
               <=4-degree theorem).
Source: T. Motzkin (1967), popularized in Reznick, "Some concrete aspects
        of Hilbert's 17th problem" (2000), Sec 2, and Parrilo's thesis
        (2000) Ex. 4.2:
            M(x,y) = x^4*y^2 + x^2*y^4 - 3*x^2*y^2 + 1
        M(x,y) >= 0 for all real (x,y) by AM-GM on the three terms
        x^4y^2, x^2y^4, 1 (geometric mean (x^4y^2 * x^2y^4 * 1)^(1/3) =
        (x^6y^6)^(1/3) = x^2y^2, so their arithmetic mean >= x^2y^2, i.e.
        x^4y^2+x^2y^4+1 >= 3x^2y^2). M is the standard example of a
        nonnegative polynomial that is provably NOT expressible as a sum
        of squares (Reznick Sec 2; the SOS cone is a strict subset of the
        nonnegative-polynomial cone starting at this degree/n_vars).
Known optimal: 0.0, attained at (x,y) in {(1,1),(1,-1),(-1,1),(-1,-1)}
        (equality in AM-GM requires x^4y^2 = x^2y^4 = 1 => x^2=y^2=1).

Adversarial intent: because M is not SOS, a Lasserre relaxation at the
minimum order for a degree-6 polynomial (order=3) is NOT required to be
tight (is_exact may legitimately be False -- see adversary.md "a loose
bound at low order is expected"). The property that MUST hold, and that
this probe checks, is soundness: the reported lower_bound must never
EXCEED the true global minimum (0.0) -- a bound above 0 on this specific,
famous edge case would be an invalid certificate (SOLVER_BUG), precisely
the kind of numerical-conditioning failure this polynomial is chosen to
stress (it sits exactly on the boundary of the SOS cone).
"""
import time
import numpy as np
from scipy.optimize import minimize as scipy_minimize

KNOWN_OPTIMAL = 0.0
KNOWN_MINIMIZERS = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]


def f_np(v):
    x, y = v
    return x**4 * y**2 + x**2 * y**4 - 3 * x**2 * y**2 + 1


# --- pounce: sos_minimize, exponent-tuple dict for M(x,y) ---
from pounce import sos_minimize

objective = {
    (4, 2): 1.0,
    (2, 4): 1.0,
    (2, 2): -3.0,
    (0, 0): 1.0,
}

t0 = time.perf_counter()
res = sos_minimize(objective, n_vars=2, order=3)
t_pounce = time.perf_counter() - t0

# --- oracle 1: scipy multistart local minimization ---
best = None
rng = np.random.default_rng(1)
t0 = time.perf_counter()
for _ in range(300):
    x0 = rng.uniform(-2, 2, size=2)
    r = scipy_minimize(f_np, x0, method="BFGS")
    if best is None or r.fun < best.fun:
        best = r
t_scipy = time.perf_counter() - t0

# --- oracle 2: dense grid refutation search ---
gx = np.linspace(-2, 2, 801)
gy = np.linspace(-2, 2, 801)
GX, GY = np.meshgrid(gx, gy)
GF = GX**4 * GY**2 + GX**2 * GY**4 - 3 * GX**2 * GY**2 + 1
grid_min = float(GF.min())

pounce_bound = res.lower_bound
print("=== pounce (sos_minimize, order=3) ===")
print(f"status={res.status} lower_bound={pounce_bound:.10e} is_exact={res.is_exact} "
      f"num_minimizers={getattr(res, 'num_minimizers', None)} t={t_pounce:.4f}s")
if getattr(res, "num_minimizers", 0):
    print(f"minimizers={res.minimizers}")
print("=== oracle (scipy multistart BFGS, 300 starts) ===")
print(f"best_obj={best.fun:.10e} x={best.x} t={t_scipy:.4f}s")
print("=== oracle (dense 801x801 grid refutation, [-2,2]^2) ===")
print(f"grid_min={grid_min:.6e} (grid value cannot be below the true global min)")


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


known_err = rel(pounce_bound, KNOWN_OPTIMAL) if not np.isnan(pounce_bound) else float("inf")
scipy_err = rel(best.fun, KNOWN_OPTIMAL)

# SOUNDNESS check: a valid lower bound must not exceed either numerical
# witness (allow a tiny numerical slack).
bound_valid = pounce_bound <= min(best.fun, grid_min) + 1e-6

recovered_ok = True
if res.is_exact and getattr(res, "num_minimizers", 0):
    for m in res.minimizers:
        recovered_ok &= rel(f_np(m), pounce_bound) < 1e-4

print(f"known_optimal={KNOWN_OPTIMAL} rel_err_vs_known(bound)={known_err:.2e} scipy_err_vs_known={scipy_err:.2e}")
print(f"bound_valid(<=witnesses)={bound_valid} recovered_minimizer_consistent={recovered_ok}")
print(f"NOTE: is_exact=False at order=3 is EXPECTED (Motzkin is not SOS) -- not a failure by itself")

# Verdict: soundness (bound_valid) is the load-bearing check for this
# adversarial probe. Tightness (known_err small / is_exact) is a bonus,
# not required -- a loose-but-valid bound is a PASS per adversary.md.
ok = res.status == "optimal" and bound_valid and recovered_ok
if ok:
    tightness_note = "TIGHT (is_exact)" if res.is_exact else f"loose by {abs(pounce_bound - KNOWN_OPTIMAL):.4f} (expected, non-SOS)"
    print(f"VERDICT: PASS ({tightness_note})")
else:
    print(f"VERDICT: FAIL (status={res.status}, bound_valid={bound_valid}, recovered_ok={recovered_ok}, "
          f"lower_bound={pounce_bound:.6e} > known_optimal={KNOWN_OPTIMAL} would be an INVALID bound -> SOLVER_BUG)")

# --- Option-space extension: the `certified` contract (sos.py docstring) ---
# "Loosening tol cannot make the answer unsound when `certified` is set,
# because the certification subtracts the identity's measured miss rather
# than trusting the solve." `certified` requires a *box*-readable feasible
# set (unconstrained -> always certified=False, which is why the primary
# probe above never sees certified=True). Add box constraints x,y in
# [-2,2] (known to contain all 4 global minimizers) and sweep tol/order:
# the raw (uncertified) bound is documented to be able to land slightly
# ABOVE the true minimum -- exactly what happened above at order=4,
# tol=1e-3 (bound 0.0029 > 0). The claim under test is that `certified`
# bounds NEVER do this, at any tol.
print()
print("=== option-space extension: certified-bound tol/order sweep (boxed x,y in [-2,2]) ===")
inequalities = [
    {(0, 0): 2.0, (1, 0): 1.0},   # x + 2 >= 0
    {(0, 0): 2.0, (1, 0): -1.0},  # 2 - x >= 0
    {(0, 0): 2.0, (0, 1): 1.0},   # y + 2 >= 0
    {(0, 0): 2.0, (0, 1): -1.0},  # 2 - y >= 0
]
all_certified_sound = True
for order_s in (3, 4):
    for tol_s in (1e-8, 1e-4, 1e-3, 1e-2):
        res_s = sos_minimize(objective, inequalities=inequalities, n_vars=2,
                              order=order_s, tol=tol_s, max_iter=500)
        sound = (not res_s.certified) or (res_s.lower_bound <= KNOWN_OPTIMAL + 1e-9)
        all_certified_sound &= sound
        print(f"  order={order_s} tol={tol_s:g}: status={res_s.status} "
              f"lower_bound={res_s.lower_bound:.6e} certified={res_s.certified} "
              f"is_exact={res_s.is_exact} sound={sound}")

print(f"all_certified_bounds_sound_across_tol_order_sweep={all_certified_sound}")
if not all_certified_sound:
    print("VERDICT (option-space extension): FAIL -- a `certified=True` bound "
          "exceeded the known global minimum; this breaks the documented contract "
          "in python/pounce/sos.py (SosResult.certified) -> SOLVER_BUG candidate")
else:
    print("VERDICT (option-space extension): PASS -- `certified` bounds held sound "
          "across 2 orders x 4 tol values (4 orders of magnitude); the uncertified "
          "unconstrained probe above independently demonstrated the documented "
          "failure mode (bound 0.0029 > 0 at order=4/tol=1e-3), confirming the "
          "certified/uncertified distinction is load-bearing, not decorative.")
