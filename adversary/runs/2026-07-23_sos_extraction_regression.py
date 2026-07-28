"""Adversary iteration 3: SOS minimizer/atom-extraction self-consistency regression.
Family: sos   Class: global polynomial min + minimizer recovery
Direction: the extraction bug found on Himmelblau (2026-07-19) and Rosenbrock-2D
(2026-07-22) -- is_exact=True but returned minimizer does NOT attain the bound,
degrading with order -- appears FIXED at the current commit. Re-verify across a
battery and hunt for any residual failure.

Oracles (no pounce-trusting):
  (A) VALIDITY: lower_bound must NOT exceed the true global min (known/multistart).
  (B) SELF-CONSISTENCY: if is_exact, EVERY returned minimizer must satisfy
      f(minimizer) ~= lower_bound (this is an internal contract, needs no oracle).
  (C) known global minima + scipy multistart refutation.
"""
import numpy as np, sympy as sp
from scipy.optimize import minimize as smin
from pounce import sos_minimize

x, y = sp.symbols('x y')

def to_dict(expr):
    p = sp.Poly(sp.expand(expr), x, y)
    return {tuple(m): float(c) for m, c in zip(p.monoms(), p.coeffs())}

def box(r):   # |x|,|y| <= r  as two inequalities g>=0
    return [{(0,0): r*r, (2,0): -1.0}, {(0,0): r*r, (0,2): -1.0}]

# name -> (expr, box_radius, true_min, known_minimizers(list) or None)
BATTERY = {
    "Himmelblau":       ((x**2+y-11)**2+(x+y**2-7)**2, 6.0, 0.0,
                         [(3,2),(-2.805118,3.131312),(-3.779310,-3.283186),(3.584428,-1.848126)]),
    "DoubleWell2D":     ((x**2-1)**2+(y**2-1)**2, 3.0, 0.0,
                         [(1,1),(1,-1),(-1,1),(-1,-1)]),
    "SixHumpCamel":     ((4-2.1*x**2+x**4/3)*x**2 + x*y + (-4+4*y**2)*y**2, 2.0, -1.0316284535,
                         [(0.0898,-0.7126),(-0.0898,0.7126)]),
    "RosenbrockBoxed":  ((1-x)**2+100*(y-x**2)**2, 2.0, 0.0, [(1,1)]),
    "AsymQuartic":      (x**4-3*x**2+x + y**4-3*y**2+y, 3.0, None, None),  # true min via multistart
}

def fnum(expr):
    fl = sp.lambdify((x,y), expr, 'numpy')
    return lambda p: float(fl(p[0], p[1]))

def multistart_min(expr, r, n=200, seed=0):
    f = fnum(expr); rng = np.random.RandomState(seed); best = np.inf
    for _ in range(n):
        x0 = rng.uniform(-r, r, 2)
        res = smin(f, x0, method='Nelder-Mead',
                   options=dict(xatol=1e-10, fatol=1e-12, maxiter=5000))
        if res.fun < best and np.all(np.abs(res.x) <= r + 1e-6):
            best = res.fun
    return best

TOLC = 1e-3   # self-consistency tolerance on f(minimizer) - lower_bound
overall_fail = []
for name, (expr, r, tmin, kmins) in BATTERY.items():
    f = fnum(expr)
    true_min = tmin if tmin is not None else multistart_min(expr, r)
    fd = to_dict(expr); cons = box(r)
    print(f"\n### {name}  (true global min = {true_min:.6f})")
    for o in [2,3,4]:
        res = sos_minimize(fd, inequalities=cons, order=o)
        lb = res.lower_bound
        # (A) validity: lb must not exceed true min (allow SDP slack)
        valid = (lb <= true_min + 1e-4) or np.isnan(lb)
        # (B) self-consistency if is_exact
        fvals = [f(np.asarray(m)) for m in res.minimizers]
        consistent = True; worst = 0.0
        if res.is_exact and res.minimizers:
            for fv in fvals:
                worst = max(worst, abs(fv - lb))
            consistent = worst <= TOLC
        tag = "OK"
        if not valid: tag = "INVALID_BOUND(BUG)"; overall_fail.append((name,o,"invalid_bound"))
        elif res.is_exact and not consistent: tag = "EXTRACTION_BUG"; overall_fail.append((name,o,f"f(min)-lb={worst:.2e}"))
        mstr = ", ".join(f"({m[0]:.3f},{m[1]:.3f}):f={fv:.2e}" for m,fv in zip(res.minimizers,fvals)) or "none"
        print(f"  o{o}: status={res.status:<18} lb={lb: .4e} is_exact={res.is_exact!s:<5} "
              f"n_min={res.num_minimizers} worst|f-lb|={worst:.2e} [{tag}]")
        print(f"       minimizers: {mstr}")

print("\n" + "="*60)
if overall_fail:
    print("VERDICT: FAIL")
    for nm,o,why in overall_fail: print(f"  {nm} order {o}: {why}")
else:
    print("VERDICT: PASS (no invalid bounds; every is_exact minimizer attains its lower_bound)")
