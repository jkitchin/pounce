"""Adversary cross-check: Dixon-Price function, n=3, global polynomial minimum
Family: sos   Class: unconstrained quartic, 3 vars, 4 global minimizers (sign pairs)
Source: Dixon & Price, J. Optim. Theory Appl. 61(3) 465-468 (1989);
        Jamil & Yang, Int. J. Math. Model. Numer. Optim. 4(2) 2013, fn 48:
        f(x) = (x1-1)^2 + sum_{i=2}^n i (2 x_i^2 - x_{i-1})^2,  f* = 0 at
        x_i = 2^{-(2^i - 2)/2^i} (up to signs of x_2.. under the squares).
        For n=3: x1=1, x2=+-2^{-1/2}, x3=+-2^{-3/4}. f is a sum of squares, so
        order-2 Lasserre is exact (bound 0). Not in the log.
Known optimal: 0
"""
import time, itertools, numpy as np, sympy as sp
x1, x2, x3 = X = sp.symbols("x1 x2 x3")
fexpr = (x1 - 1) ** 2 + 2 * (2 * x2 ** 2 - x1) ** 2 + 3 * (2 * x3 ** 2 - x2) ** 2
poly = sp.Poly(sp.expand(fexpr), *X)
obj = {tuple(int(e) for e in mon): float(cf) for mon, cf in poly.terms()}
fnum = sp.lambdify([X], fexpr, "numpy")
KNOWN = 0.0
from pounce import sos_minimize
t0 = time.perf_counter(); r = sos_minimize(obj, n_vars=3, order=2); tp = time.perf_counter() - t0
attrs = {a: getattr(r, a) for a in dir(r) if not a.startswith("_") and not callable(getattr(r, a))}
print("pounce SosResult:", {k: (v if not isinstance(v, (list, np.ndarray)) or len(v) < 8 else "...") for k, v in attrs.items()})
lb = r.lower_bound
# refutation: dense multistart local search for the global min
from scipy.optimize import minimize
best = np.inf; t0 = time.perf_counter()
for x0 in itertools.product(np.linspace(-2, 2, 7), repeat=3):
    s = minimize(lambda v: fnum(v), x0, method="BFGS", options={"gtol": 1e-12})
    best = min(best, s.fun)
to = time.perf_counter() - t0
mins = getattr(r, "minimizers", None) or []
mvals = [float(fnum(np.asarray(m))) for m in mins]
print(f"pounce lower_bound={lb:.3e} t={tp:.4f}s minimizers={len(mins)} f(minimizers)={['%.2e' % v for v in mvals]}")
print(f"multistart scipy best={best:.3e} t={to:.3f}s known={KNOWN}")
valid = lb <= best + 1e-6
ok = valid and abs(lb - KNOWN) < 1e-5 and all(v < 1e-5 for v in mvals)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (valid_lower_bound={valid})")
