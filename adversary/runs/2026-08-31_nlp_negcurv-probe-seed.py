"""Adversary probe: the negative-curvature escape declines on a coordinate-aligned saddle.
Family: nlp   Class: nonconvex / second-order certification
Targets: PR #802 (gh#797), PR #807 (gh#805)
Known answer: analytic. min 0.5*(p*x0^2 - q*x1^2) on [-2,2]^2 has minimum -2q at (0,+-2).

FINDING: at DEFAULT options POUNCE reports "Optimal Solution Found" at the
origin (obj 0) for a large fraction of these, and the outcome flips when the
two variables are RENAMED -- the same model, permuted.
"""
import numpy as np, warnings, pounce
warnings.simplefilter("ignore")

def solve(P, n):
    f  = lambda x: 0.5*x@P@x
    g  = lambda x: P@x
    hs = lambda x, *A: P*(A[1] if len(A) >= 2 else 1.0)
    return pounce.minimize(f, np.zeros(n), jac=g, hess=hs, bounds=[(-2, 2)]*n,
                           print_level=0, max_iter=500)

print("=== 1. permutation sensitivity: one model, variables renamed ===")
for lbl, d in (("f = 0.5( x0^2 - 1.05 x1^2)", [1.0, -1.05]),
               ("f = 0.5(-1.05 x0^2 +  x1^2)", [-1.05, 1.0])):
    r = solve(np.diag(d), 2)
    print(f"  {lbl}: obj={r.fun:+.6f} status={r.status} x={np.round(r.x,4)}  (true min -2.1)")

print("\n=== 2. prevalence, diagonal indefinite, one negative eigenvalue ===")
rng = np.random.default_rng(7); tot_bad = tot_all = 0
for n in (2, 3, 4, 5, 6):
    bad = tot = 0
    for _ in range(60):
        d = rng.uniform(0.5, 3.0, size=n); k = rng.integers(0, n); d[k] = -d[k]
        r = solve(np.diag(d), n)
        true = sum(0.5*min(di*4, 0.0) for di in d)
        tot += 1
        if r.fun > true + 1e-4: bad += 1
    tot_bad += bad; tot_all += tot
    print(f"  n={n}: saddle certified Optimal on {bad}/{tot} ({100*bad/tot:.0f}%)")

print("\n=== 3. escape budget does not help ===")
P = np.diag([1.0, -1.05])
for k in (0, 1, 2, 5, 20):
    f = lambda x: 0.5*x@P@x; g = lambda x: P@x
    hs = lambda x, *A: P*(A[1] if len(A) >= 2 else 1.0)
    r = pounce.minimize(f, np.zeros(2), jac=g, hess=hs, bounds=[(-2,2)]*2,
                        neg_curv_escapes=k, print_level=0, max_iter=500)
    print(f"  neg_curv_escapes={k:>2}: obj={r.fun:+.6f} status={r.status}")

print("\n=== 4. an infinitesimal start perturbation fixes it ===")
for eps in (0.0, 1e-12, 1e-8, 1e-3):
    f = lambda x: 0.5*x@P@x; g = lambda x: P@x
    hs = lambda x, *A: P*(A[1] if len(A) >= 2 else 1.0)
    r = pounce.minimize(f, np.full(2, eps), jac=g, hess=hs, bounds=[(-2,2)]*2,
                        print_level=0, max_iter=500)
    print(f"  x0 = {eps:g}: obj={r.fun:+.6f}")

print(f"\nVERDICT: {'FAIL' if tot_bad else 'PASS'} "
      f"({tot_bad}/{tot_all} models certified a saddle as Optimal at defaults)")
