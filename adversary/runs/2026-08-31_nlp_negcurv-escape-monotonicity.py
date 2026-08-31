"""Adversary probe: the negative-curvature escape at points it actually reaches.
Family: nlp   Class: nonconvex / second-order certification
Targets: PR #802 (gh#797 neg_curv_escapes), PR #807 (gh#805 best-certificate floor)

`neg_curv_escapes`' option doc states a hard, oracle-free guarantee:

  "The stationary point is snapshotted first and is restored and reported
   unless the continuation comes back with a certificate of its own at a
   better point ... CANNOT return a worse answer than reporting the
   stationary point immediately would have ... Above 1 the floor keeps the
   BEST certificate the escapes have left rather than the most recent one
   (gh #805), so that guarantee holds at any value."

Three contracts are checked per model:
  C1 MONOTONICITY  obj(k) <= obj(0) + tol for every k          [the quoted guarantee]
  C2 FEASIBILITY   the escaped point respects eq rows + bounds [the constr_viol_tol refusal]
  C3 FLOOR         obj(k) <= obj(1) + tol for k >= 2           [gh#805, the best-certificate floor]

ENGAGEMENT IS ASSERTED, not assumed. A first version of this probe swept 40
random indefinite models and reported a clean PASS on all 160 comparisons --
while the escape fired on exactly zero of them, because a random start on a
nonconvex box converges to a genuine local minimum and there is nothing to
escape. Reaching the branch needs a model whose barrier subproblem is
SYMMETRIC about the start, so the symmetric inertia correction cannot break
the tie (the mechanism the option doc describes). Models here are built that
way, and the run fails loudly if the escape does not change the answer on
enough of them.
"""
import numpy as np, warnings
import pounce
from scipy.optimize import LinearConstraint, minimize as sp_minimize
warnings.simplefilter("ignore")

ESCAPES = [0, 1, 2, 3, 5]
OK      = (0, 1)          # Solve_Succeeded, Solved_To_Acceptable_Level
TOL     = 1e-6

def solve(P, c, lo, hi, x0, k, A=None, b=None):
    f  = lambda x: 0.5*x@P@x + c@x
    g  = lambda x: P@x + c
    hs = lambda x, *a: P*(a[1] if len(a) >= 2 else 1.0)
    return pounce.minimize(f, x0, jac=g, hess=hs, bounds=list(zip(lo, hi)),
                           constraints=LinearConstraint(A, b, b) if A is not None else None,
                           neg_curv_escapes=k, print_level=0, max_iter=500)

def feas(x, A, b, lo, hi):
    v = 0.0
    if A is not None:
        v = max(v, float(np.max(np.abs(A@x - b))))
    return max(v, float(np.max(np.maximum(0, lo - x))), float(np.max(np.maximum(0, x - hi))))

def reduced_min_eig(P, x, A, lo, hi, atol=1e-6):
    """Smallest eigenvalue of P on the null space of the active constraints."""
    rows = [] if A is None else [r for r in np.atleast_2d(A)]
    n = len(x)
    for i in range(n):
        if abs(x[i]-lo[i]) < atol or abs(x[i]-hi[i]) < atol:
            e = np.zeros(n); e[i] = 1.0; rows.append(e)
    if not rows:
        return float(np.linalg.eigvalsh(P)[0])
    M = np.array(rows)
    _, s, Vt = np.linalg.svd(M)
    rank = int((s > 1e-10).sum())
    Z = Vt[rank:].T
    if Z.shape[1] == 0:
        return np.inf                       # point fully determined; nothing to move
    return float(np.linalg.eigvalsh(Z.T@P@Z)[0])

def multistart(P, c, lo, hi, A, b, tries=400, seed=0):
    """Independent global-min oracle: dense multistart SLSQP."""
    rng = np.random.default_rng(seed)
    cons = [] if A is None else [{'type':'eq','fun':(lambda x, A=A, b=b: A@x - b)}]
    best = np.inf
    for _ in range(tries):
        x0 = rng.uniform(lo, hi)
        if A is not None:                    # project onto the equality rows
            x0 = x0 - np.linalg.pinv(A)@(A@x0 - b)
            x0 = np.clip(x0, lo, hi)
        r = sp_minimize(lambda x: 0.5*x@P@x + c@x, x0, jac=lambda x: P@x + c,
                        bounds=list(zip(lo, hi)), constraints=cons, method='SLSQP')
        if r.success and feas(r.x, A, b, lo, hi) < 1e-7:
            best = min(best, float(r.fun))
    return best

# ---- corpus: symmetric barrier subproblems, started at a stationary point ----
def corpus():
    rng = np.random.default_rng(20260831)
    out = []
    # 1. pure off-diagonal forms on a symmetric box, started at the origin
    for n in (2, 3, 4, 5, 6):
        P = np.zeros((n, n))
        for i in range(0, n-1, 2):
            P[i, i+1] = P[i+1, i] = 1.0
        for i in range(2*(n//2), n):
            P[i, i] = 2.0
        out.append((f"offdiag_n{n}", P, np.zeros(n), -2*np.ones(n), 2*np.ones(n),
                    np.zeros(n), None, None))
    # 2. symmetric spectra with genuine negative directions, symmetric box, origin start
    for t in range(10):
        n = int(rng.integers(2, 7))
        B = rng.normal(size=(n, n)); Q, _ = np.linalg.qr(B)
        w = np.concatenate([-np.abs(rng.normal(size=(n+1)//2))-0.5,
                             np.abs(rng.normal(size=n//2))+0.5])
        P = Q@np.diag(w)@Q.T; P = (P+P.T)/2
        out.append((f"spec_n{n}_t{t}", P, np.zeros(n), -2*np.ones(n), 2*np.ones(n),
                    np.zeros(n), None, None))
    # 3. the documented equality form and symmetric relatives
    for n, s in ((2, 2.0), (4, 2.0), (6, 3.0)):
        P = np.zeros((n, n))
        for i in range(0, n-1, 2):
            P[i, i+1] = P[i+1, i] = 1.0
        A = np.ones((1, n)); b = np.array([s])
        x0 = np.full(n, s/n)
        out.append((f"eqsym_n{n}", P, np.zeros(n), np.zeros(n), 4*np.ones(n), x0, A, b))
    return out

rows, viol_c1, viol_c2, viol_c3 = [], [], [], []
engaged = 0
models  = corpus()
for name, P, c, lo, hi, x0, A, b in models:
    base = solve(P, c, lo, hi, x0, 0, A, b)
    if base.status not in OK:
        print(f"[skip] {name}: base status {base.status}")
        continue
    per_k = {0: base}
    for k in ESCAPES[1:]:
        per_k[k] = solve(P, c, lo, hi, x0, k, A, b)
    if any(per_k[k].status in OK and per_k[k].fun < base.fun - 1e-9 for k in ESCAPES[1:]):
        engaged += 1
    for k in ESCAPES[1:]:
        r = per_k[k]
        if r.status not in OK:
            rows.append((name, k, base.fun, None, r.status, "-", "-")); continue
        v  = feas(r.x, A, b, lo, hi)
        me = reduced_min_eig(P, r.x, A, lo, hi)
        rows.append((name, k, base.fun, r.fun, r.status, f"{v:.1e}", f"{me:+.2e}"))
        if r.fun > base.fun + TOL*max(1.0, abs(base.fun)):
            viol_c1.append((name, k, base.fun, r.fun))
        if v > 1e-6:
            viol_c2.append((name, k, r.fun, v))
        if k >= 2 and per_k[1].status in OK and r.fun > per_k[1].fun + TOL*max(1.0, abs(per_k[1].fun)):
            viol_c3.append((name, k, per_k[1].fun, r.fun))

print(f"{'model':<14} {'k':>2} {'obj(k=0)':>14} {'obj(k)':>14} {'st':>3} {'feas':>8} {'redEig':>10}")
for nm, k, f0, fk, st, v, me in rows:
    print(f"{nm:<14} {k:>2} {f0:>14.6e} {('%14.6e'%fk) if fk is not None else '           n/a'} {st:>3} {v:>8} {me:>10}")

print(f"\nmodels: {len(models)}   escape changed the answer on: {engaged}")
print(f"C1 monotonicity violations : {len(viol_c1)}")
for v in viol_c1: print(f"   {v[0]} k={v[1]}: {v[2]:.10e} -> {v[3]:.10e}  WORSE by {v[3]-v[2]:.3e}")
print(f"C2 feasibility  violations : {len(viol_c2)}")
for v in viol_c2: print(f"   {v[0]} k={v[1]}: obj={v[2]:.6e} constraint violation {v[3]:.2e}")
print(f"C3 floor (gh#805) viols    : {len(viol_c3)}")
for v in viol_c3: print(f"   {v[0]} k={v[1]}: obj(1)={v[2]:.10e} -> obj(k)={v[3]:.10e}  WORSE by {v[3]-v[2]:.3e}")

# global-optimum context on the models the escape engaged (not a pass criterion:
# the escape is a LOCAL method and the doc says so explicitly)
print("\n-- escaped objective vs multistart global oracle (context only) --")
for name, P, c, lo, hi, x0, A, b in models[:8]:
    r = solve(P, c, lo, hi, x0, 5, A, b)
    if r.status in OK:
        gb = multistart(P, c, lo, hi, A, b, seed=1)
        print(f"  {name:<14} pounce(k=5)={r.fun:+.6e}  multistart={gb:+.6e}  gap={r.fun-gb:+.2e}")

fail = viol_c1 or viol_c2 or viol_c3 or engaged < len(models)//3
print("\nVERDICT: PASS" if not fail else
      f"VERDICT: FAIL (C1={len(viol_c1)} C2={len(viol_c2)} C3={len(viol_c3)} engaged={engaged}/{len(models)})")
