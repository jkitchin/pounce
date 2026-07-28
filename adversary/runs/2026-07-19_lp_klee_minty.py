"""Adversary cross-check: Klee-Minty cube (LP)
Family: lp   Class: exponentially-degenerate / badly-scaled LP with known closed-form optimum

Source: V. Klee and G. J. Minty, "How good is the simplex algorithm?",
in Inequalities III (O. Shisha, ed.), Academic Press, 1972, pp. 159-175.
Standard perturbed-cube form as given in Chvatal, "Linear Programming" (1983),
Ch. 4 problem 4.2, and Nocedal & Wright, "Numerical Optimization" 2e, S13.4
(discussion of simplex worst-case complexity).

    maximize    sum_{j=1..n} 10^(n-j) x_j
    subject to  2 * sum_{j=1..i-1} 10^(i-j) x_j  +  x_i  <=  100^(i-1),  i=1..n
                x >= 0

KNOWN OPTIMAL (exact, closed form): the unique optimal vertex is
    x* = (0, 0, ..., 0, 100^(n-1)),   objective z* = 100^(n-1).
The feasible region is a combinatorially-perturbed n-cube with 2^n vertices;
the classic Dantzig-rule simplex path visits all of them.  For an IPM this is
instead a *scaling/conditioning* stress test: constraint RHS ranges over
1 .. 100^(n-1) and objective coefficients over 1 .. 10^(n-1).

pounce minimizes, so we pass c = -(10^(n-j)) and expect obj = -100^(n-1).
"""
import time
import numpy as np


def klee_minty(n):
    """Return (c_min, G, h, known_x, known_obj_max) for the n-dim Klee-Minty cube."""
    G = np.zeros((n, n))
    h = np.zeros(n)
    for i in range(1, n + 1):
        for j in range(1, i):
            G[i - 1, j - 1] = 2.0 * 10.0 ** (i - j)
        G[i - 1, i - 1] = 1.0
        h[i - 1] = 100.0 ** (i - 1)
    c_max = np.array([10.0 ** (n - j) for j in range(1, n + 1)])
    known_x = np.zeros(n)
    known_x[-1] = 100.0 ** (n - 1)
    return -c_max, G, h, known_x, 100.0 ** (n - 1)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


import pounce
from scipy.optimize import linprog
import cvxpy as cp

results = []
for n in (3, 4, 5, 6, 7):
    c, G, h, known_x, known_zmax = klee_minty(n)
    known_obj = -known_zmax  # minimization form
    lb = np.zeros(n)

    # --- pounce (LP: P=None) ---
    t0 = time.perf_counter()
    rp = pounce.solve_qp(P=None, c=c, G=G, h=h, lb=lb)
    t_pounce = time.perf_counter() - t0
    x_p, obj_p, st_p = np.asarray(rp.x, dtype=float), float(rp.obj), rp.status

    # --- oracle 1: scipy linprog (HiGHS dual simplex, exact vertex arithmetic) ---
    t0 = time.perf_counter()
    ls = linprog(c, A_ub=G, b_ub=h, bounds=[(0, None)] * n, method="highs")
    t_scipy = time.perf_counter() - t0

    # --- oracle 2: cvxpy / CLARABEL (independent IPM) ---
    xv = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(c @ xv), [G @ xv <= h, xv >= 0])
    t0 = time.perf_counter()
    try:
        prob.solve(solver=cp.CLARABEL)
        cvx_status, cvx_obj = prob.status, float(prob.value)
    except Exception as exc:  # CLARABEL can fail outright on the badly-scaled cube
        cvx_status, cvx_obj = f"SolverError({type(exc).__name__})", float("nan")
    t_cvx = time.perf_counter() - t0

    e_known = rel(obj_p, known_obj)
    e_scipy = rel(obj_p, ls.fun)
    e_cvx = rel(obj_p, cvx_obj)
    # solution error measured relative to the scale of the optimal vertex
    x_err = float(np.max(np.abs(x_p - known_x)) / max(1.0, known_zmax))

    print(f"--- n={n}  (known z* = {known_zmax:.6e}) ---")
    print(f"  pounce  status={st_p:12s} obj={obj_p: .12e}  t={t_pounce:.4f}s")
    print(f"  scipy   status={ls.status}            obj={ls.fun: .12e}  t={t_scipy:.4f}s")
    print(f"  clarabel status={cvx_status:24s} obj={cvx_obj: .12e}  t={t_cvx:.4f}s")
    print(f"  rel_err_vs_known={e_known:.3e}  vs_scipy={e_scipy:.3e}  vs_clarabel={e_cvx:.3e}"
          f"  scaled_x_inf_err={x_err:.3e}")

    ok = (st_p == "optimal") and e_known < 1e-4 and e_scipy < 1e-4 and x_err < 1e-4
    results.append((n, ok, st_p, e_known, e_scipy, e_cvx, t_pounce, t_scipy, t_cvx))
    print(f"  n={n}: {'PASS' if ok else 'FAIL'}")

print()
print("=== summary ===")
for n, ok, st, ek, es, ec, tp, ts, tc in results:
    print(f"n={n} {'PASS' if ok else 'FAIL'} status={st} err_known={ek:.2e} "
          f"t_pounce={tp:.4f} t_scipy={ts:.4f} t_clarabel={tc:.4f}")

all_ok = all(r[1] for r in results)
worst = max(r[3] for r in results)
print(f"max_rel_err_vs_known={worst:.3e}")
print("VERDICT: PASS" if all_ok else
      f"VERDICT: FAIL (max rel err vs known optimum = {worst:.3e})")
