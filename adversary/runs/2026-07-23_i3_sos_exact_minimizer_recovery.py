"""i3 Test 5 — sos_minimize is_exact minimizer recovery (#281/#302).

#281/#302 fixed sos_minimize reporting is_exact=True with a WRONG recovered
minimizer on Rosenbrock-2D. This probes an SOS-EXACT polynomial with TWO global
minimizers whose recovery is non-trivial:

    p(x,y) = (x^2 - 1)^2 + 10 (y - x^2)^2
           = 11 x^4 - 20 x^2 y + 10 y^2 - 2 x^2 + 1
p is a manifest sum of squares, so the SOS/Lasserre lower bound is EXACT and
equals 0, attained at the two global minimizers (+1, 1) and (-1, 1).

Check: (1) lower_bound ~ 0; (2) EVERY recovered minimizer m must actually
attain the bound, i.e. p(m) ~ 0 and m near {(±1,1)}. A recovered "minimizer"
that does not attain the bound is the #281 failure mode.

Refutation oracle: dense 601x601 grid over [-2,2]^2 + 60-start scipy L-BFGS-B
multistart -> independent global minimum and its argmin set.
"""
from __future__ import annotations
import numpy as np
import pounce
from scipy.optimize import minimize as smin

POLY = {(4, 0): 11.0, (2, 1): -20.0, (0, 2): 10.0, (2, 0): -2.0, (0, 0): 1.0}


def p(xy):
    x, y = xy
    return (x ** 2 - 1) ** 2 + 10.0 * (y - x ** 2) ** 2


def grid_min():
    g = np.linspace(-2, 2, 601)
    X, Y = np.meshgrid(g, g)
    Z = (X ** 2 - 1) ** 2 + 10.0 * (Y - X ** 2) ** 2
    return float(Z.min())


def multistart_min():
    best = np.inf
    args = []
    rng = np.random.default_rng(0)
    for _ in range(60):
        x0 = rng.uniform(-2, 2, size=2)
        r = smin(p, x0, method="L-BFGS-B")
        if r.fun < best:
            best = r.fun
        if r.fun < 1e-8:
            args.append(r.x)
    return best, args


def main():
    res = pounce.sos_minimize(POLY, n_vars=2)
    lb = res.lower_bound
    mins = [np.asarray(m) for m in res.minimizers]
    gmin = grid_min()
    ms_min, ms_args = multistart_min()
    print(f"pounce SOS: lower_bound={lb:.3e} is_exact={res.is_exact} "
          f"status={res.status} num_minimizers={res.num_minimizers}")
    print(f"grid min={gmin:.3e}   multistart min={ms_min:.3e}  "
          f"(true global = 0 at (+-1, 1))")
    for m in mins:
        print(f"  recovered minimizer {m} -> p(m)={p(m):.3e}, "
              f"|.|-to-(1,1)={np.linalg.norm(m-[1,1]):.3e}, "
              f"to-(-1,1)={np.linalg.norm(m-[-1,1]):.3e}")

    bound_ok = abs(lb - 0.0) < 1e-5 and abs(gmin - 0.0) < 1e-9
    # each recovered minimizer must attain the (exact) bound AND be a true argmin
    if res.is_exact and mins:
        recov_ok = all(
            abs(p(m) - lb) < 1e-4 and
            min(np.linalg.norm(m - [1, 1]), np.linalg.norm(m - [-1, 1])) < 1e-3
            for m in mins
        )
    else:
        recov_ok = res.num_minimizers == 0  # not certified exact -> nothing to attain

    if bound_ok and (recov_ok or not res.is_exact):
        print("VERDICT: PASS (SOS-exact bound = 0 and every certified minimizer "
              "attains it at a true argmin (+-1,1))")
    else:
        print(f"VERDICT: FAIL (bound_ok={bound_ok} recov_ok={recov_ok}; "
              f"is_exact={res.is_exact} but a recovered minimizer does NOT attain "
              f"the bound / is not a true global argmin — #281 residual)")


if __name__ == "__main__":
    main()
