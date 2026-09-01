"""The sigma-path census, with a reference the census can actually resolve.

gh#875 and gh#880 both used this family: unconstrained ill-conditioned QPs,
`cond` x magnitude x `n` x rotated-or-not, where

    min 1/2 x'Px + c'x,   P = Q diag(eig) Q',   c = -P t

is built so that `x* = t` "exactly, by construction, with no solver in the loop
and no reliance on an oracle".

That claim is false at the top of the range, and gh#882 is the consequence. `t`
is the optimum of the problem in *exact* arithmetic, but the problem handed to
the solver is the one whose coefficients are the float64 values `P` and `c`
actually hold, and `c = -P @ t` is itself rounded. The realised optimum
therefore sits away from `t` by, measured here,

    cond   1e2      1e4      1e6      1e8      1e10     1e12
    drift  2.6e-15  4.0e-13  4.2e-11  2.6e-09  2.8e-07  4.3e-05

against a fixed 1e-6 threshold. At `1e12` the reference is wrong by more than
the threshold outright; at `1e10` it is within a factor of a few of the errors
being judged. So in exactly the regime the defect lives in, the census was
mixing its own reference error into the solver's.

That mattered in practice, not in principle: read with the `t` reference, eight
of the nine instances that survive the gh#880 fix looked like they were at or
under the measurement floor, and the residue looked like mostly ruler. Read
against the exact optimum, all nine are real solver errors. The conclusion
reversed.

The fix here is not higher precision, it is **exact** arithmetic. Every float64
is a rational, so the realised problem `(P_fl, c_fl)` is an exactly-specified
rational linear system, and `x* = -P_fl^-1 c_fl` is computed with
`fractions.Fraction` and no rounding anywhere. The reference is then exact to
the last bit and the census resolves down to the float64 representation of `x*`
itself, ~1e-16 relative, at every conditioning.

Two things make that reference the *solver's* problem and not a neighbouring
one, and both are checked rather than assumed:

  * `assert_nl_roundtrip` re-reads the `.nl` that was just written and requires
    every numeric literal in it to be exactly a float64 the generator intended.
    Pyomo's writer emits shortest-round-trip `repr`, so this passes today; if it
    ever switches to a fixed `%g`, the census fails loudly here rather than
    quietly reacquiring the bug this file exists to remove.
  * the tree Pyomo writes omits coefficients equal to `0.0` and `1.0`, so the
    check compares against the multiset of *written* values, not all n^2.

Run `gen.py` to write the instances and `run.py <binary>` to score one.
"""

import itertools
import json
import os
import re
from fractions import Fraction

import numpy as np
import pyomo.environ as pyo

# Instances are regenerable from SEED by construction, so they live in an
# ignored sibling directory (`runs/*/`) rather than in the published history --
# the same rule adversary/.gitignore applies to fuzz/instances.jsonl.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2026-08-31_convex_sigma-census")

# Unchanged from the gh#875 census, so instance k here is instance k there.
SEED = 875
CONDS = [1e2, 1e4, 1e6, 1e8, 1e10, 1e12]
MAGS = [1e-3, 1.0, 1e3]
NS = [2, 5]
ROTS = [False, True]


def exact_solve(P, c):
    """`x` solving `P x = -c` in exact rational arithmetic.

    Gaussian elimination with partial pivoting over `Fraction`. Every float64 is
    exactly a rational, so nothing here rounds: the result is *the* optimum of
    the realised problem, not an estimate of it. `P` is symmetric positive
    definite, so a pivot can only vanish on a singular matrix.
    """
    n = len(c)
    a = [[Fraction(P[i][j]) for j in range(n)] + [Fraction(-c[i])] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if a[piv][col] == 0:
            raise ZeroDivisionError(f"singular at column {col}")
        a[col], a[piv] = a[piv], a[col]
        inv = a[col][col]
        for r in range(col + 1, n):
            f = a[r][col] / inv
            if f:
                for j in range(col, n + 1):
                    a[r][j] -= f * a[col][j]
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        s = a[i][n] - sum(a[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / a[i][i]
    return x


def assert_nl_roundtrip(path, P, c, n):
    """The `.nl` must carry `P` and `c` bit-for-bit, or the reference is wrong.

    Pyomo drops coefficients of `0.0` and `1.0` from the expression tree, so the
    objective literals are compared as a multiset against the values that are
    actually written. The gradient block carries every entry of `c`.
    """
    txt = open(path).read()
    lits = [float(v) for v in re.findall(r"^n(-?[\d.eE+-]+)$", txt, re.M)]
    assert lits and lits[0] == 0.5, f"{path}: expected a leading 0.5 factor"
    want = sorted(P[i][j] for i in range(n) for j in range(n) if P[i][j] not in (0.0, 1.0))
    got = sorted(v for v in lits[1:] if v not in (0.0, 1.0))
    assert got == want, f"{path}: objective coefficients did not round-trip"
    block = txt.split(f"G0 {n}\n")[1].strip().splitlines()
    grad = [float(line.split()[1]) for line in block]
    assert grad == list(c), f"{path}: gradient coefficients did not round-trip"


def build():
    rng = np.random.default_rng(SEED)
    cases = []
    for cond, mag, n, rot in itertools.product(CONDS, MAGS, NS, ROTS):
        eig = np.logspace(0, np.log10(cond), n) * mag
        Q = np.linalg.qr(rng.standard_normal((n, n)))[0] if rot else np.eye(n)
        P = Q @ np.diag(eig) @ Q.T
        P = 0.5 * (P + P.T)
        t = rng.uniform(-3.0, 3.0, n)
        cases.append(dict(cond=cond, mag=mag, n=n, rot=rot, P=P, t=t))
    return cases


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = []
    for k, cs in enumerate(build()):
        P, t, n = cs["P"], cs["t"], cs["n"]
        c = -P @ t

        m = pyo.ConcreteModel()
        m.I = pyo.RangeSet(0, n - 1)
        m.x = pyo.Var(m.I, initialize=0.0)
        m.obj = pyo.Objective(
            expr=0.5 * sum(P[i, j] * m.x[i] * m.x[j] for i in range(n) for j in range(n))
            + sum(c[i] * m.x[i] for i in range(n))
        )
        path = os.path.join(OUT, f"c{k:03d}.nl")
        m.write(path, io_options={"symbolic_solver_labels": False})

        Pl = P.tolist()
        cl = c.tolist()
        assert_nl_roundtrip(path, Pl, cl, n)
        xstar = exact_solve(Pl, cl)

        # How far the old reference was from the true one, in the census's own
        # relative-error units. This is the bug of gh#882 as a number, per
        # instance, and `run.py` prints it beside each result.
        scale = max(1, max(abs(v) for v in xstar))
        drift = float(max(abs(xstar[i] - Fraction(t[i])) for i in range(n)) / scale)

        meta.append(
            dict(
                k=k,
                cond=cs["cond"],
                mag=cs["mag"],
                n=n,
                rot=cs["rot"],
                # exact, as num/den strings -- float64 would reintroduce the
                # very rounding this file exists to remove.
                xstar=[[str(v.numerator), str(v.denominator)] for v in xstar],
                xstar_float=[float(v) for v in xstar],
                t=list(map(float, t)),
                old_reference_drift=drift,
            )
        )

    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"wrote {len(meta)} instances with exact references")
    worst = {}
    for e in meta:
        worst[e["cond"]] = max(worst.get(e["cond"], 0.0), e["old_reference_drift"])
    print("\nhow far `x* = t` was from the realised optimum (the gh#882 bug):")
    for cond in sorted(worst):
        print(f"  cond {cond:8.0e}   up to {worst[cond]:.2e}")


if __name__ == "__main__":
    main()
