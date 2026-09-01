"""Score one pounce binary against the census. See the `_gen.py` companion.

    python3 2026-08-31_convex_sigma-census_score.py <path-to-pounce> [--threshold 1e-6]

Relative error is computed in exact rational arithmetic against the exact
optimum of the realised problem, so the number printed is the solver's error and
nothing else. The `drift` column is how far the old `x* = t` reference sat from
that optimum -- any row whose error is at or below its own drift is a row the
pre-gh#882 census could not have resolved.

The `err/eps.cond` column is the other floor, and the one that turned out to
matter: `sigma_forward_error_is_small` accumulates `||delta||` in float64, so it
cannot resolve a relative error below `eps * cond` and cannot reject what it
cannot see. A flagged row below 1.0 there is a genuine error that no
double-precision forward-error guard can be asked to catch.
"""

import collections
import json
import os
import subprocess
import sys
from fractions import Fraction

EPS = 2.220446049250313e-16

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2026-08-31_convex_sigma-census")


def read_sol(path, n):
    lines = open(path).read().splitlines()
    i = next(j for j, ln in enumerate(lines) if ln.strip() == "Options")
    j = i + 1
    j += 1 + int(lines[j])
    vals = []
    for ln in lines[j:]:
        try:
            vals.append(float(ln.strip()))
        except ValueError:
            break
    return vals[-n:]


def main():
    binary = sys.argv[1]
    threshold = float(sys.argv[sys.argv.index("--threshold") + 1]) if "--threshold" in sys.argv else 1e-6
    meta = json.load(open(os.path.join(OUT, "meta.json")))

    bad = 0
    by_cond = collections.defaultdict(lambda: [0, 0, 0.0])
    rows = []
    for e in meta:
        nl = os.path.join(OUT, f"c{e['k']:03d}.nl")
        sol = nl[:-3] + ".sol"
        if os.path.exists(sol):
            os.remove(sol)
        r = subprocess.run([binary, nl], capture_output=True, text=True, timeout=120)
        ok = "Optimal Solution Found" in r.stdout
        iters = next(
            (ln.split(":")[-1].strip() for ln in r.stdout.splitlines() if "Number of Iterations" in ln),
            "-",
        )
        xstar = [Fraction(int(a), int(b)) for a, b in e["xstar"]]
        x = [Fraction(v) for v in read_sol(sol, e["n"])]
        scale = max(1, max(abs(v) for v in xstar))
        rel = float(max(abs(x[i] - xstar[i]) for i in range(e["n"])) / scale)
        wrong = ok and rel > threshold
        bad += wrong
        s = by_cond[e["cond"]]
        s[0] += wrong
        s[1] += 1
        s[2] = max(s[2], rel)
        rows.append((e["k"], e["cond"], e["mag"], e["n"], e["rot"], rel, e["old_reference_drift"], iters, wrong))

    print(binary)
    print(f"claimed-optimal-but-wrong (rel > {threshold:g}): {bad} / {len(meta)}")
    print(f"{'cond':>9} {'wrong':>7} {'max rel err':>12}")
    for cond in sorted(by_cond):
        w, t, worst = by_cond[cond]
        print(f"{cond:>9.0e} {w:>3d}/{t:<3d} {worst:>12.3e}")

    flagged = [r for r in rows if r[8]]
    if flagged:
        print(f"\n{'k':>4} {'cond':>8} {'mag':>7} {'n':>2} {'rot':>5} {'rel err':>11} {'old drift':>11} {'err/drift':>9} {'err/eps.cond':>13} {'it':>4}")
        for k, cond, mag, n, rot, rel, drift, iters, _ in flagged:
            ratio = rel / drift if drift else float("inf")
            floor = rel / (EPS * cond)
            print(f"{k:>4} {cond:>8.0e} {mag:>7.0e} {n:>2} {str(rot):>5} {rel:>11.3e} {drift:>11.3e} {ratio:>9.2f} {floor:>13.2f} {iters:>4}")


if __name__ == "__main__":
    main()
