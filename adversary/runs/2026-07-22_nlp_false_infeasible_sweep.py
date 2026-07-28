"""How often does POUNCE declare local infeasibility on a solvable problem?

Family: nlp   Class: false LocalInfeasibility verdict (prevalence)

Rapid infeasibility detection fires when BOTH
  * the constraint violation is bounded away from zero, and
  * the infeasibility stationarity ||J^T c|| / max(1, ||c||) is ~zero.

The two halves are evaluated in different spaces: the violation on the UNSCALED
residual (pounce#173), the stationarity on the SCALED one. Under an aggressive
constraint scaling `dc`, the stationarity carries a factor dc^2 and collapses
toward zero while the violation stays large -- so both gates pass at a point that
is not remotely stationary for the infeasibility. HS13 from x0=(1e4,1e4) is the
worked example: dc ~ 3.3e-7, unscaled violation 0.51, scaled stationarity ~5e-14.

This sweep measures how often that happens across the corpus, by comparing the
default against `nlp_scaling_method=none` (which removes the mismatch by removing
the scaling). A model that reports INFEASIBLE by default but SOLVES unscaled is a
candidate false verdict.

A false infeasibility verdict is worse than a false unbounded one for a
branch-and-bound driver: it prunes a node that may contain the optimum, silently.
"""

import concurrent.futures as cf
import glob
import os
import subprocess
import sys

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
CORPUS = os.path.expanduser("~/.cache/discopt/minlplib/current/nl")


def run(nl, extra, tag):
    sol = f"/tmp/inf_{tag}_{os.path.basename(nl)}.sol"
    try:
        p = subprocess.run(
            [POUNCE, nl, sol, "-AMPL", "max_wall_time=10", *extra],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None, None
    res = None
    if os.path.exists(sol):
        for line in reversed(open(sol).read().strip().splitlines()):
            if line.startswith("objno"):
                res = int(line.split()[-1])
                break
        os.remove(sol)
    obj = None
    for line in (p.stdout + p.stderr).splitlines():
        if line.strip().startswith("Objective"):
            try:
                obj = float(line.split()[-1])
            except ValueError:
                pass
    return res, obj


def band(res):
    if res is None:
        return "none"
    if 0 <= res < 100:
        return "SOLVED"
    if 100 <= res < 200:
        return "ACCEPTABLE"
    if 200 <= res < 300:
        return "INFEASIBLE"
    if 300 <= res < 400:
        return "UNBOUNDED"
    return "LIMIT/FAIL"


def one(nl):
    d_res, d_obj = run(nl, [], "d")
    if band(d_res) != "INFEASIBLE":
        return None  # only models the default calls infeasible are interesting
    n_res, n_obj = run(nl, ["nlp_scaling_method=none"], "n")
    return (os.path.basename(nl), band(d_res), band(n_res), d_obj, n_obj)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1600
    files = sorted(glob.glob(os.path.join(CORPUS, "*.nl")))
    files = [f for f in files if os.path.getsize(f) < 200_000][:limit]
    print(f"scanning {len(files)} models for INFEASIBLE verdicts", flush=True)

    hits = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(one, files)):
            if r:
                hits.append(r)
            if (i + 1) % 200 == 0:
                print(f"  ... {i+1}/{len(files)}  ({len(hits)} infeasible so far)", flush=True)

    print()
    print(f"models the default calls INFEASIBLE: {len(hits)}")
    suspicious = [h for h in hits if h[2] in ("SOLVED", "ACCEPTABLE")]
    print(f"  of those, SOLVED/ACCEPTABLE with nlp_scaling_method=none: {len(suspicious)}")
    print()
    if hits:
        print(f"  {'model':<32} {'default':<12} {'unscaled':<12} {'obj (unscaled)':>18}")
        for n, d, u, do, uo in sorted(hits, key=lambda h: h[2] != "SOLVED"):
            mark = "  <-- candidate false verdict" if u in ("SOLVED", "ACCEPTABLE") else ""
            print(f"  {n:<32} {d:<12} {u:<12} {str(uo):>18}{mark}")
    print()
    print("VERDICT: FAIL (false infeasibility)" if suspicious else "VERDICT: PASS")


if __name__ == "__main__":
    main()
