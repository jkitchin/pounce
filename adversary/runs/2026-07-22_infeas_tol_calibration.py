"""Calibrate `infeas_stationarity_tol` for the UNSCALED stationarity measure.

Separability analysis on the targeted cases says the unscaled measure admits a
tolerance in roughly [1.0e-2, 1.0):

    ceq   (must fire)      fire-threshold 1.006e-2
    cubic (must fire)      fire-threshold 3.407e-3
    hs13  (must NOT fire)  fire-threshold 1.000e+0

but three problems is not a calibration. The risk of raising the tolerance from
1e-8 is FALSE POSITIVES: declaring local infeasibility on solvable models, which
is the dangerous direction (a B&B driver prunes a node that may hold the
optimum).

So sweep candidate tolerances across the corpus and count how many models newly
report INFEASIBLE relative to the pre-fix binary. The acceptable answer is a
tolerance where the targeted cases behave AND no corpus model newly reports
infeasibility.
"""

import concurrent.futures as cf
import glob
import os
import subprocess
import sys

HEAD = "/Users/jkitchin/projects/pounce/target/release/pounce"
PREFIX = "/tmp/pounce-prefix/target/release/pounce"
CORPUS = os.path.expanduser("~/.cache/discopt/minlplib/current/nl")
TOLS = ["1e-2", "3e-2", "1e-1", "3e-1"]


def run(binary, nl, extra, tag):
    sol = f"/tmp/cal_{tag}_{os.path.basename(nl)}.sol"
    try:
        subprocess.run(
            [binary, nl, sol, "-AMPL", "max_wall_time=10", *extra],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    res = None
    if os.path.exists(sol):
        for line in reversed(open(sol).read().strip().splitlines()):
            if line.startswith("objno"):
                res = int(line.split()[-1])
                break
        os.remove(sol)
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
    base = run(PREFIX, nl, [], "base")
    out = {"base": base}
    for t in TOLS:
        out[t] = run(HEAD, nl, [f"infeas_stationarity_tol={t}"], t.replace("-", ""))
    return os.path.basename(nl), out


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    files = sorted(glob.glob(os.path.join(CORPUS, "*.nl")))
    files = [f for f in files if os.path.getsize(f) < 200_000][:limit]
    print(f"calibrating over {len(files)} models; tolerances {TOLS}", flush=True)

    rows = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(one, files)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  ... {i+1}/{len(files)}", flush=True)

    base_infeas = sum(1 for _, o in rows if o["base"] == "INFEASIBLE")
    print()
    print(f"pre-fix INFEASIBLE verdicts: {base_infeas} / {len(rows)}")
    print()
    print(f"  {'tol':<8} {'INFEAS':>8} {'new-false-pos':>14} {'lost':>6}   examples")
    for t in TOLS:
        newly = [n for n, o in rows if o["base"] != "INFEASIBLE" and o[t] == "INFEASIBLE"]
        lost = [n for n, o in rows if o["base"] == "INFEASIBLE" and o[t] != "INFEASIBLE"]
        total = sum(1 for _, o in rows if o[t] == "INFEASIBLE")
        ex_s = ", ".join(newly[:3]) if newly else ""
        print(f"  {t:<8} {total:>8} {len(newly):>14} {len(lost):>6}   {ex_s}")
        if lost:
            print(f"           lost: {', '.join(lost[:6])}")


if __name__ == "__main__":
    main()
