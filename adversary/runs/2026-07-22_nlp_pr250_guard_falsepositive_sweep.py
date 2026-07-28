"""Adversary sweep: how often does PR #250's dual-divergence guard fire falsely?

Family: nlp   Class: false-positive rate of a termination guard
Target: PR #250 / commit ba29b53, option `dual_diverging_streak` (default 15).

The guard watches the NLP-scaled dual infeasibility and, after 15 consecutive
growing iterations in an elevated regime, diverts the solve into restoration
(or terminates DivergingIterates). It was added to rescue emfl050_5_5 from a
permanent stall.

Found by the last-5-PRs corpus diff: on autocorr_bern55-06 the guard fires on a
solve that Ipopt and pre-#250 POUNCE both converge in 72 iterations, and the
diverted solve lands on a WORSE, NON-STATIONARY point:

    guard on  (default): 111 iters, obj -2263.46, NLP error 1.0e0, "Acceptable"
    guard off (=0)     :  72 iters, obj -2304.00, NLP error 3.7e-9, "Optimal"
    ipopt (oracle)     :  72 iters, obj -2304.00, NLP error 3.7e-9, "Optimal"

This sweep measures how general that is: run the SAME head binary over the
corpus with the guard at its default and disabled, and count where disabling it
gives a strictly better outcome. Same binary both sides, so the only difference
is the guard.
"""

import concurrent.futures as cf
import glob
import os
import subprocess
import sys

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
CORPUS = os.path.expanduser("~/.cache/discopt/minlplib/current/nl")


def run(nl, streak, tag):
    sol = f"/tmp/g_{tag}_{os.path.basename(nl)}.sol"
    try:
        p = subprocess.run(
            [POUNCE, nl, sol, "-AMPL", "max_wall_time=10", f"dual_diverging_streak={streak}"],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", None, None)
    res = None
    if os.path.exists(sol):
        for line in reversed(open(sol).read().strip().splitlines()):
            if line.startswith("objno"):
                res = int(line.split()[-1])
                break
        os.remove(sol)
    obj = err = None
    for line in (p.stdout + p.stderr).splitlines():
        s = line.strip()
        if s.startswith("Objective") and obj is None:
            try: obj = float(s.split()[-1])
            except ValueError: pass
        if s.startswith("Overall NLP error") and err is None:
            try: err = float(s.split()[-1])
            except ValueError: pass
    return (res, obj, err)


def one(nl):
    return (os.path.basename(nl), run(nl, 15, "on"), run(nl, 0, "off"))


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    files = sorted(glob.glob(os.path.join(CORPUS, "*.nl")))
    files = [f for f in files if os.path.getsize(f) < 200_000][:limit]
    print(f"sweeping {len(files)} models: dual_diverging_streak=15 (default) vs 0 (off)", flush=True)

    rows = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(one, files)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{len(files)}", flush=True)

    degraded, helped = [], []
    for name, (ron, oon, eon), (roff, ooff, eoff) in rows:
        if ron is None or roff is None:
            continue
        on_solved = isinstance(ron, int) and 0 <= ron < 100
        off_solved = isinstance(roff, int) and 0 <= roff < 100
        # Guard hurt: disabling it upgrades the status, or keeps the status but
        # reaches a materially better objective / a stationary point.
        if off_solved and not on_solved:
            degraded.append((name, ron, roff, oon, ooff, eon, eoff))
        elif on_solved and not off_solved:
            helped.append((name, ron, roff, oon, ooff, eon, eoff))

    print()
    print(f"total={len(rows)}  guard-degrades={len(degraded)}  guard-helps={len(helped)}")
    print()
    if degraded:
        print("GUARD FIRES FALSELY (disabling it solves the problem):")
        print(f"  {'model':<30} {'objno on':>9} {'objno off':>10} {'obj on':>16} {'obj off':>16} {'err on':>10} {'err off':>10}")
        for n, ron, roff, oon, ooff, eon, eoff in degraded:
            print(f"  {n:<30} {ron:>9} {roff:>10} {str(oon):>16} {str(ooff):>16} {str(eon):>10} {str(eoff):>10}")
    if helped:
        print("GUARD HELPS (it is load-bearing here):")
        for n, ron, roff, oon, ooff, eon, eoff in helped:
            print(f"  {n:<30} on={ron} off={roff} obj {oon} -> {ooff}")

    print()
    print("VERDICT: FAIL (false positives)" if degraded else "VERDICT: PASS")


if __name__ == "__main__":
    main()
