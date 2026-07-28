"""Adversary regression sweep: the last 5 merged PRs, over the MINLPLib corpus.

Family: nlp   Class: differential / regression sweep
Baseline: f5aea43 (right after PR #249, i.e. BEFORE the last five PRs)
Head:     current main (includes #251 docs, #250, #253, #258, #256/#255)

WHY THIS SWEEP. Four of the five PRs change TERMINATION logic on the shared
numerical backbone, so they can affect every solve, not just their own repro:

  #250  dual-infeasibility divergence guard   -> can fire DivergingIterates
  #253  divergence guard needs non-decelerating descent
  #258  barrier floor scaled by |obj_scaling_factor|  (MINE -- lowers the floor
        whenever the objective is deflated)
  #256  proactive wall-time deadline checks around KKT factorizations

#258 is the one I am most suspicious of, because the floor it lowers was
introduced deliberately: without it "mu collapses to mu_min (1e-11) while primal
infeasibility is still large -- observed on SSINE/DECONVBNE, where the next
direction is dominated by ill-conditioned barrier terms and the line search
stalls." My change lowers that floor on exactly the ill-scaled problems the
comment is about. If it reintroduces those stalls, this sweep will show
problems that used to solve and now do not.

METHOD. Run both binaries over the corpus with identical options and a hard
time limit, and diff the AMPL solve_result_num band and the objective. Report:
  - REGRESSION: baseline solved, head did not
  - IMPROVEMENT: head solved, baseline did not
  - OBJ_DIFF:   both solved but objectives disagree beyond tolerance
"""

import concurrent.futures as cf
import glob
import os
import subprocess
import sys
import time

BEFORE = "/tmp/pounce-prefix/target/release/pounce"
AFTER = "/Users/jkitchin/projects/pounce/target/release/pounce"
CORPUS = os.path.expanduser("~/.cache/discopt/minlplib/current/nl")
TIME_LIMIT = "10"
WORKERS = 8


def run(binary, nl, tag):
    sol = f"/tmp/sweep_{tag}_{os.path.basename(nl)}.sol"
    try:
        p = subprocess.run(
            [binary, nl, sol, "-AMPL", f"max_wall_time={TIME_LIMIT}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return ("HARD_TIMEOUT", None, 60.0)
    res, obj = None, None
    if os.path.exists(sol):
        txt = open(sol).read()
        for line in reversed(txt.strip().splitlines()):
            if line.startswith("objno"):
                res = int(line.split()[-1])
                break
        os.remove(sol)
    for line in (p.stdout + p.stderr).splitlines():
        if line.strip().startswith("Objective"):
            try:
                obj = float(line.split()[-1])
            except ValueError:
                pass
    return (band(res), obj, 0.0)


def band(res):
    if res is None:
        return "none"
    if 0 <= res < 100:
        return "SOLVED"
    if 100 <= res < 200:
        return "SOLVED?"
    if 200 <= res < 300:
        return "INFEAS"
    if 300 <= res < 400:
        return "UNBOUNDED"
    return "LIMIT/FAIL"


def one(nl):
    t0 = time.perf_counter()
    b = run(BEFORE, nl, "b")
    tb = time.perf_counter() - t0
    t0 = time.perf_counter()
    a = run(AFTER, nl, "a")
    ta = time.perf_counter() - t0
    return (os.path.basename(nl), b, a, tb, ta)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    files = sorted(glob.glob(os.path.join(CORPUS, "*.nl")))
    # Keep the sweep to small models so the whole run stays bounded.
    files = [f for f in files if os.path.getsize(f) < 200_000][:limit]
    print(f"sweeping {len(files)} models; baseline=f5aea43 head=main", flush=True)

    results = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(one, files)):
            results.append(r)
            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(files)}", flush=True)

    regress, improve, objdiff = [], [], []
    for name, (bb, bo, _), (ab, ao, _), tb, ta in results:
        if bb == "SOLVED" and ab != "SOLVED":
            regress.append((name, bb, ab, bo, ao))
        elif ab == "SOLVED" and bb != "SOLVED":
            improve.append((name, bb, ab, bo, ao))
        # Compare objectives whenever BOTH sides returned a point they call
        # solved at some level -- including the acceptable band (objno 100-199).
        # Restricting this to the strict band hid the autocorr_bern55-06
        # regression, whose 1.8 % objective loss happened entirely inside the
        # acceptable band on both sides.
        elif bb in ("SOLVED", "SOLVED?") and ab in ("SOLVED", "SOLVED?") \
                and bo is not None and ao is not None:
            denom = max(1.0, abs(bo))
            if abs(bo - ao) / denom > 1e-6:
                objdiff.append((name, bo, ao, abs(bo - ao) / denom))

    print()
    print(f"total={len(results)}  regressions={len(regress)}  improvements={len(improve)}  obj-diffs={len(objdiff)}")
    print()
    if regress:
        print("REGRESSIONS (baseline solved, head did not):")
        for n, bb, ab, bo, ao in regress:
            print(f"  {n:<28} {bb} -> {ab}   obj {bo} -> {ao}")
    if improve:
        print("IMPROVEMENTS (head solved, baseline did not):")
        for n, bb, ab, bo, ao in improve[:40]:
            print(f"  {n:<28} {bb} -> {ab}   obj {bo} -> {ao}")
    if objdiff:
        print("OBJECTIVE DIFFERENCES (both solved):")
        for n, bo, ao, rel in sorted(objdiff, key=lambda r: -r[3])[:40]:
            print(f"  {n:<28} {bo!r} -> {ao!r}   rel={rel:.3e}")

    print()
    print("VERDICT: FAIL (regressions found)" if regress else "VERDICT: PASS (no regressions)")


if __name__ == "__main__":
    main()
