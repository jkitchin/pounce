#!/usr/bin/env python3
"""Compare the NLP filter-IPM solver against the convex LP/QP IPM on a
suite of .nl files.

For each problem we solve it twice through the same pounce binary:
  - solver_selection=nlp        (the Ipopt-derived filter line-search IPM)
  - solver_selection=<lp-ipm|qp-ipm>  (the convex/conic HSDE IPM, pounce-convex)

and compare final objective, iteration count, wall-clock, and status,
using each solver's --json-output report (uniform schema across paths).

Usage:
  compare_solvers.py <bin> <nl_dir> <convex_sel> <out_json>
    convex_sel in {lp-ipm, qp-ipm}
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def solve(bin_path, nl, selection, time_limit=120):
    """Run one solve; return (record_dict, wall_seconds)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    start = time.time()
    try:
        subprocess.run(
            [bin_path, nl, f"solver_selection={selection}",
             "--json-output", out],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=time_limit,
        )
    except subprocess.TimeoutExpired:
        return {"status": "TimeOut", "objective": None,
                "iteration_count": None, "wall": time_limit}, time_limit
    wall = time.time() - start
    try:
        with open(out) as fh:
            data = json.load(fh)
        sol = data.get("solution", {})
        stat = data.get("statistics", {})
        return {
            "status": sol.get("status"),
            "objective": stat.get("final_objective", sol.get("objective")),
            "iteration_count": stat.get("iteration_count"),
            "wall": stat.get("total_wallclock_time_secs", wall),
        }, wall
    except Exception as e:
        return {"status": f"ParseError:{e}", "objective": None,
                "iteration_count": None, "wall": wall}, wall
    finally:
        Path(out).unlink(missing_ok=True)


def main():
    bin_path, nl_dir, convex_sel, out_json = sys.argv[1:5]
    nls = sorted(Path(nl_dir).glob("*.nl"))
    rows = []
    print(f"{'problem':<14}{'nlp_obj':>16}{'cvx_obj':>16}"
          f"{'nlp_it':>8}{'cvx_it':>8}{'nlp_s':>9}{'cvx_s':>9}{'  reldiff':>12}")
    for nl in nls:
        name = nl.stem
        nlp, _ = solve(bin_path, str(nl), "nlp")
        cvx, _ = solve(bin_path, str(nl), convex_sel)
        a, b = nlp["objective"], cvx["objective"]
        if a is not None and b is not None:
            denom = max(abs(a), abs(b), 1e-10)
            reldiff = abs(a - b) / denom
        else:
            reldiff = None
        rows.append({"name": name, "nlp": nlp, "convex": cvx,
                     "reldiff": reldiff})
        fa = f"{a:.6e}" if a is not None else "n/a"
        fb = f"{b:.6e}" if b is not None else "n/a"
        fr = f"{reldiff:.2e}" if reldiff is not None else "n/a"
        print(f"{name:<14}{fa:>16}{fb:>16}"
              f"{str(nlp['iteration_count']):>8}{str(cvx['iteration_count']):>8}"
              f"{nlp['wall']:>9.3f}{cvx['wall']:>9.3f}{fr:>12}")
    with open(out_json, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
