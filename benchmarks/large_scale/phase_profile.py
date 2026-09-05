#!/usr/bin/env python3
"""Phase profile on `laptime`: how much of a solve is spent finding the
active set, and how much converging on one?

`IterRecord::active_bounds` / `active_set_changes` (in the JSON solve
report at `--json-detail full`) count the bound and slack indices the
barrier treats as active, and how many changed class since the previous
row. This script runs `laptime` across a refinement path on both Hessian
legs and reduces those series to one line per run.

Why both legs. `laptime` declares an analytic Hessian, so the default
`exact` path is available to it -- and is not available to the FMU-backed
models this profile exists to reason about. `limited-memory` is their
analogue and is the leg to read; `exact` is the reference for what the
iteration count would be with real curvature.

Two rows are dropped from every run. At a cold start the bound
multipliers are `bound_mult_init_val`, one constant for every index,
rather than a dual estimate, so row 0 classifies nearly every index with
a small starting slack as active and row 1's change count is that
collapse. Neither is a measurement.

What the numbers do not carry: the activity split is taken in the
solver's scaled frame and is not scale-invariant, so counts are
comparable within a run and not across models (see the schema doc).
And `laptime`'s cost profile is not a collocation model's in general --
its AMPL-AD evaluations are ~9% of wall time here against ~77% in the
linear algebra, which is the reverse of an FMU-backed model. Iteration
counts and active-set shape transfer; wall-clock shares do not.

    python3 benchmarks/large_scale/phase_profile.py --out-dir /tmp/prof
"""

import argparse
import json
import os
import subprocess
import sys
import time

MESHES = [("0.08", 80), ("0.16", 160), ("0.32", 320)]
LEGS = ["exact", "limited-memory"]


def generate(out_dir, scale):
    d = os.path.join(out_dir, f"lap_{scale}")
    nl = os.path.join(d, "laptime.nl")
    if os.path.exists(nl):
        return nl
    subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "generate_nl.py"),
         "laptime", "--scale", scale, "--out-dir", d],
        check=True,
    )
    return nl


def solve(binary, nl, leg, report, max_iter, extra=()):
    t0 = time.time()
    subprocess.run(
        [binary, nl, "--json-output", report, "--json-detail", "full",
         "print_level=0", "sb=yes", f"max_iter={max_iter}",
         f"hessian_approximation={leg}", *extra],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return time.time() - t0


def profile(report):
    r = json.load(open(report))
    st, sol, rows = r["statistics"], r["solution"], r["iterations"]
    body = rows[2:]                      # see the module docstring
    ch = [g.get("active_set_changes") or 0 for g in body]
    n = len(ch)
    deciles = []
    for d in range(10):
        lo = d * n // 10
        hi = max((d + 1) * n // 10, lo + 1)
        seg = ch[lo:hi]
        deciles.append(sum(seg) / max(len(seg), 1))
    tail = ch[max(0, n - max(n // 10, 1)):]
    # Where the active set last moved splits the run: everything up to
    # and including that row is the approach, the rest is convergence on
    # a set that has stopped moving.
    last_move = max((i for i, c in enumerate(ch) if c > 0), default=-1)
    return {
        "status": sol["status"],
        "iters": st["iteration_count"],
        "objective": st["final_objective"],
        "wall": st["total_wallclock_time_secs"],
        "moves": sum(ch),
        "zero_pct": 100.0 * sum(1 for c in ch if c == 0) / n,
        "tail_mean": sum(tail) / len(tail),
        "active_end": body[-1]["active_bounds"],
        "approach": last_move + 1,
        "settled": n - last_move - 1,
        "deciles": deciles,
    }


def emit(results, key_label="N"):
    hdr = (f"{key_label:>5} {'leg':<16}{'status':<18}{'it':>5}{'objective':>12}"
           f"{'wall':>8}{'moves':>8}{'zero%':>7}{'tail':>7}{'appr':>6}{'settled':>8}")
    print(hdr)
    print("-" * len(hdr))
    for (key, leg), p in results.items():
        print(f"{key:>5} {leg:<16}{p['status'][:17]:<18}{p['iters']:>5}"
              f"{p['objective']:>12.4f}{p['wall']:>7.1f}s{p['moves']:>8}"
              f"{p['zero_pct']:>6.0f}%{p['tail_mean']:>7.1f}"
              f"{p['approach']:>6}{p['settled']:>8}")

    print("\nmean index-moves per iteration, by decile of the run:")
    print(f"{key_label:>5} {'leg':<16}" + "".join(f"{i + 1:>7}" for i in range(10)))
    for (key, leg), p in results.items():
        print(f"{key:>5} {leg:<16}" + "".join(f"{d:>7.0f}" for d in p["deciles"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/tmp/laptime-phase-profile")
    ap.add_argument("--binary", default="target/release/pounce")
    ap.add_argument("--max-iter", type=int, default=1200)
    ap.add_argument("--leg", action="append", choices=LEGS)
    ap.add_argument(
        "--history",
        help="comma-separated limited_memory_max_history values. Sweeps the "
             "curvature memory at one mesh (--mesh) instead of the mesh path: "
             "more pairs is the cheapest test of whether the settled tail is "
             "curvature-starved.",
    )
    ap.add_argument("--mesh", default="0.16",
                    help="scale for --history (default 0.16, i.e. N=160)")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    if args.history:
        nl = generate(args.out_dir, args.mesh)
        for h in [int(v) for v in args.history.split(",")]:
            report = os.path.join(args.out_dir, f"hist_{args.mesh}_{h}.json")
            wall = solve(args.binary, nl, "limited-memory", report,
                         args.max_iter, [f"limited_memory_max_history={h}"])
            if not os.path.exists(report):
                print(f"history={h}: no report after {wall:.0f}s", file=sys.stderr)
                continue
            results[(h, f"lbfgs m={h}")] = profile(report)
        emit(results, "m")
        return 0

    legs = args.leg or LEGS
    for scale, n_int in MESHES:
        nl = generate(args.out_dir, scale)
        for leg in legs:
            report = os.path.join(args.out_dir, f"prof_{scale}_{leg}.json")
            wall = solve(args.binary, nl, leg, report, args.max_iter)
            if not os.path.exists(report):
                print(f"N={n_int} {leg}: no report after {wall:.0f}s", file=sys.stderr)
                continue
            results[(n_int, leg)] = profile(report)

    emit(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
