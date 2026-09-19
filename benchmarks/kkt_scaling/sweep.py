#!/usr/bin/env python3
"""KKT scaling sweep: where does superlinear solve time come from?

Structured-KKT decomposition, Phase 0b. Solves a *family* of .nl instances that
differ in one size parameter (the number of blocks, e.g. horizon hours) under
several fill-reducing orderings, and fits an exponent in that parameter to each
quantity that could carry the growth:

  iterations        IPM iterations (restoration rows included)
  factor_nnz        nnz(L) of the final main-solve factorization
  factor_flops      FERAL work proxy, per factorization
  delayed_per_fac   delayed-column entries per factorization
  fac_per_iter      factorization seconds per iteration (main + restoration)
  other_per_iter    everything else per iteration (wall - factorization)
  wall              total wall-clock seconds

Each maps to a row of the attribution table in the structured-KKT plan: fill,
pivoting, iterations, or other per-iteration work. Only the first two are
things a block-structured KKT solve can change.

Usage
-----
  sweep.py run  --bin PATH --out DIR [--ordering auto metis ...]
                [--timeout S] [--opt KEY=VALUE ...] INSTANCE.nl ...
  sweep.py fit  DIR [--by size|n]

`run` writes DIR/<instance>__<ordering>.json (the solve report) and appends one
row per solve to DIR/runs.csv. The size parameter is the integer after the last
`_H` / `_T` / `_K` in the file name (e.g. gaslib40_H048.nl -> 48); pass
instances whose names follow that. `fit` reads runs.csv and prints a
least-squares slope of log(metric) against log(size) (or log(n)) per ordering,
over the solves that converged.

Nothing here is gas-specific; it needs only the pounce CLI and the linear-solver
summary its --json-output report carries.
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time

FIELDS = [
    "instance", "size", "ordering", "status", "n", "m", "iterations",
    "resto_calls", "wall", "n_factors", "fac_secs", "resto_fac_secs",
    "factor_nnz", "factor_flops_total", "delayed_total", "two_by_two_total",
    "tiny_total", "max_front_rows", "peak_bytes", "ordering_used",
]

METRICS = {
    "iterations": lambda r: r["iterations"],
    "factor_nnz": lambda r: r["factor_nnz"],
    "factor_flops": lambda r: r["factor_flops_total"] / max(r["n_factors"], 1),
    "delayed_per_fac": lambda r: r["delayed_total"] / max(r["n_factors"], 1),
    "fac_per_iter": lambda r: (r["fac_secs"] + r["resto_fac_secs"]) / max(r["iterations"], 1),
    "other_per_iter": lambda r: max(r["wall"] - r["fac_secs"] - r["resto_fac_secs"], 0.0)
    / max(r["iterations"], 1),
    "wall": lambda r: r["wall"],
}

CONVERGED = {"SolveSucceeded", "SolvedToAcceptableLevel"}
# A solve stopped by the time limit still paid for every iteration it ran, so
# the per-iteration and per-factorization metrics stay valid; only the totals
# (iterations, wall) need convergence to mean anything.
TIME_LIMITED = {"MaximumWallTimeExceeded", "MaximumCpuTimeExceeded"}
TOTALS = {"iterations", "wall"}


def size_of(path):
    m = re.findall(r"_[HTK](\d+)", os.path.basename(path))
    if not m:
        sys.exit(f"cannot read a size parameter from {path!r} (want ..._H048.nl)")
    return int(m[-1])


def row_from_report(instance, size, ordering, report, wall_fallback):
    ls = report.get("linear_solver") or {}
    resto = ls.get("restoration") or {}
    st = report.get("statistics", {})
    pb = report.get("problem", {})
    # Delayed / 2x2 / tiny and flops are totals over main + restoration: both
    # are factorizations the solve paid for.
    tot = lambda k: (ls.get(k) or 0) + (resto.get(k) or 0)  # noqa: E731
    return {
        "instance": instance,
        "size": size,
        "ordering": ordering,
        "status": report.get("solution", {}).get("status", "?"),
        "n": pb.get("n_variables", 0),
        "m": pb.get("n_constraints", 0),
        "iterations": st.get("iteration_count", 0),
        "resto_calls": st.get("restoration_calls", 0),
        "wall": st.get("total_wallclock_time_secs") or wall_fallback,
        "n_factors": tot("n_factors"),
        "fac_secs": ls.get("total_factor_secs") or 0.0,
        "resto_fac_secs": resto.get("total_factor_secs") or 0.0,
        "factor_nnz": ls.get("last_nnz_l") or 0,
        "factor_flops_total": tot("total_factor_flops"),
        "delayed_total": tot("total_delayed_cols"),
        "two_by_two_total": tot("total_two_by_two"),
        "tiny_total": tot("total_n_tiny"),
        "max_front_rows": ls.get("last_max_front_rows") or 0,
        "peak_bytes": ls.get("last_peak_bytes") or 0,
        "ordering_used": ls.get("last_ordering") or "",
    }


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, "runs.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        # Smallest first, so a timeout on the largest instance costs least.
        for nl in sorted(a.instances, key=size_of):
            size = size_of(nl)
            stem = os.path.splitext(os.path.basename(nl))[0]
            for ordering in a.ordering:
                rep = os.path.join(a.out, f"{stem}__{ordering}.json")
                sol = os.path.join(a.out, f"{stem}__{ordering}.sol")
                # pounce's own limit sits under the harness timeout, so a
                # long solve still writes its report instead of being killed.
                cmd = [a.bin, nl, sol, "--json-output", rep, "print_level=0",
                       f"feral_ordering={ordering}",
                       f"max_wall_time={max(a.timeout - 120.0, 60.0)}", *a.opt]
                t0 = time.time()
                try:
                    subprocess.run(cmd, capture_output=True, timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    pass
                wall = time.time() - t0
                if not os.path.exists(rep):
                    print(f"{stem:>22} {ordering:>9}  no report ({wall:.0f}s)", flush=True)
                    continue
                with open(rep) as f:
                    row = row_from_report(stem, size, ordering, json.load(f), wall)
                w.writerow(row)
                fh.flush()
                print(
                    f"{stem:>22} {ordering:>9}  {row['status']:<24} it={row['iterations']:<5} "
                    f"wall={row['wall']:8.1f}s  fac={row['fac_secs'] + row['resto_fac_secs']:8.1f}s  "
                    f"nnzL={row['factor_nnz']}",
                    flush=True,
                )


def slope(xs, ys):
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 2:
        return float("nan"), len(pts)
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    return (sxy / sxx if sxx > 0 else float("nan")), len(pts)


def cmd_fit(a):
    with open(os.path.join(a.dir, "runs.csv")) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in FIELDS[4:-1]:
            if k != "status":
                r[k] = float(r[k])
        r["size"] = float(r["size"])
    orderings = sorted({r["ordering"] for r in rows})
    x_key = "size" if a.by == "size" else "n"
    print(
        f"exponent of each metric in {x_key} (log-log slope; count in parentheses).\n"
        "iterations and wall use converged solves only; the per-iteration and\n"
        "per-factorization metrics also use time-limited ones.\n"
    )
    print(f"{'metric':<16}" + "".join(f"{o:>14}" for o in orderings))
    for name, fn in METRICS.items():
        cells = []
        for o in orderings:
            ok = CONVERGED if name in TOTALS else CONVERGED | TIME_LIMITED
            rs = sorted(
                (r for r in rows if r["ordering"] == o and r["status"] in ok),
                key=lambda r: r[x_key],
            )
            s, k = slope([r[x_key] for r in rs], [fn(r) for r in rs])
            cells.append(f"{s:>10.2f} ({k})" if not math.isnan(s) else f"{'-':>14}")
        print(f"{name:<16}" + "".join(cells))
    dropped = [r for r in rows if r["status"] not in CONVERGED]
    if dropped:
        print("\nnot converged (excluded from the totals; time-limited ones still\n"
              "count toward the per-iteration metrics):")
        for r in dropped:
            print(f"  {r['instance']} {r['ordering']}: {r['status']} it={int(r['iterations'])}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bin", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--ordering", nargs="+", default=["auto", "metis", "auto_race"])
    r.add_argument("--timeout", type=float, default=3600.0)
    r.add_argument("--opt", nargs="*", default=[], help="extra KEY=VALUE solver options")
    r.add_argument("instances", nargs="+")
    f = sub.add_parser("fit")
    f.add_argument("dir")
    f.add_argument("--by", choices=["size", "n"], default="size")
    a = p.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_fit(a)


if __name__ == "__main__":
    main()
