#!/usr/bin/env python3
"""Run the CBLIB exponential/power-cone conic tier through POUNCE.

Unlike the other suites (which are `.nl`-driven through the main `pounce`
binary), CBLIB ships *conic* programs in Conic Benchmark Format (`.cbf`),
solved through POUNCE's convex conic driver via the `pounce_cblib` binary.
Each instance is solved and recorded in the same schema the composite
report consumes:

    {solver, name, n, m, status, objective, iterations, solve_time}

Out:  benchmarks/cblib/pounce.json

By default this runs the small instances vendored with the repo (the
exp-cone GPs demb761/beck751/fang88 and a synthetic power-cone problem,
under crates/pounce-cli/tests/data/cblib). Point `--dir` at a folder of
additional `.cbf` files (e.g. a local CBLIB checkout) to run more.

Run:  python3 benchmarks/cblib/run_cblib.py [--dir PATH] [--detail full]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BIN = os.path.join(ROOT, "target", "release", "pounce_cblib")
VENDORED = os.path.join(
    ROOT, "crates", "pounce-cli", "tests", "data", "cblib"
)


def status_underscored(s: str) -> str:
    """`SolveSucceeded` -> `Solve_Succeeded` (the composite-report form)."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s)


def build_binary() -> None:
    print("Building pounce_cblib (release)…", file=sys.stderr)
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "pounce_cblib"],
        cwd=ROOT,
        check=True,
    )


def instances(extra_dir):
    """Yield (name, path) for every .cbf to run, vendored first."""
    seen = set()
    for d in [VENDORED] + ([extra_dir] if extra_dir else []):
        if not d or not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".cbf") and fn not in seen:
                seen.add(fn)
                yield fn[:-4], os.path.join(d, fn)


def run_one(name, path, detail):
    """Solve one instance; return the standard-schema record (or None)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    try:
        proc = subprocess.run(
            [BIN, path, "--json-output", out, "--json-detail", detail],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            print(f"  {name}: no report ({proc.stderr.strip()})", file=sys.stderr)
            return None
        with open(out) as f:
            r = json.load(f)
        return {
            "solver": "pounce",
            "name": name,
            "n": r["problem"]["n_variables"],
            "m": r["problem"]["n_constraints"],
            "status": status_underscored(r["solution"]["status"]),
            "objective": r["solution"]["objective"],
            "iterations": r["statistics"]["iteration_count"],
            "solve_time": r["statistics"]["total_wallclock_time_secs"],
        }
    finally:
        if os.path.exists(out):
            os.remove(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", help="extra directory of .cbf instances")
    ap.add_argument("--detail", default="summary", choices=["summary", "full"])
    ap.add_argument("--no-build", action="store_true", help="skip cargo build")
    args = ap.parse_args()

    if not args.no_build:
        build_binary()
    if not os.path.exists(BIN):
        sys.exit(f"binary not found: {BIN} (drop --no-build to build it)")

    records = []
    for name, path in instances(args.dir):
        rec = run_one(name, path, args.detail)
        if rec is not None:
            records.append(rec)
            print(
                f"  {rec['name']:<20} {rec['status']:<20} "
                f"obj={rec['objective']:.6g}  iters={rec['iterations']}  "
                f"{rec['solve_time']:.3f}s"
            )

    out_path = os.path.join(HERE, "pounce.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
