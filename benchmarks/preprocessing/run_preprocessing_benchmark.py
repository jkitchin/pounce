#!/usr/bin/env python3
"""Compare pounce iteration counts with and without
`presolve_auxiliary` on an `.nl` problem.

Usage:

    python benchmarks/preprocessing/run_preprocessing_benchmark.py \
        crates/pounce-cli/tests/fixtures/parametric.nl

Optional flags:

    --pounce PATH       Path to the pounce binary. Defaults to
                        `target/release/pounce` then `cargo run
                        --release -p pounce-cli`.
    --tol VALUE         Solve tolerance, passed as `tol=VALUE`.
    --print-level N     Console verbosity for each solve.

The script runs the same `.nl` problem twice:
  * `presolve_auxiliary=no` (baseline);
  * `presolve_auxiliary=yes presolve_auxiliary_diagnostics=yes`.

It parses the per-run JSON solve report (written to a temp file)
and prints a table comparing iteration count, final objective, and
total wall time.

Issue: <https://github.com/jkitchin/pounce/issues/53>.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def default_pounce_path():
    """Find a pounce binary. Prefer a release build; fall back to
    `cargo run -- ...`."""
    repo = Path(__file__).resolve().parents[2]
    release = repo / "target" / "release" / "pounce"
    if release.exists():
        return [str(release)]
    debug = repo / "target" / "debug" / "pounce"
    if debug.exists():
        return [str(debug)]
    return ["cargo", "run", "--release", "--quiet", "-p", "pounce-cli", "--bin", "pounce", "--"]


def run_one(pounce_cmd, nl_path, json_path, options):
    """Run pounce once. Returns (elapsed_seconds, parsed_report,
    stderr_text)."""
    cmd = list(pounce_cmd) + [
        str(nl_path),
        "--json-output",
        str(json_path),
        "--json-detail",
        "summary",
    ] + options
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode not in (0, 1):
        print(f"pounce exited with code {proc.returncode}", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)
    with open(json_path) as f:
        report = json.load(f)
    return elapsed, report, proc.stderr


def fmt_row(label, iters, obj, wall, extra=""):
    return f"{label:<20} {iters:>8}  {obj:>16.9e}  {wall:>9.3f}s  {extra}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("nl_path", type=Path, help="Path to a `.nl` problem.")
    p.add_argument(
        "--pounce",
        type=str,
        default=None,
        help="Path to the pounce binary.",
    )
    p.add_argument("--tol", type=str, default=None, help="Solve tolerance.")
    p.add_argument(
        "--print-level", type=int, default=0, help="Console verbosity per run."
    )
    args = p.parse_args()

    if not args.nl_path.exists():
        sys.exit(f"error: {args.nl_path} does not exist")

    pounce_cmd = [args.pounce] if args.pounce else default_pounce_path()

    common = []
    if args.tol is not None:
        common.append(f"tol={args.tol}")
    if args.print_level:
        common.append(f"print_level={args.print_level}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        baseline_opts = common + ["presolve=yes", "presolve_auxiliary=no"]
        aux_opts = (
            common
            + [
                "presolve=yes",
                "presolve_auxiliary=yes",
                "presolve_auxiliary_diagnostics=yes",
            ]
        )

        bw, b_report, _ = run_one(pounce_cmd, args.nl_path, td / "baseline.json", baseline_opts)
        aw, a_report, a_err = run_one(pounce_cmd, args.nl_path, td / "aux.json", aux_opts)

    b_iters = b_report["statistics"]["iteration_count"]
    a_iters = a_report["statistics"]["iteration_count"]
    b_obj = b_report["statistics"]["final_objective"]
    a_obj = a_report["statistics"]["final_objective"]

    print(f"Problem: {args.nl_path}")
    print()
    print(f"{'config':<20} {'iters':>8}  {'final_obj':>16}  {'wall':>9}")
    print(fmt_row("baseline", b_iters, b_obj, bw))
    print(fmt_row("auxiliary=yes", a_iters, a_obj, aw))
    print()
    delta = b_iters - a_iters
    pct = (delta / b_iters * 100.0) if b_iters else 0.0
    print(f"iteration delta: {-delta:+d}  ({pct:+.1f}%)")

    # Reproduce the diagnostics line from stderr if present.
    for line in a_err.splitlines():
        if line.startswith("auxiliary-preprocessing:"):
            print(line)


if __name__ == "__main__":
    main()
