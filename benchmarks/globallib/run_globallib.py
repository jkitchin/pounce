#!/usr/bin/env python3
"""GLOBALLib global-optimization benchmark harness for `pounce-global`.

Drives `pounce <model>.nl solver_selection=global` on the GLOBALLib subset that
has a *proven* global optimum (MINLPLib `=opt=`), and checks the **certified**
objective the spatial branch-and-bound solver returns against that ground truth.

Unlike the synthetic Rust suite (`crates/pounce-global/examples/benchmark.rs`),
this runs real AMPL `.nl` files through the same CLI path users hit, so it tests
the whole pipeline: parse -> classify -> bound-capping -> B&B -> certificate.

Ground truth lives in `optima.txt` (one `<stem> <objective>` per line, from
MINLPLib's `minlplib.solu`, `=opt=` entries only). The `.nl` files are supplied
via the bench-data tree (see README for the AMPL translation recipe).

Usage:
  run_globallib.py [--bin PATH] [--nl-dir DIR] [--timeout SECS]
                   [--max-vars N] [--out report.json] [stems...]

Default nl-dir: $POUNCE_BENCH_DATA/globallib/nl or
                ~/Dropbox/projects/pounce-bench-data/globallib/nl
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

# "POUNCE (global B&B, pounce-global): <msg>  obj=..  gap=..  nodes=N  peak_frontier=.."
RESULT_RE = re.compile(
    r"obj=(?P<obj>[-+0-9.eE]+)\s+gap=(?P<gap>[-+0-9.eE]+)\s+nodes=(?P<nodes>\d+)"
)
STATUS_RE = re.compile(r"pounce-global\):\s*(?P<msg>[^.]+\.)")


def default_nl_dir():
    env = os.environ.get("POUNCE_BENCH_DATA")
    if env:
        return Path(env) / "globallib" / "nl"
    return Path.home() / "Dropbox/projects/pounce-bench-data/globallib/nl"


def load_optima(path):
    opt = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stem, val = line.split()
        opt[stem] = float(val)
    return opt


def run_one(bin_path, nl, timeout):
    start = time.time()
    try:
        p = subprocess.run(
            [bin_path, str(nl), "solver_selection=global"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, text=True,
        )
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "obj": None, "gap": None,
                "nodes": None, "wall": timeout}
    wall = time.time() - start
    out = p.stdout
    rec = {"status": None, "obj": None, "gap": None, "nodes": None, "wall": wall}
    ms = STATUS_RE.search(out)
    if ms:
        rec["status"] = ms.group("msg").strip()
    mr = RESULT_RE.search(out)
    if mr:
        rec["obj"] = float(mr.group("obj"))
        rec["gap"] = float(mr.group("gap"))
        rec["nodes"] = int(mr.group("nodes"))
    if rec["status"] is None:
        # crash / panic / no result line
        rec["status"] = f"NO-RESULT(rc={p.returncode})"
    return rec


def var_count(nl):
    try:
        with open(nl) as fh:
            fh.readline()
            return int(fh.readline().split()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./target/release/pounce")
    ap.add_argument("--nl-dir", default=str(default_nl_dir()))
    ap.add_argument("--optima", default=str(Path(__file__).with_name("optima.txt")))
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-vars", type=int, default=None,
                    help="skip problems with more than this many variables")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="relative tolerance for the certified-vs-known check")
    ap.add_argument("--atol", type=float, default=1e-6,
                    help="absolute tolerance floor (so a proven optimum of 0 is "
                         "not failed for a correct certified value of ~1e-7)")
    ap.add_argument("--out", default=None)
    ap.add_argument("stems", nargs="*", help="restrict to these stems")
    args = ap.parse_args()

    nl_dir = Path(args.nl_dir)
    optima = load_optima(args.optima)
    stems = args.stems or sorted(optima)

    rows = []
    print(f"{'problem':<14}{'n':>4}  {'status':<24}{'certified':>16}"
          f"{'known':>16}{'gap':>9}{'nodes':>8}{'s':>8}  verdict")
    n_ok = n_to = n_wrong = n_other = 0
    for stem in stems:
        nl = nl_dir / f"{stem}.nl"
        known = optima.get(stem)
        if not nl.exists() or known is None:
            continue
        nv = var_count(nl)
        if args.max_vars is not None and nv is not None and nv > args.max_vars:
            continue
        rec = run_one(args.bin, nl, args.timeout)
        cert = rec["obj"]
        # verdict
        if rec["status"] == "TIMEOUT":
            verdict, n_to = "TIMEOUT", n_to + 1
        elif "Global optimum found" in (rec["status"] or "") and cert is not None:
            # Combined absolute+relative check: a proven optimum of exactly 0
            # (common here — ex14_1_*, ex9_2_3) makes a pure *relative* metric
            # explode for a certified value of ~1e-7 that is in fact correct to
            # ~1e-6 absolute. Accept when EITHER the absolute gap is within the
            # floor OR the relative gap is within tol.
            abs_err = abs(cert - known)
            rel = abs_err / max(abs(known), abs(cert), 1e-6)
            if abs_err <= args.atol or rel <= args.tol:
                verdict, n_ok = "OK", n_ok + 1
            else:
                verdict, n_wrong = f"WRONG(rel={rel:.1e})", n_wrong + 1
        else:
            verdict, n_other = rec["status"] or "??", n_other + 1
        rows.append({"stem": stem, "n": nv, "known": known, **rec,
                     "verdict": verdict})
        c = f"{cert:.6e}" if cert is not None else "n/a"
        g = f"{rec['gap']:.1e}" if rec["gap"] is not None else "n/a"
        print(f"{stem:<14}{str(nv):>4}  {(rec['status'] or '')[:23]:<24}{c:>16}"
              f"{known:>16.6e}{g:>9}{str(rec['nodes']):>8}{rec['wall']:>8.2f}  {verdict}")

    total = len(rows)
    print(f"\n{'='*70}\nSUMMARY ({total} problems, timeout={args.timeout}s, "
          f"tol={args.tol})\n{'='*70}")
    print(f"  certified correct global optimum : {n_ok}")
    print(f"  timed out                        : {n_to}")
    print(f"  wrong certified value            : {n_wrong}")
    print(f"  other (node-limit/infeas/crash)  : {n_other}")
    if n_wrong:
        print("\n  *** WRONG (certified value disagrees with proven optimum) ***")
        for r in rows:
            if r["verdict"].startswith("WRONG"):
                print(f"    {r['stem']}: certified {r['obj']} vs known {r['known']}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
