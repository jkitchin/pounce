#!/usr/bin/env python3
"""Cross-check the two OBBT LP engines on GLOBALLib.

`pounce-global`'s spatial branch-and-bound tightens variable bounds with
optimality-based bound tightening (OBBT), and the LP solves inside OBBT can be
driven by either engine:

  * the default conic interior-point solver (`global_obbt_lp=ipm`), or
  * the bounded-variable revised simplex (`global_obbt_lp=simplex`, gated behind
    the off-by-default `simplex-obbt` cargo feature).

OBBT only narrows boxes; it must never cut off the global optimum. A bug in
either LP engine can produce a too-tight (wrong) bound that prunes the true
minimizer, so the branch-and-bound then *certifies the wrong optimum*. This is a
silent soundness failure: the run reports "Global optimum found" with a bogus
value.

This harness runs the GLOBALLib proven-optimum subset twice — once per engine —
and asserts the two engines certify the **same** optimum on every model either
of them solves. Concretely it fails (nonzero exit) when:

  1. either engine returns a WRONG certified value (disagrees with the MINLPLib
     proven optimum beyond tolerance), or
  2. both engines certify "Global optimum found" but disagree with **each
     other** beyond tolerance.

A model that one engine solves and the other times out is reported but is not a
failure (timeouts are a performance difference, not a soundness one). This is
the validation gate before graduating `simplex-obbt` to the default engine.

Usage:
  compare_obbt_engines.py [--timeout SECS] [--max-vars N] [--tol REL]
                          [--atol ABS] [--bin PATH] [--nl-dir DIR]
                          [--out-dir DIR] [stems...]

  # Or compare two already-generated run_globallib.py reports:
  compare_obbt_engines.py --ipm-json ipm.json --simplex-json simplex.json

Exit code 0 iff the two engines agree everywhere (soundness gate passes).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
RUNNER = HERE / "run_globallib.py"


def run_engine(args, engine_opts, out_path):
    """Invoke run_globallib.py for one engine, returning its parsed rows."""
    cmd = [
        sys.executable, str(RUNNER),
        "--bin", args.bin,
        "--nl-dir", args.nl_dir,
        "--timeout", str(args.timeout),
        "--tol", str(args.tol),
        "--atol", str(args.atol),
        "--out", str(out_path),
    ]
    if args.max_vars is not None:
        cmd += ["--max-vars", str(args.max_vars)]
    if args.stems_file:
        cmd += ["--stems-file", args.stems_file]
    for o in engine_opts:
        cmd += ["--opt", o]
    cmd += args.stems
    print(f"\n{'#'*72}\n# running: {' '.join(cmd)}\n{'#'*72}", flush=True)
    subprocess.run(cmd, check=True)
    return {r["stem"]: r for r in json.loads(Path(out_path).read_text())}


def agree(a, b, tol, atol):
    """True if two certified objectives agree within abs OR rel tolerance."""
    if a is None or b is None:
        return False
    abs_err = abs(a - b)
    rel = abs_err / max(abs(a), abs(b), 1e-6)
    return abs_err <= atol or rel <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="./target/release/pounce")
    ap.add_argument("--nl-dir",
                    default=str(__import__("os").environ.get(
                        "POUNCE_BENCH_DATA",
                        str(Path.home() / "Dropbox/projects/pounce-bench-data"))
                        ) + "/globallib/nl")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-vars", type=int, default=None)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--out-dir", default="/tmp")
    ap.add_argument("--ipm-json", default=None,
                    help="skip running; load this IPM report instead")
    ap.add_argument("--simplex-json", default=None,
                    help="skip running; load this simplex report instead")
    ap.add_argument("--stems-file", default=None,
                    help="newline-separated stem list (e.g. tiers/micro.txt)")
    ap.add_argument("stems", nargs="*")
    args = ap.parse_args()

    if args.ipm_json and args.simplex_json:
        ipm = {r["stem"]: r for r in json.loads(Path(args.ipm_json).read_text())}
        spx = {r["stem"]: r
               for r in json.loads(Path(args.simplex_json).read_text())}
    else:
        out = Path(args.out_dir)
        ipm = run_engine(args, ["global_obbt_lp=ipm"], out / "globallib_ipm.json")
        spx = run_engine(args, ["global_obbt_lp=simplex"],
                         out / "globallib_simplex.json")

    stems = sorted(set(ipm) | set(spx))
    wrong = []        # engine certified a value disagreeing with proven optimum
    disagree = []     # engines disagree with each other
    both_ok = 0
    only_ipm = only_spx = neither = 0

    print(f"\n{'='*94}")
    print(f"{'problem':<14}{'known':>15}{'ipm':>16}{'simplex':>16}  verdict")
    print(f"{'='*94}")
    for stem in stems:
        ri, rs = ipm.get(stem), spx.get(stem)
        known = (ri or rs).get("known")
        oi = ri["obj"] if ri else None
        os_ = rs["obj"] if rs else None
        vi = ri["verdict"] if ri else "MISSING"
        vs = rs["verdict"] if rs else "MISSING"
        ok_i = vi == "OK"
        ok_s = vs == "OK"

        notes = []
        if vi.startswith("WRONG"):
            wrong.append((stem, "ipm", oi, known))
            notes.append(f"IPM {vi}")
        if vs.startswith("WRONG"):
            wrong.append((stem, "simplex", os_, known))
            notes.append(f"SIMPLEX {vs}")
        if ok_i and ok_s:
            both_ok += 1
            if not agree(oi, os_, args.tol, args.atol):
                disagree.append((stem, oi, os_))
                notes.append("ENGINES DISAGREE")
        elif ok_i and not ok_s:
            only_ipm += 1
            notes.append(f"only IPM solved (spx={vs})")
        elif ok_s and not ok_i:
            only_spx += 1
            notes.append(f"only simplex solved (ipm={vi})")
        else:
            neither += 1

        ci = f"{oi:.6e}" if oi is not None else "n/a"
        cs = f"{os_:.6e}" if os_ is not None else "n/a"
        kn = f"{known:.6e}" if known is not None else "n/a"
        print(f"{stem:<14}{kn:>15}{ci:>16}{cs:>16}  {'; '.join(notes)}")

    print(f"\n{'='*94}\nSUMMARY ({len(stems)} models, timeout={args.timeout}s)")
    print(f"  both engines certified correct optimum : {both_ok}")
    print(f"  only IPM solved (simplex timed out)    : {only_ipm}")
    print(f"  only simplex solved (IPM timed out)    : {only_spx}")
    print(f"  neither solved                         : {neither}")
    print(f"  WRONG certified values                 : {len(wrong)}")
    print(f"  engine-vs-engine disagreements         : {len(disagree)}")

    if wrong:
        print("\n  *** WRONG (certified value disagrees with proven optimum) ***")
        for stem, eng, got, known in wrong:
            print(f"    {stem} [{eng}]: certified {got} vs known {known}")
    if disagree:
        print("\n  *** ENGINE DISAGREEMENT (ipm vs simplex) ***")
        for stem, oi, os_ in disagree:
            print(f"    {stem}: ipm {oi} vs simplex {os_}")

    if wrong or disagree:
        print("\nSOUNDNESS GATE: FAIL")
        return 1
    print("\nSOUNDNESS GATE: PASS — both engines certify identical optima.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
