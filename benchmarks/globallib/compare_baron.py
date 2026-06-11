#!/usr/bin/env python3
"""Timing/correctness cross-check of pounce-global against BARON on GLOBALLib.

BARON is a true spatial-branch-and-bound global solver — the canonical
reference for this Floudas/GAMS test set. Unlike HiGHS (an LP/convex-QP solver
whose AMPL driver only *piecewise-linearly approximates* nonconvex terms),
BARON certifies global optima, so it is both a correctness peer *and* a timing
yardstick. The BARON used here is AMPL's bundled **demo** build, capped at 10
variables / 10 constraints for nonlinear models, so it can only solve the small
subset — for those it is the gold standard.

Inputs:
  * optima.txt            — proven optima (MINLPLib ``=opt=``, ground truth)
  * pounce.json (--pounce)   — the pounce-global harness report (obj, wall, nodes)
  * baron_sweep.tsv (--baron) — `stem  proven  result  obj  time`

Reports, over the subset BARON's demo could solve, a side-by-side of the
certified objective (vs ground truth) and the wall-clock time, so the headline
is "where both certify, do they agree, and how do the solve times compare."
"""
import argparse
import json
from pathlib import Path


def load_optima(path):
    opt = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stem, val = line.split()
        opt[stem] = float(val)
    return opt


def load_baron(path):
    rows = {}
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if i == 0:
            continue
        parts = (line.split("\t") + ["", "", "", "", ""])[:5]
        stem, _proven, result, obj, tim = parts
        rows[stem] = {
            "result": result,
            "obj": _f(obj),
            "time": _f(tim),
        }
    return rows


def load_pounce(path):
    data = json.loads(Path(path).read_text())
    records = data if isinstance(data, list) else data.get("results", data)
    rows = {}
    for r in records:
        stem = r.get("stem") or r.get("problem")
        rows[stem] = {
            "verdict": r.get("verdict") or r.get("status"),
            "obj": r.get("obj"),
            "wall": r.get("wall"),
            "nodes": r.get("nodes"),
        }
    return rows


def _f(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def rel_ok(a, b, tol):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--optima", default=str(here / "optima.txt"))
    ap.add_argument("--pounce", default=str(here / "pounce.json"))
    ap.add_argument("--baron", default=str(here / "baron_sweep.tsv"))
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    opt = load_optima(args.optima)
    baron = load_baron(args.baron)
    pounce = load_pounce(args.pounce) if Path(args.pounce).exists() else {}

    # The interesting set: problems BARON's demo actually solved.
    solved = sorted(s for s in opt if baron.get(s, {}).get("result") == "solved")

    print(f"BARON solved {len(solved)}/{len(opt)} (demo: ≤10 vars/cons nonlinear)\n")
    hdr = f"{'problem':<14}{'proven':>14}{'baron_obj':>14}{'baron_s':>9}" \
          f"{'pounce_obj':>14}{'pounce_s':>10}  {'verdict'}"
    print(hdr)
    print("-" * len(hdr))

    n_both_agree = 0
    baron_t = []
    pounce_t = []
    for stem in solved:
        proven = opt[stem]
        b = baron[stem]
        p = pounce.get(stem, {})
        pobj, pwall, pv = p.get("obj"), p.get("wall"), p.get("verdict")
        b_ok = rel_ok(b["obj"], proven, args.tol)
        p_ok = rel_ok(pobj, proven, args.tol) if pobj is not None else False
        if b_ok and p_ok:
            n_both_agree += 1
        if b["time"] is not None:
            baron_t.append(b["time"])
        if p_ok and pwall is not None:
            pounce_t.append(pwall)
        verdict = "both✓" if (b_ok and p_ok) else (
            f"pounce={pv}" if not p_ok else "baron-off")
        ps = f"{pobj:.5g}" if isinstance(pobj, (int, float)) else "-"
        pw = f"{pwall:.2f}" if isinstance(pwall, (int, float)) else "-"
        print(f"{stem:<14}{proven:>14.5g}{b['obj']:>14.5g}{b['time']:>9.3f}"
              f"{ps:>14}{pw:>10}  {verdict}")

    print(f"\n{'='*70}")
    print(f"on BARON's {len(solved)}-problem demo subset:")
    print(f"  both certify the proven optimum : {n_both_agree}/{len(solved)}")
    if baron_t:
        print(f"  BARON  wall: median {median(baron_t):.3f}s  max {max(baron_t):.3f}s")
    if pounce_t:
        print(f"  pounce wall: median {median(pounce_t):.3f}s  max {max(pounce_t):.3f}s"
              f"  (n={len(pounce_t)} it also solved)")


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


if __name__ == "__main__":
    main()
