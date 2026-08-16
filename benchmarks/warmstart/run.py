"""Command line entry point.

    python -m warmstart.run                       # everything, default solver
    python -m warmstart.run --quick               # one scale, fewer families
    python -m warmstart.run --families simplex_proj --scales small -v

Writes a JSON result file (default ``warmstart/results.json``) and,
unless ``--no-report``, renders ``warmstart/report.md`` beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import List

from .adapters import ARMS, get_adapter
from .families import REGISTRY
from .report import render
from .runner import run_family_scale
from .spec import SCALES

_HERE = os.path.dirname(os.path.abspath(__file__))
_QUICK_FAMILIES = ["simplex_proj", "rosenbrock_ring", "nmpc_vanderpol"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_HERE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _solver_version(name: str) -> str:
    if name == "pounce":
        try:
            import pounce

            return getattr(pounce, "__version__", "unknown")
        except Exception:
            return "unknown"
    return "unknown"


def _csv(value: str, valid, what: str) -> List[str]:
    if value == "all":
        return list(valid)
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [i for i in items if i not in valid]
    if bad:
        raise SystemExit(
            f"unknown {what}: {', '.join(bad)}\nknown: {', '.join(valid)}"
        )
    return items


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="warmstart.run", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--solver", default="pounce", help="adapter name (default: pounce)")
    p.add_argument("--families", default="all", help="comma-separated, or 'all'")
    p.add_argument("--scales", default="all", help="comma-separated, or 'all'")
    p.add_argument("--arms", default="all", help="comma-separated, or 'all'")
    p.add_argument("--tol", type=float, default=1e-8, help="solver tolerance")
    p.add_argument(
        "--kkt-gate",
        type=float,
        default=1e-4,
        help="KKT residual a step must actually achieve to count as "
        "converged, independent of the solver's own status code",
    )
    p.add_argument(
        "--viol-gate",
        type=float,
        default=1e-5,
        help="constraint violation a step must achieve to count as converged",
    )
    p.add_argument(
        "--obj-tol",
        type=float,
        default=1e-6,
        help="relative objective margin: a step whose objective is worse "
        "than the reference arm's by more than this is counted incorrect",
    )
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument(
        "--recentering",
        default="residual",
        choices=("residual", "none"),
        help="warm_start_recentering (pounce#606). The warm-start "
        "baseline moves with this, and so does any margin measured "
        "against it, so a predictor claim has to name which setting it "
        "was taken at",
    )
    p.add_argument(
        "--tier",
        default="default",
        choices=("default", "large", "all"),
        help="which size tier to run. `default` is the standard sweep; "
        "`large` is the opt-in n = 602-2402 MPC horizons, where a single "
        "active-set solve takes seconds; `all` runs both",
    )
    p.add_argument("--quick", action="store_true", help="3 families, one scale")
    p.add_argument("--out", default=os.path.join(_HERE, "results.json"))
    p.add_argument("--no-report", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    families = _csv(args.families, list(REGISTRY), "family")
    if args.tier != "all" and args.families == "all":
        # An explicitly named family is always honored; the tier filter
        # only prunes the "all" default.
        families = [f for f in families if REGISTRY[f].tier == args.tier]
    scales = _csv(args.scales, list(SCALES), "scale")
    arms = _csv(args.arms, ARMS, "arm")
    if args.quick:
        families = [f for f in families if f in _QUICK_FAMILIES]
        scales = ["small"]

    adapter = get_adapter(args.solver, max_iter=args.max_iter,
                          recentering=args.recentering)

    meta = {
        "solver": args.solver,
        "solver_version": _solver_version(args.solver),
        "git_sha": _git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "tol": args.tol,
        "obj_tol": args.obj_tol,
        "kkt_gate": args.kkt_gate,
        "viol_gate": args.viol_gate,
        "max_iter": args.max_iter,
        "recentering": args.recentering,
        "arms": arms,
    }

    runs = []
    total = len(families) * len(scales)
    done = 0
    for family in families:
        for scale in scales:
            done += 1
            print(f"[{done}/{total}] {family} @ {scale} ... ", end="", flush=True)
            t0 = time.perf_counter()
            run = run_family_scale(
                adapter,
                family,
                scale,
                arms,
                tol=args.tol,
                obj_tol=args.obj_tol,
                kkt_gate=args.kkt_gate,
                viol_gate=args.viol_gate,
            )
            runs.append(run)
            print(f"{time.perf_counter() - t0:.1f}s")
            if args.verbose:
                for arm, steps in run["arms"].items():
                    it = sum(s["iters"] for s in steps)
                    bad = sum(0 if s["correct"] else 1 for s in steps)
                    print(f"      {arm:<9} iters={it:<6} incorrect={bad}")

    payload = {"meta": meta, "runs": runs}
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {args.out}")

    if not args.no_report:
        report_path = os.path.splitext(args.out)[0] + ".md"
        with open(report_path, "w") as fh:
            fh.write(render(payload))
        print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
