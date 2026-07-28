"""Can ANY threshold on a given stationarity measure separate the two cases?

The detector fires when, for `infeas_max_streak` (default 5) consecutive
iterations:

    constr_viol > infeas_viol_kappa * constr_viol_tol     (default 1e2 * 1e-4)
    stationarity <= infeas_stationarity_tol

So a measure is usable only if some threshold makes the streak reachable on
genuinely infeasible problems and unreachable on feasible ones. This replays the
instrumented traces and answers that directly, per candidate measure, rather than
eyeballing a couple of sampled iterations.

MUST FIRE  : contradictory equalities, contradictory cubic, bound contradiction
MUST NOT   : hs13_bigstart (feasible, f* = 1)
"""

import os
import re
import subprocess
import sys

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
VIOL_GATE = 1e2 * 1e-4  # infeas_viol_kappa * constr_viol_tol
STREAK = 5

LINE = re.compile(
    r"INFEASPROBE iter=(\d+) viol_unscaled=(\S+) s_scaled=(\S+) s_unscaled=(\S+) s_rel=(\S+)"
)


def trace(nl):
    env = dict(os.environ, RUST_LOG="pounce::infeas=debug")
    p = subprocess.run(
        [POUNCE, nl, "/tmp/sep.sol", "-AMPL", "max_wall_time=30"],
        capture_output=True, text=True, env=env, timeout=180,
    )
    rows = []
    for m in LINE.finditer(p.stdout + p.stderr):
        rows.append(
            dict(
                iter=int(m.group(1)),
                viol=float(m.group(2)),
                s_scaled=float(m.group(3)),
                s_unscaled=float(m.group(4)),
                s_rel=float(m.group(5)),
            )
        )
    return rows


def fires(rows, key, tol):
    """Would the detector fire with this measure and tolerance?"""
    streak = 0
    for r in rows:
        if r["viol"] > VIOL_GATE and r[key] <= tol:
            streak += 1
            if streak >= STREAK:
                return True
        else:
            streak = 0
    return False


def main():
    cases = {
        "ceq (MUST fire)": ("/tmp/ceq.nl", True),
        "cubic (MUST fire)": ("/tmp/cubic.nl", True),
        "boundcon (MUST fire)": ("/tmp/boundcon.nl", True),
        "hs13 (must NOT fire)": (
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "pr250_work", "hs13_bigstart.nl"),
            False,
        ),
    }
    traces = {}
    for label, (nl, _) in cases.items():
        if not os.path.exists(nl):
            print(f"  (skipping {label}: {nl} missing)")
            continue
        traces[label] = trace(nl)
        print(f"{label:<24} {len(traces[label])} instrumented iterations")

    print()
    for key in ("s_scaled", "s_unscaled", "s_rel"):
        print(f"=== measure: {key} ===")
        # Report, per case, the smallest value reached on a run of STREAK
        # consecutive gate-passing iterations -- i.e. the tolerance at which the
        # detector would just barely fire.
        for label, rows in traces.items():
            gated = [r[key] for r in rows if r["viol"] > VIOL_GATE]
            best = None
            run = []
            for r in rows:
                if r["viol"] > VIOL_GATE:
                    run.append(r[key])
                    if len(run) >= STREAK:
                        window_max = max(run[-STREAK:])
                        best = window_max if best is None else min(best, window_max)
                else:
                    run = []
            print(
                f"  {label:<24} min={min(gated):.3e} max={max(gated):.3e} "
                f"fire-threshold={'n/a' if best is None else f'{best:.3e}'}"
            )
        # Separability: need tol >= fire-threshold for all MUST, < for MUST NOT.
        need_fire, need_quiet = [], []
        for label, rows in traces.items():
            must = cases[label][1]
            run, best = [], None
            for r in rows:
                if r["viol"] > VIOL_GATE:
                    run.append(r[key])
                    if len(run) >= STREAK:
                        w = max(run[-STREAK:])
                        best = w if best is None else min(best, w)
                else:
                    run = []
            (need_fire if must else need_quiet).append((label, best))
        lo = max((b for _, b in need_fire if b is not None), default=None)
        hi = min((b for _, b in need_quiet if b is not None), default=float("inf"))
        if lo is None:
            print("  -> no MUST-fire case reachable at any tolerance\n")
        elif lo < hi:
            print(f"  -> SEPARABLE: any tol in [{lo:.3e}, {hi:.3e}) works\n")
        else:
            print(f"  -> NOT SEPARABLE: must-fire needs >= {lo:.3e}, must-not-fire needs < {hi:.3e}\n")


if __name__ == "__main__":
    main()
