"""Reproduce the emfl050 bad-warm-start stall that PR #250's dual-divergence
guard was built for, and measure whether the guard still earns its keep.

Family: nlp   Class: termination-guard justification / cost control
Target: PR #250 / commit ba29b53, option `dual_diverging_streak` (default 15).

WHY THIS EXISTS. Nothing in the repo pins the guard's firing behaviour -- only
option-wiring tests. Meanwhile:

  * emfl050_5_5 now solves cold-start in 91 iters / 0.577 s with the guard OFF,
  * the guard's stated purpose (bound an uninterruptible-factorization grind) is
    also targeted by #245 and #256's predictive overshoot check,
  * its only measured value on 500 corpus models is landing deb7/deb9/deb8 in
    better basins.

So the question "should the guard exist at all?" cannot be answered from what is
checked in. Issue #246 describes the missing repro precisely:

    emfl050_3_3 (1611 vars, 1593 cons), warm-started via solve_nlp(evaluator,
    x0, ...) with a deliberately-bad start (x0 = 10 on unbounded-above vars),
    budget 2 s:
        max_wall_time=2.0s -> ACTUAL 11.7s  status=TIME_LIMIT  iterations=0

and PR #250 claims for emfl050_5_5:

    "converts a permanent stall (diverges forever) into Solve_Succeeded in ~4 s,
     and at a 2 s budget the overshoot drops from 2.6x to 1.0x"

This script rebuilds that setup and runs it with the guard ON (default 15) and
OFF (0) at the same build, so the comparison isolates the guard exactly.

READING THE RESULT.
  * If guard-OFF overshoots badly (>= ~2x) and guard-ON does not, the guard is
    doing real work the deadline machinery cannot do -- keep it and pin it.
  * If both sit near 1.0x, the deadline work (#242/#244/#245/#246/#256) has
    superseded it, and the guard's remaining value is basin luck.
"""

import os
import sys
import time

import numpy as np

sys.setrecursionlimit(100000)

import discopt.modeling as dm
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.solvers.nlp_ipopt import _infer_constraint_bounds
import discopt.solvers.nlp_pounce as NP

CORPUS = os.path.expanduser("~/.cache/discopt/minlplib/current/nl")
# A pre-#250 build has no `dual_diverging_streak` option at all; set
# PR250_NO_GUARD_OPTION=1 to run the single "guard absent" arm against it.
ARMS = ([(None, "absent")] if os.environ.get("PR250_NO_GUARD_OPTION")
        else [(15, "on"), (0, "off")])
BAD_START = 10.0  # issue #246: "x0 = 10 on unbounded-above vars"


def build(name):
    model = dm.from_nl(os.path.join(CORPUS, f"{name}.nl"))
    ev = NLPEvaluator(model)
    # `_infer_constraint_bounds` returns (cl, cu) arrays; solve_nlp wants the
    # per-row [(lo, hi), ...] form the B&B driver passes.
    cl, cu = _infer_constraint_bounds(ev)
    cb = [(float(lo), float(hi)) for lo, hi in zip(cl, cu)]
    lb, ub = (np.asarray(b, float) for b in ev.variable_bounds)
    # The deliberately-bad warm start: push every variable that is unbounded
    # above out to BAD_START, leave the rest at a bound-respecting midpoint.
    x0 = np.clip(np.zeros_like(lb), lb, ub)
    free_above = np.isinf(ub)
    x0[free_above] = BAD_START
    x0 = np.clip(x0, lb, ub)
    return ev, x0, cb, int(ev.n_variables), int(ev.n_constraints)


def run(ev, x0, cb, budget, streak):
    opts = {
        "print_level": 0,
        "max_iter": 3000,
        "tol": 1e-7,
        "max_wall_time": budget,
    }
    # `streak=None` omits the option entirely, so the same harness runs against a
    # pre-#250 build where `dual_diverging_streak` does not exist yet (setting it
    # there is a hard OPTION_INVALID error, not a no-op).
    if streak is not None:
        opts["dual_diverging_streak"] = streak
    t0 = time.perf_counter()
    r = NP.solve_nlp(ev, x0, constraint_bounds=cb, options=opts)
    dt = time.perf_counter() - t0
    iters = None
    for attr in ("iterations", "n_iterations", "iter_count"):
        if hasattr(r, attr):
            iters = getattr(r, attr)
            break
    return dt, r.status.name, r.objective, iters


def main():
    models = sys.argv[1:] or ["emfl050_3_3", "emfl050_5_5"]
    budgets = [2.0, 5.0]

    print(f"bad warm start: x0 = {BAD_START} on every ub=+inf variable\n")
    rows = []
    for name in models:
        ev, x0, cb, n, m = build(name)
        nfree = int(np.isinf(np.asarray(ev.variable_bounds[1], float)).sum())
        print(f"=== {name}: {n} vars ({nfree} with ub=+inf), {m} cons ===")

        # CRITICAL: the first solve on a fresh evaluator pays JAX tracing and
        # compilation for every callback -- 16 s on emfl050_3_3 and 71 s on
        # emfl050_5_5, with `iterations=0`, which looks exactly like the stall
        # this script is trying to measure. Issue #246 notes the callbacks are
        # jitted and cost 0.1-0.2 ms each, i.e. the reported numbers are
        # post-warm-up. Burn the compile on a generous budget and discard it, or
        # every measurement below is really a compile-time measurement.
        t0 = time.perf_counter()
        run(ev, x0, cb, 600.0, ARMS[0][0])
        print(f"  (warm-up / JIT compile discarded: {time.perf_counter() - t0:.1f} s)")

        print(f"{'budget':>7} {'guard':>6} {'wall':>8} {'ratio':>7} {'status':<14} {'iters':>6}  objective")
        for budget in budgets:
            # Interleave and repeat so a slow first call in either arm cannot be
            # mistaken for a guard effect; report the faster of two reps.
            for streak, label in ARMS:
                reps = [run(ev, x0, cb, budget, streak) for _ in range(2)]
                dt, status, obj, iters = min(reps, key=lambda r: r[0])
                ratio = dt / budget
                rows.append((name, budget, label, dt, ratio, status, iters, obj))
                print(
                    f"{budget:>7.1f} {label:>6} {dt:>8.2f} {ratio:>6.2f}x "
                    f"{status:<14} {str(iters):>6}  {obj}"
                )
        print()

    print("=" * 78)
    worst_on = max((r[4] for r in rows if r[2] == "on"), default=0.0)
    worst_off = max((r[4] for r in rows if r[2] == "off"), default=0.0)
    print(f"worst overshoot ratio  guard ON: {worst_on:.2f}x   guard OFF: {worst_off:.2f}x")
    if worst_off >= 2.0 and worst_on < 1.5:
        print("VERDICT: guard EARNS ITS KEEP (bounds an overshoot the deadline cannot)")
    elif worst_off < 1.5:
        print("VERDICT: guard REDUNDANT here (deadline machinery already bounds it)")
    else:
        print("VERDICT: INCONCLUSIVE")


if __name__ == "__main__":
    main()
