"""Was issue #246's "0-iteration bad-start stall" actually JAX compile time?

Issue #246 reported, for emfl050_3_3 warm-started with x0 = 10 on unbounded-above
variables at a 2 s budget:

    max_wall_time=2.0s -> ACTUAL 11.7s   status=TIME_LIMIT   iterations=0

and concluded:

    "The callbacks are fast (jitted; 0.1-0.2 ms each), so the time is inside
     POUNCE's own initialization/restoration linear algebra, not the caller."

That conclusion holds only if the callbacks were already compiled when the clock
started. On a freshly built NLPEvaluator they are not.

THE EXPERIMENT. Cost that belongs to first-call compilation follows whichever
arm runs FIRST, and is independent of any solver option. Cost that belongs to
the dual-divergence guard follows the guard. So run the identical pair in both
orders on cold evaluators:

    cold evaluator A: guard OFF first, then guard ON
    cold evaluator B: guard ON  first, then guard OFF

If the slow, `iterations=0` run is always the FIRST one regardless of the guard
setting, the reported stall is caller-side compilation and PR #250's guard was
built for a measurement artifact. If it follows the guard, the stall is real.

Each order needs its own freshly built evaluator, since compilation is cached on
the evaluator.
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
BUDGET = 2.0


def build(name):
    model = dm.from_nl(os.path.join(CORPUS, f"{name}.nl"))
    ev = NLPEvaluator(model)
    cl, cu = _infer_constraint_bounds(ev)
    cb = [(float(lo), float(hi)) for lo, hi in zip(cl, cu)]
    lb, ub = (np.asarray(b, float) for b in ev.variable_bounds)
    x0 = np.clip(np.zeros_like(lb), lb, ub)
    x0[np.isinf(ub)] = 10.0
    return ev, np.clip(x0, lb, ub), cb


def run(ev, x0, cb, streak):
    opts = {
        "print_level": 0,
        "max_iter": 3000,
        "tol": 1e-7,
        "max_wall_time": BUDGET,
        "dual_diverging_streak": streak,
    }
    t0 = time.perf_counter()
    r = NP.solve_nlp(ev, x0, constraint_bounds=cb, options=opts)
    dt = time.perf_counter() - t0
    iters = next(
        (getattr(r, a) for a in ("iterations", "n_iterations", "iter_count") if hasattr(r, a)),
        None,
    )
    return dt, r.status.name, iters, r.objective


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "emfl050_3_3"
    print(f"=== {name}, budget {BUDGET} s, x0 = 10 on ub=+inf vars ===\n")

    results = {}
    for order in (("off", 0), ("on", 15)), (("on", 15), ("off", 0)):
        first, second = order
        ev, x0, cb = build(name)  # fresh evaluator per order
        tag = f"{first[0]}-first"
        print(f"-- cold evaluator, {first[0]} runs first --")
        for label, streak in (first, second):
            dt, status, iters, obj = run(ev, x0, cb, streak)
            slot = "1st" if (label, streak) == first else "2nd"
            results[(tag, label)] = (dt, status, iters, slot)
            print(f"   {slot} call  guard={label:<3} {dt:7.2f} s  {status:<12} iters={iters}  obj={obj}")
        print()

    print("=" * 72)
    slow = [(k, v) for k, v in results.items() if v[0] > 5.0]
    if slow and all(v[3] == "1st" for _, v in slow):
        print("Every slow, iterations=0 run is the FIRST call on a cold evaluator,")
        print("in BOTH orders and under BOTH guard settings.")
        print("VERDICT: the reported stall is caller-side JAX compilation,")
        print("         not a POUNCE stall and not the dual-divergence guard.")
    elif slow and all(v[3] != "1st" for _, v in slow):
        print("VERDICT: slowness follows the guard, not call order -- stall is real.")
    else:
        print("VERDICT: INCONCLUSIVE (no run exceeded 5 s, or the pattern is mixed).")


if __name__ == "__main__":
    main()
