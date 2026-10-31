"""Experiment 1 driver: where does POUNCE fail on feasible models, and does
the AL rung (`al_fallback.py`) rescue those runs?

    python exp_al_corpus.py <fixture-name> <n-starts> [infeasible]

For a fixture POUNCE solves from its own start (so it is feasible), solve it
from its own start and `n-starts` perturbed ones with default options; every run that fails
(status not success/acceptable, or source violation > 1e-6) is retried
through the AL rung. With `infeasible`, the fixture is a known-infeasible
model and only the AL rung's verdict is recorded -- it must not report a
feasible success there (the false-rescue check).

Starts: the model's own x0 (start 0), then x0 + N(0, 1) * max(1, |x0|),
projected into the variable bounds --
a perturbation of the order of the variables themselves, so a start moves
to a genuinely different basin rather than a nearby copy.

One JSON line per start on stdout.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pounce  # noqa: E402

import al_fallback as al  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "crates", "pounce-cli",
                   "tests", "fixtures")


def starts(nl, k):
    x0 = np.asarray(nl.x0, float)
    xl, xu = np.asarray(nl.x_l, float), np.asarray(nl.x_u, float)
    rng = np.random.default_rng(12345)
    out = [np.clip(x0, np.where(np.isfinite(xl), xl, -np.inf), np.where(np.isfinite(xu), xu, np.inf))]
    for _ in range(k):
        x = x0 + rng.standard_normal(x0.size) * np.maximum(1.0, np.abs(x0))
        out.append(np.clip(x, np.where(np.isfinite(xl), xl, -np.inf), np.where(np.isfinite(xu), xu, np.inf)))
    return out


def run(name, k, infeasible=False):
    nl = pounce.read_nl(os.path.join(FIX, name + ".nl"))
    for s, x0 in enumerate(starts(nl, k)):
        rec = {"fixture": name, "start": s, "n": nl.n, "m": nl.m}
        if not infeasible:
            t = time.perf_counter()
            x, info = al.solve_plain(nl, x0)
            v = al.constraint_violation(nl, x)
            rec.update(plain_status=info["status_msg"], plain_iters=int(info["iter_count"]),
                       plain_viol=v, plain_obj=float(nl.objective(x)), plain_wall=time.perf_counter() - t)
            failed = int(info["status"]) not in al.OK or v > 1e-6
            rec["plain_failed"] = failed
            if not failed:
                print(json.dumps(rec), flush=True)
                continue
        t = time.perf_counter()
        try:
            x, info, alrec = al.solve_al(nl, x0)
            v = al.constraint_violation(nl, x)
            rec.update(al_status=info["status_msg"], al_viol=v, al_obj=float(nl.objective(x)),
                       al_inner=alrec["inner_iters"], al_polish=alrec["polish_iters"],
                       al_outer=len(alrec["outer"]), al_point_viol=alrec["al_point_viol"],
                       al_wall=time.perf_counter() - t,
                       al_rescued=int(info["status"]) in al.OK and v <= 1e-6)
        except Exception as e:
            rec.update(al_error=repr(e)[:300])
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), len(sys.argv) > 3 and sys.argv[3] == "infeasible")
