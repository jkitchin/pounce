"""Experiment 1 control: would POUNCE's existing robustness options rescue
the same failures the AL rung was tried on?

    python exp_al_control.py results_al_corpus.jsonl > results_al_control.jsonl

For every run in the corpus file whose plain solve failed, re-solve from the
identical start (regenerated with exp_al_corpus.starts) with each existing
option, scoring rescue exactly as the AL rung was scored: success or
acceptable status and source violation <= 1e-6. If an existing option
rescues as many runs, the AL rung buys nothing new.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pounce  # noqa: E402

import al_fallback as al  # noqa: E402
import exp_al_corpus as corpus  # noqa: E402

CONTROLS = {
    "l1_exact_penalty_barrier": {"l1_exact_penalty_barrier": "yes"},
    "l1_fallback_on_restoration_failure": {"l1_fallback_on_restoration_failure": "yes"},
    "mu_strategy_adaptive": {"mu_strategy": "adaptive"},
}


def main(path):
    failed = {}
    for line in open(path):
        if line.startswith("{"):
            r = json.loads(line)
            if r.get("plain_failed"):
                failed.setdefault(r["fixture"], []).append(r["start"])
    for name, idx in sorted(failed.items()):
        nl = pounce.read_nl(os.path.join(corpus.FIX, name + ".nl"))
        starts = corpus.starts(nl, max(idx))
        for s in idx:
            for ctrl, opts in CONTROLS.items():
                rec = {"fixture": name, "start": s, "control": ctrl}
                try:
                    x, info = al.solve_plain(nl, starts[s], opts)
                    v = al.constraint_violation(nl, x)
                    rec.update(status=info["status_msg"], viol=v, obj=float(nl.objective(x)),
                               iters=int(info["iter_count"]),
                               rescued=int(info["status"]) in al.OK and v <= 1e-6)
                except Exception as e:
                    rec["error"] = repr(e)[:200]
                print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
