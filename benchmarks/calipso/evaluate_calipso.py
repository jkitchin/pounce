"""Score CALIPSO.jl's own answers with this port.

    python evaluate_calipso.py results_calipso.jsonl > results_calipso_eval.jsonl

Each record from `calipso_baseline.jl` is mapped into the port's variable
vector (`models.get(...)[3]`) and evaluated with the port's own `report`
and `stationarity.classify`, so CALIPSO's rows carry exactly the columns
the POUNCE and Ipopt rows do, measured the same way.

This is also the end-to-end check of every port: at CALIPSO's answer the
port's source-level violation should be of the order of CALIPSO's own
reported residual (its default tolerance is 1e-4), and its objective is the
number to compare with the paper's table. A port that transcribed the model
differently shows up here as a large violation at a point CALIPSO calls
converged.

CALIPSO is classified with ``solver_tol=1e-4``, the tolerance it was run at
(see stationarity.py on why judging it at 1e-8 would be unfair).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models  # noqa: E402
import stationarity  # noqa: E402

EXAMPLE_TO_MODELS = {}
for name in models.ALL:
    try:
        _, _, example, _ = models.get(name)
    except Exception:
        continue
    EXAMPLE_TO_MODELS.setdefault(example, []).append(name)


def main(path):
    for line in open(path):
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        for name in EXAMPLE_TO_MODELS.get(rec["example"], []):
            model, _, _, from_calipso = models.get(name)
            z = from_calipso(rec)
            rep = model.report(z)
            try:
                st = stationarity.classify(model, z, solver_tol=1e-4)
            except Exception as e:
                st = {"klass": "error", "reason": repr(e)}
            rep["stationarity"] = {k: st[k] for k in ("klass", "n_biactive", "residuals", "resid_tol", "note", "reason")
                                   if k in st}
            out = {"solver": "calipso", "model": name, "route": "calipso", "seed": rec["seed"],
                   "status": 0 if rec["converged"] else -1,
                   "status_msg": "converged" if rec["converged"] else "not_converged",
                   "iters": rec["iters"], "stages": 1, "solve_wall": round(rec["solve_wall"], 2),
                   "calipso_residual": rec["residual"], "calipso_equality": rec["equality"],
                   "calipso_comp": rec["comp"], **rep}
            print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_calipso.jsonl")
