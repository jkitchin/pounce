"""python run_one.py <model> <route> <seed>  -> one JSON line on stdout.

model: any name in models.ALL
"""
import json
import os
import sys
import time

import numpy as np


def main():
    model_name, route, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    import models
    from mpcc_nlp import run_route

    model, initial_point, _, _ = models.get(model_name)
    z0 = initial_point(seed)
    t0 = time.perf_counter()
    try:
        r = run_route(model, route, z0)
    except Exception as e:  # report, don't hide
        print(json.dumps({"model": model_name, "route": route, "seed": seed, "error": repr(e)}))
        return
    rep = model.report(r["x"])
    import stationarity
    try:
        st = stationarity.classify(model, r["x"], solver_tol=1e-8)
    except Exception as e:  # a classification failure must not lose the run
        st = {"klass": "error", "reason": repr(e)}
    rep["stationarity"] = {k: st[k] for k in ("klass", "n_biactive", "residuals", "resid_tol", "note", "reason")
                           if k in st}
    solver = os.environ.get("CALIPSO_SOLVER", "pounce")
    if solver == "ipopt":
        solver += "-" + os.environ.get("IPOPT_LINEAR_SOLVER", "mumps")
    else:
        solver += "-" + os.environ.get("POUNCE_LINEAR_SOLVER", "feral")
    out = {"solver": solver, "model": model_name, "route": route, "seed": seed,
           "status": r["status"], "status_msg": r["status_msg"], "iters": r["iters"],
           "stages": r["stages"], "solve_wall": round(r["wall"], 2),
           "total_wall": round(time.perf_counter() - t0, 2),
           "finish_failed": r.get("finish_failed", False),
           "log": [list(map(str, s)) for s in r.get("log", [])], **rep}
    if "declared" in r:
        out["declared"] = r["declared"]
    print(json.dumps(out))
    np.save(f"sol_{solver}_{model_name}_{route}_{seed}.npy", r["x"])


if __name__ == "__main__":
    main()
