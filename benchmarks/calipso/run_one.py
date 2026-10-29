"""python run_one.py <model> <route> <seed>  -> one JSON line on stdout.

model: bunnyhop | drifting_norm | drifting_squared
"""
import json
import sys
import time

import numpy as np


def main():
    model_name, route, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if model_name == "bunnyhop":
        import bunnyhop as mod
        model = mod.model
    else:
        import drifting as mod
        model = mod.make_model(model_name.split("_", 1)[1])
    from mpcc_nlp import run_route

    z0 = mod.initial_point(seed)
    t0 = time.perf_counter()
    try:
        r = run_route(model, route, z0)
    except Exception as e:  # report, don't hide
        print(json.dumps({"model": model_name, "route": route, "seed": seed, "error": repr(e)}))
        return
    rep = model.report(r["x"])
    out = {"model": model_name, "route": route, "seed": seed,
           "status": r["status"], "status_msg": r["status_msg"], "iters": r["iters"],
           "stages": r["stages"], "solve_wall": round(r["wall"], 2),
           "total_wall": round(time.perf_counter() - t0, 2),
           "finish_failed": r.get("finish_failed", False),
           "log": [list(map(str, s)) for s in r.get("log", [])], **rep}
    print(json.dumps(out))
    np.save(f"sol_{model_name}_{route}_{seed}.npy", r["x"])


if __name__ == "__main__":
    main()
