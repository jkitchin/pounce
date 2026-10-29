"""Summarize one or more results files.

    python summarize.py results_pounce.jsonl results_ipopt_mumps.jsonl ...

Prints, per model, one row per route and one column group per solver:
`physical/seeds` (source-level violation <= 1e-4), `converged/seeds`
(status 0 or 1), median objective over the physical runs, and median
iterations over all runs.
"""
import json
import sys
from collections import defaultdict

import numpy as np

PAPER = {
    "bunnyhop": {"CALIPSO": (462.61, 1.76e-6, 101), "Ipopt": (1503.42, 8.96e-8, 1409)},
    "drifting": {"CALIPSO": (0.24, 1.38e-5, 189), "Ipopt": (8.66, 1.00e-1, 194)},
}
VIOL_OK = 1e-4
ROUTES = ("defaults_eq", "direct", "ncp_eq", "ncp_eq_l1", "jordan_eq",
          "scholtes_then_ncp", "scholtes_then_ineq")

files = sys.argv[1:] or ["results.jsonl"]
g = defaultdict(list)
solvers = []
for fn in files:
    for line in open(fn):
        if not line.strip():
            continue
        r = json.loads(line)
        s = r.get("solver", fn)
        if s not in solvers:
            solvers.append(s)
        g[(r["model"], r["route"], s)].append(r)


def cell(rs):
    if not rs:
        return "n/a"
    ok = [r for r in rs if "error" not in r]
    phys = [r for r in ok if r["violation"] <= VIOL_OK]
    conv = sum(r["status"] in (0, 1) for r in ok)
    obj = f"{np.median([r['obj'] for r in phys]):.4g}" if phys else "–"
    it = int(np.median([r["iters"] for r in ok])) if ok else "–"
    return f"{len(phys)}/{len(rs)} · {conv}/{len(rs)} · {obj} · {it}"


out = ["cells: physical/seeds · converged/seeds · median obj (physical runs) · median iters"]
for model in ("bunnyhop", "drifting_squared", "drifting_norm"):
    base = model.split("_")[0]
    out.append(f"\n### {model}\n")
    out.append("paper: " + "; ".join(
        f"{k} obj {v[0]}, viol {v[1]:.2e}, {v[2]} it" for k, v in PAPER[base].items()))
    out.append("")
    out.append("| route | " + " | ".join(solvers) + " |")
    out.append("|---|" + "---|" * len(solvers))
    for route in ROUTES:
        if not any(g.get((model, route, s)) for s in solvers):
            continue
        out.append(f"| `{route}` | " + " | ".join(cell(g.get((model, route, s))) for s in solvers) + " |")
print("\n".join(out))
