"""Summarize one or more results files.

    python summarize.py results_pounce.jsonl results_ipopt_mumps.jsonl ...

Prints, per model, one row per route and one column per solver:
`physical/seeds` (source-level violation <= 1e-4), `certified/seeds`
(S-stationary for an MPCC, KKT for a plain NLP; `stationarity.py`), median
objective over the physical runs, and median iterations over all runs.
CALIPSO.jl's answers appear as the `calipso` route of the `calipso`
solver, scored by the port (`evaluate_calipso.py`).
"""
import json
import sys
from collections import defaultdict

import numpy as np

#: The paper's Table 2: (objective, violation, iterations). Only the four
#: contact-implicit problems have numbers; rocket landing is a figure.
PAPER = {
    "bunnyhop": {"CALIPSO": (462.61, 1.76e-6, 101), "Ipopt": (1503.42, 8.96e-8, 1409)},
    "drifting": {"CALIPSO": (0.24, 1.38e-5, 189), "Ipopt": (8.66, 1.00e-1, 194)},
    "ballincup": {"CALIPSO": (11.96, 4.86e-5, 131), "Ipopt": (68.18, 2.60e-2, 205)},
    "quadruped": {"CALIPSO": (574.84, 5.32e-4, 178), "Ipopt": (1855.18, 1.13e-1, 2000)},
}
VIOL_OK = 1e-4
ROUTES = ("calipso", "defaults_eq", "direct", "ncp_eq", "ncp_eq_l1", "jordan_eq",
          "scholtes_then_ncp", "scholtes_then_ineq")
MODELS = ("bunnyhop", "drifting_squared", "drifting_norm", "ballincup", "quadruped", "rocket",
          "complementarity", "maratos", "wachter")

files = sys.argv[1:] or ["results.jsonl"]
g = defaultdict(list)
solvers = []
for fn in files:
    recs = [json.loads(line) for line in open(fn)
            if line.startswith("{")]  # Ipopt prints its banner to stdout
    # a timeout / crash record carries no solver; it belongs to its file's
    named = [r["solver"] for r in recs if "solver" in r]
    file_solver = max(set(named), key=named.count) if named else fn
    for r in recs:
        s = r.get("solver", file_solver)
        if s not in solvers:
            solvers.append(s)
        g[(r["model"], r["route"], s)].append(r)


def cell(rs):
    if not rs:
        return "n/a"
    ok = [r for r in rs if "error" not in r]
    phys = [r for r in ok if r["violation"] <= VIOL_OK]
    # A stationarity verdict says nothing about feasibility -- the KKT test
    # passes at wachter's infeasible stopping point -- so a run counts as
    # certified only when it is also physical.
    cert = sum(r.get("stationarity", {}).get("klass") in ("S", "KKT") for r in phys)
    klasses = [r.get("stationarity", {}).get("klass") for r in ok]
    if not any("stationarity" in r for r in ok):
        cert_txt = "–"  # run predates the stationarity column
    elif all(k == "n/a" for k in klasses):
        cert_txt = "n/a"  # cone complementarity: the NCP classes do not apply
    else:
        cert_txt = f"{cert}/{len(rs)}"
    obj = f"{np.median([r['obj'] for r in phys]):.4g}" if phys else "–"
    it = int(np.median([r["iters"] for r in ok])) if ok else "–"
    return f"{len(phys)}/{len(rs)} · {cert_txt} · {obj} · {it}"


out = ["cells: physical/seeds · certified (S or KKT)/seeds · median obj (physical runs) · median iters"]
for model in MODELS:
    if not any(key[0] == model for key in g):
        continue
    base = model.split("_")[0]
    out.append(f"\n### {model}\n")
    if base in PAPER:
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
