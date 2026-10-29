"""Summarize results.jsonl -> per (model, route) table and a markdown file."""
import json
import sys
from collections import defaultdict

import numpy as np

PAPER = {
    "bunnyhop": {"CALIPSO": (462.61, 1.76e-6, 101), "Ipopt": (1503.42, 8.96e-8, 1409)},
    "drifting": {"CALIPSO": (0.24, 1.38e-5, 189), "Ipopt": (8.66, 1.00e-1, 194)},
}
VIOL_OK = 1e-4  # a solution "satisfies physics" if max violation <= this

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl") if l.strip()]
g = defaultdict(list)
for r in rows:
    g[(r["model"], r["route"])].append(r)

out = []
for model in ("bunnyhop", "drifting_squared", "drifting_norm"):
    base = model.split("_")[0]
    out.append(f"\n### {model}\n")
    out.append("paper: " + "; ".join(f"{k} obj {v[0]}, viol {v[1]:.2e}, {v[2]} it" for k, v in PAPER[base].items()))
    out.append("")
    out.append("| route | seeds | converged (status 0/1) | viol <= 1e-4 | obj (median, best) | max viol (median) | iters (median) | solve s (median) | statuses |")
    out.append("|---|---:|---:|---:|---|---|---:|---:|---|")
    for route in ("ipopt_defaults_eq", "direct", "ncp_eq", "ncp_eq_l1", "jordan_eq",
                  "scholtes_then_ncp", "scholtes_then_ineq"):
        rs = g.get((model, route))
        if not rs:
            continue
        ok = [r for r in rs if "error" not in r]
        errs = len(rs) - len(ok)
        conv = sum(r["status"] in (0, 1) for r in ok)
        good = [r for r in ok if r["violation"] <= VIOL_OK]
        objs = [r["obj"] for r in good]
        st = defaultdict(int)
        for r in ok:
            st[r["status_msg"]] += 1
        if errs:
            st["timeout/crash"] += errs
        stx = ", ".join(f"{k} x{v}" for k, v in st.items())
        objtxt = f"{np.median(objs):.4g}, {min(objs):.4g}" if objs else "-"
        out.append(
            f"| {route} | {len(rs)} | {conv} | {len(good)} | {objtxt} | "
            f"{np.median([r['violation'] for r in ok]):.1e} | {int(np.median([r['iters'] for r in ok])) if ok else '-'} | "
            f"{np.median([r['solve_wall'] for r in ok]):.1f} | {stx} |" if ok else
            f"| {route} | {len(rs)} | 0 | 0 | - | - | - | - | {stx} |")
print("\n".join(out))
