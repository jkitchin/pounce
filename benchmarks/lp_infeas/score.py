#!/usr/bin/env python3
"""Score lp_infeas results against the instance labels.

Every instance in this suite has no optimal solution, so a run is one of:

* **correct** — the verdict the label calls for: ``Infeasible_Problem_Detected``
  on a ``primal_infeasible`` instance, ``Diverging_Iterates`` (pounce's
  unbounded status) on a ``dual_infeasible`` one;
* **wrong** — any other definite verdict: a success status (the solver
  claims an optimum of a model that has none), or the opposite class;
* **unresolved** — no verdict: a time or iteration limit, a numerical
  failure, a crash.

Wrong verdicts are what the convex path's documentation promises never to
produce ("these statuses are never reported in error"), so this script exits
non-zero when there is one. Unresolved instances are a speed / robustness
measure and do not fail it.

Usage:
    python3 score.py nl/labels.json pounce.json [more.json ...] [--md out.md]

Several result files may be given (e.g. one per shard of a split run);
later files override earlier ones for the same instance.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict

CORRECT = {
    "primal_infeasible": {"Infeasible_Problem_Detected"},
    "dual_infeasible": {"Diverging_Iterates"},
}
# Definite verdicts. A status here that is not CORRECT for the label is wrong.
DEFINITE = {
    "Solve_Succeeded",
    "Solved_To_Acceptable_Level",
    "Feasible_Point_Found",
    "Infeasible_Problem_Detected",
    "Diverging_Iterates",
}
HIGHS_CORRECT = {"primal_infeasible": "Infeasible", "dual_infeasible": "Unbounded"}
HIGHS_DEFINITE = {"Optimal", "Infeasible", "Unbounded"}


def classify(expected, status):
    if status in CORRECT[expected]:
        return "correct"
    if status in DEFINITE:
        return "wrong"
    return "unresolved"


def highs_classify(expected, status):
    if status == HIGHS_CORRECT[expected]:
        return "correct"
    if status in HIGHS_DEFINITE:
        return "wrong"
    return "unresolved"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("labels")
    ap.add_argument("results", nargs="+")
    ap.add_argument("--md", help="also write the table to this markdown file")
    args = ap.parse_args()

    with open(args.labels) as f:
        labels = json.load(f)
    runs = {}
    for path in args.results:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            # run_nl_bench.sh writes a `<results>.env.json` sidecar beside
            # each results array; a shell glob over a results directory
            # picks it up.
            print(f"score.py: skipping {path} (not a results array)", file=sys.stderr)
            continue
        for r in data:
            runs[r["name"]] = r

    groups = defaultdict(list)
    missing = []
    for name, lab in sorted(labels.items()):
        if name not in runs:
            missing.append(name)
            continue
        r = runs[name]
        groups[(lab["source"], lab["expected"])].append(
            (name, lab, r, classify(lab["expected"], r["status"]))
        )

    have_highs = any("highs" in lab for lab in labels.values())
    out = []
    head = "| source | class | n | correct | wrong | unresolved | median time, correct (s) |"
    sep = "|---|---|---:|---:|---:|---:|---:|"
    if have_highs:
        head += " HiGHS correct / wrong / unresolved |"
        sep += "---:|"
    out += [head, sep]
    total = defaultdict(int)
    wrong, unresolved = [], []
    for (source, expected), rows in sorted(groups.items()):
        c = defaultdict(int)
        hc = defaultdict(int)
        times = []
        for name, lab, r, verdict in rows:
            c[verdict] += 1
            total[verdict] += 1
            if verdict == "correct":
                times.append(float(r["solve_time"]))
            elif verdict == "wrong":
                wrong.append((name, expected, r["status"]))
            else:
                unresolved.append((name, expected, r["status"]))
            if "highs" in lab:
                hc[highs_classify(expected, lab["highs"])] += 1
        med = f"{statistics.median(times):.1f}" if times else "-"
        line = (f"| {source} | {expected} | {len(rows)} | {c['correct']} | {c['wrong']} | "
                f"{c['unresolved']} | {med} |")
        if have_highs:
            line += f" {hc['correct']} / {hc['wrong']} / {hc['unresolved']} |"
        out.append(line)
    n = sum(total.values())
    out.append(f"| **all** | | {n} | {total['correct']} | {total['wrong']} | "
               f"{total['unresolved']} | |" + (" |" if have_highs else ""))
    out.append("")
    if wrong:
        out.append("**Wrong verdicts:** " + ", ".join(f"`{a}` ({b}: {s})" for a, b, s in wrong))
    else:
        out.append("**Wrong verdicts:** none.")
    if unresolved:
        out.append("")
        out.append("**Unresolved:** " + ", ".join(f"`{a}` ({s})" for a, _, s in unresolved))
    if missing:
        out.append("")
        out.append(f"**Not run:** {len(missing)} labelled instance(s) have no result.")
    text = "\n".join(out)
    print(text)
    if args.md:
        with open(args.md, "w") as f:
            f.write(text + "\n")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
