#!/usr/bin/env python3
"""Check a run_nl_bench.sh results file against a suite's answers.json.

Usage:
    check_nl_answers.py <answers.json> <pounce.json>

`answers.json` lives beside the data in the benchmark data root
(`$POUNCE_BENCH_DATA/<suite>/answers.json`): a suite-level `rel_tol` (default
1e-6) and one entry per problem with `expected_status`, `f_star` and
`f_star_kind`. Objectives match when |f - f_star| <= rel_tol * max(1, |f_star|).

How a row is judged depends on what the reference is:

* `f_star_kind: "global"` -- f_star is a proven optimum (the mpcc_nl suite).
  The run must converge and match it; anything else FAILs.
* `f_star_kind: "best_known"` -- f_star is the best value published, found by
  local solvers (MacMPEC). A converged run is classified, not failed:
  `at` the best known, `worse` (another local solution) or `BETTER` (a
  finding: check the point's feasibility before believing it), in the
  direction of the problem's `sense` (`minimize` unless it says
  `maximize`; objectives are compared in the model's own sense). A run
  that does not converge is reported as `unsolved`.

Independent of kind, a run that contradicts the reference FAILs: converging
on a problem whose `expected_status` is `InfeasibleProblemDetected`, not
reporting infeasible there, or anything but the CLI's own refusal
(`Complementarity_Refused`, which run_nl_bench.sh records from the CLI's
message) on a problem that must be `refused` -- a crash is not a refusal.
Problems with no reference (`expected_status: "unknown"`) are listed only.

Exits 1 if any row FAILs. A problem missing from pounce.json FAILs.
"""

import collections
import json
import math
import sys

CONVERGED = {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
INFEASIBLE = "Infeasible_Problem_Detected"
REFUSED = "Complementarity_Refused"


def judge(ans, row, rel_tol):
    """(outcome, detail); outcome is FAIL, pass, at, worse, BETTER, unsolved, note."""
    if row is None:
        return "FAIL", "no result row"
    status, f = row.get("status"), row.get("objective")
    want = ans["expected_status"]
    if want == "unknown":
        return "note", f"{status}, f = {f} (no reference)"
    if want == "refused":
        if status == REFUSED:
            return "pass", "refused"
        if status in CONVERGED:
            return "FAIL", f"solved ({status}); must be refused"
        return "FAIL", f"{status}; must be refused by the CLI, not fail otherwise"
    if want == "InfeasibleProblemDetected":
        if status == INFEASIBLE:
            return "pass", "infeasible"
        return "FAIL", f"{status}; reference says infeasible"
    # A feasible reference.
    if status == INFEASIBLE:
        return "FAIL", "reported infeasible; reference has a feasible point"
    kind = ans.get("f_star_kind", "global")
    if status not in CONVERGED:
        return ("FAIL" if kind == "global" else "unsolved"), status
    f_star = ans.get("f_star")
    if f_star is None:
        return "pass", "converged"
    if not isinstance(f, (int, float)) or not math.isfinite(f):
        return "FAIL", f"converged with objective {f!r}"
    tol = rel_tol * max(1.0, abs(f_star))
    if abs(f - f_star) <= tol:
        return ("pass" if kind == "global" else "at"), f"f = {f:.10g}"
    detail = f"f = {f:.10g}, f* = {f_star:.10g}"
    if kind == "global":
        return "FAIL", detail
    # The reported objective is in the model's own sense, so "better" is
    # lower for a minimization and higher for a maximization.
    improved = f > f_star if ans.get("sense") == "maximize" else f < f_star
    return ("BETTER" if improved else "worse"), detail


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    doc = json.load(open(sys.argv[1]))
    rel_tol = float(doc.get("rel_tol", 1e-6))
    answers = doc["problems"]
    rows = {r["name"]: r for r in json.load(open(sys.argv[2])) if r.get("solver") == "pounce"}
    counts = collections.Counter()
    for ans in answers:
        row = rows.get(ans["name"])
        outcome, why = judge(ans, row, rel_tol)
        counts[outcome] += 1
        status = row["status"] if row else "-"
        print(f"{outcome:8s} {ans['name']:20s} {status:28s} {why}")
    for name in sorted(set(rows) - {a["name"] for a in answers}):
        print(f"{'note':8s} {name:20s} has a result row but no entry in answers.json")
    order = ["pass", "at", "BETTER", "worse", "unsolved", "note", "FAIL"]
    summary = ", ".join(f"{counts[k]} {k}" for k in order if counts[k])
    print(f"{doc.get('suite', '?')}: {len(answers)} problems -- {summary}")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
