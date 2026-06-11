#!/usr/bin/env python3
"""Compact one-suite summary of a pounce.json produced by run_nl_bench.sh.

Usage: summarize_pounce.py <suite> <pounce.json>
Prints a short status breakdown, totals, and the notable problems
(failures + slowest) for at-a-glance reporting as each set finishes.
"""
import json
import sys
from collections import Counter

SUCCESS = {"Solve_Succeeded"}
ACCEPT = {"Solved_To_Acceptable_Level"}


def main():
    suite, path = sys.argv[1], sys.argv[2]
    with open(path) as fh:
        rows = json.load(fh)
    rows = [r for r in rows if r.get("solver") == "pounce"]
    n = len(rows)
    status = Counter(r["status"] for r in rows)
    succ = sum(status[s] for s in SUCCESS)
    acc = sum(status[s] for s in ACCEPT)
    solved = succ + acc
    tot_it = sum(r["iterations"] or 0 for r in rows)
    tot_t = sum(r["solve_time"] or 0.0 for r in rows)

    print(f"### {suite} — {n} problems")
    print(f"- solved: {solved}/{n}  "
          f"({succ} optimal, {acc} acceptable)  "
          f"= {100*solved/n:.0f}%")
    # status breakdown (non-success first)
    other = {s: c for s, c in status.items()
             if s not in SUCCESS and s not in ACCEPT}
    if other:
        bd = ", ".join(f"{s} x{c}" for s, c in
                       sorted(other.items(), key=lambda kv: -kv[1]))
        print(f"- not solved: {bd}")
    print(f"- totals: {tot_it} iters, {tot_t:.1f}s wall "
          f"({tot_t/n:.2f}s/problem avg)")

    # notable: non-solved problems
    bad = [r for r in rows
           if r["status"] not in SUCCESS and r["status"] not in ACCEPT]
    if bad:
        names = ", ".join(sorted(r["name"] for r in bad)[:20])
        more = "" if len(bad) <= 20 else f" (+{len(bad)-20} more)"
        print(f"- failures: {names}{more}")

    # slowest 5
    slow = sorted(rows, key=lambda r: -(r["solve_time"] or 0))[:5]
    slow_s = ", ".join(f"{r['name']} {r['solve_time']:.1f}s/{r['iterations']}it"
                       for r in slow if (r["solve_time"] or 0) > 1.0)
    if slow_s:
        print(f"- slowest: {slow_s}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
