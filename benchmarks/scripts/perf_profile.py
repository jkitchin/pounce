#!/usr/bin/env python3
"""Solver comparison figures from the benchmark JSONs.

Two figure kinds, both "fraction of problems vs cost":

  performance  Dolan & Moré (2002) performance profile. x = performance
               ratio τ (each solver's cost / the best solver's cost on that
               problem), y = fraction of problems solved within factor τ.
               τ=1 reads as "fraction where this solver was fastest";
               τ→∞ as "fraction solved at all" (robustness). x is log2.

  data         Raw ECDF / data profile. x = absolute cost (seconds or
               iterations), y = fraction of problems solved within that
               budget. No best-solver normalization — directly answers
               "how many problems by 1s? by 10s?". x is log10.

Cost metric is --metric {time,iters}. A problem counts as solved by a
solver only if its status is a success (optionally including
"acceptable"); a failure is assigned infinite cost so it never sits below
a finite budget. Problems no solver solved are dropped (performance mode)
since the ratio is undefined.

Reads benchmarks/<suite>/{pounce.json,ipopt_ma57.json}. Pass one or more
suites; they are pooled into one figure (the standard way to profile over
a whole test set).

Usage:
  perf_profile.py vanderbei --metric time
  perf_profile.py vanderbei qp lp --metric iters --mode performance -o prof.png
  perf_profile.py mittelmann --mode data --metric time
"""
import argparse
import json
import os
import sys

import numpy as np

# Status -> solved? (mirrors summarize_pounce.py)
SUCCESS = {"Solve_Succeeded"}
ACCEPT = {"Solved_To_Acceptable_Level"}

# Pretty names + stable colors for the legend.
SOLVER_LABEL = {"pounce": "POUNCE (feral)", "ipopt": "Ipopt (MA57)"}
SOLVER_COLOR = {"pounce": "#1f77b4", "ipopt": "#d62728"}

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)


def load_suite(suite, count_acceptable, bench_dir=BENCH):
    """Return {solver: {name: cost_or_None}} for one suite.

    cost is solve_time/iterations when the solver solved the problem,
    else None (=failed/timed-out -> treated as +inf downstream).
    """
    out = {}
    for solver, fname in (("pounce", "pounce.json"),
                          ("ipopt", "ipopt_ma57.json")):
        path = os.path.join(bench_dir, suite, fname)
        if not os.path.exists(path):
            continue
        rows = [r for r in json.load(open(path)) if r.get("solver") == solver]
        ok = SUCCESS | (ACCEPT if count_acceptable else set())
        out[solver] = {r["name"]: r for r in rows}
        out[solver]["_ok"] = ok
    return out


def cost(row, metric):
    v = row.get("solve_time" if metric == "time" else "iterations")
    return None if v is None else float(v)


def build_matrix(suites, metric, count_acceptable, bench_dir=BENCH):
    """Pool suites. Return (solvers, names, C) where C[s][p] is cost or inf."""
    per = {s: load_suite(s, count_acceptable, bench_dir) for s in suites}
    solvers = []
    for s in suites:
        for sv in per[s]:
            if sv not in solvers:
                solvers.append(sv)
    if not solvers:
        sys.exit("no solver JSONs found for: " + ", ".join(suites))

    # union of problem keys (suite-qualified so names can't collide)
    names = []
    for s in suites:
        any_solver = next(iter(per[s].values()))
        for nm in any_solver:
            if nm == "_ok":
                continue
            names.append(f"{s}/{nm}")
    names = sorted(set(names))

    C = {sv: np.full(len(names), np.inf) for sv in solvers}
    for j, qual in enumerate(names):
        s, nm = qual.split("/", 1)
        for sv in solvers:
            book = per[s].get(sv)
            if not book or nm not in book:
                continue
            row, ok = book[nm], book["_ok"]
            if row["status"] in ok:
                c = cost(row, metric)
                if c is not None and c >= 0:
                    # Floor at a tiny epsilon so log/ratio stay finite for
                    # sub-millisecond / zero-iteration solves.
                    C[sv][j] = max(c, 1e-6)
    return solvers, names, C


def performance_profile(solvers, C):
    """ρ_s(τ): fraction with cost ≤ τ × best-on-that-problem."""
    stack = np.vstack([C[sv] for sv in solvers])
    best = stack.min(axis=0)                      # best cost per problem
    keep = np.isfinite(best)                       # someone solved it
    if not keep.any():
        sys.exit("no problem was solved by any solver — nothing to plot")
    ratios = {sv: C[sv][keep] / best[keep] for sv in solvers}
    nprob = keep.sum()
    # τ grid up to the largest finite ratio.
    finite = np.concatenate([r[np.isfinite(r)] for r in ratios.values()])
    tau_max = max(2.0, finite.max())
    taus = np.geomspace(1.0, tau_max * 1.05, 400)
    curves = {sv: np.array([(ratios[sv] <= t).sum() / nprob for t in taus])
              for sv in solvers}
    return taus, curves, nprob


def data_profile(solvers, C):
    """ECDF: fraction with absolute cost ≤ budget."""
    finite = np.concatenate([C[sv][np.isfinite(C[sv])] for sv in solvers])
    if finite.size == 0:
        sys.exit("no solved problems — nothing to plot")
    nprob = len(next(iter(C.values())))
    lo, hi = max(finite.min(), 1e-6), finite.max()
    budgets = np.geomspace(lo * 0.9, hi * 1.1, 400)
    curves = {sv: np.array([(C[sv] <= b).sum() / nprob for b in budgets])
              for sv in solvers}
    return budgets, curves, nprob


def render_profile(suites, output, metric="time", mode="performance",
                   count_acceptable=True, bench_dir=BENCH, title=None):
    """Build one profile figure and save it to `output`.

    Importable entry point (the report generator calls this). Returns
    (nprob, solvers) or None if there is nothing to plot (e.g. fewer than
    two solvers for a performance profile, or no solved problems).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solvers, names, C = build_matrix(suites, metric, count_acceptable, bench_dir)
    if mode == "performance" and len(solvers) < 2:
        return None  # a single-solver performance profile is degenerate
    metric_label = "solve time (s)" if metric == "time" else "iterations"
    label = title or ", ".join(suites)

    fig, ax = plt.subplots(figsize=(7, 5))
    if mode == "performance":
        x, curves, nprob = performance_profile(solvers, C)
        for sv in solvers:
            ax.step(x, curves[sv], where="post",
                    label=SOLVER_LABEL.get(sv, sv),
                    color=SOLVER_COLOR.get(sv), lw=2)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"performance ratio $\tau$ "
                      f"(× best {metric_label})")
        ax.set_ylabel(r"fraction of problems  $\rho_s(\tau)$")
        ax.set_title(f"Performance profile — {label} "
                     f"({nprob} problems, by {metric})")
        ax.set_xlim(1, x[-1])
    else:
        x, curves, nprob = data_profile(solvers, C)
        for sv in solvers:
            ax.step(x, curves[sv], where="post",
                    label=SOLVER_LABEL.get(sv, sv),
                    color=SOLVER_COLOR.get(sv), lw=2)
        ax.set_xscale("log")
        ax.set_xlabel(f"{metric_label} budget")
        ax.set_ylabel("fraction of problems solved")
        ax.set_title(f"Data profile — {label} "
                     f"({nprob} problems, by {metric})")

    ax.set_ylim(0, 1.02)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return nprob, solvers


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="+", help="suite name(s), e.g. vanderbei qp")
    ap.add_argument("--metric", choices=("time", "iters"), default="time")
    ap.add_argument("--mode", choices=("performance", "data"),
                    default="performance")
    ap.add_argument("--no-acceptable", action="store_true",
                    help="count only Solve_Succeeded as solved "
                         "(default also counts Solved_To_Acceptable_Level)")
    ap.add_argument("-o", "--output", default=None,
                    help="output image path (default profile_<mode>_<metric>.png)")
    args = ap.parse_args()

    out = args.output or f"profile_{args.mode}_{args.metric}.png"
    res = render_profile(args.suites, out, metric=args.metric, mode=args.mode,
                         count_acceptable=not args.no_acceptable)
    if res is None:
        sys.exit("nothing to plot (need ≥2 solvers for a performance "
                 "profile, or no solved problems)")
    nprob, solvers = res
    print(f"wrote {out}  ({nprob} problems, solvers: {', '.join(solvers)})")


if __name__ == "__main__":
    main()
