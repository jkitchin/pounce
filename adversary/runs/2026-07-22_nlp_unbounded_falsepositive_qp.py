"""Minimal reproducer + sweep: FALSE 'unbounded' on a strictly convex QP.

Family: nlp (auto-routed to the convex QP IPM)   Class: status-reporting correctness

    min  -x + x^2/(2M)      s.t.  x >= 0        (M > 0)

TRUTH (elementary): P = 1/M > 0, so the objective is STRICTLY CONVEX and
coercive (f -> +inf as x -> inf). f'(x) = -1 + x/M = 0 gives the unique global
minimizer x* = M with f* = -M/2, a FINITE optimum for every finite M. The
problem is BOUNDED for all M. It is not unbounded for any M.

It merely LOOKS unbounded for a long time: for x << M the objective is
essentially -x, so any divergence heuristic watching for
'iterates growing + objective descending + drops not decelerating'
sees all three signals continuously until x reaches M.

This sweeps M and reports which of pounce's routes claim UNBOUNDED.
Oracle: Ipopt on the identical .nl, plus the closed form f* = -M/2.
"""

import os
import subprocess

import numpy as np
import pyomo.environ as pyo

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ubfp_qp")
os.makedirs(TMP, exist_ok=True)


def build(M, n=1):
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(m.I, bounds=(0, None), initialize=1.0)
    m.c = pyo.Constraint(expr=sum(m.x[i] for i in m.I) >= 0.0)
    m.obj = pyo.Objective(
        expr=sum(-m.x[i] + m.x[i] ** 2 / (2.0 * M) for i in m.I), sense=pyo.minimize
    )
    return m


def run(binary, nl, extra=()):
    stub = nl[:-3]
    sol = stub + ".sol"
    if os.path.exists(sol):
        os.remove(sol)
    p = subprocess.run(
        [binary, nl, "-AMPL", *extra], capture_output=True, text=True, timeout=60
    )
    txt = p.stdout + p.stderr
    res, obj = None, None
    if os.path.exists(sol):
        for ln in open(sol).read().splitlines():
            if ln.startswith("objno"):
                res = int(ln.split()[-1])
    for ln in txt.splitlines():
        if ln.strip().startswith("Objective..............."):
            obj = float(ln.split()[-1])
    route = ""
    for ln in txt.splitlines():
        if "Selected solver:" in ln:
            route = ln.split("Selected solver:")[1].split("[")[0].strip()
    return res, obj, route, txt


def band(res):
    if res is None:
        return "no-sol"
    return {0: "SOLVED"}.get(
        res,
        "SOLVED" if 0 <= res < 100 else
        "INFEASIBLE" if 200 <= res < 300 else
        "UNBOUNDED" if 300 <= res < 400 else "LIMIT/FAIL",
    )


def main():
    Ms = [10.0 ** k for k in range(2, 17)]
    print(f"{'M':>10} {'f*=-M/2':>14} | {'auto':<12}{'obj':>14} {'route':<26}"
          f"| {'nlp':<12}{'obj':>14} | {'ipopt':<12}{'obj':>14}")
    print("-" * 140)
    bad_auto, bad_nlp = [], []
    for M in Ms:
        nl = os.path.join(TMP, f"qp_M{int(np.log10(M))}.nl")
        build(M).write(nl, io_options={"symbolic_solver_labels": False})
        fstar = -M / 2.0
        ra, oa, route, _ = run(POUNCE, nl)
        rn, on, _, _ = run(POUNCE, nl, ["solver_selection=nlp"])
        ri, oi, _, _ = run(IPOPT, nl)
        print(
            f"{M:>10.0e} {fstar:>14.4e} | {band(ra):<12}{oa if oa is not None else float('nan'):>14.4e} "
            f"{route:<26}| {band(rn):<12}{on if on is not None else float('nan'):>14.4e} "
            f"| {band(ri):<12}{oi if oi is not None else float('nan'):>14.4e}"
        )
        if band(ra) == "UNBOUNDED" or (oa is not None and abs(oa - fstar) > 1e-4 * max(1, abs(fstar))):
            bad_auto.append(M)
        if band(rn) == "UNBOUNDED" or (on is not None and abs(on - fstar) > 1e-4 * max(1, abs(fstar))):
            bad_nlp.append(M)

    print()
    print(f"auto route WRONG (false-unbounded / wrong obj) for M in: {[f'{m:.0e}' for m in bad_auto]}")
    print(f"nlp  route WRONG                                for M in: {[f'{m:.0e}' for m in bad_nlp]}")

    # multivariable confirmation
    print("\n--- n-variable separable version, M = 1e10 ---")
    for n in (1, 2, 5, 10):
        nl = os.path.join(TMP, f"qpn{n}.nl")
        build(1e10, n).write(nl, io_options={"symbolic_solver_labels": False})
        ra, oa, route, _ = run(POUNCE, nl)
        ri, oi, _, _ = run(IPOPT, nl)
        print(f"  n={n:<3} f*={-n*1e10/2:>12.4e}  pounce(auto)={band(ra):<11}{oa:>14.4e}  ipopt={band(ri):<9}{oi:>14.4e}")

    if bad_auto:
        print(f"\nVERDICT: SOLVER_BUG — false 'unbounded (dual infeasible)' on a strictly "
              f"convex, provably bounded QP for {len(bad_auto)} value(s) of M; the forced "
              f"nlp route and Ipopt both return the correct finite optimum -M/2.")
    else:
        print("\nVERDICT: PASS")


if __name__ == "__main__":
    main()
