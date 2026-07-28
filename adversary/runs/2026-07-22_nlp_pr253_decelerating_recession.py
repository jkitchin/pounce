"""Adversary cross-check: unbounded-below objectives with DECELERATING descent.

Family: nlp   Class: unboundedness detection (false negative)
Target:  PR #253 (issue #252) — the divergence guard now requires the objective's
         per-step drop to be *non-decelerating* before it will report
         DivergingIterates:

             keeping_up = decrease >= prev_decrease * 0.9
             streak accumulates only on   growing AND descending AND keeping_up

         The stated rationale is "a genuine recession ray drives f toward -inf
         with a per-step drop that keeps up as |x| grows geometrically, whereas
         an excursion converging to a finite optimum decelerates toward zero."

HYPOTHESIS: that premise is false for an important class of genuinely unbounded
problems. Any objective that diverges to -inf SUB-LINEARLY in |x| has per-step
drops that decelerate to zero while still being unbounded below. Examples:

    min -log(x)     s.t. x >= 1     ->  f -> -inf,  drop per doubling = -log 2 (constant-ish)
    min -sqrt(x)    s.t. x >= 1     ->  f -> -inf,  drop per doubling GROWS
    min -x**0.25    s.t. x >= 1     ->  f -> -inf

The decisive case is -log(x): along a GEOMETRIC growth path (which is what the
guard measures against, since `growing` requires a >= 2*prev), f drops by a
CONSTANT log(2) per doubling. Constant is not "keeping up" in the strict sense
only if the ratio dips below 0.9 -- but any noise in the step sizes makes the
ratio oscillate around 1. More importantly, on a path where |x| grows FASTER
than geometrically (typical late-stage IPM behaviour on a ray), -log(x) drops
decelerate relative to the previous step whenever the growth factor shrinks.

These are all UNBOUNDED. Ipopt reports Diverging_Iterates / Unbounded. If POUNCE
now returns "optimal" or a converged status on them, PR #253 traded a false
positive for a FALSE NEGATIVE: silently claiming an optimum for a problem with
no finite optimum. For a branch-and-bound driver that is strictly worse than the
bug it fixed -- a missed unboundedness is an incorrect fathom / wrong bound,
whereas a spurious unbounded verdict was at least detectable.

Oracle: Ipopt 3.x (/opt/homebrew/bin/ipopt) on the identical .nl, via Pyomo.
"""

import os
import subprocess
import time

import pyomo.environ as pyo

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"
OUT = os.path.dirname(os.path.abspath(__file__))

# Each case: (name, objective rule, expectation). All are unbounded below.
CASES = {
    # f = -log(x): drop per doubling of x is a CONSTANT log 2.
    "neg_log": lambda m: -pyo.log(m.x),
    # f = -sqrt(x): drop per doubling GROWS (should be detected even by a
    # strict "non-decelerating" rule) -- included as the control that the
    # guard still works when its premise holds.
    "neg_sqrt": lambda m: -pyo.sqrt(m.x),
    # f = -x^(1/4): sub-linear, slower than sqrt.
    "neg_quartic_root": lambda m: -(m.x**0.25),
    # f = -log(x) - log(y): two-variable version, both free above.
    "neg_log2": lambda m: -pyo.log(m.x) - pyo.log(m.y),
}


def build(case):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1.0, None), initialize=2.0)
    m.y = pyo.Var(bounds=(1.0, None), initialize=2.0)
    m.obj = pyo.Objective(rule=CASES[case], sense=pyo.minimize)
    # A harmless constraint so the .nl has a constraint block.
    m.c = pyo.Constraint(expr=m.x + m.y >= 2.0)
    return m


def write_nl(m, path):
    m.write(path, io_options={"symbolic_solver_labels": False})


def run_solver(binary, nl, extra):
    sol = nl.replace(".nl", f".{os.path.basename(binary)}.sol")
    t0 = time.perf_counter()
    p = subprocess.run(
        [binary, nl, "-AMPL", *extra], capture_output=True, text=True, timeout=120
    )
    dt = time.perf_counter() - t0
    txt = p.stdout + p.stderr
    solres = None
    if os.path.exists(sol):
        last = open(sol).read().strip().splitlines()[-1]
        if last.startswith("objno"):
            solres = int(last.split()[-1])
    return txt, dt, solres


def classify(solres, text):
    """AMPL solve_result_num bands: 0-99 solved, 200-299 infeasible,
    300-399 unbounded, 400+ limit/failure."""
    if solres is None:
        return "no-sol-file"
    if 0 <= solres < 100:
        return f"SOLVED({solres})"
    if 200 <= solres < 300:
        return f"INFEASIBLE({solres})"
    if 300 <= solres < 400:
        return f"UNBOUNDED({solres})"
    return f"LIMIT/FAIL({solres})"


def main():
    rows = []
    for case in CASES:
        m = build(case)
        nl = os.path.join(OUT, f"_pr253_{case}.nl")
        write_nl(m, nl)

        # POUNCE at default settings, and with a low diverging_iterates_tol
        # (the setting a branch-and-bound driver uses to abort runaway nodes --
        # exactly the configuration PR #253 was tuning for).
        p_def, t_p, r_p = run_solver(POUNCE, nl, [])
        p_low, t_pl, r_pl = run_solver(POUNCE, nl, ["diverging_iterates_tol=1e6"])
        i_txt, t_i, r_i = run_solver(IPOPT, nl, [])

        rows.append(
            dict(
                case=case,
                pounce_default=classify(r_p, p_def),
                pounce_lowtol=classify(r_pl, p_low),
                ipopt=classify(r_i, i_txt),
                t_pounce=t_p,
                t_ipopt=t_i,
                pounce_exit=[
                    l for l in p_def.splitlines() if l.startswith("EXIT")
                ],
                ipopt_exit=[l for l in i_txt.splitlines() if l.startswith("EXIT")],
            )
        )

    print(f"{'case':<20} {'pounce(default)':<22} {'pounce(divtol=1e6)':<22} {'ipopt':<22}")
    print("-" * 88)
    for r in rows:
        print(
            f"{r['case']:<20} {r['pounce_default']:<22} {r['pounce_lowtol']:<22} {r['ipopt']:<22}"
        )
    print()
    for r in rows:
        print(f"{r['case']}: pounce {r['pounce_exit']}  |  ipopt {r['ipopt_exit']}")

    # A false negative = ipopt says unbounded, pounce says solved.
    fn = [
        r
        for r in rows
        if r["ipopt"].startswith("UNBOUNDED")
        and (
            r["pounce_default"].startswith("SOLVED")
            or r["pounce_lowtol"].startswith("SOLVED")
        )
    ]
    print()
    if fn:
        print(f"VERDICT: FAIL — {len(fn)} false negative(s): " + ", ".join(r["case"] for r in fn))
    else:
        print("VERDICT: PASS (no missed unboundedness)")


if __name__ == "__main__":
    main()
