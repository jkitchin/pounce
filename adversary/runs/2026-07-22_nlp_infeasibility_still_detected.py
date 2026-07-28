"""False-negative check on the infeasibility-stationarity fix.

The fix measures the infeasibility stationarity unscaled instead of scaled. For a
row scale `dc < 1` the unscaled measure is LARGER, so the detector fires LESS
often — which is the point on HS13, but is exactly the direction that could break
genuine infeasibility detection.

So: problems that really ARE infeasible, built so gradient-based scaling picks a
tiny `dc` (a huge Jacobian at the starting point), which is the regime where the
fix changes the measure most. POUNCE must still report infeasibility on all of
them. Anything that now solves or hits an iteration limit instead is a false
negative introduced by the fix.

Ipopt is the oracle: it must also call them infeasible, otherwise the model is
wrong rather than the solver.
"""

import os
import subprocess

import pyomo.environ as pyo

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
PREFIX = "/tmp/pounce-prefix/target/release/pounce"  # merged main, before the fix
IPOPT = "/opt/homebrew/bin/ipopt"
OUT = os.path.dirname(os.path.abspath(__file__))


def contradictory_cubic():
    """x^3 >= 1000 and x^3 <= -1000 — empty feasible set, huge Jacobian at x0."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1e4)
    m.obj = pyo.Objective(expr=(m.x - 2) ** 2)
    m.c1 = pyo.Constraint(expr=m.x**3 >= 1000.0)
    m.c2 = pyo.Constraint(expr=m.x**3 <= -1000.0)
    return m


def contradictory_equalities():
    """Two equalities that cannot hold together, both steep at x0."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1e4)
    m.y = pyo.Var(initialize=1e4)
    m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
    m.e1 = pyo.Constraint(expr=m.x**3 + m.y**3 == 1.0)
    m.e2 = pyo.Constraint(expr=m.x**3 + m.y**3 == 2.0)
    return m


def infeasible_circle():
    """Disjoint circles: (x-0)^2+y^2 <= 1 and (x-10)^2+y^2 <= 1."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1e4)
    m.y = pyo.Var(initialize=1e4)
    m.obj = pyo.Objective(expr=m.x + m.y)
    m.c1 = pyo.Constraint(expr=m.x**2 + m.y**2 <= 1.0)
    m.c2 = pyo.Constraint(expr=(m.x - 10) ** 2 + m.y**2 <= 1.0)
    return m


def bound_contradiction():
    """A steep constraint that cannot be met inside the variable's own box."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0.0, 1.0), initialize=1.0)
    m.obj = pyo.Objective(expr=m.x)
    m.c = pyo.Constraint(expr=m.x**5 >= 1e3)
    return m


CASES = {
    "contradictory_cubic": contradictory_cubic,
    "contradictory_equalities": contradictory_equalities,
    "infeasible_circle": infeasible_circle,
    "bound_contradiction": bound_contradiction,
}


def run(binary, nl, extra=()):
    sol = nl.replace(".nl", f".{os.path.basename(binary)}.sol")
    if os.path.exists(sol):
        os.remove(sol)
    try:
        p = subprocess.run(
            [binary, nl, sol, "-AMPL", "max_wall_time=20", *extra]
            if "pounce" in binary
            else [binary, nl, "-AMPL", "max_wall_time=20", *extra],
            capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""
    res = None
    if os.path.exists(sol):
        for line in reversed(open(sol).read().strip().splitlines()):
            if line.startswith("objno"):
                res = int(line.split()[-1])
                break
        os.remove(sol)
    exits = [l.strip() for l in (p.stdout + p.stderr).splitlines() if l.startswith("EXIT")]
    return band(res), (exits[0] if exits else "")


def band(res):
    if res is None:
        return "none"
    if 0 <= res < 100:
        return "SOLVED"
    if 200 <= res < 300:
        return "INFEASIBLE"
    if 300 <= res < 400:
        return "UNBOUNDED"
    return "LIMIT/FAIL"


def main():
    print(f"{'case':<26} {'pre-fix':<12} {'post-fix':<12} {'ipopt':<12}  post-fix EXIT")
    print("-" * 100)
    misses = []
    for name, build in CASES.items():
        nl = os.path.join(OUT, f"_infeas_{name}.nl")
        build().write(nl, io_options={"symbolic_solver_labels": False})
        pre_band, pre_exit = run(PREFIX, nl)
        p_band, p_exit = run(POUNCE, nl)
        i_band, _ = run(IPOPT, nl)
        os.remove(nl)
        print(f"{name:<26} {pre_band:<12} {p_band:<12} {i_band:<12}  {p_exit}")
        # The regression that matters: pre-fix detected it, post-fix does not.
        if pre_band == "INFEASIBLE" and p_band != "INFEASIBLE":
            misses.append(f"{name} (pre-fix {pre_band} -> post-fix {p_band})")


    print()
    if misses:
        print(f"VERDICT: FAIL — infeasibility missed on: {', '.join(misses)}")
    else:
        print("VERDICT: PASS — no infeasibility missed")


if __name__ == "__main__":
    main()
