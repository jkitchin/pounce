"""Does the false-infeasibility fix cover the EQUALITY-constraint block too?

HS13 has one inequality and no equalities, so the regression tests for
`curr_unscaled_infeasibility_stationarity` only ever exercised the `d` block and
its scale vector `dd`. The `c` block and `dc` went untested, and a silent
downcast failure there would hand the caller the scaled residual — reinstating
the bug on exactly the problems the fix is supposed to cover.

This builds the equality analogue of the HS13 case: a feasible, equality-
constrained NLP whose Jacobian at a remote starting point is enormous, so
gradient-based scaling picks a tiny `dc`. If the fix covers the `c` block, POUNCE
must not report infeasibility and must agree with Ipopt.

    min  (x - 2)^2 + (y - 1)^2
    s.t. (1 - x)^3 - y == 0                        (equality version of HS13)
         x, y >= 0
    x0 = (1e4, 1e4)

The equality makes the feasible set a curve, and the cubic gives the same
vanishing-gradient structure that makes HS13 hard.
"""

import os
import subprocess

import pyomo.environ as pyo

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"
OUT = os.path.dirname(os.path.abspath(__file__))


def build(nl):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=1e4)
    m.y = pyo.Var(bounds=(0, None), initialize=1e4)
    m.obj = pyo.Objective(expr=(m.x - 2) ** 2 + (m.y - 1) ** 2, sense=pyo.minimize)
    m.eq = pyo.Constraint(expr=(1 - m.x) ** 3 - m.y == 0)
    m.write(nl, io_options={"symbolic_solver_labels": False})


def run(binary, nl, extra=()):
    sol = nl.replace(".nl", f".{os.path.basename(binary)}.sol")
    if os.path.exists(sol):
        os.remove(sol)
    p = subprocess.run([binary, nl, "-AMPL", *extra], capture_output=True, text=True, timeout=120)
    res = None
    if os.path.exists(sol):
        for line in reversed(open(sol).read().strip().splitlines()):
            if line.startswith("objno"):
                res = int(line.split()[-1])
                break
        os.remove(sol)
    txt = p.stdout + p.stderr
    exits = [l.strip() for l in txt.splitlines() if l.startswith("EXIT")]
    obj = None
    for line in txt.splitlines():
        if line.strip().startswith("Objective"):
            try:
                obj = float(line.split()[-1])
            except ValueError:
                pass
    return res, obj, (exits[0] if exits else "")


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
    nl = os.path.join(OUT, "_eq_hs13.nl")
    build(nl)
    rows = []
    for label, binary, extra in (
        ("pounce (default)", POUNCE, ()),
        ("pounce (no scaling)", POUNCE, ("nlp_scaling_method=none",)),
        ("ipopt", IPOPT, ()),
    ):
        res, obj, exit_line = run(binary, nl, extra)
        rows.append((label, band(res), obj, exit_line))
        print(f"{label:<22} {band(res):<12} obj={obj}   {exit_line}")
    os.remove(nl)

    print()
    pounce_default = rows[0][1]
    if pounce_default == "INFEASIBLE" and rows[2][1] != "INFEASIBLE":
        print("VERDICT: FAIL — equality (dc) block still reports false infeasibility;")
        print("         the fix does not cover it.")
    elif pounce_default == rows[2][1]:
        print("VERDICT: PASS — equality (dc) block agrees with Ipopt.")
    else:
        print(f"VERDICT: INCONCLUSIVE — pounce {pounce_default}, ipopt {rows[2][1]}")


if __name__ == "__main__":
    main()
