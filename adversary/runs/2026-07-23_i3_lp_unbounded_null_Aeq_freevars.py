"""i3 Test 7 — unbounded LP with a recession ray in null(A_eq) over FREE
variables (#285/#306).

#285/#306 fixed the divergence detector missing an unbounded LP whose recession
ray lies in null(A_eq) with free variables. This probes just outside: a
rank-deficient equality system so the null space is MULTI-dimensional, and the
descent recession ray is a *combination* of null directions.

n=5, all free. Equality block (rank 2):
    x1 + x2 + x3 + x4 + x5 = 0
    x1 - x2                = 0
(a third dependent row x1+x2+x3+x4+x5 = 0 duplicated -> still rank 2)
null(A_eq) is 3-dimensional. Objective c = (0,0,1,-1,0):
along d=(0,0,1,-1,0) we have A_eq d = 0 (d in null) and c.d = 1-(-1)=2>0, so
-d is a descent recession ray -> min c'x is UNBOUNDED below.

Bounded control: add x3 <= 100 -> optimum becomes finite (attained), must NOT
be reported unbounded.

Oracle: analytic (unbounded / bounded). scipy linprog confirms status.
"""
from __future__ import annotations
import subprocess
import numpy as np
import pyomo.environ as pyo
from scipy.optimize import linprog

PB = "/Users/jkitchin/projects/pounce/target/release/pounce"


def build(fname, capped):
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 4)
    # capped -> box ALL vars into [-50,50] so the feasible set is compact and the
    # optimum is finite (a proper bounded control). Uncapped -> all free.
    bnds = (-50.0, 50.0) if capped else (None, None)
    m.x = pyo.Var(m.I, bounds=bnds, initialize=0.0)
    c = [0, 0, 1, -1, 0]
    m.obj = pyo.Objective(expr=sum(c[i] * m.x[i] for i in m.I))
    m.e1 = pyo.Constraint(expr=sum(m.x[i] for i in m.I) == 0)
    m.e2 = pyo.Constraint(expr=m.x[0] - m.x[1] == 0)
    m.e3 = pyo.Constraint(expr=sum(m.x[i] for i in m.I) == 0)  # dependent dup
    m.write(fname, io_options={"symbolic_solver_labels": True})


def run(name, capped, expect_unbounded):
    ok = True
    for sel in ("auto", "nlp"):
        nl, sol = f"/tmp/i3t7_{name}.nl", f"/tmp/i3t7_{name}_{sel}.sol"
        build(nl, capped)
        p = subprocess.run([PB, nl, sol, f"solver_selection={sel}"],
                           capture_output=True, text=True)
        head = open(sol).readline().strip()
        objno = next((l for l in open(sol) if l.startswith("objno")), "?").strip()
        unb = "300" in objno or "unbounded" in head.lower() or "Diverging" in head
        print(f"[{name}/{sel}] exit={p.returncode} | {head} | {objno} | "
              f"{'UNBOUNDED' if unb else 'BOUNDED/OTHER'}")
        ok = ok and (unb == expect_unbounded)
    return ok


def main():
    # scipy oracle
    c = [0, 0, 1, -1, 0]
    Aeq = [[1, 1, 1, 1, 1], [1, -1, 0, 0, 0]]
    beq = [0, 0]
    r = linprog(c, A_eq=Aeq, b_eq=beq, bounds=[(None, None)] * 5)
    print(f"scipy linprog (uncapped): status={r.status} ({r.message.strip()}) "
          f"[3=unbounded expected]")
    u = run("uncapped", False, True)
    b = run("capped", True, False)
    if u and b:
        print("VERDICT: PASS (null(A_eq) free-var recession ray -> unbounded; "
              "capped control -> bounded)")
    else:
        print(f"VERDICT: FAIL (uncapped_unbounded={u} capped_bounded={b}; "
              f"scipy says uncapped unbounded — #285 residual if uncapped missed)")


if __name__ == "__main__":
    main()
