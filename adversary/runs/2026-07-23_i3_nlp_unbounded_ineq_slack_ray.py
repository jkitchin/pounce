"""i3 Test 4 — genuine unbounded-below via an INEQUALITY-slack recession ray
(#314/#322). Correct behavior: report unbounded (objno 300 / DivergingIterates),
exit 1.

  min -x  s.t.  x - y <= 3,  y - x <= 3   (x,y free)   [a corridor |x-y|<=3]
The recession ray d=(1,1) keeps BOTH inequality slacks constant at 3 while the
objective -> -inf: a recession ray that lives on the *inequality* slack surface
(distinct from a null(A_eq) ray). Unbounded below.

Oracle: analytic (unbounded). A bounded control (min -x s.t. x<=10, corridor)
is checked to confirm the detector is not trigger-happy.
"""
from __future__ import annotations
import subprocess
import pyomo.environ as pyo

PB = "/Users/jkitchin/projects/pounce/target/release/pounce"


def unbounded(fname):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=0.0)
    m.y = pyo.Var(initialize=0.0)
    m.obj = pyo.Objective(expr=-m.x)
    m.c1 = pyo.Constraint(expr=m.x - m.y <= 3)
    m.c2 = pyo.Constraint(expr=m.y - m.x <= 3)
    m.write(fname, io_options={"symbolic_solver_labels": True})


def bounded_control(fname):
    # same corridor but cap x -> optimum x=10 attained
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(None, 10.0), initialize=0.0)
    m.y = pyo.Var(initialize=0.0)
    m.obj = pyo.Objective(expr=-m.x)
    m.c1 = pyo.Constraint(expr=m.x - m.y <= 3)
    m.c2 = pyo.Constraint(expr=m.y - m.x <= 3)
    m.write(fname, io_options={"symbolic_solver_labels": True})


def run(name, builder, expect_unbounded):
    ok = True
    for sel in ("auto", "nlp"):
        nl, sol = f"/tmp/i3t4_{name}.nl", f"/tmp/i3t4_{name}_{sel}.sol"
        builder(nl)
        p = subprocess.run([PB, nl, sol, f"solver_selection={sel}"],
                           capture_output=True, text=True)
        head = open(sol).readline().strip()
        objno = next((l for l in open(sol) if l.startswith("objno")), "?").strip()
        unb = ("300" in objno) or ("unbounded" in head.lower()) or \
              ("Diverging" in head)
        got = "UNBOUNDED" if unb else "BOUNDED/OTHER"
        print(f"[{name}/{sel}] exit={p.returncode} | {head} | {objno} | {got}")
        ok = ok and (unb == expect_unbounded)
    return ok


def main():
    u = run("corridor_unbounded", unbounded, True)
    b = run("corridor_bounded", bounded_control, False)
    if u and b:
        print("VERDICT: PASS (inequality-slack recession ray correctly unbounded; "
              "capped control correctly bounded)")
    else:
        print(f"VERDICT: FAIL (unbounded_case_ok={u} bounded_control_ok={b})")


if __name__ == "__main__":
    main()
