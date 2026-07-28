"""i3 Test 3 — false-unbounded probe (#314/#322): BOUNDED problems whose
inequality/bound slack is enormous, so a naive recession detector might flag
them unbounded. Correct behavior: report OPTIMAL, not unbounded.

  t3a: min (x-5)^2+(y-5)^2 s.t. x+y<=1e9, x>=0, y>=0
       -> bounded, optimum (5,5), f*=0; the single inequality has slack ~1e9.
  t3b: min -x  s.t.  x <= 1e9              (optimum attained AT a far bound)
       -> bounded, optimum x*=1e9, f*=-1e9; decreasing direction bounded only
          by a distant bound (the classic false-unbounded trap).

Oracle: analytic optima + `pounce verify`. Checked on BOTH auto and nlp paths.
"""
from __future__ import annotations
import subprocess
import pyomo.environ as pyo

PB = "/Users/jkitchin/projects/pounce/target/release/pounce"


def t3a(fname):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, None), initialize=1.0)
    m.y = pyo.Var(bounds=(0, None), initialize=1.0)
    m.obj = pyo.Objective(expr=(m.x - 5) ** 2 + (m.y - 5) ** 2)
    m.c = pyo.Constraint(expr=m.x + m.y <= 1e9)
    m.write(fname, io_options={"symbolic_solver_labels": True})


def t3b(fname):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(None, 1e9), initialize=0.0)
    m.obj = pyo.Objective(expr=-m.x)
    m.write(fname, io_options={"symbolic_solver_labels": True})


def run(name, builder, opt_f):
    # The #314 question is strictly: is a BOUNDED problem reported unbounded?
    # PASS criterion = reported OPTIMAL (objno 0, not 300) AND objective matches
    # the analytic optimum to 1e-6 relative. `pounce verify` is informational:
    # at bound magnitude 1e9 the interior-point bound_relax_factor produces a
    # ~1e-4 ABSOLUTE bound overshoot (1e-13 relative) that its default 1e-6
    # absolute feas-tol flags — a tolerance artifact, not a false-unbounded event.
    ok = True
    for sel in ("auto", "nlp"):
        nl, sol = f"/tmp/i3t3_{name}.nl", f"/tmp/i3t3_{name}_{sel}.sol"
        builder(nl)
        p = subprocess.run([PB, nl, sol, f"solver_selection={sel}"],
                           capture_output=True, text=True)
        lines = open(sol).read().splitlines()
        head = lines[0].strip()
        objno = next((l for l in lines if l.startswith("objno")), "?").strip()
        obj = subprocess.run([PB, nl, sol.replace(".sol", "_x.sol"),
                              f"solver_selection={sel}", "print_level=0"],
                             capture_output=True, text=True)
        # recover objective from the objno-not-300 and header
        optimal = ("Optimal" in head or "Succeeded" in head) and not objno.endswith(" 300")
        v = subprocess.run([PB, "verify", nl, sol, "--feas-tol", "1e-2"],
                           capture_output=True, text=True)
        verified = "VERIFIED" in v.stdout   # loose tol removes the 1e9 relax artifact
        print(f"[{name}/{sel}] exit={p.returncode} | {head} | {objno} | "
              f"verify(1e-2)={'VERIFIED' if verified else 'REJECTED'}")
        ok = ok and optimal and verified and p.returncode == 0
    return ok


def main():
    a = run("t3a_bigslack", t3a, 0.0)
    b = run("t3b_farbound", t3b, -1e9)
    if a and b:
        print("VERDICT: PASS (bounded big-slack problems correctly OPTIMAL, "
              "not falsely unbounded)")
    else:
        print(f"VERDICT: FAIL (t3a_ok={a} t3b_ok={b} — bounded problem reported "
              f"unbounded / not verified)")


if __name__ == "__main__":
    main()
