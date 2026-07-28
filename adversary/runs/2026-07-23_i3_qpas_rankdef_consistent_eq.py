"""i3 Test 1 — qp-active-set InternalError on some rank-deficient CONSISTENT
equality blocks (residual gap of #313/#321/#323).

#313/#321/#323 stopped the dedicated active-set QP engine (pounce-qp, reached
via CLI `solver_selection=qp-active-set`) from raising an internal error / exit-1
on an exactly rank-deficient but *consistent* equality block. This probes just
outside that patch.

All four instances below constrain the SAME feasible line x+y=2 in 2 variables,
so all share the analytic optimum of  min (x-1)^2+(y-2)^2 :
    x* = (0.5, 1.5),  f* = 0.5   (project (1,2) onto x+y=2; KKT-unique).
The only difference is how the (redundant) equality rows are written:

  r_exactdup : [x+y=2 ; x+y=2]                (2 identical rows)
  r_scaled3  : [x+y=2 ; 2x+2y=4 ; 3x+3y=6]    (3 rows, distinct scalings)
  r_triple   : [x+y=2 ; x+y=2 ; x+y=2]        (3 identical rows)
  r_combo    : [x+y=2 ; 3x+3y=6 ; 4x+4y=8]    (3 rows, r3 = r1+r2)

Oracle for the optimum: analytic KKT (unassailable) + pounce's OWN success on
the r_scaled3 encoding of the *identical* feasible set (self-consistency).
`pounce verify` independently checks each returned .sol against the .nl.

Finding shape: pounce SOLVES r_exactdup and r_scaled3 (SolveSucceeded, verify
VERIFIED) but returns `InternalError` (objno 500, exit 1, verify REJECTED) on
r_triple and r_combo — different encodings of the SAME feasible set and SAME
optimum. That residual InternalError is exactly the symptom #313/#323 targeted.
"""
from __future__ import annotations
import subprocess
import numpy as np
import pyomo.environ as pyo

PB = "/Users/jkitchin/projects/pounce/target/release/pounce"
OPT_X = np.array([0.5, 1.5])
OPT_F = 0.5

ROWS = {
    "r_exactdup": [(1, 1, 2), (1, 1, 2)],
    "r_scaled3":  [(1, 1, 2), (2, 2, 4), (3, 3, 6)],
    "r_triple":   [(1, 1, 2), (1, 1, 2), (1, 1, 2)],
    "r_combo":    [(1, 1, 2), (3, 3, 6), (4, 4, 8)],
}


def build_nl(fname, rows):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=0.0)
    m.y = pyo.Var(initialize=0.0)
    m.obj = pyo.Objective(expr=(m.x - 1) ** 2 + (m.y - 2) ** 2)
    m.C = pyo.ConstraintList()
    for a, b, r in rows:
        m.C.add(a * m.x + b * m.y == r)
    m.write(fname, io_options={"symbolic_solver_labels": True})


def run(name, rows):
    nl = f"/tmp/i3t1_{name}.nl"
    sol = f"/tmp/i3t1_{name}.sol"
    build_nl(nl, rows)
    p = subprocess.run([PB, nl, sol, "solver_selection=qp-active-set"],
                       capture_output=True, text=True)
    with open(sol) as fh:
        head = fh.readline().strip()
        txt = fh.read()
    objno = next((ln for ln in txt.splitlines() if ln.startswith("objno")), "objno ?")
    v = subprocess.run([PB, "verify", nl, sol], capture_output=True, text=True)
    verified = "VERIFIED" in v.stdout
    print(f"[{name}] exit={p.returncode} | {head} | {objno} | "
          f"verify={'VERIFIED' if verified else 'REJECTED'}")
    solved = ("SolveSucceeded" in head) and verified and p.returncode == 0
    return solved


def main():
    print("Analytic optimum for all encodings: x*=(0.5,1.5), f*=0.5")
    res = {name: run(name, rows) for name, rows in ROWS.items()}
    print("Solved:", res)
    # residual gap = at least one consistent rank-deficient encoding InternalErrors
    # while another encoding of the SAME feasible set succeeds.
    any_ok = any(res.values())
    any_bad = not all(res.values())
    if any_ok and any_bad:
        bad = [k for k, ok in res.items() if not ok]
        print(f"VERDICT: FAIL (qp-active-set InternalError/exit-1 on consistent "
              f"rank-deficient equality encodings {bad}, while other encodings of "
              f"the SAME feasible set (x+y=2, opt f=0.5) solve cleanly — residual "
              f"gap of #313/#321/#323)")
    elif all(res.values()):
        print("VERDICT: PASS (all consistent rank-deficient encodings solved)")
    else:
        print("VERDICT: FAIL (all encodings failed)")


if __name__ == "__main__":
    main()
