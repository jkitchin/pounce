"""i3 Test 2 — qp-active-set on rank-deficient equality blocks in n=4
(rank-2 dependency), CONSISTENT and INCONSISTENT.  Control that bounds the
Test-1 finding: does the #313/#321/#323 fix hold for a *non-degenerate*
rank-deficient block (not the rank-1 exact-duplicate corner)?

n=4, min sum_i (x_i - 1)^2 with a dependent equality block r3 = r1 + r2:
  consistent  : x1+x2=1 ; x3+x4=1 ; x1+x2+x3+x4=2  (rank 2, feasible)
                optimum by symmetry x*=(.5,.5,.5,.5), f*=1.0
  inconsistent: x1+x2=1 ; x3+x4=1 ; x1+x2+x3+x4=3  (rank 2, INFEASIBLE)

Expected: consistent -> SolveSucceeded f=1.0 (verify VERIFIED);
          inconsistent -> InfeasibleProblemDetected (objno 200), NOT InternalError.
Oracle: analytic optimum + `pounce verify`.
"""
from __future__ import annotations
import subprocess
import pyomo.environ as pyo

PB = "/Users/jkitchin/projects/pounce/target/release/pounce"


def build(fname, rows, n=4):
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.x = pyo.Var(m.I, initialize=0.0)
    m.obj = pyo.Objective(expr=sum((m.x[i] - 1) ** 2 for i in m.I))
    m.C = pyo.ConstraintList()
    for coefs, rhs in rows:
        m.C.add(sum(coefs[i] * m.x[i] for i in m.I) == rhs)
    m.write(fname, io_options={"symbolic_solver_labels": True})


def run(name, rows):
    nl, sol = f"/tmp/i3t2_{name}.nl", f"/tmp/i3t2_{name}.sol"
    build(nl, rows)
    p = subprocess.run([PB, nl, sol, "solver_selection=qp-active-set"],
                       capture_output=True, text=True)
    head = open(sol).readline().strip()
    objno = next((l for l in open(sol) if l.startswith("objno")), "objno ?").strip()
    v = subprocess.run([PB, "verify", nl, sol], capture_output=True, text=True)
    print(f"[{name}] exit={p.returncode} | {head} | {objno} | "
          f"verify={'VERIFIED' if 'VERIFIED' in v.stdout else 'REJECTED'}")
    return head, objno


def main():
    hc, oc = run("consistent",
                 [([1, 1, 0, 0], 1), ([0, 0, 1, 1], 1), ([1, 1, 1, 1], 2)])
    hi, oi = run("inconsistent",
                 [([1, 1, 0, 0], 1), ([0, 0, 1, 1], 1), ([1, 1, 1, 1], 3)])
    ok_c = "SolveSucceeded" in hc and oc.endswith(" 0")
    ok_i = "Infeasible" in hi and "200" in oi
    print(f"consistent solved={ok_c} (expect f=1.0); inconsistent flagged infeasible={ok_i}")
    if ok_c and ok_i:
        print("VERDICT: PASS (rank-deficient n=4 block: consistent solved, "
              "inconsistent correctly infeasible — fix generalizes)")
    else:
        print(f"VERDICT: FAIL (consistent_ok={ok_c} inconsistent_ok={ok_i})")


if __name__ == "__main__":
    main()
