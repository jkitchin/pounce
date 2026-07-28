"""i3 Test 6 — active-set false infeasibility when m/n >> 1 on a FEASIBLE QP
(#282/#303).

#282/#303 fixed the active-set QP engine certifying infeasibility on a FEASIBLE
problem once m/n >= 5. This probes the same regime with a redundant, strictly
feasible inequality stack.

n=3, min ||x - c||^2 with c=(0.5,0.5,0.5) STRICTLY interior to [0,1]^3.
The box's 6 facets are each replicated with a tiny random inward shrink, giving
m = 18 inequalities (m/n = 6). The optimum is unconstrained: x* = c, f* = 0,
with NO active inequality (all 18 slack). A false "infeasible" here is the bug.

Two more feasible instances at m/n = 8 and 12 (more replicas) for breadth.

Oracle: analytic optimum (x*=c, f*=0) + pounce's convex QP IPM path (solve_qp).
Solved through the dedicated active-set engine via CLI solver_selection=qp-active-set.
"""
from __future__ import annotations
import subprocess
import numpy as np
import pounce
import pyomo.environ as pyo

PB = "/Users/jkitchin/projects/pounce/target/release/pounce"
C = np.array([0.5, 0.5, 0.5])


def build(fname, reps):
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 2)
    m.x = pyo.Var(m.I, bounds=(-10, 10), initialize=0.0)
    m.obj = pyo.Objective(expr=sum((m.x[i] - C[i]) ** 2 for i in m.I))
    m.C = pyo.ConstraintList()
    rng = np.random.default_rng(1)
    for i in range(3):
        for k in range(reps):
            eps = 1e-6 * rng.random()
            m.C.add(m.x[i] <= 1.0 - eps)     # x_i <= 1 (feasible, slack ~0.5)
            m.C.add(m.x[i] >= 0.0 + eps)      # x_i >= 0 (feasible, slack ~0.5)
    m.write(fname, io_options={"symbolic_solver_labels": True})
    return 6 * reps  # m


def run(reps):
    nl, sol = f"/tmp/i3t6_{reps}.nl", f"/tmp/i3t6_{reps}.sol"
    mcount = build(nl, reps)
    p = subprocess.run([PB, nl, sol, "solver_selection=qp-active-set"],
                       capture_output=True, text=True)
    head = open(sol).readline().strip()
    objno = next((l for l in open(sol) if l.startswith("objno")), "?").strip()
    v = subprocess.run([PB, "verify", nl, sol], capture_output=True, text=True)
    verified = "VERIFIED" in v.stdout
    infeas_claim = "Infeasible" in head or "200" in objno
    print(f"[m/n={mcount/3:.0f}, m={mcount}] {head} | {objno} | "
          f"verify={'VERIFIED' if verified else 'REJECTED'} | "
          f"infeasible_claim={infeas_claim}")
    ok = ("Succeeded" in head) and verified and not infeas_claim
    return ok


def main():
    # sanity: solve_qp (IPM) confirms feasibility/optimum
    qp = pounce.solve_qp(P=2 * np.eye(3), c=-2 * C,
                         lb=np.zeros(3), ub=np.ones(3))
    print(f"solve_qp IPM (box only): x={np.round(qp.x,4)} status={qp.status} "
          f"(analytic x*=c=(0.5,0.5,0.5), f*=0)")
    res = {r: run(r) for r in (3, 4, 6)}  # m/n = 6, 8, 12
    print("per-instance ok:", res)
    if all(res.values()):
        print("VERDICT: PASS (high-m/n feasible QP solved, no false infeasibility)")
    else:
        bad = [r for r, ok in res.items() if not ok]
        print(f"VERDICT: FAIL (active-set false-infeasible / unverified on feasible "
              f"high-m/n QP at reps {bad}; analytic opt x*=c f*=0 — #282 residual)")


if __name__ == "__main__":
    main()
