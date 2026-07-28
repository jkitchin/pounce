"""Extended two-sided tangency sweep on the convex-QCQP conic path.

The status battery found that two unit disks centred at +-d, separated by a gap
of -1e-7 (i.e. provably disjoint, hence infeasible), drive POUNCE's convex QCQP
conic IPM to `Numerical failure in KKT factorization` with solve_result_num 500,
where Ipopt returns 200 (local infeasibility).  At -1e-6 POUNCE is correct.

This sweep locates the boundary on BOTH sides and, crucially, checks whether the
*feasible* side degrades the same way -- a 500 on a feasible instance would be a
much worse outcome than a 500 on an infeasible one, and a `solved` with a wrong
objective would be worse still.

Geometry (exact):
    disks (x+d)^2+y^2 <= 1 and (x-d)^2+y^2 <= 1, objective min y
    gap := 2*(1-d)
    gap > 0 -> feasible; the lens is  |x| <= 1-d,  min y = -sqrt(1-d^2)
    gap < 0 -> disks are disjoint by |gap|; infeasible
"""

import math
import os
import subprocess
import tempfile

import pyomo.environ as pyo

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"
WORK = tempfile.mkdtemp(prefix="adv_tang_")


def band(res):
    if res is None:
        return "NONE"
    return ["SOLVED", "ACCEPT", "INFEAS", "UNBND", "LIMIT", "FAILURE"][min(res // 100, 5)]


def solve(binary, d, tag):
    wd = os.path.join(WORK, tag)
    os.makedirs(wd, exist_ok=True)
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-5, 5), initialize=0.0)
    m.y = pyo.Var(bounds=(-5, 5), initialize=0.0)
    m.obj = pyo.Objective(expr=m.y)
    m.c1 = pyo.Constraint(expr=(m.x + d) ** 2 + m.y**2 <= 1.0)
    m.c2 = pyo.Constraint(expr=(m.x - d) ** 2 + m.y**2 <= 1.0)
    nl = os.path.join(wd, "m.nl")
    m.write(nl, io_options={"symbolic_solver_labels": False})
    p = subprocess.run([binary, nl, "-AMPL", "max_wall_time=8"],
                       capture_output=True, text=True, timeout=30, cwd=wd)
    sol = os.path.join(wd, "m.sol")
    res = None
    if os.path.exists(sol):
        for line in reversed(open(sol).read().strip().splitlines()):
            if line.startswith("objno"):
                res = int(line.split()[-1])
                break
    obj = None
    for line in (p.stdout + p.stderr).splitlines():
        s = line.strip()
        if s.startswith("Objective") and ":" in s:
            try:
                obj = float(s.split()[-1])
            except ValueError:
                pass
    msg = ""
    for line in (p.stdout + p.stderr).splitlines():
        s = line.strip()
        if "POUNCE (" in s or s.startswith("EXIT"):
            msg = s.split("):", 1)[-1].strip() if "):" in s else s
    return res, band(res), obj, msg


print("=" * 100)
print("FEASIBLE side: gap > 0, exact min y = -sqrt(1-d^2)")
print("=" * 100)
print(f"{'gap':<10} {'pounce':<9} {'ipopt':<9} {'pounce_obj':>14} {'exact':>14} {'rel_err':>10}  note")
feas_bad = []
for gap in (1e-2, 1e-4, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10):
    d = 1.0 - gap / 2.0
    exact = -math.sqrt(max(0.0, 1.0 - d * d))
    pr, pb, po, pm = solve(POUNCE, d, f"pf{gap:.0e}")
    ir, ib, io_, _ = solve(IPOPT, d, f"if{gap:.0e}")
    err = abs(po - exact) / max(1.0, abs(exact)) if po is not None else float("nan")
    note = ""
    if pb in ("INFEAS", "UNBND"):
        note = "FALSE INFEASIBLE"
        feas_bad.append((gap, pb))
    elif pb == "FAILURE":
        note = "numerical failure"
    elif pb == "SOLVED" and err > 1e-5:
        note = "SOLVED but objective wrong"
        feas_bad.append((gap, err))
    print(f"{gap:<10.0e} {pb:<9} {ib:<9} {po!s:>14.14} {exact:>14.6e} {err:>10.2e}  {note}")

print()
print("=" * 100)
print("INFEASIBLE side: gap < 0, disks disjoint by |gap|")
print("=" * 100)
print(f"{'gap':<10} {'pounce':<9} {'ipopt':<9}  pounce message")
infeas_bad = []
for gap in (-1e-2, -1e-4, -1e-6, -3e-7, -1e-7, -1e-8, -1e-9):
    d = 1.0 - gap / 2.0
    pr, pb, po, pm = solve(POUNCE, d, f"pi{abs(gap):.0e}_{gap<0}")
    ir, ib, io_, _ = solve(IPOPT, d, f"ii{abs(gap):.0e}_{gap<0}")
    flag = ""
    if pb == "SOLVED" and abs(gap) > 1e-6:
        flag = "  <== FALSE OPTIMAL"
        infeas_bad.append((gap, "false optimal"))
    elif pb == "FAILURE" and ib == "INFEAS":
        flag = "  <== degraded vs Ipopt"
        infeas_bad.append((gap, "500 vs ipopt 200"))
    print(f"{gap:<10.0e} {pb:<9} {ib:<9}  {pm[:60]}{flag}")

print()
print("=" * 100)
print(f"feasible-side failures : {feas_bad}")
print(f"infeasible-side degrade: {infeas_bad}")
if feas_bad:
    print("VERDICT: FAIL (false infeasibility or wrong objective on a feasible instance)")
elif infeas_bad:
    print("VERDICT: DEGRADED (honest failure status, but weaker than the oracle)")
else:
    print("VERDICT: PASS")
