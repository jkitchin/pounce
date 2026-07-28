#!/usr/bin/env python
"""i2-9 (#283/#304, family socp): HSDE unboundedness detectors.

Unbounded (below) conic programs must be reported dual_infeasible (the HSDE
unboundedness certificate), not a silent large 'optimal'. Each objective
decreases without bound along a recession ray inside the cone. Oracle = cvxpy
CLARABEL 'unbounded'.
"""
import numpy as np, cvxpy as cp
from pounce import solve_socp
def run(name, c, G, h, cones):
    r=solve_socp(c=np.array(c,float),G=np.array(G,float),h=np.array(h,float),cones=cones)
    good = r.status=="dual_infeasible"
    print(f"  [{name}] status={r.status} success={r.success} obj={r.obj:.3e}  {'OK' if good else 'BAD'}")
    return good
ok=True
# min -x0 s.t. (x0,x1) in soc  => x0>=|x1|>=0, x0->inf : unbounded
ok &= run("soc min -x0", [-1,0], [[-1,0],[0,-1]], [0,0], [("soc",2)])
# min -x0 s.t. (x0,x1) in exp region : unbounded
ok &= run("exp min -x0", [-1,0], [[0,0],[-1,0],[0,-1]], [0,0,0], [("exp",3)])
# min -(X00+X11) over X psd : unbounded
ok &= run("psd min -trace", [-1,0,-1], [[-1,0,0],[0,-1,0],[0,0,-1]], [0,0,0], [("psd",2)])
# cvxpy cross-check for the soc case
x=cp.Variable(2); p=cp.Problem(cp.Minimize(-x[0]),[cp.SOC(x[0],x[1:])]); p.solve(solver=cp.CLARABEL)
print(f"  cvxpy soc oracle: {p.status}")
print("VERDICT: PASS" if ok else "VERDICT: FAIL (HSDE missed an unbounded ray)")
