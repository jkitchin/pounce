#!/usr/bin/env python
"""i2-8 (#283/#304, family socp): HSDE cone-infeasibility detectors.

Primal-infeasible instances on each cone family must be reported
primal_infeasible (not a silent 'optimal'). Slack s = h - Gx forced into a
point outside the cone via constant rows (G=0). Oracle = correct expected
status (infeasible).
"""
import numpy as np
from pounce import solve_socp
def run(name, cones, h, n=1):
    G=np.zeros((len(h),n))
    r=solve_socp(c=np.zeros(n),G=G,h=np.array(h,float),cones=cones)
    good = r.status=="primal_infeasible"
    print(f"  [{name}] status={r.status} success={r.success}  {'OK' if good else 'BAD'}")
    return good
ok=True
ok &= run("soc t=-1>=|.|", [("soc",2)], [-1.0, 0.0])      # s0=-1 < |s1|
ok &= run("exp z=-1",      [("exp",3)], [0.0,1.0,-1.0])   # y=1,x=0 needs z>=1, z=-1
ok &= run("psd X00=-1",    [("psd",2)], [-1.0,0.0,0.0])   # negative diagonal
ok &= run("pow y=-1",      [("pow",0.5)], [0.0,-1.0,1.0]) # y<0
print("VERDICT: PASS" if ok else "VERDICT: FAIL (HSDE missed a primal-infeasible cone)")
