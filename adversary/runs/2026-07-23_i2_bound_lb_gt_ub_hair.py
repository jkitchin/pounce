#!/usr/bin/env python
"""i2-3 (#275/#295/#311, family qp): impossible-bound guard at tiny lb-ub gap.

min 0.5 x^2 with lb=1.0, ub=1.0-gap  =>  lb>ub, feasible set empty (should be
primal_infeasible). pounce's impossible-bound guard fires for gap>=1e-6, but:
  gap=1e-8  -> numerical_failure, x=nan
  gap<=1e-10 -> silently returns status='optimal' at x=[1.0], violating ub.
The violation (1e-10) is below the ~1e-8 feasibility tolerance, so a solver may
legitimately treat lb~=ub as a fixed variable => TOLERANCE/borderline, reported
as an observation. Oracle: cvxpy CLARABEL reports 'infeasible' for all gaps.
"""
import numpy as np, cvxpy as cp
from pounce import solve_qp
print("== lb=1.0, ub=1.0-gap (lb>ub, feasible set empty) ==")
for gap in [1e-12,1e-10,1e-8,1e-6,1e-4,1e-2]:
    lb, ub = 1.0, 1.0-gap
    try:
        r=solve_qp(P=np.eye(1),c=np.array([0.0]),lb=np.array([lb]),ub=np.array([ub]))
        st, xv = r.status, r.x
    except Exception as e:
        st, xv = f"RAISED {type(e).__name__}", None
    x=cp.Variable(1); p=cp.Problem(cp.Minimize(0.5*cp.square(x)),[x>=lb,x<=ub]); p.solve(solver=cp.CLARABEL)
    print(f"  gap={gap:.0e}: pounce={st} x={xv} | cvxpy={p.status}")
print("VERDICT: TOLERANCE (borderline: gap<=1e-10 returns 'optimal' at ub-violating point, "
      "but violation < feastol; gap=1e-8 -> numerical_failure not silent-optimal; "
      "gap>=1e-6 correctly primal_infeasible)")
