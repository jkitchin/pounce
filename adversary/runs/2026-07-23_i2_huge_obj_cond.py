#!/usr/bin/env python
"""i2-2 (#286/#307, family qp): huge-magnitude objective + cond(P) normalization.

Neighbor of the huge-objective normalization fix. min 0.5 x'Px + c'x with
P=diag(1e10,1), c=[-1e18,-1]. Closed form x=[1e8, 1.0], obj = -5e25 - 0.5.
pounce recovers x0 and the objective to full relative precision, but the
small-weight coordinate x1 comes back ~1.4e-7 instead of 1.0 (100% rel error
on that coordinate) — the objective normalization masks the negligible-weight
variable. Objective passes tol, so this is TOLERANCE/secondary, not a hard bug.
"""
import numpy as np, cvxpy as cp
from pounce import solve_qp

def rel(a,b): return abs(a-b)/max(1.0,abs(b))
P=np.diag([1e10,1.0]); c=np.array([-1e18,-1.0])
r=solve_qp(P=P,c=c)
x=cp.Variable(2); p=cp.Problem(cp.Minimize(0.5*cp.quad_form(x,cp.psd_wrap(P))+c@x),[]); p.solve(solver=cp.CLARABEL)
true_x=np.array([1e8,1.0]); true_obj=-5e25-0.5
print(f"pounce st={r.status} x={r.x} obj={r.obj:.6e}")
print(f"cvxpy  x={x.value} obj={p.value:.6e}")
print(f"true   x={true_x} obj={true_obj:.6e}")
obj_err=rel(r.obj,true_obj); x1_err=rel(r.x[1],true_x[1])
print(f"obj_rel_err={obj_err:.3e}  x1_rel_err={x1_err:.3e}")
if obj_err<1e-4 and x1_err<1e-4:
    print("VERDICT: PASS")
elif obj_err<1e-4:
    print("VERDICT: TOLERANCE (objective correct to 1e-4 but minimizer coord x1 wrong "
          f"({r.x[1]:.3e} vs 1.0); objective normalization drops the tiny-weight variable)")
else:
    print("VERDICT: FAIL (SOLVER_BUG)")
