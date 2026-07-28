#!/usr/bin/env python
"""i2-1 (#293/#309/#320, family qp): mixed-scale diagonal Hessian.

Probes the neighbor of the tiny/mixed-scale Hessian fix. A QP whose Hessian
simultaneously has a TINY curvature (1e-10) and a HUGE curvature (1e10) — so
cond(P) ~ 1e20 — is well-posed and Clarabel solves it, but pounce returns the
STARTING point x=[0,0] labelled status='optimal', success=True, while its OWN
reported kkt_error is 1.0 (dual_infeasibility 1.0). Silent wrong answer.

Closed form: min 0.5 x'Px + c'x, P=diag(a,b), c=[-1,-1]  =>  x=[1/a, 1/b],
obj = -0.5*(1/a + 1/b).
"""
import numpy as np, cvxpy as cp
from pounce import solve_qp

def oracle(P, c):
    x = cp.Variable(len(c))
    p = cp.Problem(cp.Minimize(0.5*cp.quad_form(x, cp.psd_wrap(P)) + c@x), [])
    p.solve(solver=cp.CLARABEL)
    return p.value, x.value

def rel(a, b): return abs(a-b)/max(1.0, abs(b))

fails = []
c = np.array([-1.0, -1.0])
print("== mixed-scale diag(a,b), c=[-1,-1] : true obj = -0.5*(1/a+1/b) ==")
for a, b in [(1e-6,1e6),(1e-8,1e8),(1e-10,1e10),(1e-12,1e12)]:
    P = np.diag([a, b])
    r = solve_qp(P=P, c=c)
    true_obj = -0.5*(1.0/a + 1.0/b)
    ov, xv = oracle(P, c)
    grad = P @ r.x + c
    err = rel(r.obj, true_obj)
    bad = (r.status == "optimal") and err > 1e-4
    if bad: fails.append((a, b))
    print(f"  diag({a:.0e},{b:.0e}) cond~{b/a:.0e}: pounce st={r.status} success={r.success} "
          f"kkt={r.kkt_error:g} x={r.x} obj={r.obj:.4e} | true={true_obj:.4e} cvxpy={ov:.4e} "
          f"|grad(x)|={np.linalg.norm(grad):.3e} rel_err={err:.3e} {'<-- WRONG' if bad else ''}")

print(f"\nfailing (a,b) pairs: {fails}")
if fails:
    print("VERDICT: FAIL (SOLVER_BUG: mixed-scale QP returns success=True/status=optimal "
          "at x=0 with kkt_error=1.0; obj off by orders of magnitude; confirmed by cvxpy CLARABEL)")
else:
    print("VERDICT: PASS")
