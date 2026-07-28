#!/usr/bin/env python
"""i2-4 (#277/#298, family api): tol/max_iter validation across convex entry points.

#277/#298 added tol/max_iter validation and explicitly names the convex entry
points "solve_qp, solve_socp, minimize, batch". solve_qp and solve_socp now
correctly REJECT every unsatisfiable tol (0, <0, inf, nan, >=1) and non-positive
max_iter. But `minimize` was NOT given the same guard: it ACCEPTS tol=inf,
tol=1.0, tol=1e6, and returns status=0 / success=True at a NON-stationary,
WRONG point -- the exact "tol>=1 accepts the non-stationary iterate and returns
a wrong point labeled optimal" failure #277 documents. Demonstrated on
Rosenbrock (true min x=[1,1], f=0): with tol>=1 minimize stops at f~0.199 and
reports success=True.
"""
import numpy as np
from pounce import solve_qp, solve_socp, minimize

def raises(fn):
    try: fn(); return False
    except (ValueError,) : return True
    except Exception: return False

P=np.eye(1); c=np.array([-1.0])
qp   = lambda **k: solve_qp(P=P, c=c, **k)
socp = lambda **k: solve_socp(P=P, c=c, cones=[], **k)
bad_tol=[0.0,-1.0,np.inf,np.nan,1.0,2.0]; bad_iter=[0,-5]

print("== solve_qp / solve_socp: every bad tol/max_iter must raise ==")
core_ok=True
for name,fn in [("solve_qp",qp),("solve_socp",socp)]:
    for t in bad_tol:
        r=raises(lambda: fn(tol=t)); core_ok&=r; print(f"  {name}(tol={t}) raises={r}")
    for m in bad_iter:
        r=raises(lambda: fn(max_iter=m)); core_ok&=r; print(f"  {name}(max_iter={m}) raises={r}")

print("\n== minimize: does it reject tol>=1 / inf ? (Rosenbrock, true f=0 at [1,1]) ==")
def rosen(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
x0=np.array([-1.2,1.0])
minimize_gap=False
import warnings
for t in [np.inf, 1e6, 1.0]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r=minimize(rosen, x0=x0.copy(), tol=t)
            wrong = bool(r.success) and r.fun > 1e-4
            if wrong: minimize_gap=True
            print(f"  minimize(tol={t}): status={r.status} success={r.success} "
                  f"x={np.round(r.x,4)} f={r.fun:.4e} {'<-- WRONG-OPTIMAL (success at non-stationary pt)' if wrong else ''}")
        except (ValueError,) as e:
            print(f"  minimize(tol={t}): raised ValueError (validated) OK")

print()
if core_ok and not minimize_gap:
    print("VERDICT: PASS")
elif minimize_gap:
    print("VERDICT: FAIL (SOLVER_BUG: `minimize` does not validate tol -- accepts tol>=1/inf and "
          "returns success=True/status=0 at a non-stationary wrong point; #277/#298 tol guard was "
          "not propagated to the minimize entry point that #277 explicitly lists)")
else:
    print("VERDICT: FAIL (solve_qp/solve_socp accepted a bad tol/max_iter)")
