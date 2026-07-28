#!/usr/bin/env python
"""i2-6 (#278/#299, family socp): zero-dimension cone blocks across FFI.

Zero-dim ("soc",0)/("psd",0) and empty cone lists and m=0 problems must not
panic across the FFI. Expected: clean ValueError for a 0-dim structured cone, or
a correct solve for an empty/nonneg-0 block. baseline min 0.5||x||^2 + [1,1]'x
=> x=[-1,-1], obj=-1 (unconstrained).
"""
import numpy as np
from pounce import solve_socp
def run(name, fn, want):
    try:
        r=fn(); got=f"status={r.status} x={r.x} obj={r.obj}"
        good = want=="solve" and r.status=="optimal"
    except ValueError as e:
        got=f"RAISED ValueError: {str(e)[:50]}"; good = want=="raise"
    except Exception as e:
        got=f"RAISED {type(e).__name__}: {str(e)[:50]}"; good=False
    print(f"  [{name}] {got}  {'OK' if good else 'BAD'}")
    return good
P2=np.eye(2); c2=np.array([1.0,1.0]); P1=np.eye(1); c1=np.array([1.0])
z=lambda m,n: (np.zeros((m,n)), np.zeros(m))
ok=True
ok &= run("empty cones m=0", lambda: solve_socp(P=P2,c=c2,cones=[]), "solve")
ok &= run("soc dim0", lambda: solve_socp(P=P1,c=c1,G=z(0,1)[0],h=z(0,1)[1],cones=[("soc",0)]), "raise")
ok &= run("psd dim0", lambda: solve_socp(P=P1,c=c1,G=z(0,1)[0],h=z(0,1)[1],cones=[("psd",0)]), "raise")
ok &= run("nonneg dim0", lambda: solve_socp(P=P1,c=c1,G=z(0,1)[0],h=z(0,1)[1],cones=[("nonneg",0)]), "solve")
ok &= run("exp wrong dim (2)", lambda: solve_socp(c=c2,G=z(2,2)[0],h=z(2,2)[1],cones=[("exp",2)]), "raise")
print("VERDICT: PASS" if ok else "VERDICT: FAIL (zero-dim cone panicked or mis-handled)")
