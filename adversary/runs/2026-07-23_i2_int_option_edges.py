#!/usr/bin/env python
"""i2-5 (#276/#297, family api): integer option range / truncation edges.

max_iter at/over int32 (2^31) and int64 (2^63) edges must NOT silently truncate
to a negative/zero iteration budget and return a wrong 'optimal'. Expected:
either accept (huge budget, solves normally) or raise cleanly. baseline min
0.5x^2 - x => x=1, obj=-0.5.
"""
import numpy as np
from pounce import solve_qp
P=np.eye(1); c=np.array([-1.0])
def check(mi):
    try:
        r=solve_qp(P=P,c=c,max_iter=mi)
        good = (r.status=="optimal" and abs(r.x[0]-1.0)<1e-6)
        return f"status={r.status} x={r.x} obj={r.obj}", good
    except (ValueError, OverflowError) as e:
        return f"RAISED {type(e).__name__}: {str(e)[:50]}", True   # clean raise is acceptable
    except Exception as e:
        return f"RAISED {type(e).__name__}: {str(e)[:50]}", False
ok=True
for mi in [2**31-1, 2**31, 2**31+1, 2**32, 3_000_000_000, 2**63-1, 2**63, 10**20]:
    msg,good=check(mi); ok &= good
    print(f"  max_iter={mi}: {msg}  {'OK' if good else 'BAD'}")
print("VERDICT: PASS" if ok else "VERDICT: FAIL (silent int truncation -> wrong/empty budget)")
