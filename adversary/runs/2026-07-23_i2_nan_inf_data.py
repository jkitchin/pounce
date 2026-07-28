#!/usr/bin/env python
"""i2-10 (#295/#311, family qp): NaN/Inf data + inf-rhs equality + inf-bound mixes.

Every NaN/Inf appearing in P, c, A, b, G, h or a bound (lb=+inf, ub=-inf, inf
equality rhs, nan bound) must RAISE a clear ValueError, while legitimate
one-sided bounds (lb=-inf meaning "no lower bound") must SOLVE. Expected
behavior is the oracle.
"""
import numpy as np
from pounce import solve_qp
P=np.eye(1); c0=np.array([0.0])
def raises(fn):
    try: fn(); return False
    except ValueError: return True
    except Exception: return False
def solves(fn):
    try: return fn().status=="optimal"
    except Exception: return False
ok=True
cases_raise = {
 "nan c": lambda: solve_qp(P=P,c=np.array([np.nan])),
 "inf c": lambda: solve_qp(P=P,c=np.array([np.inf])),
 "nan P": lambda: solve_qp(P=np.array([[np.nan]]),c=c0),
 "eq inf rhs": lambda: solve_qp(P=P,c=c0,A=np.array([[1.0]]),b=np.array([np.inf])),
 "nan b": lambda: solve_qp(P=P,c=c0,A=np.array([[1.0]]),b=np.array([np.nan])),
 "nan h": lambda: solve_qp(P=P,c=c0,G=np.array([[1.0]]),h=np.array([np.nan])),
 "inf A": lambda: solve_qp(P=P,c=c0,A=np.array([[np.inf]]),b=np.array([1.0])),
 "lb=+inf": lambda: solve_qp(P=P,c=c0,lb=np.array([np.inf]),ub=np.array([2.0])),
 "ub=-inf": lambda: solve_qp(P=P,c=c0,lb=np.array([0.0]),ub=np.array([-np.inf])),
 "nan lb": lambda: solve_qp(P=P,c=c0,lb=np.array([np.nan]),ub=np.array([2.0])),
}
cases_solve = {
 "lb=-inf,ub=2 (no lower bound)": lambda: solve_qp(P=P,c=c0,lb=np.array([-np.inf]),ub=np.array([2.0])),
}
for n,f in cases_raise.items():
    r=raises(f); ok &= r; print(f"  raise[{n}] -> {r} {'OK' if r else 'BAD'}")
for n,f in cases_solve.items():
    s=solves(f); ok &= s; print(f"  solve[{n}] -> {s} {'OK' if s else 'BAD'}")
print("VERDICT: PASS" if ok else "VERDICT: FAIL (a NaN/Inf slipped through or a valid one-sided bound rejected)")
