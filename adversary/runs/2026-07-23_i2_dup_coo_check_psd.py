#!/usr/bin/env python
"""i2-7 (#279, family qp): duplicate COO entries — check_psd must validate the
SAME matrix that is solved.

scipy COO sums duplicate (i,j) entries. If the PSD guard deduped/overwrote
instead of summing, it would validate a different matrix than the solver uses.
Two probes:
 (a) duplicates SUM to an indefinite P (diag(1,-1)) -> must be rejected (not a
     silent 'optimal').
 (b) duplicates SUM to a PSD P that any single entry alone would look indefinite
     (diag(1,1) via -1 + 2) -> must SOLVE, matching cvxpy.
"""
import numpy as np, scipy.sparse as sp, cvxpy as cp
from pounce import solve_qp

# (a) sum -> [[1,0],[0,-1]] indefinite
Pa=sp.coo_matrix((np.array([1.0,2.0,-3.0]),(np.array([0,1,1]),np.array([0,1,1]))),shape=(2,2))
print("(a) dense sum =", Pa.toarray().tolist(), "(indefinite, min eig -1)")
try:
    r=solve_qp(P=Pa,c=np.array([0.0,0.0]),check_psd=True)
    a_ok=False; print(f"    pounce st={r.status} x={r.x}  <-- accepted indefinite!")
except ValueError as e:
    a_ok=True; print(f"    RAISED ValueError: {str(e)[:70]}  OK")

# (b) sum -> [[1,0],[0,1]] PSD; c=[-1,-1] -> x=[1,1], obj=-1
Pb=sp.coo_matrix((np.array([1.0,-1.0,2.0]),(np.array([0,1,1]),np.array([0,1,1]))),shape=(2,2))
print("(b) dense sum =", Pb.toarray().tolist(), "(PSD)")
r=solve_qp(P=Pb,c=np.array([-1.0,-1.0]),check_psd=True)
x=cp.Variable(2); p=cp.Problem(cp.Minimize(0.5*cp.quad_form(x,cp.psd_wrap(Pb.toarray()))+np.array([-1.0,-1.0])@x),[]); p.solve(solver=cp.CLARABEL)
b_ok = r.status=="optimal" and abs(r.obj-p.value)<1e-6
print(f"    pounce st={r.status} x={r.x} obj={r.obj} | cvxpy obj={p.value}  {'OK' if b_ok else 'BAD'}")
print("VERDICT: PASS" if a_ok and b_ok else "VERDICT: FAIL (check_psd validated a different matrix than solved)")
