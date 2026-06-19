import time, numpy as np, pounce
from scipy.integrate import solve_bvp as sp
def t(f,r=8):
    f(); t0=time.perf_counter()
    for _ in range(r): f()
    return 1e3*(time.perf_counter()-t0)/r

problems = {}
# A: linear smooth
def fA(x,y): return np.vstack((y[1],-y[0]))
def bA(ya,yb): return np.array([ya[0],yb[0]-1.0])
problems['linear y\"=-y']=(fA,bA,np.pi/2,lambda x,m:(lambda y:(y.__setitem__((0,slice(None)),x/(np.pi/2)) or y))(np.zeros((2,m))))
# B: Bratu nonlinear
def fB(x,y): return np.vstack((y[1],-np.exp(y[0])))
def bB(ya,yb): return np.array([ya[0],yb[0]])
problems['Bratu nonlin']=(fB,bB,1.0,lambda x,m:np.zeros((2,m)))
# C: boundary layer  eps y'' = y, y(0)=1,y(1)=0  (sharp layer near x=1 for small eps)
eps=1e-3
def fC(x,y): return np.vstack((y[1], y[0]/eps))
def bC(ya,yb): return np.array([ya[0]-1.0, yb[0]])
problems['bdry-layer e=1e-3']=(fC,bC,1.0,lambda x,m:np.zeros((2,m)))

print("=== Fixed-mesh scaling: time (ms) vs nodes; both on same mesh ===")
for name,(fun,bc,dom,yg) in problems.items():
    print(f"\n# {name}")
    print(f"{'m':>6} | {'scipy ms':>9} {'pounce ms':>10} {'ratio':>6} | {'p.iters':>7} {'max|Δ|':>8}")
    for m in (51,201,801,3201):
        x=np.linspace(0,dom,m); y0=yg(x,m)
        ts=t(lambda:sp(fun,bc,x,y0,tol=1e-6,max_nodes=m))
        r=pounce.solve_bvp(fun,bc,x,y0,tol=1e-6,method='newton')
        tp=t(lambda:pounce.solve_bvp(fun,bc,x,y0,tol=1e-6,method='newton'))
        xt=np.linspace(0,dom,400)
        rs=sp(fun,bc,x,y0,tol=1e-6,max_nodes=m)
        d=np.max(np.abs(r.sol(xt)[0]-rs.sol(xt)[0])) if rs.success and r.success else float('nan')
        print(f"{m:>6} | {ts:>9.2f} {tp:>10.2f} {tp/ts:>5.2f}x | {r.niter:>7} {d:>8.1e}")
