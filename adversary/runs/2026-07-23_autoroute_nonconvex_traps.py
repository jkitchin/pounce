"""Adversary iteration 5: autoroute transparency on nonconvex/indefinite traps.
Family: autoroute   Class: misrouting-resistance
Direction: the router must NOT send a subtly nonconvex problem to a convex
specialized solver (qp-ipm / socp) that would return a spuriously wrong answer.
Contract: auto and forced-nlp ANSWERS must agree (objective to tol); only
disagreeing answers are a ROUTING_ERROR. A conservative NLP fall-through that
still gets the right answer is logged, not filed.

Fresh shapes (vs logged: indefinite-BOX refusal, ill-scaled, non-unique face,
convex QP/LP/QCQP): indefinite objective with LINEAR inequalities, concave
(nonconvex) QCQP inequality, convex ball via nonlinear fun, degenerate LP.
"""
import numpy as np, pounce
from scipy.optimize import minimize as smin

def rel(a, b): return abs(a - b) / max(1.0, abs(b))

def run(name, f, x0, jac=None, constraints=(), bounds=None, known=None, oracle=None):
    res_auto = pounce.minimize(f, x0, jac=jac, constraints=constraints, bounds=bounds,
                               options={"solver_selection": "auto"})
    res_nlp  = pounce.minimize(f, x0, jac=jac, constraints=constraints, bounds=bounds,
                               options={"solver_selection": "nlp"})
    route = res_auto.info.get("solver") if isinstance(res_auto.info, dict) else None
    route = route or "nlp(fallthrough)"
    obj_dis = rel(res_auto.fun, res_nlp.fun)
    x_dis   = float(np.linalg.norm(np.asarray(res_auto.x) - np.asarray(res_nlp.x), np.inf))
    # correctness vs an independent oracle / known optimum (objective)
    ref = known
    if ref is None and oracle is not None:
        ref = oracle()
    auto_ok = (ref is None) or (rel(res_auto.fun, ref) < 1e-4)
    verdict = "PASS"
    if obj_dis > 1e-4:
        verdict = "ROUTING_ERROR"          # auto vs nlp disagree in objective
    elif not auto_ok:
        verdict = "WRONG_ANSWER"           # both agree but both wrong (not a routing issue)
    print(f"[{name}]")
    print(f"   route={route:22} auto.f={res_auto.fun:.8f}  nlp.f={res_nlp.fun:.8f}  "
          f"ref={('%.8f'%ref) if ref is not None else 'n/a'}")
    print(f"   obj_auto_vs_nlp={obj_dis:.2e}  x_auto_vs_nlp={x_dis:.2e}  -> {verdict}")
    return verdict, route

verdicts = []

# ---- 1. INDEFINITE objective, LINEAR inequalities (bounded polytope) ----
# min 0.5 x'Px + c'x, P indefinite (eigs +3,-1), on box [0,1]^2. Nonconvex; global min at a vertex.
P = np.array([[1.0, 2.0],[2.0, 1.0]])   # eigenvalues 3, -1 -> indefinite
c = np.array([0.0, 0.0])
def f1(x): x=np.asarray(x,float); return 0.5*x@P@x
def j1(x): x=np.asarray(x,float); return P@x
# global min over [0,1]^2: check the 4 vertices + evaluate; -1 eigenvector direction (1,-1)...
verts=[np.array(v,float) for v in [(0,0),(1,0),(0,1),(1,1)]]
known1=min(f1(v) for v in verts)   # vertices: 0,0.5,0.5, and (1,1)->0.5+2=... 0.5*(1+4+1)=... compute
v11=f1(np.array([1.0,1.0]))        # 0.5*(1+1)+2 = 0.5*(1*1+2*1*1*... ) -> use f1
known1=min(f1(v) for v in verts)
verdicts.append(run("1 indefinite-obj + linear box", f1, np.array([0.4,0.6]), jac=j1,
                    bounds=[(0,1),(0,1)], known=known1)[0])

# ---- 2. Concave (nonconvex) QCQP inequality: min c'x s.t. x'x >= 1 (outside unit ball), box ----
# feasible region nonconvex. min -x0 - x1 s.t. x0^2+x1^2 >= 1, 0<=xi<=1. global min at (1,1)? check feasibility:1+1>=1 ok, f=-2.
def f2(x): x=np.asarray(x,float); return -(x[0]+x[1])
def j2(x): return np.array([-1.0,-1.0])
cons2=[{"type":"ineq","fun":lambda x: float(x[0]**2+x[1]**2-1.0),
        "jac":lambda x: np.array([2*x[0],2*x[1]])}]
verdicts.append(run("2 concave QCQP (outside ball)", f2, np.array([0.9,0.9]), jac=j2,
                    constraints=cons2, bounds=[(0,1),(0,1)], known=-2.0)[0])

# ---- 3. CONVEX ball via nonlinear fun: min c'x s.t. ||x||<=1 (as smooth g=1-x'x>=0) ----
# min -(x0+x1) s.t. x0^2+x1^2<=1 -> x*=(1/sqrt2,1/sqrt2), f*=-sqrt2.
def f3(x): x=np.asarray(x,float); return -(x[0]+x[1])
def j3(x): return np.array([-1.0,-1.0])
cons3=[{"type":"ineq","fun":lambda x: float(1.0-(x[0]**2+x[1]**2)),
        "jac":lambda x: np.array([-2*x[0],-2*x[1]])}]
verdicts.append(run("3 convex ball via nl-fun", f3, np.array([0.1,0.1]), jac=j3,
                    constraints=cons3, known=-np.sqrt(2))[0])

# ---- 4. Degenerate LP: min c'x s.t. equalities with a redundant row + bounds ----
# min -x0 - x1 s.t. x0+x1<=1, x0+x1<=1 (duplicate), 0<=xi. optimum -1 on the whole face x0+x1=1.
def f4(x): x=np.asarray(x,float); return -(x[0]+x[1])
def j4(x): return np.array([-1.0,-1.0])
cons4=[{"type":"ineq","fun":lambda x: float(1.0-(x[0]+x[1])),"jac":lambda x:np.array([-1.0,-1.0])},
       {"type":"ineq","fun":lambda x: float(1.0-(x[0]+x[1])),"jac":lambda x:np.array([-1.0,-1.0])}]
verdicts.append(run("4 degenerate LP (dup row)", f4, np.array([0.2,0.2]), jac=j4,
                    constraints=cons4, bounds=[(0,None),(0,None)], known=-1.0)[0])

# ---- 5. Convex QP that is POSITIVE SEMIDEFINITE (singular P), unique-on-a-line -> objective must agree ----
# min 0.5 (x0 - x1)^2 s.t. x0 + x1 = 2. P=[[1,-1],[-1,1]] PSD (eig 0,2). optimum f*=0 on x0=x1=1.
Pp=np.array([[1.0,-1.0],[-1.0,1.0]])
def f5(x): x=np.asarray(x,float); return 0.5*x@Pp@x
def j5(x): x=np.asarray(x,float); return Pp@x
cons5=[{"type":"eq","fun":lambda x: float(x[0]+x[1]-2.0),"jac":lambda x:np.array([1.0,1.0])}]
verdicts.append(run("5 PSD-singular QP (eq)", f5, np.array([0.0,2.0]), jac=j5,
                    constraints=cons5, known=0.0)[0])

print("\n" + "="*60)
for i,v in enumerate(verdicts,1): print(f"  test {i}: {v}")
bad=[v for v in verdicts if v in ("ROUTING_ERROR","WRONG_ANSWER")]
print(f"\nVERDICT: {'PASS' if not bad else 'FAIL: '+','.join(bad)} ({verdicts.count('PASS')}/{len(verdicts)})")
