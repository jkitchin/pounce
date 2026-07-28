"""Adversary: SQP filter globalization + exact-Hessian option.
Family: nlp   Class: active-set-SQP filter globalization (Fletcher-Leyffer)
Target: crates/pounce-algorithm/src/sqp/filter.rs (filter, no second-order
correction, no restoration phase) + the sqp_hessian=exact option path.
Oracle: Ipopt (via Pyomo) on identical problems + known optima.

Two findings:
  A [SOLVER_BUG]        sqp_hessian=exact -> Internal_Error on ANY constrained
                        problem (incl. the documented linear-constraint case).
  B [SOLVER_LIMITATION] filter globalization (no SOC) fails/stalls on Maratos
                        and HS6 from standard starts where Ipopt converges.
"""
import numpy as np, warnings, pounce
warnings.filterwarnings("ignore")
import pyomo.environ as pyo

def sqp(f, x0, jac=None, hess=None, cons=(), bounds=None, glob="filter", he=None):
    opts = {"algorithm": "active-set-sqp", "sqp_globalization": glob}
    if he: opts["sqp_hessian"] = he
    r = pounce.minimize(f, np.asarray(x0, float), jac=jac, hess=hess,
                        constraints=cons, bounds=bounds, options=opts)
    info = r.info if isinstance(r.info, dict) else {}
    return bool(r.success), info.get("status_msg"), info.get("iter_count"), np.asarray(r.x), r.fun

def ipopt(build, keys, tol=1e-10):
    m = build(); s = pyo.SolverFactory("ipopt"); s.options["tol"] = tol
    res = s.solve(m, tee=False)
    return np.array([pyo.value(m.x[k]) for k in keys]), pyo.value(m.o), str(res.solver.termination_condition)

# ============================================================ FINDING A
print("="*70)
print("FINDING A: sqp_hessian=exact -> Internal_Error on any constrained problem")
print("="*70)
fu = lambda x: (x[0]-2)**2 + (x[1]-1)**2
ju = lambda x: np.array([2*(x[0]-2), 2*(x[1]-1)])
hu = lambda x: np.array([[2.0, 0.0], [0.0, 2.0]])            # PD, exact objective Hessian
cases = [
    ("unconstrained", dict()),
    ("bounds-only",   dict(bounds=[(-5, 5), (-5, 5)])),
    ("linear-eq",     dict(cons=[{"type": "eq",   "fun": lambda x: x[0]+x[1]-3, "jac": lambda x: np.array([1.0, 1.0])}])),
    ("linear-ineq",   dict(cons=[{"type": "ineq", "fun": lambda x: 5-x[0]-x[1], "jac": lambda x: np.array([-1.0, -1.0])}])),
]
A_bug = False
for lbl, kw in cases:
    okE, stE, _, _, _ = sqp(fu, [0., 0.], jac=ju, hess=hu, he="exact", **kw)
    okB, stB, _, _, _ = sqp(fu, [0., 0.], jac=ju, hess=hu, he="damped-bfgs", **kw)
    flag = "" if okE else "  <-- exact FAILS (bfgs works)"
    if not okE and okB: A_bug = True
    print(f"  {lbl:14}  exact: ok={okE!s:5} {str(stE):24}  damped-bfgs: ok={okB!s:5}{flag}")
print(f"\n  => Finding A reproduced: {A_bug}  "
      "(exact works unconstrained/bounds, Internal_Error with ANY general constraint)")

# ============================================================ FINDING B
print("\n" + "="*70)
print("FINDING B: filter globalization (no SOC) fails where Ipopt succeeds")
print("="*70)

# --- B1: Maratos example  min 2(x1^2+x2^2-1)-x1 s.t. x1^2+x2^2=1, x*=(1,0) ---
fM = lambda x: 2*(x[0]**2+x[1]**2-1) - x[0]
jM = lambda x: np.array([4*x[0]-1, 4*x[1]])
cM = [{"type": "eq", "fun": lambda x: x[0]**2+x[1]**2-1, "jac": lambda x: np.array([2*x[0], 2*x[1]])}]
def maratos_ip(t):
    def b():
        m = pyo.ConcreteModel(); m.x = pyo.Var([0, 1], initialize={0: np.cos(t), 1: np.sin(t)})
        m.o = pyo.Objective(expr=2*(m.x[0]**2+m.x[1]**2-1)-m.x[0])
        m.c = pyo.Constraint(expr=m.x[0]**2+m.x[1]**2 == 1); return m
    return b
print("\n  Maratos (x*=(1,0)); start = (cos t, sin t):")
print(f"  {'t':>5} | {'filter ok':>9} {'nit':>4} {'|x-x*|':>9} {'status':>34} | {'ipopt |x-x*|':>12}")
B_fail = 0
for t in [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]:
    ok, st, nit, x, fv = sqp(fM, [np.cos(t), np.sin(t)], jac=jM, cons=cM, glob="filter")
    err = np.linalg.norm(x - np.array([1.0, 0.0]))
    xi, oi, ti = ipopt(maratos_ip(t), [0, 1]); erri = np.linalg.norm(xi - np.array([1.0, 0.0]))
    if not ok and erri < 1e-6: B_fail += 1
    print(f"  {t:>5.2f} | {ok!s:>9} {nit:>4} {err:>9.1e} {str(st):>34} | {erri:>12.1e}")

# --- B2: HS6  min (1-x1)^2 s.t. 10(x2-x1^2)=0, x*=(1,1) (Ipopt: trivial) ---
f6 = lambda x: (1-x[0])**2
j6 = lambda x: np.array([-2*(1-x[0]), 0.0])
c6 = [{"type": "eq", "fun": lambda x: 10*(x[1]-x[0]**2), "jac": lambda x: np.array([-20*x[0], 10.0])}]
def hs6_ip():
    m = pyo.ConcreteModel(); m.x = pyo.Var([0, 1], initialize={0: -1.2, 1: 1.0})
    m.o = pyo.Objective(expr=(1-m.x[0])**2); m.c = pyo.Constraint(expr=10*(m.x[1]-m.x[0]**2) == 0); return m
xi, oi, ti = ipopt(hs6_ip, [0, 1])
print(f"\n  HS6 (x*=(1,1), f*=0):  Ipopt from (-1.2,1) -> x={np.round(xi,4)} f={oi:.1e} ({ti})")
for x0 in [[-1.2, 1.0], [0.5, 0.5]]:
    for glob in ["filter", "l1-elastic"]:
        ok, st, nit, x, fv = sqp(f6, x0, jac=j6, cons=c6, glob=glob)
        err = np.linalg.norm(x - np.array([1.0, 1.0]))
        if not ok: B_fail += 1
        print(f"    start={str(x0):11} {glob:11}: ok={ok!s:5} nit={nit:>3} f={fv:.4f} xerr={err:.1e}  {st}")

print(f"\n  => Finding B: filter globalization failures on Ipopt-solvable problems = {B_fail}")

print("\n" + "="*70)
print(f"VERDICT: A={'SOLVER_BUG (reproduced)' if A_bug else 'not reproduced'}; "
      f"B={'SOLVER_LIMITATION (%d failures)' % B_fail if B_fail else 'not reproduced'}")
