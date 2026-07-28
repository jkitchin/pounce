"""Adversary cross-check: pyomo-pounce dual/rc sign convention.

Family: nlp (pyomo plugin contract)   Class: duals / suffixes
Source: Hillier & Lieberman "Introduction to Operations Research",
        Wyndor Glass Co. LP (max 3x1+5x2 s.t. x1<=4, 2x2<=12, 3x1+2x2<=18);
        known optimum 36 at (2,6), shadow prices (0, 1.5, 1).
Oracle: ANALYTIC dz/db (finite-difference re-solve of the rhs) + glpk + cbc + ipopt.

The analytic invariant that settles the sign, independent of any solver:
  * minimization, `<=` constraint: relaxing (increasing b) can never increase
    the optimum  =>  dz/db <= 0.
  * minimization, `>=` constraint: tightening... => dz/db >= 0.
If a solver reports dual with the opposite sign of dz/db while every other
solver on the SAME Pyomo path reports dz/db, that solver's duals are wrong.
"""
import time

import pyomo.environ as pe
import pyomo_pounce  # noqa: F401  registers SolverFactory('pounce')

SOLVERS = ["pounce", "ipopt", "glpk", "cbc"]
NLP_ONLY = {"glpk", "cbc"}


def solve(model_fn, solver, rhs=None):
    m = model_fn() if rhs is None else model_fn(rhs)
    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    m.rc = pe.Suffix(direction=pe.Suffix.IMPORT)
    opt = pe.SolverFactory(solver)
    t0 = time.perf_counter()
    res = opt.solve(m, tee=False)
    dt = time.perf_counter() - t0
    return m, res, dt


def duals(m):
    out = {}
    for c in m.component_data_objects(pe.Constraint, active=True):
        out[c.name] = m.dual.get(c)
    return out


def rcs(m):
    out = {}
    for v in m.component_data_objects(pe.Var, active=True):
        out[v.name] = m.rc.get(v)
    return out


def fd_shadow(model_fn, base, i, solver="glpk", eps=1e-4):
    """Central finite difference of the optimal objective wrt rhs[i]."""
    def obj_at(b):
        m = model_fn(b)
        pe.SolverFactory(solver).solve(m)
        return pe.value(m.obj)
    bp = list(base); bp[i] += eps
    bm = list(base); bm[i] -= eps
    return (obj_at(bp) - obj_at(bm)) / (2 * eps)


# ---------------------------------------------------------------- case 1: LP
def wyndor_min(rhs=(4.0, 12.0, 18.0)):
    """min -3x1-5x2 (= -max 3x1+5x2), all `<=`. Optimum -36 at (2,6)."""
    m = pe.ConcreteModel()
    m.x1 = pe.Var(bounds=(0, None), initialize=0.5)
    m.x2 = pe.Var(bounds=(0, None), initialize=0.5)
    m.obj = pe.Objective(expr=-3 * m.x1 - 5 * m.x2, sense=pe.minimize)
    m.c1 = pe.Constraint(expr=m.x1 <= rhs[0])
    m.c2 = pe.Constraint(expr=2 * m.x2 <= rhs[1])
    m.c3 = pe.Constraint(expr=3 * m.x1 + 2 * m.x2 <= rhs[2])
    return m


def wyndor_max(rhs=(4.0, 12.0, 18.0)):
    m = wyndor_min(rhs)
    m.del_component(m.obj)
    m.obj = pe.Objective(expr=3 * m.x1 + 5 * m.x2, sense=pe.maximize)
    return m


# --------------------------------------------------- case 2: QP <= (the lead)
def qp_le(rhs=(3.0,)):
    """min (x-3)^2+(y-2)^2 s.t. x+y <= b. b=3 -> x=2,y=1, z=2, dz/db=-2."""
    m = pe.ConcreteModel()
    m.x = pe.Var(initialize=0.0)
    m.y = pe.Var(initialize=0.0)
    m.obj = pe.Objective(expr=(m.x - 3) ** 2 + (m.y - 2) ** 2, sense=pe.minimize)
    m.c = pe.Constraint(expr=m.x + m.y <= rhs[0])
    return m


# --------------------------------------------------------- case 3: QP >= / ==
def qp_ge(rhs=(2.0,)):
    """min x^2+y^2 s.t. x+y >= b. b=2 -> x=y=1, z=2, dz/db=b=+2."""
    m = pe.ConcreteModel()
    m.x = pe.Var(initialize=0.0)
    m.y = pe.Var(initialize=0.0)
    m.obj = pe.Objective(expr=m.x ** 2 + m.y ** 2, sense=pe.minimize)
    m.c = pe.Constraint(expr=m.x + m.y >= rhs[0])
    return m


def qp_eq(rhs=(2.0,)):
    """min x^2+y^2 s.t. x+y == b. dz/db = b = +2."""
    m = pe.ConcreteModel()
    m.x = pe.Var(initialize=0.0)
    m.y = pe.Var(initialize=0.0)
    m.obj = pe.Objective(expr=m.x ** 2 + m.y ** 2, sense=pe.minimize)
    m.c = pe.Constraint(expr=m.x + m.y == rhs[0])
    return m


# ------------------------------------------------------ case 4: variable bound
def qp_bound(rhs=(1.0,)):
    """min (x-3)^2 s.t. x <= ub (a BOUND, not a constraint).
    ub=1 -> x=1, z=4, dz/dub = -4  => rc should be -4 under dz/db."""
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(None, rhs[0]), initialize=0.0)
    m.obj = pe.Objective(expr=(m.x - 3) ** 2, sense=pe.minimize)
    m.keep = pe.Constraint(expr=m.x >= -100)
    return m


def report(title, model_fn, base, fd_solver, solvers, want_rc=False):
    print("=" * 74)
    print(title)
    print("=" * 74)
    ana = {}
    for i in range(len(base)):
        ana[i] = fd_shadow(model_fn, base, i, solver=fd_solver)
    print("analytic dz/db (central FD, %s):" % fd_solver,
          {i: round(v, 6) for i, v in ana.items()})
    rows = {}
    for s in solvers:
        try:
            m, res, dt = solve(model_fn, s, rhs=tuple(base))
            d = duals(m)
            r = rcs(m)
            rows[s] = (pe.value(m.obj), d, r, dt,
                       str(res.solver.termination_condition))
            print(f"  {s:7s} obj={pe.value(m.obj):+.8f} t={dt:.3f}s "
                  f"tc={res.solver.termination_condition}")
            print(f"          duals={ {k: (None if v is None else round(v, 6)) for k, v in d.items()} }")
            if want_rc:
                print(f"          rc   ={ {k: (None if v is None else round(v, 6)) for k, v in r.items()} }")
        except Exception as e:  # pragma: no cover
            print(f"  {s:7s} ERROR {type(e).__name__}: {e}")
    print()
    return ana, rows


if __name__ == "__main__":
    report("CASE 1  LP (Wyndor, minimize -3x1-5x2, three `<=`)  known z*=-36",
           wyndor_min, [4.0, 12.0, 18.0], "glpk", SOLVERS)
    report("CASE 2  QP `<=`  min (x-3)^2+(y-2)^2 s.t. x+y<=3   z*=2, dz/db=-2",
           qp_le, [3.0], "ipopt", [s for s in SOLVERS if s not in NLP_ONLY])
    report("CASE 3  QP `>=`  min x^2+y^2 s.t. x+y>=2           z*=2, dz/db=+2",
           qp_ge, [2.0], "ipopt", [s for s in SOLVERS if s not in NLP_ONLY])
    report("CASE 4  QP `==`  min x^2+y^2 s.t. x+y==2           z*=2, dz/db=+2",
           qp_eq, [2.0], "ipopt", [s for s in SOLVERS if s not in NLP_ONLY])
    report("CASE 5  BOUND    min (x-3)^2 s.t. x<=ub=1          z*=4, dz/dub=-4",
           qp_bound, [1.0], "ipopt", [s for s in SOLVERS if s not in NLP_ONLY],
           want_rc=True)
    report("CASE 6  LP MAXIMIZE (Wyndor)  z*=+36, dz/db2=+1.5, dz/db3=+1",
           wyndor_max, [4.0, 12.0, 18.0], "glpk", SOLVERS)
