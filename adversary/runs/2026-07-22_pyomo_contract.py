"""Adversary cross-check: pyomo-pounce plugin contract (parts a-e).

Family: nlp (pyomo plugin)   Class: API contract / suffixes / status mapping
Companion to 2026-07-22_pyomo_dual_signs.py.

Part A2: self-contained dual-sign oracle -- the finite difference of the
optimal objective wrt the rhs is computed with POUNCE ITSELF, so the sign
verdict depends on no other solver at all.
Parts B-E: status/TerminationCondition mapping, options= passthrough,
load_solutions=False, tee=True, and re-solving the same model twice.
"""
import io
import time
from contextlib import redirect_stdout

import pyomo.environ as pe
from pyomo.opt import TerminationCondition as TC
import pyomo_pounce  # noqa: F401


def qp_le(b):
    m = pe.ConcreteModel()
    m.x = pe.Var(initialize=0.0)
    m.y = pe.Var(initialize=0.0)
    m.obj = pe.Objective(expr=(m.x - 3) ** 2 + (m.y - 2) ** 2)
    m.c = pe.Constraint(expr=m.x + m.y <= b)
    return m


def qp_ge(b):
    m = pe.ConcreteModel()
    m.x = pe.Var(initialize=0.0)
    m.y = pe.Var(initialize=0.0)
    m.obj = pe.Objective(expr=m.x ** 2 + m.y ** 2)
    m.c = pe.Constraint(expr=m.x + m.y >= b)
    return m


def pounce_obj(fn, b):
    m = fn(b)
    pe.SolverFactory("pounce").solve(m)
    return pe.value(m.obj)


def pounce_dual(fn, b):
    m = fn(b)
    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    pe.SolverFactory("pounce").solve(m)
    return m.dual[m.c]


print("=" * 74)
print("A2  SELF-CONTAINED SIGN ORACLE (finite difference uses POUNCE only)")
print("=" * 74)
eps = 1e-3
for name, fn, b0 in [("x+y <= b", qp_le, 3.0), ("x+y >= b", qp_ge, 2.0)]:
    fd = (pounce_obj(fn, b0 + eps) - pounce_obj(fn, b0 - eps)) / (2 * eps)
    d = pounce_dual(fn, b0)
    print(f"  {name}:  dz/db (pounce FD) = {fd:+.6f}   pounce dual = {d:+.6f}"
          f"   -> {'MATCH' if abs(fd - d) < 1e-4 else 'SIGN FLIPPED'}")
print()

print("=" * 74)
print("B  STATUS / TerminationCondition MAPPING")
print("=" * 74)


def infeasible():
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(0, 1), initialize=0.5)
    m.obj = pe.Objective(expr=m.x)
    m.c1 = pe.Constraint(expr=m.x >= 3)
    return m


def unbounded():
    m = pe.ConcreteModel()
    m.x = pe.Var(initialize=0.0)
    m.obj = pe.Objective(expr=m.x)
    m.c = pe.Constraint(expr=m.x <= 10)
    return m


def hard_nlp():
    # Rosenbrock chain, hard enough that max_iter=3 cannot converge.
    m = pe.ConcreteModel()
    n = 20
    m.I = pe.RangeSet(1, n)
    m.x = pe.Var(m.I, initialize=-1.2)
    m.obj = pe.Objective(expr=sum(
        100 * (m.x[i + 1] - m.x[i] ** 2) ** 2 + (1 - m.x[i]) ** 2
        for i in range(1, n)))
    return m


cases = [
    ("optimal", lambda: qp_le(3.0), {}, TC.optimal),
    ("infeasible", infeasible, {}, TC.infeasible),
    ("unbounded", unbounded, {}, TC.unbounded),
    ("iteration-limit", hard_nlp, {"max_iter": 3}, TC.maxIterations),
]
for label, fn, opts, expected in cases:
    for solver in ("pounce", "ipopt"):
        m = fn()
        try:
            t0 = time.perf_counter()
            res = pe.SolverFactory(solver).solve(
                m, options=opts, load_solutions=False)
            dt = time.perf_counter() - t0
            tc = res.solver.termination_condition
            st = res.solver.status
            mark = "ok " if tc == expected else "DIFF"
            print(f"  [{mark}] {label:16s} {solver:7s} status={st} "
                  f"tc={tc} (expected {expected}) t={dt:.3f}s")
        except Exception as e:
            print(f"  [ERR] {label:16s} {solver:7s} {type(e).__name__}: {e}")
print()

print("=" * 74)
print("C  options= PASSTHROUGH via SolverFactory(...).solve(m, options={...})")
print("=" * 74)
m = hard_nlp()
res = pe.SolverFactory("pounce").solve(m, options={"max_iter": 2},
                                       load_solutions=False)
print(f"  max_iter=2 -> tc={res.solver.termination_condition} "
      f"(honored if not 'optimal')")
m = hard_nlp()
res = pe.SolverFactory("pounce").solve(m, options={"max_iter": 5000},
                                       load_solutions=False)
print(f"  max_iter=5000 -> tc={res.solver.termination_condition}")
m = qp_le(3.0)
res = pe.SolverFactory("pounce").solve(m, options={"tol": 1e-12},
                                       load_solutions=False)
print(f"  tol=1e-12 -> tc={res.solver.termination_condition}")
m = qp_le(3.0)
try:
    res = pe.SolverFactory("pounce").solve(
        m, options={"no_such_option_xyz": 1}, load_solutions=False)
    print(f"  bogus option -> tc={res.solver.termination_condition} "
          "(accepted silently)")
except Exception as e:
    print(f"  bogus option -> raised {type(e).__name__}: {e}")
print()

print("=" * 74)
print("D  load_solutions=False / tee=True / RE-SOLVE TWICE")
print("=" * 74)
m = qp_le(3.0)
m.x.set_value(99.0)
m.y.set_value(99.0)
res = pe.SolverFactory("pounce").solve(m, load_solutions=False)
print(f"  load_solutions=False: x={pe.value(m.x)} y={pe.value(m.y)} "
      f"(expect 99/99, untouched) -> "
      f"{'ok' if pe.value(m.x) == 99.0 else 'DIFF: values were loaded'}")
print(f"    result has {len(res.solution)} solution(s) attached")

buf = io.StringIO()
with redirect_stdout(buf):
    m2 = qp_le(3.0)
    pe.SolverFactory("pounce").solve(m2, tee=True)
out = buf.getvalue()
print(f"  tee=True captured {len(out)} chars, {len(out.splitlines())} lines; "
      f"looks like solver log: {'iter' in out.lower()}")

m3 = qp_le(3.0)
m3.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
opt = pe.SolverFactory("pounce")
r1 = opt.solve(m3)
o1, d1 = pe.value(m3.obj), m3.dual[m3.c]
r2 = opt.solve(m3)  # warm start from the previous solution
o2, d2 = pe.value(m3.obj), m3.dual[m3.c]
print(f"  re-solve: obj {o1:.10f} -> {o2:.10f}  dual {d1:+.8f} -> {d2:+.8f}  "
      f"{'stable' if abs(o1 - o2) < 1e-6 and abs(d1 - d2) < 1e-6 else 'DIFF'}")
r3 = opt.solve(m3, options={"max_iter": 1000})
print(f"  third solve with options: tc={r3.solver.termination_condition} "
      f"obj={pe.value(m3.obj):.10f}")
