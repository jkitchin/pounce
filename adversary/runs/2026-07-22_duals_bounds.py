"""Follow-up: do VARIABLE BOUND multipliers (reduced costs) carry the same
flip, and does `pounce verify` detect a deliberately mis-signed .sol?

Bound-active LP (min form):
    min  x1 + 2*x2   s.t.  x1 + x2 >= 3,  0 <= x1 <= 1,  x2 >= 0
    x* = (1, 2), obj* = 5.
    Analytic: c1 dual = d obj/d rhs = +2 (>= row, min problem).
              x1 at UPPER bound 1; reduced cost d obj/d ub = 1 - 2 = -1.
"""
import os
import subprocess
import tempfile

import numpy as np
import pyomo.environ as pyo

import pounce

POUNCE = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT = "/opt/homebrew/bin/ipopt"
WORK = tempfile.mkdtemp(prefix="adv_bnd_")


def build():
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(bounds=(0, 1), initialize=0.5)
    m.x2 = pyo.Var(bounds=(0, None), initialize=0.5)
    m.obj = pyo.Objective(expr=m.x1 + 2 * m.x2, sense=pyo.minimize)
    m.c1 = pyo.Constraint(expr=m.x1 + m.x2 >= 3)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


print("ANALYTIC: obj*=5 at x*=(1,2); c1 dual = +2; rc(x1 @ ub) = -1")

# FD oracle with pounce itself (solve_qp), perturbing rhs and the x1 upper bound
from pounce import solve_qp


def f(rhs, ub):
    G = np.array([[-1.0, -1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([-rhs, ub, 0.0, 0.0])
    r = solve_qp(P=np.zeros((2, 2)), c=np.array([1.0, 2.0]), G=G, h=h)
    return float(r.obj), r


f0, r0 = f(3.0, 1.0)
d = 1e-5
d_rhs = (f(3 + d, 1)[0] - f(3 - d, 1)[0]) / (2 * d)
d_ub = (f(3, 1 + d)[0] - f(3, 1 - d)[0]) / (2 * d)
print(f"pounce FD: obj={f0:.8f}  d/d rhs={d_rhs:+.6f}  d/d ub(x1)={d_ub:+.6f}")
print(f"pounce solve_qp z (G rows: -x1-x2<=-3, x1<=1, -x1<=0, -x2<=0) = {np.round(r0.z,6)}")

for name in ("ipopt", "pounce"):
    m = build()
    res = pyo.SolverFactory(name).solve(m)
    print(
        f"{name:6s}: obj={pyo.value(m.obj):.8f} x={[pyo.value(m.x1), pyo.value(m.x2)]} "
        f"dual(c1)={m.dual[m.c1]:+.6f} rc(x1)={m.rc.get(m.x1, float('nan')):+.6f} "
        f"rc(x2)={m.rc.get(m.x2, float('nan')):+.6f}"
    )

# Raw pounce library mult_x_L / mult_x_U on the same .nl
nl = os.path.join(WORK, "bnd.nl")
build().write(nl, io_options={"symbolic_solver_labels": True})
nlp = pounce.read_nl(nl)


class W:
    objective = staticmethod(lambda x: nlp.objective(x))
    gradient = staticmethod(lambda x: nlp.gradient(x))
    constraints = staticmethod(lambda x: nlp.constraints(x))
    jacobian = staticmethod(lambda x: nlp.jacobian(x))
    jacobianstructure = staticmethod(lambda: nlp.jacobian_structure())


p = pounce.Problem(n=nlp.n, m=nlp.m, problem_obj=W(), lb=nlp.x_l, ub=nlp.x_u,
                   cl=nlp.g_l, cu=nlp.g_u)
p.add_option("print_level", 0)
x, info = p.solve(np.array([0.5, 0.5]))
print(f"pounce lib: x={np.round(x,6)} mult_g={np.round(np.asarray(info['mult_g']),6)} "
      f"mult_x_L={np.round(np.asarray(info['mult_x_L']),6)} "
      f"mult_x_U={np.round(np.asarray(info['mult_x_U']),6)}")
print("  GAMS link computes var_marg = mult_x_L - mult_x_U =",
      np.round(np.asarray(info['mult_x_L']) - np.asarray(info['mult_x_U']), 6))
print("  correct GAMS/AMPL rc for x1 @ ub in a min problem is -1")

# ---- (d) can `pounce verify` detect a DELIBERATELY mis-signed .sol? ----
print("\n=== (d) verify on a hand-negated .sol ===")
base = nl[:-3]
subprocess.run([POUNCE, base, "-AMPL"], capture_output=True, cwd=WORK, timeout=30)
sol = open(base + ".sol").read()
lines = sol.splitlines()
i = [k for k, l in enumerate(lines) if l.strip() == "Options"][0]
j = i + 2 + int(lines[i + 1])
counts = [int(lines[j + k]) for k in range(4)]
j += 4
nd = counts[0]
orig = [float(lines[j + k]) for k in range(nd)]
print("pounce .sol duals:", np.round(orig, 6))
for k in range(nd):
    lines[j + k] = f"{-orig[k]:.17e}"
flipped = os.path.join(WORK, "flipped.sol")
open(flipped, "w").write("\n".join(lines) + "\n")
for label, s in (("pounce-native", base + ".sol"), ("hand-negated", flipped)):
    cp = subprocess.run([POUNCE, "verify", nl, s], capture_output=True, text=True, timeout=30)
    out = [l for l in (cp.stdout + cp.stderr).splitlines() if "stationarity" in l or "VERDICT" in l]
    print(f"  verify({label}) rc={cp.returncode}: {' | '.join(o.strip() for o in out)}")
print(f"\nworkdir: {WORK}")
