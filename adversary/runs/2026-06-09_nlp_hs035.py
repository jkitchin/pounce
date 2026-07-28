"""Adversary cross-check: Hock-Schittkowski problem 35 (HS035, Beale's problem)
Family: nlp   Class: inequality-constrained convex QP (one linear ineq + bounds)
Source: W. Hock & K. Schittkowski, "Test Examples for Nonlinear Programming
        Codes", Lecture Notes in Economics and Mathematical Systems 187,
        Springer 1981, problem 35.  (Originally E.M.L. Beale.)
Formulation:
    min  9 - 8 x1 - 6 x2 - 4 x3
         + 2 x1^2 + 2 x2^2 + x3^2 + 2 x1 x2 + 2 x1 x3
    s.t. x1 + x2 + 2 x3 <= 3
         x1, x2, x3 >= 0
    start x0 = (0.5, 0.5, 0.5)
Known optimal:  x* = (4/3, 7/9, 4/9),  f* = 1/9 = 0.111111...
N_VARIABLES = 3,  N_CONSTRAINTS = 1 (linear inequality) + 3 lower bounds.

Strategy: ONE Pyomo model; solve with pyomo-pounce plugin and Ipopt; emit
.nl and run `pounce verify` as a derivative-independent feasibility oracle.
"""
import os
import time
import subprocess
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.solvers.plugins.solvers.ASL import ASL

HERE = os.path.dirname(os.path.abspath(__file__))
POUNCE_BIN = "/Users/jkitchin/projects/pounce/target/release/pounce"
IPOPT_BIN = "/opt/homebrew/bin/ipopt"

KNOWN_X = np.array([4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0])
KNOWN_OPTIMAL = 1.0 / 9.0


@SolverFactory.register("pounce_adv35", doc="POUNCE via AMPL protocol (adversary)")
class POUNCEADV(ASL):
    def __init__(self, **kwds):
        kwds["type"] = "pounce_adv35"
        super().__init__(**kwds)
        self._metasolver = False
        self.options.solver = "pounce"

    def _default_executable(self):
        return POUNCE_BIN


def build():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], domain=pyo.NonNegativeReals, initialize=0.5)
    x = m.x
    m.obj = pyo.Objective(
        expr=9.0 - 8.0 * x[1] - 6.0 * x[2] - 4.0 * x[3]
        + 2.0 * x[1] ** 2 + 2.0 * x[2] ** 2 + x[3] ** 2
        + 2.0 * x[1] * x[2] + 2.0 * x[1] * x[3]
    )
    m.c = pyo.Constraint(expr=x[1] + x[2] + 2.0 * x[3] <= 3.0)
    return m


def solve_with(solver_name, exe=None):
    m = build()
    opt = SolverFactory(solver_name)
    if exe:
        opt.set_executable(exe, validate=False)
    t0 = time.perf_counter()
    res = opt.solve(m, tee=False)
    dt = time.perf_counter() - t0
    x = np.array([pyo.value(m.x[i]) for i in (1, 2, 3)])
    obj = pyo.value(m.obj)
    status = str(res.solver.termination_condition)
    return x, obj, status, dt


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


x_p, obj_p, st_p, t_p = solve_with("pounce_adv35")
x_o, obj_o, st_o, t_o = solve_with("ipopt", exe=IPOPT_BIN)

m = build()
nl_path = os.path.join(HERE, "hs035.nl")
sol_path = os.path.join(HERE, "hs035.sol")
m.write(nl_path, format="nl", io_options={"symbolic_solver_labels": True})
subprocess.run([POUNCE_BIN, nl_path, sol_path], check=True,
               capture_output=True, text=True)
vr = subprocess.run([POUNCE_BIN, "verify", nl_path, sol_path],
                    capture_output=True, text=True)
verify_rc = vr.returncode

obj_err = rel(obj_p, obj_o)
x_err = float(np.linalg.norm(x_p - x_o, np.inf))
known_err = rel(obj_p, KNOWN_OPTIMAL)
x_known_err = float(np.linalg.norm(x_p - KNOWN_X, np.inf))

print("=== pounce (Pyomo plugin) ===")
print(f"status={st_p} obj={obj_p:.10e} x={x_p} t={t_p:.4f}s")
print("=== oracle (Ipopt) ===")
print(f"status={st_o} obj={obj_o:.10e} x={x_o} t={t_o:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} known_x={KNOWN_X}")
print(f"rel_err_vs_known={known_err:.2e}  x_inf_err_vs_known={x_known_err:.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e}  x_inf_err_vs_oracle={x_err:.2e}")
print(f"pounce_verify_exit={verify_rc}  (0=feasible)")
print("verify_stdout:", vr.stdout.strip()[:400])

ok_status = ("optimal" in st_p.lower())
ok = (ok_status and known_err < 1e-4 and x_known_err < 1e-4
      and obj_err < 1e-4 and verify_rc == 0)
if ok:
    print("VERDICT: PASS")
else:
    print(f"VERDICT: FAIL (status={st_p}, known_err={known_err:.2e}, "
          f"x_known_err={x_known_err:.2e}, obj_err={obj_err:.2e}, "
          f"verify_rc={verify_rc})")
