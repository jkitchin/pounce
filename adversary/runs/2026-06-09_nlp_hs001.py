"""Adversary cross-check: Hock-Schittkowski problem 1 (HS001)
Family: nlp   Class: bound-constrained (one lower bound), nonconvex Rosenbrock
Source: W. Hock & K. Schittkowski, "Test Examples for Nonlinear Programming
        Codes", Lecture Notes in Economics and Mathematical Systems 187,
        Springer 1981, problem 1.
Formulation:
    min  100 (x2 - x1^2)^2 + (1 - x1)^2
    s.t. x2 >= -1.5
    start x0 = (-2, 1)
Known optimal:  x* = (1, 1),  f* = 0.
N_VARIABLES = 2,  N_CONSTRAINTS = 0 (one variable lower bound only).

Strategy: Build ONE Pyomo model. Solve with the pyomo-pounce plugin
(pounce via the AMPL NL/SOL protocol, exact analytic derivatives) and with
Ipopt. Also emit the .nl and run `pounce verify` as a derivative-independent
KKT/feasibility oracle.
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

KNOWN_X = np.array([1.0, 1.0])
KNOWN_OPTIMAL = 0.0


# --- register the pyomo-pounce plugin, pinned to the working release binary ---
@SolverFactory.register("pounce_adv", doc="POUNCE via AMPL protocol (adversary)")
class POUNCEADV(ASL):
    def __init__(self, **kwds):
        kwds["type"] = "pounce_adv"
        super().__init__(**kwds)
        self._metasolver = False
        self.options.solver = "pounce"

    def _default_executable(self):
        return POUNCE_BIN


def build():
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(initialize=-2.0)
    m.x2 = pyo.Var(initialize=1.0, bounds=(-1.5, None))
    m.obj = pyo.Objective(
        expr=100.0 * (m.x2 - m.x1**2) ** 2 + (1.0 - m.x1) ** 2
    )
    return m


def solve_with(solver_name, exe=None):
    m = build()
    opt = SolverFactory(solver_name)
    if exe:
        opt.set_executable(exe, validate=False)
    t0 = time.perf_counter()
    res = opt.solve(m, tee=False)
    dt = time.perf_counter() - t0
    x = np.array([pyo.value(m.x1), pyo.value(m.x2)])
    obj = pyo.value(m.obj)
    status = str(res.solver.termination_condition)
    return x, obj, status, dt


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# --- pounce (Pyomo path) ---
x_p, obj_p, st_p, t_p = solve_with("pounce_adv")

# --- oracle: Ipopt (Pyomo path, exact derivatives) ---
x_o, obj_o, st_o, t_o = solve_with("ipopt", exe=IPOPT_BIN)

# --- derivative-independent oracle: write .nl, run `pounce verify` ---
m = build()
nl_path = os.path.join(HERE, "hs001.nl")
sol_path = os.path.join(HERE, "hs001.sol")
m.write(nl_path, format="nl", io_options={"symbolic_solver_labels": True})
# solve to produce a .sol via pounce binary directly
subprocess.run([POUNCE_BIN, nl_path, sol_path], check=True,
               capture_output=True, text=True)
vr = subprocess.run([POUNCE_BIN, "verify", nl_path, sol_path],
                    capture_output=True, text=True)
verify_rc = vr.returncode

obj_err = rel(obj_p, obj_o)
x_err = float(np.linalg.norm(x_p - x_o, np.inf))
known_err = rel(obj_p, KNOWN_OPTIMAL)  # KNOWN=0 -> abs error
x_known_err = float(np.linalg.norm(x_p - KNOWN_X, np.inf))

print("=== pounce (Pyomo plugin) ===")
print(f"status={st_p} obj={obj_p:.10e} x={x_p} t={t_p:.4f}s")
print("=== oracle (Ipopt) ===")
print(f"status={st_o} obj={obj_o:.10e} x={x_o} t={t_o:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} known_x={KNOWN_X}")
print(f"abs_err_vs_known={known_err:.2e}  x_inf_err_vs_known={x_known_err:.2e}")
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
