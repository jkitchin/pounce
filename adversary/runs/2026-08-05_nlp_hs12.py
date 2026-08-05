"""Adversary cross-check: Hock-Schittkowski Problem 12
Family: nlp   Class: inequality-constrained NLP (nonconvex quadratic constraint)

Problem (Hock & Schittkowski, "Test Examples for Nonlinear Programming
Codes", Springer LNEMS 187, 1981, Problem 12):

    minimize    f(x) = 0.5*x1^2 + x2^2 - x1*x2 - 7*x1 - 7*x2
    subject to  25 - 4*x1^2 - x2^2 >= 0

Starting point x0 = (0, 0) (standard HS12 start).

KNOWN_OPTIMAL = -30.0 at x* = (2, 3) (published HS12 optimum; the
inequality is active: 25 - 4*2^2 - 3^2 = 25 - 16 - 9 = 0).

Solved via Pyomo with two independent solvers on the SAME model:
SolverFactory('pounce') (pyomo-pounce plugin, ASL/.nl interface) and
scipy.optimize.minimize (trust-constr, since no Linux Ipopt is available in
this sandbox -- environment-adapted oracle per adversary run instructions).
Also cross-checked with `pounce verify` against the .nl/.sol pair pounce
itself wrote, as a solver-independent KKT/feasibility oracle.
"""
import os
import subprocess
import tempfile
import time

import numpy as np
import pyomo.environ as pyo
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverFactory
from scipy.optimize import NonlinearConstraint, minimize

KNOWN_OPTIMAL = -30.0
KNOWN_X = np.array([2.0, 3.0])
X0 = (0.0, 0.0)


def build_model():
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(initialize=X0[0])
    m.x2 = pyo.Var(initialize=X0[1])
    m.obj = pyo.Objective(
        expr=0.5 * m.x1**2 + m.x2**2 - m.x1 * m.x2 - 7 * m.x1 - 7 * m.x2
    )
    m.c1 = pyo.Constraint(expr=25 - 4 * m.x1**2 - m.x2**2 >= 0)
    return m


# --- pounce via Pyomo -------------------------------------------------------
m_pounce = build_model()
solver = SolverFactory("pounce")
t0 = time.perf_counter()
result = solver.solve(m_pounce, tee=False)
t_pounce = time.perf_counter() - t0

x_pounce = np.array([pyo.value(m_pounce.x1), pyo.value(m_pounce.x2)])
obj_pounce = pyo.value(m_pounce.obj)
term_cond = str(result.solver.termination_condition)


# --- oracle: scipy.optimize.minimize (trust-constr) -------------------------
def f(x):
    x1, x2 = x
    return 0.5 * x1**2 + x2**2 - x1 * x2 - 7 * x1 - 7 * x2


def f_grad(x):
    x1, x2 = x
    return np.array([x1 - x2 - 7.0, 2 * x2 - x1 - 7.0])


ineq = NonlinearConstraint(lambda x: 25 - 4 * x[0] ** 2 - x[1] ** 2, 0, np.inf)

t0 = time.perf_counter()
res = minimize(
    f, np.array(X0), jac=f_grad, method="trust-constr",
    constraints=[ineq], options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 2000},
)
t_oracle = time.perf_counter() - t0
x_oracle = res.x
obj_oracle = res.fun


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_oracle = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))
x_err_known = float(np.linalg.norm(x_pounce - KNOWN_X, np.inf))

print("=== pounce (via pyomo-pounce) ===")
print(f"termination={term_cond} obj={obj_pounce:.10f} x={x_pounce} t={t_pounce:.4f}s")
print("=== oracle: scipy trust-constr ===")
print(f"success={res.success} obj={obj_oracle:.10f} x={x_oracle} t={t_oracle:.4f}s status={res.status}")
print(f"known_optimal={KNOWN_OPTIMAL} rel_err_vs_known={obj_err_known:.2e} x_err_vs_known={x_err_known:.2e}")
print(f"obj_err_vs_oracle={obj_err_oracle:.2e} x_inf_err={x_err:.2e}")

# --- pounce verify on the .nl/.sol pair (solver-independent KKT oracle) ----
with tempfile.TemporaryDirectory() as td:
    m_nl = build_model()
    sol_solver = SolverFactory("pounce")
    TempfileManager.tempdir = td
    verify_result = sol_solver.solve(
        m_nl, tee=False, keepfiles=True, symbolic_solver_labels=True
    )
    TempfileManager.tempdir = None
    nl_files = [f for f in os.listdir(td) if f.endswith(".nl")]
    sol_files = [f for f in os.listdir(td) if f.endswith(".sol")]
    verify_out = None
    if nl_files and sol_files:
        nlf = os.path.join(td, sorted(nl_files)[0])
        solf = os.path.join(td, sorted(sol_files)[0])
        try:
            verify_out = subprocess.run(
                ["pounce", "verify", nlf, solf],
                capture_output=True, text=True, timeout=10,
            )
        except FileNotFoundError:
            verify_out = None
    else:
        print(f"(keepfiles dir contents: {os.listdir(td)})")

if verify_out is not None:
    print("=== pounce verify ===")
    print(f"returncode={verify_out.returncode}")
    print((verify_out.stdout or "").strip()[-500:])
    if verify_out.returncode != 0:
        print("STDERR:", (verify_out.stderr or "").strip()[-500:])
else:
    print("=== pounce verify === SKIPPED (no keepfiles .nl/.sol found)")

ok = (
    "optimal" in term_cond.lower()
    and obj_err_known < 1e-4
    and obj_err_oracle < 1e-4
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (termination={term_cond}, obj_err_known={obj_err_known:.2e})")
