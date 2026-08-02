"""Adversary cross-check: Hock-Schittkowski Problem 71 (the classic IPOPT tutorial example)
Family: nlp   Class: bound- + equality- + inequality-constrained NLP

Problem (Hock & Schittkowski, "Test Examples for Nonlinear Programming
Codes", Springer LNEMS 187, 1981, Problem 71; also the canonical worked
example in the Ipopt documentation / COIN-OR tutorial):

    minimize    x1*x4*(x1+x2+x3) + x3
    subject to  x1*x2*x3*x4 >= 25
                x1^2 + x2^2 + x3^2 + x4^2 = 40
                1 <= x1,x2,x3,x4 <= 5

Starting point x0 = (1, 5, 5, 1).

KNOWN_OPTIMAL = 17.0140173  (published HS71 optimum)
Known optimizer x* ~= (1.0, 4.7429994, 3.8211503, 1.3794082)

Solved via Pyomo with two independent solvers on the SAME model:
SolverFactory('pounce') (pyomo-pounce plugin, ASL/.nl interface) and
scipy.optimize.minimize (trust-constr, since no Linux Ipopt is available in
this sandbox -- environment-adapted oracle per adversary run instructions).
Also cross-checked with `pounce verify` against the .nl/.sol pair pounce
itself wrote, as a solver-independent KKT/feasibility oracle.
"""
import subprocess
import time

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from scipy.optimize import minimize, NonlinearConstraint

KNOWN_OPTIMAL = 17.0140173
X0 = (1.0, 5.0, 5.0, 1.0)


def build_model():
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(bounds=(1, 5), initialize=X0[0])
    m.x2 = pyo.Var(bounds=(1, 5), initialize=X0[1])
    m.x3 = pyo.Var(bounds=(1, 5), initialize=X0[2])
    m.x4 = pyo.Var(bounds=(1, 5), initialize=X0[3])
    m.obj = pyo.Objective(expr=m.x1 * m.x4 * (m.x1 + m.x2 + m.x3) + m.x3)
    m.c1 = pyo.Constraint(expr=m.x1 * m.x2 * m.x3 * m.x4 >= 25)
    m.c2 = pyo.Constraint(expr=m.x1**2 + m.x2**2 + m.x3**2 + m.x4**2 == 40)
    return m


# --- pounce via Pyomo -------------------------------------------------------
m_pounce = build_model()
solver = SolverFactory("pounce")
t0 = time.perf_counter()
result = solver.solve(m_pounce, tee=False)
t_pounce = time.perf_counter() - t0

x_pounce = np.array([pyo.value(m_pounce.x1), pyo.value(m_pounce.x2),
                      pyo.value(m_pounce.x3), pyo.value(m_pounce.x4)])
obj_pounce = pyo.value(m_pounce.obj)
term_cond = str(result.solver.termination_condition)

# --- oracle: scipy.optimize.minimize (trust-constr) -------------------------
def f(x):
    x1, x2, x3, x4 = x
    return x1 * x4 * (x1 + x2 + x3) + x3


def f_grad(x):
    x1, x2, x3, x4 = x
    return np.array([
        x4 * (2 * x1 + x2 + x3),
        x1 * x4,
        x1 * x4 + 1.0,
        x1 * (x1 + x2 + x3),
    ])


ineq = NonlinearConstraint(lambda x: x[0] * x[1] * x[2] * x[3], 25, np.inf)
eq = NonlinearConstraint(lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2, 40, 40)
bounds = [(1, 5)] * 4

t0 = time.perf_counter()
res = minimize(f, np.array(X0), jac=f_grad, method="trust-constr",
                bounds=bounds, constraints=[ineq, eq],
                options={"gtol": 1e-10, "xtol": 1e-12, "maxiter": 2000})
t_oracle = time.perf_counter() - t0
x_oracle = res.x
obj_oracle = res.fun


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_oracle = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

print("=== pounce (via pyomo-pounce) ===")
print(f"termination={term_cond} obj={obj_pounce:.10f} x={x_pounce} t={t_pounce:.4f}s")
print("=== oracle: scipy trust-constr ===")
print(f"success={res.success} obj={obj_oracle:.10f} x={x_oracle} t={t_oracle:.4f}s status={res.status}")
print(f"known_optimal={KNOWN_OPTIMAL} rel_err_vs_known={obj_err_known:.2e}")
print(f"obj_err_vs_oracle={obj_err_oracle:.2e} x_inf_err={x_err:.2e}")

# --- pounce verify on the .nl/.sol pair (solver-independent KKT oracle) ----
import tempfile
import os
from pyomo.common.tempfiles import TempfileManager

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
