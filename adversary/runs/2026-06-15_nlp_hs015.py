"""Adversary cross-check: Hock-Schittkowski problem 15 (HS015)
Family: nlp   Class: inequality + upper-bounded variable, nonconvex objective
Source: Hock & Schittkowski, "Test Examples for Nonlinear Programming
        Codes" (1981), problem 15.

  minimize  100 (x2 - x1^2)^2 + (1 - x1)^2
  s.t.      x1 * x2 - 1 >= 0          (inequality, active at optimum)
            x1 + x2^2  >= 0           (inequality, inactive at optimum)
            x1 <= 0.5                 (upper bound, active at optimum)

Known optimum (Hock & Schittkowski 1981, p. 38):
  x* = (0.5, 2.0)
  f* = 306.5
The first inequality (x1*x2 = 1) and the upper bound x1 = 0.5 are both active.

Strategy: solve ONE Pyomo model with SolverFactory('pounce') and with
SolverFactory('ipopt'); also drive pounce.minimize (with analytic jac) and
compare all three against the published optimum.  Cross-validate with the
solver-independent `pounce verify` oracle on the emitted .nl.
"""
import math
import os
import subprocess
import tempfile
import time
import numpy as np

KNOWN_OPTIMAL = 306.5
X_STAR = np.array([0.5, 2.0])

POUNCE_CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# ---------------------------------------------------------------------------
# (A) pounce.minimize, scipy-style, with ANALYTIC jac (avoids any FD fallback)
# ---------------------------------------------------------------------------
import pounce


def f(x):
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def fjac(x):
    # d/dx1: 100*2*(x2-x1^2)*(-2 x1) + 2(1-x1)(-1)
    g1 = 100.0 * 2.0 * (x[1] - x[0] ** 2) * (-2.0 * x[0]) - 2.0 * (1.0 - x[0])
    # d/dx2: 100*2*(x2-x1^2)
    g2 = 100.0 * 2.0 * (x[1] - x[0] ** 2)
    return np.array([g1, g2])


# scipy convention: 'ineq' fun >= 0
constraints = [
    {"type": "ineq",
     "fun": lambda x: x[0] * x[1] - 1.0,
     "jac": lambda x: np.array([x[1], x[0]])},
    {"type": "ineq",
     "fun": lambda x: x[0] + x[1] ** 2,
     "jac": lambda x: np.array([1.0, 2.0 * x[1]])},
]
# x1 <= 0.5 as a bound; x2 free
bounds = [(None, 0.5), (None, None)]
x0 = np.array([-2.0, 1.0])

t0 = time.perf_counter()
res = pounce.minimize(f, x0, jac=fjac, bounds=bounds, constraints=constraints,
                      options={"solver_selection": "nlp"})
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(res.x)
obj_pounce = float(res.fun)
status = res.status

# ---------------------------------------------------------------------------
# (B) ONE Pyomo model -> Ipopt oracle (and pounce via SolverFactory if avail)
# ---------------------------------------------------------------------------
import pyomo.environ as pyo


def build_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(2))
    m.x[0] = -2.0
    m.x[1] = 1.0
    m.x[0].setub(0.5)
    m.obj = pyo.Objective(
        expr=100.0 * (m.x[1] - m.x[0] ** 2) ** 2 + (1.0 - m.x[0]) ** 2)
    m.c1 = pyo.Constraint(expr=m.x[0] * m.x[1] - 1.0 >= 0.0)
    m.c2 = pyo.Constraint(expr=m.x[0] + m.x[1] ** 2 >= 0.0)
    return m


m_or = build_model()
t0 = time.perf_counter()
pyo.SolverFactory("ipopt").solve(m_or)
t_oracle = time.perf_counter() - t0
x_oracle = np.array([pyo.value(m_or.x[i]) for i in range(2)])
obj_oracle = pyo.value(m_or.obj)

# pounce via Pyomo SolverFactory (best-effort; same single model formulation)
pounce_pyomo_obj = None
pounce_pyomo_t = None
try:
    m_pp = build_model()
    t0 = time.perf_counter()
    pyo.SolverFactory("pounce").solve(m_pp)
    pounce_pyomo_t = time.perf_counter() - t0
    pounce_pyomo_obj = pyo.value(m_pp.obj)
except Exception as e:  # noqa: BLE001
    pounce_pyomo_obj = f"unavailable: {type(e).__name__}: {e}"

# ---------------------------------------------------------------------------
# (C) Solver-independent oracle: `pounce verify <nl> <sol>` on the emitted .nl
# ---------------------------------------------------------------------------
verify_rc = None
verify_out = ""
cli_obj = None
cli_status = None
try:
    tmpdir = tempfile.mkdtemp(prefix="hs015_")
    nlfile = os.path.join(tmpdir, "hs015.nl")
    m_nl = build_model()
    m_nl.write(nlfile, io_options={"symbolic_solver_labels": True})

    # Solve the .nl with the pounce CLI (exact derivatives from the .nl).
    # Positional: PATH [SOL]. A nonlinear .nl auto-routes to the NLP IPM.
    sol_path = os.path.join(tmpdir, "hs015.sol")
    cli = subprocess.run(
        [POUNCE_CLI, nlfile, sol_path],
        capture_output=True, text=True, timeout=60)
    cli_status = cli.returncode
    cli_out = cli.stdout + cli.stderr

    # Now run the solver-independent verifier on the .nl + .sol
    if os.path.exists(sol_path):
        ver = subprocess.run(
            [POUNCE_CLI, "verify", nlfile, sol_path],
            capture_output=True, text=True, timeout=60)
        verify_rc = ver.returncode
        verify_out = ver.stdout + ver.stderr
        # authoritative objective: the verifier's "objective at x*" line
        for line in verify_out.splitlines():
            if "objective at" in line.lower():
                for tok in line.replace(":", " ").split():
                    try:
                        cli_obj = float(tok)
                    except ValueError:
                        continue
    else:
        verify_out = "no .sol produced\n" + cli_out
except Exception as e:  # noqa: BLE001
    verify_out = f"verify path unavailable: {type(e).__name__}: {e}"

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_oracle = rel(obj_pounce, obj_oracle)
x_err_known = float(np.linalg.norm(x_pounce - X_STAR, np.inf))
x_err_oracle = float(np.linalg.norm(x_pounce - x_oracle, np.inf))
oracle_err_known = rel(obj_oracle, KNOWN_OPTIMAL)

print("=== pounce (pounce.minimize, analytic jac) ===")
print(f"status={status} obj={obj_pounce:.12e} x={x_pounce} "
      f"t={t_pounce:.4f}s nit={getattr(res, 'nit', '?')} "
      f"success={res.success}")
try:
    print(f"info={dict(res.info)}")
except Exception:  # noqa: BLE001
    pass
print("=== oracle (Ipopt via Pyomo) ===")
print(f"obj={obj_oracle:.12e} x={x_oracle} t={t_oracle:.4f}s")
print("=== pounce via Pyomo SolverFactory('pounce') ===")
print(f"obj={pounce_pyomo_obj} t={pounce_pyomo_t}")
print("=== pounce CLI on .nl (exact derivatives) ===")
print(f"cli_returncode={cli_status} cli_obj={cli_obj}")
print("=== pounce verify (solver-independent oracle) ===")
print(f"verify_returncode={verify_rc}")
print(verify_out.strip())
print("=== reference ===")
print(f"known_optimal={KNOWN_OPTIMAL:.12e}  x*={X_STAR}")
print(f"oracle_rel_err_vs_known = {oracle_err_known:.2e}")
print(f"pounce_rel_err_vs_known = {obj_err_known:.2e}")
print(f"pounce_rel_err_vs_oracle= {obj_err_oracle:.2e}")
print(f"pounce_x_inf_err_vs_known = {x_err_known:.2e}")
print(f"pounce_x_inf_err_vs_oracle= {x_err_oracle:.2e}")

# Answer correctness (independent of the success-flag quirk in #119/#123)
answer_ok = (obj_err_known < 1e-4 and obj_err_oracle < 1e-4
             and oracle_err_known < 1e-4)

if answer_ok and res.success:
    print("VERDICT: PASS")
elif answer_ok and not res.success:
    print(f"VERDICT: SOLVER_LIMITATION (correct optimum, rel_err_vs_known="
          f"{obj_err_known:.2e}, but success=False status={status})")
else:
    print(f"VERDICT: FAIL (pounce_rel_err_vs_known={obj_err_known:.2e}, "
          f"vs_oracle={obj_err_oracle:.2e}, status={status})")
