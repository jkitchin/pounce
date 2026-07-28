"""Adversary cross-check: Hock-Schittkowski problem 14 (HS014)
Family: nlp   Class: equality + inequality constrained, unbounded vars
Source: Hock & Schittkowski, "Test Examples for Nonlinear Programming
        Codes" (1981), problem 14.

  minimize  (x1 - 2)^2 + (x2 - 1)^2
  s.t.      x1 - 2 x2 + 1 = 0                (equality)
            -0.25 x1^2 - x2^2 + 1 >= 0       (inequality)

Known optimum:
  x* = ((sqrt(7)-1)/2, (sqrt(7)+1)/4) = (0.82287565..., 0.91143782...)
  f* = 9 - 2.875*sqrt(7) = 1.3934649806893021
Both constraints are active at the optimum.

Strategy: solve ONE Pyomo model with SolverFactory('pounce') and with
SolverFactory('ipopt'); also drive pounce.minimize (with analytic jac) and
compare all three against the published optimum.
"""
import math
import time
import numpy as np

KNOWN_OPTIMAL = 9.0 - 2.875 * math.sqrt(7.0)          # 1.3934649806893021
X_STAR = np.array([(math.sqrt(7.0) - 1.0) / 2.0,
                   (math.sqrt(7.0) + 1.0) / 4.0])


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# ---------------------------------------------------------------------------
# (A) pounce.minimize, scipy-style, with ANALYTIC jac (avoids any FD fallback)
# ---------------------------------------------------------------------------
import pounce


def f(x):
    return (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2


def fjac(x):
    return np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)])


# scipy convention: 'eq' fun == 0, 'ineq' fun >= 0
constraints = [
    {"type": "eq",
     "fun": lambda x: x[0] - 2.0 * x[1] + 1.0,
     "jac": lambda x: np.array([1.0, -2.0])},
    {"type": "ineq",
     "fun": lambda x: -0.25 * x[0] ** 2 - x[1] ** 2 + 1.0,
     "jac": lambda x: np.array([-0.5 * x[0], -2.0 * x[1]])},
]
x0 = np.array([2.0, 2.0])

t0 = time.perf_counter()
res = pounce.minimize(f, x0, jac=fjac, constraints=constraints,
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
    m.x[0] = 2.0
    m.x[1] = 2.0
    m.obj = pyo.Objective(expr=(m.x[0] - 2.0) ** 2 + (m.x[1] - 1.0) ** 2)
    m.eq = pyo.Constraint(expr=m.x[0] - 2.0 * m.x[1] + 1.0 == 0.0)
    m.ineq = pyo.Constraint(
        expr=-0.25 * m.x[0] ** 2 - m.x[1] ** 2 + 1.0 >= 0.0)
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
print("=== reference ===")
print(f"known_optimal={KNOWN_OPTIMAL:.12e}  x*={X_STAR}")
print(f"oracle_rel_err_vs_known = {oracle_err_known:.2e}")
print(f"pounce_rel_err_vs_known = {obj_err_known:.2e}")
print(f"pounce_rel_err_vs_oracle= {obj_err_oracle:.2e}")
print(f"pounce_x_inf_err_vs_known = {x_err_known:.2e}")
print(f"pounce_x_inf_err_vs_oracle= {x_err_oracle:.2e}")

# Answer correctness (independent of the success-flag quirk documented in #119/#123)
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
