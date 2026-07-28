"""Adversary cross-check: routing transparency on a pure LP (eq + ineq)
Family: autoroute   Class: pure linear program -> expected route lp-ipm

Source: self-contained standard-form-ish LP with one equality and two
inequalities, plus nonneg bounds. Constructed so the unique optimum is a
vertex with a closed-form value, independently verified by scipy.linprog.

  minimize    c.x  with c = [-1, -2, 1]
  subject to  x0 + x1 + x2 = 4            (equality)
              x0 + 2 x1        <= 5        (inequality)
              x1 + x2          <= 3        (inequality)
              x >= 0

Oracle optimum (verified below with scipy.linprog/HiGHS):
  The optimal vertex is x* = (3, 1, 0), obj* = -1*3 - 2*1 + 0 = -5.0.
  Feasibility: eq 3+1+0 = 4 OK; row2 3 + 2*1 = 5 <= 5 (active); row3 1 + 0 = 1 <= 3.
  (An earlier hand guess of (0,2.5,1.5)->-3.5 was a REFERENCE error: it is
  feasible but NOT optimal, since -5.0 < -3.5. The scipy oracle is the source
  of truth here and agrees with both pounce paths.)
Known optimal: -5.0
"""
import time
import numpy as np
import pounce
from scipy.optimize import linprog

KNOWN_OPTIMAL = -5.0
X_STAR = np.array([3.0, 1.0, 0.0])

c = np.array([-1.0, -2.0, 1.0])


def f(x):
    return float(c @ x)


def jac(x):
    return c.copy()


# scipy-style constraints (ineq: fun(x) >= 0; eq: fun(x) == 0)
constraints = [
    {"type": "eq",   "fun": lambda x: x[0] + x[1] + x[2] - 4.0},
    {"type": "ineq", "fun": lambda x: 5.0 - (x[0] + 2.0 * x[1])},
    {"type": "ineq", "fun": lambda x: 3.0 - (x[1] + x[2])},
]
bounds = [(0.0, None), (0.0, None), (0.0, None)]
x0 = np.array([4.0, 0.0, 0.0])  # feasible vertex (eq satisfied)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# --- independent oracle: scipy.linprog ---
A_ub = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]])
b_ub = np.array([5.0, 3.0])
A_eq = np.array([[1.0, 1.0, 1.0]])
b_eq = np.array([4.0])
lp = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
             bounds=[(0, None)] * 3, method="highs")
oracle_obj = float(lp.fun)
oracle_x = np.asarray(lp.x)

# --- auto route ---
# Routing is OPT-IN since PR #97 (default solver_selection is "nlp"); request "auto" explicitly.
t0 = time.perf_counter()
res_auto = pounce.minimize(f, x0, jac=jac, bounds=bounds, constraints=constraints,
                           options={"solver_selection": "auto"})
t_auto = time.perf_counter() - t0

# --- forced NLP ---
t0 = time.perf_counter()
res_nlp = pounce.minimize(f, x0, jac=jac, bounds=bounds, constraints=constraints,
                          options={"solver_selection": "nlp"})
t_nlp = time.perf_counter() - t0

auto_solver = res_auto.info.get("solver") if isinstance(res_auto.info, dict) else None
nlp_solver = res_nlp.info.get("solver") if isinstance(res_nlp.info, dict) else None
used_specialized = auto_solver == "lp-ipm"

obj_disagree = rel(res_auto.fun, res_nlp.fun)
x_disagree = float(np.linalg.norm(np.asarray(res_auto.x) - np.asarray(res_nlp.x), np.inf))

print("=== oracle (scipy.linprog/highs) ===")
print(f"obj={oracle_obj:.10e} x={oracle_x}")
print("=== auto ===")
print(f"success={res_auto.success} fun={res_auto.fun:.10e} x={res_auto.x} t={t_auto:.4f}s")
print(f"auto.info.solver={auto_solver!r}  auto.info.keys={list(res_auto.info.keys())}")
print("=== forced nlp ===")
print(f"success={res_nlp.success} fun={res_nlp.fun:.10e} x={res_nlp.x} t={t_nlp:.4f}s")
print(f"nlp.info.solver={nlp_solver!r} (None => true NLP backend, has mu={'mu' in res_nlp.info})")
print("=== cross-check ===")
print(f"known_optimal={KNOWN_OPTIMAL:.10e}")
print(f"auto_rel_err_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e}")
print(f"auto_rel_err_vs_oracle={rel(res_auto.fun, oracle_obj):.2e}")
print(f"auto_vs_nlp_obj_disagree={obj_disagree:.2e}  x_inf_disagree={x_disagree:.2e}")
print(f"used_specialized(lp-ipm)={used_specialized}")

ans_ok = (res_auto.success and res_nlp.success
          and obj_disagree < 1e-4
          and rel(res_auto.fun, KNOWN_OPTIMAL) < 1e-4
          and rel(res_auto.fun, oracle_obj) < 1e-4)
print("VERDICT: PASS" if ans_ok else
      f"VERDICT: FAIL (auto_vs_nlp_disagree={obj_disagree:.2e} "
      f"auto_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e})")
