"""Adversary cross-check: routing transparency on a convex QCQP -> SOCP
Family: autoroute   Class: convex QCQP (linear obj + quadratic ball constraint)
                    -> expected route socp (conic interior-point)

Source: classic "minimize a linear functional over a Euclidean ball", which has
a closed-form optimum and is the canonical convex QCQP. A quadratic *constraint*
(not just a quadratic objective) is what pushes the auto-router from the QP path
to the SOCP path.

  minimize    c.x          c = [1, -2, 2]
  subject to  x0^2 + x1^2 + x2^2 <= R^2     (R = 3)
  (no bounds, no other constraints)

Closed form: the minimizer of a linear functional over a ball of radius R is
  x* = -R * c / ||c||,   obj* = -R * ||c||.
  ||c|| = sqrt(1 + 4 + 4) = 3,  so obj* = -3 * 3 = -9, x* = -[1,-2,2].
Known optimal: -9.0   (independently re-verified with cvxpy below)
"""
import time
import numpy as np
import pounce

R = 3.0
c = np.array([1.0, -2.0, 2.0])
norm_c = float(np.linalg.norm(c))
KNOWN_OPTIMAL = -R * norm_c            # -9.0
X_STAR = -R * c / norm_c               # -[1,-2,2]


def f(x):
    return float(c @ x)


def jac(x):
    return c.copy()


# quadratic ball constraint as scipy ineq: g(x) = R^2 - ||x||^2 >= 0
constraints = [
    {
        "type": "ineq",
        "fun": lambda x: R * R - float(x @ x),
        "jac": lambda x: -2.0 * np.asarray(x, dtype=float),
    },
]
bounds = None
x0 = np.array([0.0, 0.0, 0.0])  # strictly interior, feasible


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# --- independent oracle: cvxpy ---
oracle_obj = None
try:
    import cvxpy as cp
    xv = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(c @ xv), [cp.sum_squares(xv) <= R * R])
    prob.solve()
    oracle_obj = float(prob.value)
    oracle_x = np.asarray(xv.value)
except Exception as e:  # noqa: BLE001
    oracle_x = None
    print(f"(cvxpy oracle unavailable: {e})")

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
used_specialized = auto_solver == "socp"

obj_disagree = rel(res_auto.fun, res_nlp.fun)
x_disagree = float(np.linalg.norm(np.asarray(res_auto.x) - np.asarray(res_nlp.x), np.inf))

print("=== oracle (cvxpy) ===")
if oracle_obj is not None:
    print(f"obj={oracle_obj:.10e} x={oracle_x}")
else:
    print("unavailable")
print("=== auto ===")
print(f"success={res_auto.success} fun={res_auto.fun:.10e} x={res_auto.x} t={t_auto:.4f}s")
print(f"auto.info.solver={auto_solver!r}  auto.info.keys={list(res_auto.info.keys())}")
print("=== forced nlp ===")
print(f"success={res_nlp.success} fun={res_nlp.fun:.10e} x={res_nlp.x} t={t_nlp:.4f}s")
print(f"nlp.info.solver={nlp_solver!r} (None => true NLP backend, has mu={'mu' in res_nlp.info})")
print("=== cross-check ===")
print(f"known_optimal={KNOWN_OPTIMAL:.10e}  x_star={X_STAR}")
print(f"auto_rel_err_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e}")
if oracle_obj is not None:
    print(f"auto_rel_err_vs_oracle={rel(res_auto.fun, oracle_obj):.2e}")
print(f"auto_vs_nlp_obj_disagree={obj_disagree:.2e}  x_inf_disagree={x_disagree:.2e}")
print(f"used_specialized(socp)={used_specialized}")

oracle_ok = (oracle_obj is None) or (rel(res_auto.fun, oracle_obj) < 1e-4)
ans_ok = (res_auto.success and res_nlp.success
          and obj_disagree < 1e-4
          and rel(res_auto.fun, KNOWN_OPTIMAL) < 1e-4
          and oracle_ok)
print("VERDICT: PASS" if ans_ok else
      f"VERDICT: FAIL (auto_vs_nlp_disagree={obj_disagree:.2e} "
      f"auto_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e})")
