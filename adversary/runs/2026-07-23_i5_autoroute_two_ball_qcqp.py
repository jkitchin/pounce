"""Adversary cross-check: auto-routing on a convex QCQP with TWO ball constraints.
Family: autoroute   Class: convex QCQP (convex-quadratic objective + 2 ellipsoid
                    constraints) -> expected specialized route socp.

  minimize   1/2 ||x - p||^2                p = [2, 2, 2]
  s.t.       ||x||^2        <= 1            (unit ball at origin)
             ||x - q||^2    <= 1            (unit ball at q = [1.2, 0, 0])
  (n = 3, no bounds)

Two intersecting balls -> the feasible set is the lens of their intersection.
Distinct from the logged single-ball QCQP autoroute test: TWO quadratic
constraints, and a convex-QUADRATIC (not linear) objective, so the router must
recognise the QCQP structure and lift it to SOCP.

Answers must agree between auto and forced-nlp; the reference optimum is
cross-checked with cvxpy (CLARABEL + ECOS). ROUTING_ERROR only if the ANSWERS
disagree; a conservative fall-through to NLP that still gets the right answer is
logged-not-filed.
"""
import time
import numpy as np
import pounce
import cvxpy as cp

p = np.array([2.0, 2.0, 2.0])
q = np.array([1.2, 0.0, 0.0])
x0 = np.array([0.0, 0.0, 0.0])   # feasible interior of both balls? check below


def f(x):
    d = np.asarray(x) - p
    return 0.5 * float(d @ d)


def jac(x):
    return np.asarray(x) - p


constraints = [
    {"type": "ineq",
     "fun": lambda x: 1.0 - float(np.asarray(x) @ np.asarray(x)),
     "jac": lambda x: -2.0 * np.asarray(x, float)},
    {"type": "ineq",
     "fun": lambda x: 1.0 - float((np.asarray(x) - q) @ (np.asarray(x) - q)),
     "jac": lambda x: -2.0 * (np.asarray(x, float) - q)},
]


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# oracle: cvxpy
def cvx(solver):
    xv = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(xv - p)),
                      [cp.sum_squares(xv) <= 1, cp.sum_squares(xv - q) <= 1])
    prob.solve(solver=solver)
    return float(prob.value), np.asarray(xv.value)


obj_clar, x_clar = cvx(cp.CLARABEL)
obj_ecos, x_ecos = cvx(cp.ECOS)
KNOWN_OPTIMAL = obj_clar

t0 = time.perf_counter()
res_auto = pounce.minimize(f, x0, jac=jac, constraints=constraints,
                           options={"solver_selection": "auto"})
t_auto = time.perf_counter() - t0
t0 = time.perf_counter()
res_nlp = pounce.minimize(f, x0, jac=jac, constraints=constraints,
                          options={"solver_selection": "nlp"})
t_nlp = time.perf_counter() - t0

auto_solver = res_auto.info.get("solver") if isinstance(res_auto.info, dict) else None
used_specialized = auto_solver is not None  # NLP backend reports solver=None

obj_disagree = rel(res_auto.fun, res_nlp.fun)
x_disagree = float(np.linalg.norm(np.asarray(res_auto.x) - np.asarray(res_nlp.x), np.inf))

print("=== oracle cvxpy ===")
print(f"CLARABEL obj={obj_clar:.10e} x={x_clar}")
print(f"ECOS     obj={obj_ecos:.10e} x={x_ecos}")
print("=== auto ===")
print(f"success={res_auto.success} fun={res_auto.fun:.10e} x={res_auto.x} t={t_auto:.4f}s")
print(f"auto.info.solver={auto_solver!r} keys={list(res_auto.info.keys()) if isinstance(res_auto.info,dict) else res_auto.info}")
print("=== forced nlp ===")
print(f"success={res_nlp.success} fun={res_nlp.fun:.10e} x={res_nlp.x} t={t_nlp:.4f}s")
print("=== cross-check ===")
print(f"known={KNOWN_OPTIMAL:.10e} auto_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e} "
      f"auto_vs_ecos={rel(res_auto.fun, obj_ecos):.2e}")
print(f"auto_vs_nlp_obj_disagree={obj_disagree:.2e} x_inf_disagree={x_disagree:.2e}")
print(f"used_specialized_route={used_specialized} (solver={auto_solver!r})")
if not used_specialized:
    print("LOG-NOT-FILE: auto took the conservative NLP fall-through (answer still checked).")

ans_ok = (res_auto.success and res_nlp.success and obj_disagree < 1e-4
          and rel(res_auto.fun, KNOWN_OPTIMAL) < 1e-4
          and rel(res_auto.fun, obj_ecos) < 1e-4)
print("VERDICT: PASS" if ans_ok else
      f"VERDICT: FAIL (ROUTING_ERROR? auto_vs_nlp={obj_disagree:.2e} "
      f"auto_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e})")
