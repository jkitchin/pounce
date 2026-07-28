"""Adversary cross-check: auto-routing on a strictly convex inequality QP at a vertex.
Family: autoroute   Class: strictly convex QP, two active linear inequalities
                    forming a vertex -> expected specialized route qp/qp-active-set.

  minimize   1/2 (x1^2 + x2^2) - 0.1 x1
  s.t.       x1 >= 1        (as a general linear inequality -x1 <= -1)
             x2 >= 1        (as a general linear inequality -x2 <= -1)

Unconstrained minimizer (0.1, 0) violates both, so both inequalities are active
at the optimum. KKT-derived optimum:
  x* = (1, 1),  f* = 0.5*(1+1) - 0.1 = 0.9,  multipliers lambda = (0.9, 1) > 0.
Independently confirmed with cvxpy (CLARABEL + OSQP).

Uses GENERAL linear inequality constraints (not variable bounds) so the router
must recognise the convex-QP structure. Distinct from logged autoroute tests
(convex QP eq, LP, single-ball QCQP, Rosenbrock, indefinite-box refusal,
ill-scaled detection, non-unique face): a two-active-inequality convex-QP vertex.

Answers must agree between auto and forced-nlp; only disagreeing ANSWERS are a
ROUTING_ERROR. A conservative NLP fall-through is logged-not-filed.
"""
import time
import numpy as np
import pounce
import cvxpy as cp

X_STAR = np.array([1.0, 1.0])
KNOWN_OPTIMAL = 0.9


def f(x):
    x = np.asarray(x, float)
    return 0.5 * (x[0] ** 2 + x[1] ** 2) - 0.1 * x[0]


def jac(x):
    x = np.asarray(x, float)
    return np.array([x[0] - 0.1, x[1]])


constraints = [
    {"type": "ineq", "fun": lambda x: float(x[0]) - 1.0,
     "jac": lambda x: np.array([1.0, 0.0])},
    {"type": "ineq", "fun": lambda x: float(x[1]) - 1.0,
     "jac": lambda x: np.array([0.0, 1.0])},
]
x0 = np.array([2.0, 2.0])   # feasible interior


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def cvx(solver):
    xv = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(xv) - 0.1 * xv[0]),
                      [xv[0] >= 1, xv[1] >= 1])
    prob.solve(solver=solver)
    return float(prob.value), np.asarray(xv.value)


obj_clar, x_clar = cvx(cp.CLARABEL)
obj_osqp, x_osqp = cvx(cp.OSQP)

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
print(f"CLARABEL obj={obj_clar:.10e} x={x_clar} ; OSQP obj={obj_osqp:.10e} x={x_osqp}")
print(f"analytic known = {KNOWN_OPTIMAL} at {X_STAR}")
print("=== auto ===")
print(f"success={res_auto.success} fun={res_auto.fun:.10e} x={res_auto.x} t={t_auto:.4f}s")
print(f"auto.info.solver={auto_solver!r} keys={list(res_auto.info.keys()) if isinstance(res_auto.info,dict) else res_auto.info}")
print("=== forced nlp ===")
print(f"success={res_nlp.success} fun={res_nlp.fun:.10e} x={res_nlp.x} t={t_nlp:.4f}s")
print("=== cross-check ===")
print(f"auto_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e} auto_vs_CLARABEL={rel(res_auto.fun, obj_clar):.2e}")
print(f"x_star_err_auto={np.max(np.abs(np.asarray(res_auto.x)-X_STAR)):.2e}")
print(f"auto_vs_nlp_obj_disagree={obj_disagree:.2e} x_inf_disagree={x_disagree:.2e}")
print(f"used_specialized_route={used_specialized} (solver={auto_solver!r})")
if not used_specialized:
    print("LOG-NOT-FILE: auto took the conservative NLP fall-through (answer still checked).")

ans_ok = (res_auto.success and res_nlp.success and obj_disagree < 1e-4
          and rel(res_auto.fun, KNOWN_OPTIMAL) < 1e-4
          and rel(res_auto.fun, obj_clar) < 1e-4)
print("VERDICT: PASS" if ans_ok else
      f"VERDICT: FAIL (ROUTING_ERROR? auto_vs_nlp={obj_disagree:.2e} "
      f"auto_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e})")
