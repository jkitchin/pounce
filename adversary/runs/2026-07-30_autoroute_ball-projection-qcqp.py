"""Adversary cross-check: routing transparency, Euclidean ball projection QCQP
Family: autoroute   Class: convex QCQP, QUADRATIC objective + quadratic ball
constraint -> expected route socp. DIFFERENT from the two prior QCQP-ball
autoroute runs (2026-06-09, 2026-07-23), both of which used a LINEAR objective
over a ball; this one uses a QUADRATIC objective (Euclidean projection),
exercising the auto-detector's probe of a genuinely quadratic (not affine)
`fun`, which is a different code path in the structure-detection finite
difference probe than a linear objective is.

Problem: Euclidean projection of a point a onto a ball of radius r:
    minimize    0.5 * ||x - a||^2
    subject to  ||x||^2 <= r^2
  a = (4, -3, 0), r = 2   (||a|| = 5 > r, so the constraint is active)

Closed form (standard projection-onto-a-ball result): since a lies outside
the ball, the projection is on the boundary along the ray from the origin
through a:
    x* = r * a / ||a|| = 2 * (4,-3,0)/5 = (1.6, -1.2, 0)
    obj* = 0.5 * ||x* - a||^2 = 0.5 * ||(-2.4, 1.8, 0)||^2 = 0.5*9 = 4.5

SOURCE: standard convex-analysis projection-onto-a-ball identity (e.g. Boyd &
Vandenberghe, "Convex Optimization", section 8.1 / example throughout the
Euclidean-projection literature). Cross-checked independently with cvxpy.
KNOWN_OPTIMAL: 4.5   X_STAR: (1.6, -1.2, 0.0)
"""
import time
import numpy as np
import pounce

a = np.array([4.0, -3.0, 0.0])
r = 2.0
norm_a = float(np.linalg.norm(a))
KNOWN_OPTIMAL = 0.5 * (norm_a - r) ** 2
X_STAR = r * a / norm_a


def f(x):
    d = np.asarray(x) - a
    return float(0.5 * d @ d)


def jac(x):
    return np.asarray(x) - a


def hess(x):
    return np.eye(3)


constraints = [
    {
        "type": "ineq",
        "fun": lambda x: r * r - float(np.asarray(x) @ np.asarray(x)),
        "jac": lambda x: -2.0 * np.asarray(x, dtype=float),
    },
]
x0 = np.array([0.0, 0.0, 0.0])


def rel(x, y):
    return abs(x - y) / max(1.0, abs(y))


# --- independent oracle: cvxpy ---
import cvxpy as cp

xv = cp.Variable(3)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(xv - a)), [cp.sum_squares(xv) <= r * r])
prob.solve(solver=cp.CLARABEL)
oracle_obj = float(prob.value)
oracle_x = np.asarray(xv.value)

# --- auto route ---
t0 = time.perf_counter()
res_auto = pounce.minimize(f, x0, jac=jac, hess=hess, constraints=constraints,
                            options={"solver_selection": "auto"})
t_auto = time.perf_counter() - t0

# --- forced NLP ---
t0 = time.perf_counter()
res_nlp = pounce.minimize(f, x0, jac=jac, hess=hess, constraints=constraints,
                           options={"solver_selection": "nlp"})
t_nlp = time.perf_counter() - t0

auto_solver = res_auto.info.get("solver") if isinstance(res_auto.info, dict) else None
nlp_solver = res_nlp.info.get("solver") if isinstance(res_nlp.info, dict) else None
used_specialized = auto_solver == "socp"

obj_disagree = rel(res_auto.fun, res_nlp.fun)
x_disagree = float(np.linalg.norm(np.asarray(res_auto.x) - np.asarray(res_nlp.x), np.inf))

print("=== oracle (cvxpy CLARABEL) ===")
print(f"obj={oracle_obj:.10e} x={oracle_x}")
print("=== auto ===")
print(f"success={res_auto.success} fun={res_auto.fun:.10e} x={res_auto.x} t={t_auto:.4f}s")
print(f"auto.info.solver={auto_solver!r}")
print("=== forced nlp ===")
print(f"success={res_nlp.success} fun={res_nlp.fun:.10e} x={res_nlp.x} t={t_nlp:.4f}s")
print(f"nlp.info.solver={nlp_solver!r}")
print("=== cross-check ===")
print(f"known_optimal={KNOWN_OPTIMAL:.10e}  x_star={X_STAR}")
print(f"auto_rel_err_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e}")
print(f"auto_rel_err_vs_oracle={rel(res_auto.fun, oracle_obj):.2e}")
print(f"auto_vs_nlp_obj_disagree={obj_disagree:.2e}  x_inf_disagree={x_disagree:.2e}")
print(f"used_specialized(socp)={used_specialized}")

ans_ok = (
    res_auto.success and res_nlp.success
    and obj_disagree < 1e-4
    and rel(res_auto.fun, KNOWN_OPTIMAL) < 1e-4
    and rel(res_auto.fun, oracle_obj) < 1e-4
)
print("VERDICT: PASS" if ans_ok else
      f"VERDICT: FAIL (auto_vs_nlp_disagree={obj_disagree:.2e} "
      f"auto_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e})")
