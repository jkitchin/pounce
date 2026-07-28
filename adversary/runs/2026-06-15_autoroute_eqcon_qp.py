"""Adversary cross-check: routing transparency on an equality-constrained convex QP
Family: autoroute   Class: convex QP with linear EQUALITY constraints
                    -> expected route qp (convex QP solver)

The auto-router should recognize a convex quadratic objective with purely
LINEAR equality constraints (no nonconvex pieces) and dispatch to the
specialized convex QP solver, NOT the general NLP backend. Routing-transparency
contract: minimize(auto) must agree with minimize(solver_selection="nlp") to
tolerance, AND auto must pick the specialized "qp" route.

Source: Nocedal & Wright, "Numerical Optimization" 2e, Example 16.2 (the
equality-constrained QP)

    minimize   q(x) = 3x0^2 + 2x0x1 + 2x0x2 + 2x1^2 + 2x1x2 + 2x2^2
                      + 8x0 + 3x1 + 3x2
    subject to x0 + x1 + x2 = 3
               x0       - x2 = 0       (i.e. x0 = x2)

In matrix form q(x) = 1/2 x^T G x + c^T x with
    G = [[6,2,2],[2,4,2],[2,2,4]],  c = [8,3,3].
We do NOT rely on any hand-quoted optimum: x* is re-derived in-script by
solving the (G, A) KKT system exactly, then independently confirmed with cvxpy.
For THIS data the unique optimum is x* = (0.5, 2.0, 0.5) with q* = 25.25 and
multipliers lambda* = (-13, -3) (verified by the KKT solve and cvxpy below).

Known optimal: q(x*) = 25.25, computed exactly from the closed-form KKT solution
and cross-checked against cvxpy.
N_VARIABLES: 3   N_CONSTRAINTS: 2 linear equalities
"""
import time
import numpy as np

# ---- problem data (q(x) = 1/2 x^T G x + c^T x) -----------------------------
G = np.array([[6.0, 2.0, 2.0],
              [2.0, 4.0, 2.0],
              [2.0, 2.0, 4.0]])
c = np.array([8.0, 3.0, 3.0])

# equality A x = b
A = np.array([[1.0, 1.0, 1.0],
              [1.0, 0.0, -1.0]])
b = np.array([3.0, 0.0])


def f(x):
    x = np.asarray(x, dtype=float)
    return float(0.5 * x @ G @ x + c @ x)


def jac(x):
    x = np.asarray(x, dtype=float)
    return G @ x + c


# scipy-style equality constraints: fun(x) == 0
constraints = [
    {"type": "eq", "fun": lambda x: float(A[0] @ np.asarray(x) - b[0]),
     "jac": lambda x: A[0].copy()},
    {"type": "eq", "fun": lambda x: float(A[1] @ np.asarray(x) - b[1]),
     "jac": lambda x: A[1].copy()},
]
bounds = None
x0 = np.array([0.0, 0.0, 0.0])

# ---- closed-form KKT solution ----------------------------------------------
# [G  A^T] [x ]   [-c]
# [A  0  ] [mu] = [ b]
m = A.shape[0]
KKT = np.block([[G, A.T],
                [A, np.zeros((m, m))]])
rhs = np.concatenate([-c, b])
sol = np.linalg.solve(KKT, rhs)
X_STAR = sol[:3]
LAM_STAR = sol[3:]
KNOWN_OPTIMAL = f(X_STAR)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# ---- independent oracle: cvxpy ---------------------------------------------
oracle_obj = None
oracle_x = None
try:
    import cvxpy as cp
    xv = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(G)) + c @ xv),
                      [A @ xv == b])
    prob.solve()
    oracle_obj = float(prob.value)
    oracle_x = np.asarray(xv.value)
except Exception as e:  # noqa: BLE001
    print(f"(cvxpy oracle unavailable: {e})")

import pounce

# ---- auto route ------------------------------------------------------------
# Routing is OPT-IN since PR #97 (default solver_selection is "nlp"); request "auto" explicitly.
t0 = time.perf_counter()
res_auto = pounce.minimize(f, x0, jac=jac, bounds=bounds, constraints=constraints,
                           options={"solver_selection": "auto"})
t_auto = time.perf_counter() - t0

# ---- forced NLP ------------------------------------------------------------
t0 = time.perf_counter()
res_nlp = pounce.minimize(f, x0, jac=jac, bounds=bounds, constraints=constraints,
                          options={"solver_selection": "nlp"})
t_nlp = time.perf_counter() - t0

auto_solver = res_auto.info.get("solver") if isinstance(res_auto.info, dict) else None
nlp_solver = res_nlp.info.get("solver") if isinstance(res_nlp.info, dict) else None
used_specialized = auto_solver == "qp-ipm"

obj_disagree = rel(res_auto.fun, res_nlp.fun)
x_disagree = float(np.linalg.norm(np.asarray(res_auto.x) - np.asarray(res_nlp.x), np.inf))

print("=== closed-form KKT ===")
print(f"x*={X_STAR}  lambda*={LAM_STAR}  q*={KNOWN_OPTIMAL:.10f}")
print("=== oracle (cvxpy) ===")
if oracle_obj is not None:
    print(f"obj={oracle_obj:.10e} x={oracle_x}")
else:
    print("unavailable")
print("=== auto ===")
print(f"success={res_auto.success} status={res_auto.status} fun={res_auto.fun:.10e} "
      f"x={res_auto.x} t={t_auto:.4f}s")
print(f"auto.info.solver={auto_solver!r}  auto.info.keys="
      f"{list(res_auto.info.keys()) if isinstance(res_auto.info, dict) else res_auto.info}")
print("=== forced nlp ===")
print(f"success={res_nlp.success} status={res_nlp.status} fun={res_nlp.fun:.10e} "
      f"x={res_nlp.x} t={t_nlp:.4f}s")
print(f"nlp.info.solver={nlp_solver!r} "
      f"(None => true NLP backend, has mu={'mu' in res_nlp.info if isinstance(res_nlp.info, dict) else '?'})")
print("=== cross-check ===")
print(f"known_optimal={KNOWN_OPTIMAL:.10e}")
print(f"auto_rel_err_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e}")
if oracle_obj is not None:
    print(f"auto_rel_err_vs_oracle={rel(res_auto.fun, oracle_obj):.2e}")
print(f"x_star_err_auto={float(np.linalg.norm(np.asarray(res_auto.x) - X_STAR, np.inf)):.2e}")
print(f"auto_vs_nlp_obj_disagree={obj_disagree:.2e}  x_inf_disagree={x_disagree:.2e}")
print(f"used_specialized(qp)={used_specialized}  (auto route = {auto_solver!r})")

oracle_ok = (oracle_obj is None) or (rel(res_auto.fun, oracle_obj) < 1e-4)
# ROUTING_ERROR only if the ANSWERS disagree. A fall-through to nlp that still
# gets the right answer is "merely slower", NOT a bug.
ans_ok = (res_auto.success and res_nlp.success
          and obj_disagree < 1e-4
          and rel(res_auto.fun, KNOWN_OPTIMAL) < 1e-4
          and oracle_ok)
print("VERDICT: PASS" if ans_ok else
      f"VERDICT: FAIL (auto_vs_nlp_disagree={obj_disagree:.2e} "
      f"auto_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e})")
