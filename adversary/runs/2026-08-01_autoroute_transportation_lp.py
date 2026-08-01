"""Adversary cross-check: auto-routing on a balanced transportation LP
Family: autoroute   Class: LP -> lp-ipm route (equalities + nonneg bounds,
    forced-nlp must agree). Fresh class for autoroute -- prior autoroute
    runs used QP (convex_qp, eqcon_qp, indefinite-box-qp, boxed-qp,
    nonunique-optimal-face), QCQP->socp (qcqp_ball, two_ball_qcqp,
    ball-projection-qcqp), a 2-var toy LP (lp_eq_ineq), and NLP-only traps
    (rosenbrock, illscaled-detection, status_certificates,
    nonconvex_traps): none used a genuine multi-row equality-constrained
    transportation LP or explicitly asserted problem_class == "lp" (as
    opposed to "qp" with P=0).
Source: classic balanced transportation problem (see e.g. Bertsimas &
    Tsitsiklis, "Introduction to Linear Optimization", Ch 1 example, or
    any LP textbook's transportation chapter). 2 supply nodes (20, 30),
    3 demand nodes (10, 25, 15), unit costs c_ij given below.
Known optimal: computed independently via scipy.optimize.linprog (HiGHS
    dual simplex), a completely different LP algorithm/codebase from
    pounce's interior-point LP path.
"""
import time
import numpy as np

supply = np.array([20.0, 30.0])
demand = np.array([10.0, 25.0, 15.0])
cost = np.array([
    [8.0, 6.0, 10.0],
    [9.0, 12.0, 7.0],
])
assert abs(supply.sum() - demand.sum()) < 1e-9   # balanced

# x_ij, i in 0..1 (supply), j in 0..2 (demand); flatten row-major: x[3*i+j]
nvar = 6


def flat(i, j):
    return 3 * i + j


c = cost.flatten()

# equality constraints: row sums = supply, column sums = demand (5 rows;
# one is redundant for a balanced problem, but pounce's presolve/LP path
# must handle that -- this is itself part of what's being probed)
Arows, brhs = [], []
for i in range(2):
    row = np.zeros(nvar)
    for j in range(3):
        row[flat(i, j)] = 1.0
    Arows.append(row)
    brhs.append(supply[i])
for j in range(3):
    row = np.zeros(nvar)
    for i in range(2):
        row[flat(i, j)] = 1.0
    Arows.append(row)
    brhs.append(demand[j])
A_lin = np.array(Arows)
b_lin = np.array(brhs)

bounds = [(0.0, None)] * nvar


def fun(x):
    return float(c @ x)


def jac(x):
    return c


import pounce
from scipy.optimize import LinearConstraint

lc = LinearConstraint(A_lin, b_lin, b_lin)   # equality: lb == ub == b_lin
x0 = np.zeros(nvar)

t0 = time.perf_counter()
r_auto = pounce.minimize(fun, x0=x0, jac=jac, bounds=bounds, constraints=[lc],
                          solver_selection="auto")
t_auto = time.perf_counter() - t0

t0 = time.perf_counter()
r_nlp = pounce.minimize(fun, x0=x0, jac=jac, bounds=bounds, constraints=[lc],
                         solver_selection="nlp")
t_nlp = time.perf_counter() - t0

routed_solver = r_auto.info.get("solver") if hasattr(r_auto.info, "get") else getattr(r_auto.info, "solver", None)
problem_class = r_auto.info.get("problem_class") if hasattr(r_auto.info, "get") else getattr(r_auto.info, "problem_class", None)

x_auto = np.asarray(r_auto.x, float)
x_nlp = np.asarray(r_nlp.x, float)

# --- oracle: scipy.optimize.linprog (HiGHS), fully independent LP codebase ---
from scipy.optimize import linprog

t0 = time.perf_counter()
res_lp = linprog(c, A_eq=A_lin, b_eq=b_lin, bounds=bounds, method="highs")
t_lp = time.perf_counter() - t0
assert res_lp.success, res_lp.message
x_lp = res_lp.x
obj_lp = res_lp.fun

def rel(a, ref):
    return abs(a - ref) / max(1.0, abs(ref))

auto_vs_nlp = float(np.linalg.norm(x_auto - x_nlp, np.inf))
auto_vs_lp = float(np.linalg.norm(x_auto - x_lp, np.inf))
obj_err_auto = rel(r_auto.fun, obj_lp)
obj_err_nlp = rel(r_nlp.fun, obj_lp)

print("=== pounce.minimize (solver_selection='auto') ===")
print(f"routed_solver={routed_solver} problem_class={problem_class}")
print(f"success={r_auto.success} obj={r_auto.fun:.10e} x={x_auto} t={t_auto:.4f}s")
print("=== pounce.minimize (solver_selection='nlp', forced) ===")
print(f"success={r_nlp.success} obj={r_nlp.fun:.10e} x={x_nlp} t={t_nlp:.4f}s")
print("=== oracle: scipy.optimize.linprog (HiGHS) ===")
print(f"obj={obj_lp:.10e} x={x_lp} t={t_lp:.4f}s")
print(f"auto_vs_nlp_inf={auto_vs_nlp:.2e} auto_vs_lp_inf={auto_vs_lp:.2e} "
      f"obj_err_auto={obj_err_auto:.2e} obj_err_nlp={obj_err_nlp:.2e}")

routed_to_lp = problem_class in ("lp", "convex_qp") or (routed_solver in ("lp-ipm", "qp-ipm"))
ok = r_auto.success and r_nlp.success and auto_vs_nlp < 1e-5 and obj_err_auto < 1e-6 \
    and obj_err_nlp < 1e-6 and routed_to_lp
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (auto_vs_nlp={auto_vs_nlp:.2e}, obj_err_auto={obj_err_auto:.2e}, "
      f"routed_solver={routed_solver}, problem_class={problem_class})")
if not routed_to_lp:
    print(f"NOTE: auto-route did not select the specialized LP path (routed_solver="
          f"{routed_solver}, problem_class={problem_class}) -- if the answer still "
          f"matches the oracle this is a performance/routing note, not a correctness bug.")
