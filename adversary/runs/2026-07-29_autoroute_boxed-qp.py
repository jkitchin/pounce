"""Adversary cross-check: auto-routing transparency on a boxed convex QP
Family: autoroute   Class: QP -> qp-ipm route, forced-nlp must agree
Source: minimize 0.5 x^T P x + c^T x,  P=diag(2,4,1), c=(-2,-6,-1),
        s.t. -1<=x_i<=3 (i=1,2,3), sum(x)<=3.
        Unconstrained minimizer x_u = -P^{-1}c = (1, 1.5, 1) has sum=3.5>3,
        so the sum-constraint is active and the box is not (verified below).
        KKT: P x + c + mu*1 = 0, mu>=0, at the active constraint:
            x_i = -(c_i+mu)/p_i,  sum_i x_i = 3  =>  mu = 2/7
            x* = (6/7, 10/7, 5/7) = (0.857142857..., 1.428571428..., 0.714285714...)
Known optimal: x* = (6/7, 10/7, 5/7)
"""
import time
import numpy as np
from fractions import Fraction

P = np.diag([2.0, 4.0, 1.0])
c = np.array([-2.0, -6.0, -1.0])

mu = Fraction(2, 7)
KNOWN_X = np.array([float(Fraction(2) - mu) / 2.0,
                     float(Fraction(6) - mu) / 4.0,
                     float(Fraction(1) - mu)])
KNOWN_OBJ = 0.5 * KNOWN_X @ P @ KNOWN_X + c @ KNOWN_X
assert abs(sum(KNOWN_X) - 3.0) < 1e-12
assert all(-1.0 <= xi <= 3.0 for xi in KNOWN_X)


def fun(x):
    return 0.5 * x @ P @ x + c @ x


def jac(x):
    return P @ x + c


bounds = [(-1, 3), (-1, 3), (-1, 3)]

import pounce
from scipy.optimize import LinearConstraint

A_lin = np.array([[1.0, 1.0, 1.0]])
lc = LinearConstraint(A_lin, -np.inf, 3.0)

t0 = time.perf_counter()
r_auto = pounce.minimize(fun, x0=np.zeros(3), jac=jac, bounds=bounds,
                          constraints=[lc], solver_selection="auto")
t_auto = time.perf_counter() - t0

t0 = time.perf_counter()
r_nlp = pounce.minimize(fun, x0=np.zeros(3), jac=jac, bounds=bounds,
                         constraints=[lc], solver_selection="nlp")
t_nlp = time.perf_counter() - t0

routed_solver = r_auto.info.get("solver") if hasattr(r_auto.info, "get") else getattr(r_auto.info, "solver", None)
problem_class = r_auto.info.get("problem_class") if hasattr(r_auto.info, "get") else getattr(r_auto.info, "problem_class", None)

# --- independent oracle: cvxpy ---
import cvxpy as cp

xv = cp.Variable(3)
constraints_cvx = [xv >= -1, xv <= 3, cp.sum(xv) <= 3]
prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(xv, P) + c @ xv), constraints_cvx)
prob.solve(solver=cp.CLARABEL)
x_cvx = np.asarray(xv.value)
obj_cvx = prob.value


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


x_auto = np.asarray(r_auto.x)
x_nlp = np.asarray(r_nlp.x)

auto_vs_nlp = float(np.linalg.norm(x_auto - x_nlp, np.inf))
auto_vs_known = float(np.linalg.norm(x_auto - KNOWN_X, np.inf))
nlp_vs_known = float(np.linalg.norm(x_nlp - KNOWN_X, np.inf))
auto_vs_cvx = float(np.linalg.norm(x_auto - x_cvx, np.inf))
obj_err_auto = rel(r_auto.fun, KNOWN_OBJ)
obj_err_nlp = rel(r_nlp.fun, KNOWN_OBJ)
obj_err_cvx = rel(r_auto.fun, obj_cvx)

print("=== pounce.minimize (solver_selection='auto') ===")
print(f"routed_solver={routed_solver} problem_class={problem_class}")
print(f"success={r_auto.success} obj={r_auto.fun:.10e} x={x_auto} t={t_auto:.4f}s")
print("=== pounce.minimize (solver_selection='nlp', forced) ===")
print(f"success={r_nlp.success} obj={r_nlp.fun:.10e} x={x_nlp} t={t_nlp:.4f}s")
print("=== independent oracle: cvxpy CLARABEL ===")
print(f"obj={obj_cvx:.10e} x={x_cvx}")
print(f"known_optimal={KNOWN_OBJ:.10e} x*={KNOWN_X}")
print(f"auto_vs_nlp_x_err={auto_vs_nlp:.2e} auto_vs_known_x_err={auto_vs_known:.2e} "
      f"nlp_vs_known_x_err={nlp_vs_known:.2e} auto_vs_cvxpy_x_err={auto_vs_cvx:.2e}")
print(f"obj_err_auto_vs_known={obj_err_auto:.2e} obj_err_nlp_vs_known={obj_err_nlp:.2e} "
      f"obj_err_auto_vs_cvxpy={obj_err_cvx:.2e}")

routed_specialized = routed_solver in ("qp-ipm", "lp-ipm", "socp")
ok = (
    r_auto.success and r_nlp.success
    and routed_specialized
    and auto_vs_nlp < 1e-5
    and auto_vs_known < 1e-4
    and nlp_vs_known < 1e-4
    and obj_err_auto < 1e-6
    and obj_err_cvx < 1e-6
)
if not routed_specialized:
    print(f"NOTE: auto route fell through to '{routed_solver}' instead of a specialized solver "
          "(not itself a bug if answers still agree, but logged as a routing observation)")
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
