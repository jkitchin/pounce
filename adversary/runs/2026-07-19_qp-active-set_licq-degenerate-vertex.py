"""Adversary cross-check: LICQ-degenerate QP vertex (3 active constraints in R^2)

Family: qp-active-set   Class: primal/dual degenerate convex QP (LICQ failure)
Source: Standard degenerate-QP construction, cf. Nocedal & Wright,
        *Numerical Optimization* 2e, Ch. 16.5 ("Degeneracy" / cycling in
        active-set QP methods) and Ch. 12.6 (LICQ). Instance built so the
        unconstrained minimizer (2,2) is cut off and the constrained
        minimizer sits at the vertex (1,1) where THREE inequalities are
        active in R^2 -> constraint gradients (1,1),(1,0),(0,1) are linearly
        dependent, LICQ fails, and the KKT multipliers are non-unique
        (a one-parameter family).  This is precisely the configuration that
        makes naive active-set methods cycle or return a wrong active set.

    min  0.5*((x1-2)^2 + (x2-2)^2)
    s.t. x1 + x2 <= 2
         x1       <= 1
               x2 <= 1

Known optimal: x* = (1,1), f* = 1.0  (projection of (2,2) onto the region;
(1,1) is feasible and is the unique closest feasible point since the region
is contained in {x1<=1} n {x2<=1} and (1,1) is the componentwise max).

Multiplier family: grad f = (-1,-1) at x*, so we need
  lam1*(1,1) + lam2*(1,0) + lam3*(0,1) = (1,1), lam >= 0
  => lam2 = lam3 = 1 - lam1, any lam1 in [0,1].  Non-unique -> dual degenerate.
"""

import time
import warnings

import numpy as np

KNOWN_X = np.array([1.0, 1.0])
KNOWN_OPTIMAL = 1.0

P = np.eye(2)
c = np.array([-2.0, -2.0])  # 0.5 x'Px + c'x = 0.5||x-2||^2 - 4
CONST = 4.0  # add back to compare with the (x-2)^2/2 form
G = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
h = np.array([2.0, 1.0, 1.0])


def f(x):
    return 0.5 * ((x[0] - 2.0) ** 2 + (x[1] - 2.0) ** 2)


def gf(x):
    return np.array([x[0] - 2.0, x[1] - 2.0])


def hf(x):
    return np.eye(2)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


x0 = np.array([-1.0, -1.0])

# --- finite-difference sanity check of my own derivatives (skepticism first) ---
eps = 1e-6
xt = np.array([0.3, -0.7])
fd = np.array(
    [
        (f(xt + eps * e) - f(xt - eps * e)) / (2 * eps)
        for e in (np.array([1.0, 0]), np.array([0, 1.0]))
    ]
)
assert np.allclose(fd, gf(xt), atol=1e-7), (fd, gf(xt))

from pounce import minimize, solve_qp

cons = [{"type": "ineq_le", "fun": lambda x: G @ x, "ub": h}]

# pounce constraint API: use scipy-style LinearConstraint
from scipy.optimize import LinearConstraint

lc = LinearConstraint(G, -np.inf, h)

# --- pounce: forced active-set SQP via solver_selection (the #213 path) ---
with warnings.catch_warnings(record=True) as w_as:
    warnings.simplefilter("always")
    t0 = time.perf_counter()
    r_as = minimize(
        f, x0, jac=gf, hess=hf, constraints=[lc], solver_selection="qp-active-set"
    )
    t_as = time.perf_counter() - t0

# --- pounce: same engine via algorithm= (must agree) ---
t0 = time.perf_counter()
r_alg = minimize(f, x0, jac=gf, hess=hf, constraints=[lc], algorithm="active-set-sqp")
t_alg = time.perf_counter() - t0

# --- pounce: convex QP IPM path (independent pounce engine) ---
t0 = time.perf_counter()
r_ipm = solve_qp(P=P, c=c, G=G, h=h)
t_ipm = time.perf_counter() - t0
obj_ipm = r_ipm.obj + CONST

# --- pounce: general NLP path ---
t0 = time.perf_counter()
r_nlp = minimize(f, x0, jac=gf, hess=hf, constraints=[lc], solver_selection="nlp")
t_nlp = time.perf_counter() - t0

# --- oracle: cvxpy (Clarabel + OSQP) ---
import cvxpy as cp

xv = cp.Variable(2)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(xv - 2.0)), [G @ xv <= h])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_orc = time.perf_counter() - t0
x_orc, obj_orc = xv.value, prob.value

prob.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10)
obj_orc2 = prob.value

print("=== known ===")
print(f"x*={KNOWN_X} f*={KNOWN_OPTIMAL:.10e}")
print("=== pounce solver_selection='qp-active-set' ===")
print(
    f"status={r_as.status} success={r_as.success} obj={r_as.fun:.10e} "
    f"x={np.asarray(r_as.x)} nit={r_as.nit} t={t_as:.4f}s"
)
print(f"warnings={[str(x.message)[:70] for x in w_as]}")
print("=== pounce algorithm='active-set-sqp' ===")
print(f"status={r_alg.status} obj={r_alg.fun:.10e} x={np.asarray(r_alg.x)} t={t_alg:.4f}s")
print("=== pounce solve_qp (convex IPM) ===")
print(f"status={r_ipm.status} obj={obj_ipm:.10e} x={np.asarray(r_ipm.x)} t={t_ipm:.4f}s")
print("=== pounce solver_selection='nlp' ===")
print(f"status={r_nlp.status} obj={r_nlp.fun:.10e} x={np.asarray(r_nlp.x)} t={t_nlp:.4f}s")
print("=== oracle cvxpy ===")
print(f"CLARABEL obj={obj_orc:.10e} x={x_orc} t={t_orc:.4f}s ; OSQP obj={obj_orc2:.10e}")

as_x_err = float(np.max(np.abs(np.asarray(r_as.x) - KNOWN_X)))
as_obj_err = rel(float(r_as.fun), KNOWN_OPTIMAL)
agree_alg = abs(float(r_as.fun) - float(r_alg.fun))
agree_ipm = rel(obj_ipm, KNOWN_OPTIMAL)
agree_orc = rel(obj_orc, KNOWN_OPTIMAL)
feas = float(np.max(G @ np.asarray(r_as.x) - h))

print(
    f"as_obj_err_vs_known={as_obj_err:.2e} as_x_inf_err={as_x_err:.2e} "
    f"as_vs_alg_obj_gap={agree_alg:.2e} ipm_err={agree_ipm:.2e} "
    f"cvxpy_err={agree_orc:.2e} as_max_violation={feas:.2e}"
)

ok = (
    bool(r_as.success)
    and as_obj_err < 1e-4
    and as_x_err < 1e-4
    and agree_alg < 1e-6
    and feas < 1e-7
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={r_as.status})")
