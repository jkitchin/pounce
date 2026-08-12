"""Adversary cross-check: linear objective minimized over a Euclidean ball,
auto-routing agreement (auto vs forced-nlp)
Family: autoroute   Class: LINEAR objective + quadratic (ball) constraint --
              distinct from the prior autoroute probes (boxed QP, quadratic-
              objective ball QCQP, transportation LP, bounds-only diagonal
              QP, equality+inactive-inequality QP, warm_start+auto contract,
              Rosenbrock unconstrained nonconvex, badly-scaled QP+LP+QCQP,
              degenerate CQ-failure, indefinite-QP refusal, status-reporting
              certificates): the prior ball-QCQP probe (2026-07-30) used a
              QUADRATIC objective (distance-to-a-point); this one uses a
              purely LINEAR objective over the same constraint SHAPE, which
              is the textbook Cauchy-Schwarz/dual-norm SOCP -- a different
              structural class for the router's QP/QCQP/SOCP classifier to
              get right (a linear objective + one quadratic constraint has
              no quadratic term for a naive QP-detector to key off of).
Source: Boyd & Vandenberghe, "Convex Optimization", example 5.1 / sec 3.1.6
        (dual norm): minimize c^T x subject to ||x||_2 <= r has closed form
        x* = -r*c/||c||_2 (Cauchy-Schwarz equality case), optimal value
        -r*||c||_2. An additional strictly-inactive general linear
        inequality is included to make the feasible region non-trivial
        while leaving the closed form unaffected.
            minimize    x1 + 2*x2 - 2*x3
            subject to  x1^2 + x2^2 + x3^2 <= 25     (r=5)
                        x1 - x2 <= 100                (inactive)
Known optimal: c=(1,2,-2), ||c||=3, r=5 -> x* = -5*c/3 = (-5/3,-10/3,10/3),
        obj* = -5*3 = -15.0 (closed form, independent of any solver).
"""
import time

import numpy as np
from pounce import minimize

c_vec = np.array([1.0, 2.0, -2.0])
r_ball = 5.0
c_norm = float(np.linalg.norm(c_vec))
KNOWN_X = -r_ball * c_vec / c_norm
KNOWN_OPTIMAL = float(-r_ball * c_norm)
assert abs(KNOWN_OPTIMAL - (-15.0)) < 1e-10


def fun(x):
    return float(c_vec @ x)


def jac(x):
    return c_vec.copy()


constraints = [
    {"type": "ineq", "fun": lambda x: r_ball ** 2 - float(x @ x),
     "jac": lambda x: -2.0 * x},
    {"type": "ineq", "fun": lambda x: 100.0 - (x[0] - x[1]),
     "jac": lambda x: np.array([-1.0, 1.0, 0.0])},
]

x0 = np.array([0.1, 0.1, 0.1])

# --- pounce: auto-routed ---
t0 = time.perf_counter()
r_auto = minimize(fun, x0, jac=jac, constraints=constraints, solver_selection="auto")
t_auto = time.perf_counter() - t0
solver_auto = r_auto.info.get("solver") if hasattr(r_auto, "info") else None

# --- pounce: forced NLP path ---
t0 = time.perf_counter()
r_nlp = minimize(fun, x0, jac=jac, constraints=constraints, solver_selection="nlp")
t_nlp = time.perf_counter() - t0
solver_nlp = r_nlp.info.get("solver") if hasattr(r_nlp, "info") else None

# --- oracle: scipy SLSQP, fully independent of pounce ---
from scipy.optimize import minimize as scipy_minimize

scipy_constraints = [
    {"type": "ineq", "fun": lambda x: r_ball ** 2 - float(x @ x)},
    {"type": "ineq", "fun": lambda x: 100.0 - (x[0] - x[1])},
]
t0 = time.perf_counter()
r_scipy = scipy_minimize(fun, x0, jac=jac, constraints=scipy_constraints, method="SLSQP")
t_scipy = time.perf_counter() - t0


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


err_auto = rel(r_auto.fun, KNOWN_OPTIMAL)
err_nlp = rel(r_nlp.fun, KNOWN_OPTIMAL)
err_scipy = rel(r_scipy.fun, KNOWN_OPTIMAL)
x_err_auto = float(np.linalg.norm(np.asarray(r_auto.x) - KNOWN_X, np.inf))
x_err_nlp = float(np.linalg.norm(np.asarray(r_nlp.x) - KNOWN_X, np.inf))
answers_agree = (
    rel(r_auto.fun, r_nlp.fun) < 1e-6
    and np.linalg.norm(np.asarray(r_auto.x) - np.asarray(r_nlp.x), np.inf) < 1e-6
)

print("=== pounce auto ===")
print(f"status={r_auto.status} obj={r_auto.fun:.10e} x={np.asarray(r_auto.x)} solver_tag={solver_auto!r} t={t_auto:.4f}s")
print("=== pounce forced-nlp ===")
print(f"status={r_nlp.status} obj={r_nlp.fun:.10e} x={np.asarray(r_nlp.x)} solver_tag={solver_nlp!r} t={t_nlp:.4f}s")
print("=== oracle (scipy SLSQP) ===")
print(f"obj={r_scipy.fun:.10e} x={r_scipy.x} t={t_scipy:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL} rel_err_auto={err_auto:.2e} rel_err_nlp={err_nlp:.2e} rel_err_scipy={err_scipy:.2e}")
print(f"x_inf_err_auto={x_err_auto:.2e} x_inf_err_nlp={x_err_nlp:.2e} answers_agree(auto vs nlp)={answers_agree}")

routed_specialized = solver_auto not in (None, "nlp")
print(f"routed_specialized(expected True, e.g. socp-ipm)={routed_specialized} tag={solver_auto!r}")

numerically_ok = err_auto < 1e-4 and err_nlp < 1e-4 and err_scipy < 1e-4
routing_ok = answers_agree and routed_specialized

ok = numerically_ok and routing_ok
if ok:
    print("VERDICT: PASS")
elif numerically_ok and not routing_ok:
    print(f"VERDICT: FAIL (ROUTING_ERROR candidate: answers_agree={answers_agree}, "
          f"routed_specialized={routed_specialized}, solver_auto={solver_auto!r})")
else:
    print(f"VERDICT: FAIL (obj mismatch: err_auto={err_auto:.2e}, err_nlp={err_nlp:.2e}, err_scipy={err_scipy:.2e})")
