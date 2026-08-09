"""Adversary cross-check: auto-routing agreement on an equality+inactive-
inequality convex QP (not box-only, non-uniform diagonal Hessian)
Family: autoroute   Class: QP -> qp-ipm specialized route vs forced NLP
Source: closed-form KKT (Nocedal & Wright, "Numerical Optimization" 2e,
        Ch. 16 style): minimize x1^2 + 2*x2^2  s.t.  x1+x2 = 3 (equality),
        x1-x2 <= 5 (inequality). Lagrangian stationarity on the equality
        alone: 2x1=lambda, 4x2=lambda => x1=lambda/2, x2=lambda/4;
        x1+x2=3 => (3/4)lambda=3 => lambda=4 => x1=2, x2=1. Check the
        inequality: x1-x2=1 <= 5, strictly inactive, so this point is the
        unconstrained-by-inequality KKT point and hence the global optimum
        of the convex QP.
Known optimal: 6.0 at x=(2,1); inequality multiplier must be 0 (inactive).

Distinct from the previously-logged autoroute probes (boxed QP, quadratic
ball QCQP, transportation LP, bounds-only diagonal QP, and the
warm_start+auto routing-decision contract): this one has no box/bound
constraints at all, uses a non-uniform diagonal Hessian (1 vs 2), a general
affine equality (not axis-aligned), and a strictly-inactive general affine
inequality (not a bound) -- a fresh structural combination for the
LP/QP-detection probe to classify correctly.
"""
import time
import numpy as np
from pounce import minimize

KNOWN_X = np.array([2.0, 1.0])
KNOWN_OPTIMAL = 6.0


def fun(x):
    return x[0] ** 2 + 2 * x[1] ** 2


def jac(x):
    return np.array([2 * x[0], 4 * x[1]])


constraints = [
    {"type": "eq", "fun": lambda x: x[0] + x[1] - 3.0,
     "jac": lambda x: np.array([1.0, 1.0])},
    {"type": "ineq", "fun": lambda x: 5.0 - (x[0] - x[1]),
     "jac": lambda x: np.array([-1.0, 1.0])},
]

x0 = np.array([0.1, 0.1])

# --- pounce: auto-routed (should detect convex QP structure -> qp-ipm) ---
t0 = time.perf_counter()
r_auto = minimize(fun, x0, jac=jac, constraints=constraints, solver_selection="auto")
t_auto = time.perf_counter() - t0
solver_auto = r_auto.info.get("solver") if hasattr(r_auto, "info") else None

# --- pounce: forced NLP path (routing-transparency reference) ---
t0 = time.perf_counter()
r_nlp = minimize(fun, x0, jac=jac, constraints=constraints, solver_selection="nlp")
t_nlp = time.perf_counter() - t0
solver_nlp = r_nlp.info.get("solver") if hasattr(r_nlp, "info") else None

# --- oracle: scipy SLSQP, fully independent of pounce ---
from scipy.optimize import minimize as scipy_minimize

scipy_constraints = [
    {"type": "eq", "fun": lambda x: x[0] + x[1] - 3.0},
    {"type": "ineq", "fun": lambda x: 5.0 - (x[0] - x[1])},
]
t0 = time.perf_counter()
r_scipy = scipy_minimize(fun, x0, jac=jac, constraints=scipy_constraints, method="SLSQP")
t_scipy = time.perf_counter() - t0


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


err_auto = rel(r_auto.fun, KNOWN_OPTIMAL)
err_nlp = rel(r_nlp.fun, KNOWN_OPTIMAL)
err_scipy = rel(r_scipy.fun, KNOWN_OPTIMAL)
x_err_auto = float(np.linalg.norm(r_auto.x - KNOWN_X, np.inf))
x_err_nlp = float(np.linalg.norm(r_nlp.x - KNOWN_X, np.inf))
answers_agree = rel(r_auto.fun, r_nlp.fun) < 1e-6 and np.linalg.norm(r_auto.x - r_nlp.x, np.inf) < 1e-6

print("=== pounce auto (expected route: qp-ipm) ===")
print(f"status={r_auto.status} obj={r_auto.fun:.10e} x={r_auto.x} solver_tag={solver_auto!r} t={t_auto:.4f}s")
print("=== pounce forced-nlp ===")
print(f"status={r_nlp.status} obj={r_nlp.fun:.10e} x={r_nlp.x} solver_tag={solver_nlp!r} t={t_nlp:.4f}s")
print("=== oracle (scipy SLSQP) ===")
print(f"obj={r_scipy.fun:.10e} x={r_scipy.x} t={t_scipy:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL} rel_err_auto={err_auto:.2e} rel_err_nlp={err_nlp:.2e} rel_err_scipy={err_scipy:.2e}")
print(f"x_inf_err_auto={x_err_auto:.2e} x_inf_err_nlp={x_err_nlp:.2e} answers_agree(auto vs nlp)={answers_agree}")

routed_specialized = solver_auto not in (None, "nlp")
print(f"routed_specialized(expected True, tag qp-ipm)={routed_specialized} tag={solver_auto!r}")

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
