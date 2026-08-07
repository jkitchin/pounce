"""Adversary cross-check: warm_start= forces the NLP path even when
solver_selection="auto" would otherwise route to the convex QP solver.
Family: autoroute   Class: routing-transparency contract (warm_start vs auto)
Source: pounce.minimize docstring ("A warm start always runs on the NLP
        path -- convex routing is skipped") and python/pounce/_minimize.py
        line ~1115 (`if warm_start is not None and selection != "nlp":
        warnings.warn(...); selection = "nlp"`). Problem itself: strictly
        convex bound-constrained QP with closed-form optimum via KKT
        (Nocedal & Wright 2e style box-constrained quadratic).
Known optimal: f(x)=0.5x1^2+0.5x2^2-x1-2x2, x1,x2>=0, x1+x2<=5 (inactive).
        Unconstrained stationary point (1,2) is feasible (bounds inactive,
        sum=3<5) -> global optimum x*=(1,2), f*=-2.5.

This is a fresh angle within the autoroute family: every prior autoroute log
entry checks that auto- and forced-nlp *answers* agree. This one checks the
*routing decision itself* against the documented/coded contract -- that
combining solver_selection="auto" with warm_start= is not silently honored
as "auto" (which would use the convex solver and skip the NLP path the
warm-start payload was built for), but is forced back to "nlp" with a
warning. A violation here would be a silent contract break: the caller
asked for auto-routing *and* supplied a warm start, and got neither what
"auto" promises (routing) nor a warning that their warm_start might be
partially honored.
"""
import time
import warnings

import numpy as np
from pounce import minimize, WarmStart


def fun(x):
    return 0.5 * x[0] ** 2 + 0.5 * x[1] ** 2 - x[0] - 2 * x[1]


def jac(x):
    return np.array([x[0] - 1.0, x[1] - 2.0])


bounds = [(0.0, None), (0.0, None)]
constraints = [{"type": "ineq", "fun": lambda x: 5.0 - (x[0] + x[1]),
                "jac": lambda x: np.array([-1.0, -1.0])}]

KNOWN_X = np.array([1.0, 2.0])
KNOWN_OPTIMAL = -2.5

x0 = np.array([0.1, 0.1])

# --- Step 1: cold solve with solver_selection="auto" (no warm_start).
#     Expect it to actually route to the convex solver (info["solver"] set).
t0 = time.perf_counter()
r1 = minimize(fun, x0, jac=jac, bounds=bounds, constraints=constraints,
              options={"solver_selection": "auto"})
t_r1 = time.perf_counter() - t0
solver1 = r1.info.get("solver") if hasattr(r1, "info") else None

# --- Step 2: capture a warm start from r1, then re-solve the SAME problem
#     with solver_selection="auto" AND warm_start=. Per the docstring/code,
#     this must warn and force the NLP path (info["solver"] must NOT be a
#     convex-solver tag on this call).
ws = WarmStart.from_info(r1.x, r1.info)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    t0 = time.perf_counter()
    r2 = minimize(fun, x0, jac=jac, bounds=bounds, constraints=constraints,
                  options={"solver_selection": "auto"}, warm_start=ws)
    t_r2 = time.perf_counter() - t0
    warned = any("warm_start" in str(w.message) and "NLP" in str(w.message) for w in caught)

solver2 = r2.info.get("solver") if hasattr(r2, "info") else None

# --- oracle: scipy SLSQP on the same problem (independent of pounce entirely) ---
from scipy.optimize import minimize as scipy_minimize

t0 = time.perf_counter()
r_scipy = scipy_minimize(fun, x0, jac=jac, bounds=bounds,
                          constraints=[{"type": "ineq",
                                        "fun": lambda x: 5.0 - (x[0] + x[1])}],
                          method="SLSQP")
t_scipy = time.perf_counter() - t0


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


err1 = rel(r1.fun, KNOWN_OPTIMAL)
err2 = rel(r2.fun, KNOWN_OPTIMAL)
errs = rel(r_scipy.fun, KNOWN_OPTIMAL)
x1_err = float(np.linalg.norm(r1.x - KNOWN_X, np.inf))
x2_err = float(np.linalg.norm(r2.x - KNOWN_X, np.inf))

print("=== pounce r1: auto, no warm_start ===")
print(f"status={r1.status} obj={r1.fun:.10e} x={r1.x} solver_tag={solver1!r} t={t_r1:.4f}s")
print("=== pounce r2: auto + warm_start (must fall back to NLP) ===")
print(f"status={r2.status} obj={r2.fun:.10e} x={r2.x} solver_tag={solver2!r} t={t_r2:.4f}s warned={warned}")
print("=== oracle (scipy SLSQP) ===")
print(f"obj={r_scipy.fun:.10e} x={r_scipy.x} t={t_scipy:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL} rel_err_r1={err1:.2e} rel_err_r2={err2:.2e} rel_err_scipy={errs:.2e}")
print(f"x_inf_err_r1={x1_err:.2e} x_inf_err_r2={x2_err:.2e}")

r1_routed_convex = solver1 not in (None, "nlp")
r2_fell_back_to_nlp = solver2 in (None, "nlp")

print(f"r1_routed_convex(expected True)={r1_routed_convex} "
      f"r2_fell_back_to_nlp(expected True)={r2_fell_back_to_nlp} "
      f"r2_warned(expected True)={warned}")

# The numerical answer must be correct in both cases regardless of routing.
numerically_ok = err1 < 1e-4 and err2 < 1e-4 and errs < 1e-4
contract_ok = r2_fell_back_to_nlp and warned

ok = numerically_ok and contract_ok
if ok:
    print("VERDICT: PASS")
elif numerically_ok and not contract_ok:
    print(f"VERDICT: FAIL (ROUTING_ERROR candidate: contract violated -- "
          f"r2_fell_back_to_nlp={r2_fell_back_to_nlp}, warned={warned}, solver2={solver2!r})")
else:
    print(f"VERDICT: FAIL (obj mismatch: err1={err1:.2e}, err2={err2:.2e}, errs={errs:.2e})")
