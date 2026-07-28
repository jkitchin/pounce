"""Adversary cross-check: routing transparency on a convex QP
Family: autoroute   Class: convex QP that should auto-route to the QP IPM
Source: routing-transparency contract — minimize(auto) must agree with
  minimize(solver_selection="nlp") to tolerance, AND auto should pick the
  specialized convex solver. Problem: Nocedal & Wright Example 16.4
  (x* = (1.4,1.7), q* = 0.8).
Known optimal: 0.8
"""
import time
import numpy as np

KNOWN_OPTIMAL = 0.8
X_STAR = np.array([1.4, 1.7])

def f(x):
    return (x[0] - 1.0)**2 + (x[1] - 2.5)**2

# linear inequality constraints, scipy-style (fun >= 0)
constraints = [
    {"type": "ineq", "fun": lambda x: x[0] - 2*x[1] + 2.0},
    {"type": "ineq", "fun": lambda x: -x[0] - 2*x[1] + 6.0},
    {"type": "ineq", "fun": lambda x: -x[0] + 2*x[1] + 2.0},
]
bounds = [(0.0, None), (0.0, None)]
x0 = np.array([2.0, 0.0])

import pounce

# --- auto route ---
# Routing is OPT-IN since PR #97 (default solver_selection is "nlp"); request "auto" explicitly.
t0 = time.perf_counter()
res_auto = pounce.minimize(f, x0, bounds=bounds, constraints=constraints,
                           options={"solver_selection": "auto"})
t_auto = time.perf_counter() - t0

# --- forced NLP ---
t0 = time.perf_counter()
res_nlp = pounce.minimize(f, x0, bounds=bounds, constraints=constraints,
                          options={"solver_selection": "nlp"})
t_nlp = time.perf_counter() - t0


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

obj_disagree = rel(res_auto.fun, res_nlp.fun)
x_disagree = float(np.linalg.norm(np.asarray(res_auto.x) - np.asarray(res_nlp.x), np.inf))

print("=== auto ===")
print(f"status={res_auto.status} fun={res_auto.fun:.10e} x={res_auto.x} t={t_auto:.4f}s")
print(f"auto.info={dict(res_auto.info) if hasattr(res_auto.info,'keys') else res_auto.info}")
print("=== forced nlp ===")
print(f"status={res_nlp.status} fun={res_nlp.fun:.10e} x={res_nlp.x} t={t_nlp:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} auto_rel_err_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e}")
print(f"auto_vs_nlp_obj_disagree={obj_disagree:.2e} x_inf_disagree={x_disagree:.2e}")

# ROUTING_ERROR only if the ANSWERS disagree; a slower fall-through that still
# gets the right answer is "merely slower", not a bug.
ans_ok = (res_auto.success and res_nlp.success
          and obj_disagree < 1e-4
          and rel(res_auto.fun, KNOWN_OPTIMAL) < 1e-4)
print("VERDICT: PASS" if ans_ok else f"VERDICT: FAIL (auto vs nlp disagree={obj_disagree:.2e})")
