"""Adversary cross-check: routing transparency on unconstrained Rosenbrock
Family: autoroute   Class: unconstrained nonlinear least-squares / Rosenbrock
Source: H. H. Rosenbrock, "An automatic method for finding the greatest or least
  value of a function," Computer Journal 3(3):175-184 (1960). The classic banana
  function f(x) = 100*(y - x^2)^2 + (1 - x)^2; global min f* = 0 at x* = (1, 1).
This is a smooth, nonconvex, unconstrained problem — it should route to the
general NLP solver, so auto and forced-nlp should agree (transparency).
Known optimal: 0.0  at (1, 1)
"""
import time
import numpy as np

KNOWN_OPTIMAL = 0.0
X_STAR = np.array([1.0, 1.0])


def f(x):
    return 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2


# unconstrained, no bounds, no constraints
bounds = None
constraints = None
x0 = np.array([-1.2, 1.0])  # Rosenbrock's classic hard start

import pounce

# --- auto route ---
# Routing is OPT-IN since PR #97 (default solver_selection is "nlp"); request "auto" explicitly.
# (Rosenbrock is unconstrained nonconvex, so "auto" correctly falls through to the NLP backend.)
t0 = time.perf_counter()
res_auto = pounce.minimize(f, x0, options={"solver_selection": "auto"})
t_auto = time.perf_counter() - t0

# --- forced NLP ---
t0 = time.perf_counter()
res_nlp = pounce.minimize(f, x0, options={"solver_selection": "nlp"})
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
print(f"x_star_err_auto={float(np.linalg.norm(np.asarray(res_auto.x)-X_STAR, np.inf)):.2e}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} auto_rel_err_vs_known={rel(res_auto.fun, KNOWN_OPTIMAL):.2e}")
print(f"auto_vs_nlp_obj_disagree={obj_disagree:.2e} x_inf_disagree={x_disagree:.2e}")

# ROUTING_ERROR only if the ANSWERS disagree; a slower fall-through that still
# gets the right answer is "merely slower", not a bug. KNOWN_OPTIMAL is 0 so use
# an absolute tolerance on the objective vs known.
ans_ok = (res_auto.success and res_nlp.success
          and obj_disagree < 1e-4
          and abs(res_auto.fun - KNOWN_OPTIMAL) < 1e-6)
print("VERDICT: PASS" if ans_ok else f"VERDICT: FAIL (auto vs nlp disagree={obj_disagree:.2e})")
