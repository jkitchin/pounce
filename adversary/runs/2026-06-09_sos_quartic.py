"""Adversary cross-check: global minimization of a double-well quartic
Family: sos   Class: unconstrained polynomial global minimization (Lasserre)
Source: classic double-well p(x) = x^4 - 3 x^2 + 1.
  p'(x) = 4x^3 - 6x = 0 -> x^2 = 3/2 -> p* = (3/2)^2 - 3(3/2) + 1 = -1.25.
Known optimal (global min): -1.25 at x = +/- sqrt(1.5).
"""
import time
import numpy as np

KNOWN_OPTIMAL = -1.25

# polynomial as {exponent_tuple: coeff}
objective = {(4,): 1.0, (2,): -3.0, (0,): 1.0}

# --- pounce SOS / Lasserre ---
import pounce
t0 = time.perf_counter()
r = pounce.sos_minimize(objective, n_vars=1, order=2)
t_pounce = time.perf_counter() - t0
lb_pounce = r.lower_bound
status = r.status

# --- oracle: dense grid refutation (independent) ---
xs = np.linspace(-3.0, 3.0, 2_000_001)
ps = xs**4 - 3 * xs**2 + 1.0
grid_min = float(ps.min())
grid_argmin = float(xs[int(ps.argmin())])


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

# SOS returns a LOWER bound. Valid iff lb <= true global min (+ tiny slack)
# and, when tight, lb ~= known optimum.
err_vs_known = rel(lb_pounce, KNOWN_OPTIMAL)
exceeds_true = lb_pounce > grid_min + 1e-6   # invalid lower bound => SOLVER_BUG

print("=== pounce SOS ===")
print(f"status={status} lower_bound={lb_pounce:.10e} is_exact={r.is_exact} "
      f"n_minimizers={r.num_minimizers} t={t_pounce:.4f}s")
print("=== oracle (dense grid) ===")
print(f"grid_min={grid_min:.10e} at x={grid_argmin:.6f}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={err_vs_known:.2e}")
print(f"lower_bound_exceeds_true_min={exceeds_true}")

if exceeds_true:
    print("VERDICT: FAIL (invalid lower bound exceeds true global minimum)")
else:
    ok = r.success and err_vs_known < 1e-4
    print("VERDICT: PASS" if ok else f"VERDICT: FAIL (loose/incorrect bound, err={err_vs_known:.2e})")
