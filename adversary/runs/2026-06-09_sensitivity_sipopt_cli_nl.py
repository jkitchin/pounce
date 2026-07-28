"""Adversary cross-check: sIPOPT CLI path on a .nl (sens_sol_state_1 suffix)
Family: sensitivity   Class: NLP parameter perturbation via the sIPOPT CLI suffixes
Source: upstream sIPOPT `parametric_cpp` example (parametricTNLP.cpp), the
        canonical sIPOPT demo (Pirnay, Lopez-Negrete, Biegler 2012). Fixture
        copied verbatim from
        crates/pounce-cli/tests/fixtures/parametric.nl into adversary/fixtures/.

Problem (decoded from the .nl):
    min  x1^2 + x2^2 + x3^2
    s.t. 6*x1 + 3*x2 + 2*x3 - p1 = 0
         p2*x1 + x2 - x3 - 1     = 0
    parameters pinned at nominal  p1 = 5, p2 = 1
    perturbed parameter values    p1 = 4.5, p2 = 1   (only p1 moves, Dp1 = -0.5)

The .nl declares the three sIPOPT suffixes:
    sens_state_1        -> tags p1 (var 3) and p2 (var 4) as parameters
    sens_state_value_1  -> perturbed values 4.5 and 1
    sens_init_constr    -> constraints C2,C3 pin p1,p2 to nominal

CLI path under test:
    pounce <nl> <sol>   ->  writes the perturbed primal as the
                            `sens_sol_state_1` real-var suffix in the .sol.

Oracle: a central finite-difference re-solve of the SAME NLP with scipy,
    dx/dp1 ~= (x*(p1+delta) - x*(p1-delta)) / (2 delta),
then the first-order predictor  x*(nominal) + dx/dp1 * Dp1  must match
pounce's sens_sol_state_1. Because the constraints are linear and the
objective quadratic, x*(p1) is affine in p1, so the first-order predictor is
also EXACT vs the full re-solve at p1=4.5 -- we check both.
"""
import os
import re
import subprocess
import time
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
NL = os.path.join(HERE, "..", "fixtures", "parametric.nl")
CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
SOL = "/tmp/adv_sipopt_param.sol"

P1_NOM, P2_NOM = 5.0, 1.0
P1_PERT = 4.5
DP1 = P1_PERT - P1_NOM       # -0.5
DELTA = 1e-6                 # FD step for dx/dp1


def resolve(p1, p2):
    """Independently re-solve the NLP for given parameters (scipy SLSQP)."""
    cons = [
        {"type": "eq", "fun": lambda x, p1=p1: 6 * x[0] + 3 * x[1] + 2 * x[2] - p1},
        {"type": "eq", "fun": lambda x, p2=p2: p2 * x[0] + x[1] - x[2] - 1.0},
    ]
    r = minimize(lambda x: x @ x, np.array([0.15, 0.15, 0.0]),
                 constraints=cons, method="SLSQP",
                 options={"maxiter": 1000, "ftol": 1e-16})
    return r.x


def parse_sens_sol_state_1(text):
    """Pull the sens_sol_state_1 real-var suffix block out of a .sol."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^suffix\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", lines[i])
        if m:
            count = int(m.group(2))
            tabline = int(m.group(5))
            name = lines[i + 1].strip()
            body_start = i + 2 + tabline
            if name == "sens_sol_state_1":
                out = {}
                for j in range(count):
                    idx, val = lines[body_start + j].split()
                    out[int(idx)] = float(val)
                n = max(out) + 1
                return np.array([out.get(k, 0.0) for k in range(n)])
            i = body_start + count
            continue
        i += 1
    return None


assert os.path.exists(NL), f"fixture missing: {NL}"

# --- pounce CLI sIPOPT path ---
t0 = time.perf_counter()
proc = subprocess.run([CLI, NL, SOL], capture_output=True, text=True)
t_pounce = time.perf_counter() - t0
assert proc.returncode == 0, f"pounce CLI failed: {proc.returncode}\n{proc.stderr}"

sol_text = open(SOL).read()
sens = parse_sens_sol_state_1(sol_text)
assert sens is not None, "sens_sol_state_1 suffix not found in .sol"
x_pounce_pert = sens[:3]          # perturbed x (first 3 vars; 3,4 are params)

# --- oracle: nominal + central-FD dx/dp1 + predictor + actual re-solve ---
t0 = time.perf_counter()
x_nom = resolve(P1_NOM, P2_NOM)
xp = resolve(P1_NOM + DELTA, P2_NOM)
xm = resolve(P1_NOM - DELTA, P2_NOM)
dxdp1 = (xp - xm) / (2 * DELTA)
x_pred = x_nom + dxdp1 * DP1                 # first-order predictor
x_actual = resolve(P1_PERT, P2_NOM)          # full nonlinear re-solve
t_oracle = time.perf_counter() - t0


def relinf(a, b):
    return float(np.linalg.norm(a - b, np.inf) / max(1.0, np.linalg.norm(b, np.inf)))


err_vs_pred = relinf(x_pounce_pert, x_pred)
err_vs_actual = relinf(x_pounce_pert, x_actual)
# implied dx/dp1 from pounce's perturbed primal (since predictor is exact here)
dxdp1_pounce = (x_pounce_pert - x_nom) / DP1
err_dxdp1 = relinf(dxdp1_pounce, dxdp1)

print("=== pounce CLI (sens_sol_state_1) ===")
print(f"return={proc.returncode}  t={t_pounce:.4f}s")
print(f"x_perturbed (pounce) = {x_pounce_pert}")
print(f"implied dx/dp1       = {dxdp1_pounce}")
print("=== oracle (scipy FD re-solve) ===")
print(f"x_nominal            = {x_nom}")
print(f"dx/dp1 (central FD)  = {dxdp1}   (delta={DELTA})")
print(f"x_predictor x_nom+dx*Dp1 = {x_pred}")
print(f"x_actual resolve(p1=4.5) = {x_actual}")
print(f"t_oracle={t_oracle:.4f}s")
print("=== cross-checks ===")
print(f"dx/dp1 analytic-vs-FD rel_inf_err   = {err_dxdp1:.2e}")
print(f"pounce vs FD predictor  rel_inf_err = {err_vs_pred:.2e}")
print(f"pounce vs actual resolve rel_inf_err= {err_vs_actual:.2e}")

ok = (err_dxdp1 < 1e-4) and (err_vs_pred < 1e-4) and (err_vs_actual < 1e-4)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (dxdp1={err_dxdp1:.2e}, pred={err_vs_pred:.2e}, actual={err_vs_actual:.2e})")
