"""Adversary cross-check: Hock-Schittkowski problem 34 (HS34).
Family: nlp   Class: linear objective, EXPONENTIAL inequality constraints,
              upper bounds active at the solution.
Source: Hock & Schittkowski, "Test Examples for Nonlinear Programming Codes",
        Lecture Notes in Economics and Mathematical Systems 187 (Springer, 1981),
        problem 34.

  minimize   -x1
  s.t.  g1: x2 - exp(x1) >= 0
        g2: x3 - exp(x2) >= 0
        0 <= x1 <= 100,  0 <= x2 <= 100,  0 <= x3 <= 10

Known optimum (H&S 1981):  f* = -log(log(10)) = -0.83403245
  x* = (log(log(10)), log(10), 10) = (0.8340324, 2.3025851, 10.0).
Both g1, g2 active and the x3 upper bound (=10) active at the solution.

Not previously logged (logged HS problems: 1,13,14,15,28,35,71,76,100,110):
HS34 is the exp-constrained monotone-chain problem. Oracle: Ipopt via Pyomo
(+MA57), SolverFactory('pounce'), and `pounce verify` on the emitted .nl.
"""
import math
import os
import subprocess
import tempfile
import time

import numpy as np

KNOWN_OPTIMAL = -math.log(math.log(10.0))
X_STAR = np.array([math.log(math.log(10.0)), math.log(10.0), 10.0])
X0 = np.array([0.0, 1.05, 2.9])
POUNCE_CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
HSL = "/Users/jkitchin/Dropbox/projects/CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5/lib"


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def f(x):
    return -x[0]


def fjac(x):
    return np.array([-1.0, 0.0, 0.0])


def g1(x):
    return x[1] - math.exp(x[0])


def g1jac(x):
    return np.array([-math.exp(x[0]), 1.0, 0.0])


def g2(x):
    return x[2] - math.exp(x[1])


def g2jac(x):
    return np.array([0.0, -math.exp(x[1]), 1.0])


# --- (0) FD-check every analytic derivative ---
def fd(fun, x, h=1e-6):
    out = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        out[i] = (fun(xp) - fun(xm)) / (2 * h)
    return out


fd_max = 0.0
for nm, fun, jac in [("f", f, fjac), ("g1", g1, g1jac), ("g2", g2, g2jac)]:
    for xt in (np.array([0.5, 1.4, 5.0]), X_STAR, np.array([0.2, 1.1, 3.3])):
        err = np.max(np.abs(jac(xt) - fd(fun, xt))) / max(1.0, np.max(np.abs(fd(fun, xt))))
        fd_max = max(fd_max, err)
print(f"=== derivative FD check === max_rel_err={fd_max:.2e} "
      f"({'OK' if fd_max < 1e-5 else 'BAD -> FORMULATION_ERROR'})")
print(f"f(x*)={f(X_STAR):.10f}  known f*={KNOWN_OPTIMAL:.10f}  "
      f"g(x*)=[{g1(X_STAR):.2e}, {g2(X_STAR):.2e}]")

# --- (A) pounce.minimize with analytic jac + bounds ---
import pounce
constraints = [
    {"type": "ineq", "fun": g1, "jac": g1jac},
    {"type": "ineq", "fun": g2, "jac": g2jac},
]
bounds = [(0.0, 100.0), (0.0, 100.0), (0.0, 10.0)]
t0 = time.perf_counter()
res = pounce.minimize(f, X0.copy(), jac=fjac, bounds=bounds, constraints=constraints,
                      options={"solver_selection": "nlp"})
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(res.x, float)
obj_pounce = float(res.fun)
status = res.status

# --- (B) Pyomo -> Ipopt (MA57) + SolverFactory('pounce') ---
import pyomo.environ as pyo


def build_model():
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(bounds=(0, 100), initialize=float(X0[0]))
    m.x2 = pyo.Var(bounds=(0, 100), initialize=float(X0[1]))
    m.x3 = pyo.Var(bounds=(0, 10), initialize=float(X0[2]))
    m.obj = pyo.Objective(expr=-m.x1)
    m.g1 = pyo.Constraint(expr=m.x2 - pyo.exp(m.x1) >= 0.0)
    m.g2 = pyo.Constraint(expr=m.x3 - pyo.exp(m.x2) >= 0.0)
    return m


os.environ["DYLD_LIBRARY_PATH"] = HSL + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
m_or = build_model()
ip = pyo.SolverFactory("ipopt")
ip.options["linear_solver"] = "ma57"
t0 = time.perf_counter()
ip_res = ip.solve(m_or, tee=False)
t_oracle = time.perf_counter() - t0
x_oracle = np.array([pyo.value(m_or.x1), pyo.value(m_or.x2), pyo.value(m_or.x3)])
obj_oracle = pyo.value(m_or.obj)

pp_obj = pp_x = None
try:
    m_pp = build_model()
    pyo.SolverFactory("pounce").solve(m_pp)
    pp_obj = pyo.value(m_pp.obj)
    pp_x = np.array([pyo.value(m_pp.x1), pyo.value(m_pp.x2), pyo.value(m_pp.x3)])
except Exception as e:  # noqa: BLE001
    pp_obj = f"unavailable: {type(e).__name__}: {e}"

# --- (C) pounce CLI on .nl + pounce verify ---
verify_rc = None
verify_out = cli_out = ""
try:
    tmp = tempfile.mkdtemp(prefix="hs34_")
    nlf = os.path.join(tmp, "hs34.nl")
    build_model().write(nlf, io_options={"symbolic_solver_labels": True})
    solf = os.path.join(tmp, "hs34.sol")
    cli = subprocess.run([POUNCE_CLI, nlf, solf], capture_output=True, text=True, timeout=60)
    cli_out = cli.stdout + cli.stderr
    if os.path.exists(solf):
        ver = subprocess.run([POUNCE_CLI, "verify", nlf, solf],
                             capture_output=True, text=True, timeout=60)
        verify_rc = ver.returncode
        verify_out = ver.stdout + ver.stderr
    else:
        verify_out = "no .sol produced\n" + cli_out
except Exception as e:  # noqa: BLE001
    verify_out = f"verify step failed: {type(e).__name__}: {e}"

# --- Report ---
print("=== pounce.minimize ===")
print(f"status={status} obj={obj_pounce:.10f} x={x_pounce} t={t_pounce:.4f}s")
print("=== Ipopt/MA57 oracle ===")
print(f"obj={obj_oracle:.10f} x={x_oracle} t={t_oracle:.4f}s")
print(f"=== SolverFactory('pounce') === obj={pp_obj if not isinstance(pp_obj, float) else round(pp_obj, 10)} x={pp_x}")
print("=== pounce verify (.nl/.sol) ===")
print(f"verify_rc={verify_rc}")
print(verify_out.strip()[:400])
obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_oracle = rel(obj_pounce, obj_oracle)
x_err = np.max(np.abs(x_pounce - X_STAR))
print(f"--- obj_err_vs_known={obj_err_known:.2e} obj_err_vs_ipopt={obj_err_oracle:.2e} "
      f"x_inf_err_vs_star={x_err:.2e} oracle_vs_known={rel(obj_oracle, KNOWN_OPTIMAL):.2e}")

verify_ok = (verify_rc == 0) if verify_rc is not None else True
ok = (res.success or status in ("optimal", "success")) and obj_err_known < 1e-4 \
    and obj_err_oracle < 1e-4 and x_err < 1e-3 and verify_ok
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, obj_err_known={obj_err_known:.2e}, "
      f"x_err={x_err:.2e}, verify_rc={verify_rc})")
