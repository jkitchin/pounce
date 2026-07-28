"""Adversary cross-check: Hock-Schittkowski problem 100 (HS100)
Family: nlp   Class: inequality-constrained, unbounded vars, high-degree
Source: Hock & Schittkowski, "Test Examples for Nonlinear Programming
        Codes", Lecture Notes in Economics and Mathematical Systems 187
        (Springer, 1981), problem 100 (originally Asaadi 1973, problem 2).

  minimize (x1-10)^2 + 5(x2-12)^2 + x3^4 + 3(x4-11)^2
           + 10 x5^6 + 7 x6^2 + x7^4 - 4 x6 x7 - 10 x6 - 8 x7
  s.t.  g1: 127 - 2 x1^2 - 3 x2^4 - x3 - 4 x4^2 - 5 x5        >= 0
        g2: 282 - 7 x1 - 3 x2 - 10 x3^2 - x4 + x5             >= 0
        g3: 196 - 23 x1 - x2^2 - 6 x6^2 + 8 x7                >= 0
        g4: -4 x1^2 - x2^2 + 3 x1 x2 - 2 x3^2 - 5 x6 + 11 x7  >= 0

  x0 = (1, 2, 0, 4, 0, 1, 1)   (f(x0) = 714)

Known optimum (H&S 1981, p. 121):
  f* = 680.6300573
  x* = (2.330499, 1.951372, -0.4775414, 4.365726,
        -0.6244870, 1.038131, 1.594227)
  g1 and g4 active at the solution.

Strategy: (A) pounce.minimize with analytic jacobians (FD-checked),
(B) ONE Pyomo model solved by SolverFactory('ipopt') and
SolverFactory('pounce'), (C) .nl written from that model, solved by the
pounce CLI and independently checked with `pounce verify`.
"""
import os
import subprocess
import tempfile
import time

import numpy as np

KNOWN_OPTIMAL = 680.6300573
X_STAR = np.array([2.330499, 1.951372, -0.4775414, 4.365726,
                   -0.6244870, 1.038131, 1.594227])
X0 = np.array([1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0])

POUNCE_CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# ---------------------------------------------------------------------------
# Analytic model + derivatives
# ---------------------------------------------------------------------------
def f(x):
    return ((x[0] - 10.0) ** 2 + 5.0 * (x[1] - 12.0) ** 2 + x[2] ** 4
            + 3.0 * (x[3] - 11.0) ** 2 + 10.0 * x[4] ** 6 + 7.0 * x[5] ** 2
            + x[6] ** 4 - 4.0 * x[5] * x[6] - 10.0 * x[5] - 8.0 * x[6])


def fjac(x):
    return np.array([
        2.0 * (x[0] - 10.0),
        10.0 * (x[1] - 12.0),
        4.0 * x[2] ** 3,
        6.0 * (x[3] - 11.0),
        60.0 * x[4] ** 5,
        14.0 * x[5] - 4.0 * x[6] - 10.0,
        4.0 * x[6] ** 3 - 4.0 * x[5] - 8.0,
    ])


def g1(x):
    return 127.0 - 2.0 * x[0] ** 2 - 3.0 * x[1] ** 4 - x[2] - 4.0 * x[3] ** 2 \
        - 5.0 * x[4]


def g1jac(x):
    return np.array([-4.0 * x[0], -12.0 * x[1] ** 3, -1.0, -8.0 * x[3],
                     -5.0, 0.0, 0.0])


def g2(x):
    return 282.0 - 7.0 * x[0] - 3.0 * x[1] - 10.0 * x[2] ** 2 - x[3] + x[4]


def g2jac(x):
    return np.array([-7.0, -3.0, -20.0 * x[2], -1.0, 1.0, 0.0, 0.0])


def g3(x):
    return 196.0 - 23.0 * x[0] - x[1] ** 2 - 6.0 * x[5] ** 2 + 8.0 * x[6]


def g3jac(x):
    return np.array([-23.0, -2.0 * x[1], 0.0, 0.0, 0.0, -12.0 * x[5], 8.0])


def g4(x):
    return (-4.0 * x[0] ** 2 - x[1] ** 2 + 3.0 * x[0] * x[1]
            - 2.0 * x[2] ** 2 - 5.0 * x[5] + 11.0 * x[6])


def g4jac(x):
    return np.array([-8.0 * x[0] + 3.0 * x[1], -2.0 * x[1] + 3.0 * x[0],
                     -4.0 * x[2], 0.0, 0.0, -5.0, 11.0])


# ---------------------------------------------------------------------------
# (0) Finite-difference check of EVERY analytic derivative (guard against the
#     #1 adversary false positive).
# ---------------------------------------------------------------------------
def fd_grad(fun, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (fun(xp) - fun(xm)) / (2.0 * h)
    return g


fd_report = []
fd_max = 0.0
for name, fun, jac in [("f", f, fjac), ("g1", g1, g1jac), ("g2", g2, g2jac),
                       ("g3", g3, g3jac), ("g4", g4, g4jac)]:
    for xt in (X0 + 0.3, X_STAR, np.array([0.7, -1.3, 0.9, 2.2,
                                           0.4, -0.8, 1.1])):
        a = jac(xt)
        n = fd_grad(fun, xt)
        err = float(np.max(np.abs(a - n)) / max(1.0, np.max(np.abs(n))))
        fd_max = max(fd_max, err)
        fd_report.append((name, err))
print(f"=== derivative FD check ===  max_rel_err={fd_max:.3e} "
      f"({'OK' if fd_max < 1e-5 else 'BAD -- FORMULATION_ERROR'})")

# Also sanity-check the published x* against the published f*.
print(f"f(published x*) = {f(X_STAR):.10f}   published f* = {KNOWN_OPTIMAL}")
print(f"g(published x*) = {[round(gg(X_STAR), 6) for gg in (g1, g2, g3, g4)]}")

# ---------------------------------------------------------------------------
# (A) pounce.minimize with analytic jac
# ---------------------------------------------------------------------------
import pounce  # noqa: E402

constraints = [
    {"type": "ineq", "fun": g1, "jac": g1jac},
    {"type": "ineq", "fun": g2, "jac": g2jac},
    {"type": "ineq", "fun": g3, "jac": g3jac},
    {"type": "ineq", "fun": g4, "jac": g4jac},
]

t0 = time.perf_counter()
res = pounce.minimize(f, X0.copy(), jac=fjac, constraints=constraints,
                      options={"solver_selection": "nlp"})
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(res.x, dtype=float)
obj_pounce = float(res.fun)
status = res.status

# ---------------------------------------------------------------------------
# (B) ONE Pyomo model -> Ipopt oracle + SolverFactory('pounce')
# ---------------------------------------------------------------------------
import pyomo.environ as pyo  # noqa: E402


def build_model():
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 6)
    m.x = pyo.Var(m.I, initialize={i: float(X0[i]) for i in range(7)})
    x = m.x
    m.obj = pyo.Objective(
        expr=(x[0] - 10.0) ** 2 + 5.0 * (x[1] - 12.0) ** 2 + x[2] ** 4
        + 3.0 * (x[3] - 11.0) ** 2 + 10.0 * x[4] ** 6 + 7.0 * x[5] ** 2
        + x[6] ** 4 - 4.0 * x[5] * x[6] - 10.0 * x[5] - 8.0 * x[6])
    m.g1 = pyo.Constraint(expr=127.0 - 2.0 * x[0] ** 2 - 3.0 * x[1] ** 4
                          - x[2] - 4.0 * x[3] ** 2 - 5.0 * x[4] >= 0.0)
    m.g2 = pyo.Constraint(expr=282.0 - 7.0 * x[0] - 3.0 * x[1]
                          - 10.0 * x[2] ** 2 - x[3] + x[4] >= 0.0)
    m.g3 = pyo.Constraint(expr=196.0 - 23.0 * x[0] - x[1] ** 2
                          - 6.0 * x[5] ** 2 + 8.0 * x[6] >= 0.0)
    m.g4 = pyo.Constraint(expr=-4.0 * x[0] ** 2 - x[1] ** 2
                          + 3.0 * x[0] * x[1] - 2.0 * x[2] ** 2
                          - 5.0 * x[5] + 11.0 * x[6] >= 0.0)
    return m


m_or = build_model()
t0 = time.perf_counter()
pyo.SolverFactory("ipopt").solve(m_or)
t_oracle = time.perf_counter() - t0
x_oracle = np.array([pyo.value(m_or.x[i]) for i in range(7)])
obj_oracle = pyo.value(m_or.obj)

pounce_pyomo_obj = None
pounce_pyomo_t = None
pounce_pyomo_x = None
try:
    m_pp = build_model()
    t0 = time.perf_counter()
    pyo.SolverFactory("pounce").solve(m_pp)
    pounce_pyomo_t = time.perf_counter() - t0
    pounce_pyomo_obj = pyo.value(m_pp.obj)
    pounce_pyomo_x = np.array([pyo.value(m_pp.x[i]) for i in range(7)])
except Exception as e:  # noqa: BLE001
    pounce_pyomo_obj = f"unavailable: {type(e).__name__}: {e}"

# ---------------------------------------------------------------------------
# (C) `pounce verify` on the emitted .nl
# ---------------------------------------------------------------------------
verify_rc = None
verify_out = ""
cli_out = ""
try:
    tmpdir = tempfile.mkdtemp(prefix="hs100_")
    nlfile = os.path.join(tmpdir, "hs100.nl")
    build_model().write(nlfile, io_options={"symbolic_solver_labels": True})
    sol_path = os.path.join(tmpdir, "hs100.sol")
    cli = subprocess.run([POUNCE_CLI, nlfile, sol_path],
                         capture_output=True, text=True, timeout=60)
    cli_out = cli.stdout + cli.stderr
    if os.path.exists(sol_path):
        ver = subprocess.run([POUNCE_CLI, "verify", nlfile, sol_path],
                             capture_output=True, text=True, timeout=60)
        verify_rc = ver.returncode
        verify_out = ver.stdout + ver.stderr
    else:
        verify_out = "no .sol produced\n" + cli_out
except Exception as e:  # noqa: BLE001
    verify_out = f"verify step failed: {type(e).__name__}: {e}"

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_oracle = rel(obj_pounce, obj_oracle)
oracle_err_known = rel(obj_oracle, KNOWN_OPTIMAL)
x_err_known = float(np.linalg.norm(x_pounce - X_STAR, np.inf))
x_err_oracle = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

print("=== pounce (pounce.minimize, analytic jac) ===")
print(f"status={status} success={res.success} obj={obj_pounce:.12e}")
print(f"x={x_pounce}")
print(f"t={t_pounce:.4f}s nit={getattr(res, 'nit', '?')}")
try:
    print(f"info={dict(res.info)}")
except Exception:  # noqa: BLE001
    pass
print("=== oracle (Ipopt via Pyomo) ===")
print(f"obj={obj_oracle:.12e} t={t_oracle:.4f}s")
print(f"x={x_oracle}")
print("=== pounce via Pyomo SolverFactory('pounce') ===")
print(f"obj={pounce_pyomo_obj} t={pounce_pyomo_t}")
print(f"x={pounce_pyomo_x}")
print("=== pounce verify (.nl / .sol) ===")
print(f"rc={verify_rc}")
print(verify_out.strip()[:3000])
print("=== reference ===")
print(f"known_optimal={KNOWN_OPTIMAL:.10f}")
print(f"oracle_rel_err_vs_known  = {oracle_err_known:.2e}")
print(f"pounce_rel_err_vs_known  = {obj_err_known:.2e}")
print(f"pounce_rel_err_vs_oracle = {obj_err_oracle:.2e}")
print(f"pounce_x_inf_err_vs_known  = {x_err_known:.2e}")
print(f"pounce_x_inf_err_vs_oracle = {x_err_oracle:.2e}")

answer_ok = (obj_err_known < 1e-4 and obj_err_oracle < 1e-4
             and oracle_err_known < 1e-4)
if fd_max >= 1e-5:
    print("VERDICT: FORMULATION_ERROR (analytic derivatives fail FD check)")
elif answer_ok and res.success:
    print("VERDICT: PASS")
elif answer_ok:
    print(f"VERDICT: SOLVER_LIMITATION (correct optimum but success=False, "
          f"status={status})")
else:
    print(f"VERDICT: FAIL (rel_err_vs_known={obj_err_known:.2e}, "
          f"vs_oracle={obj_err_oracle:.2e}, status={status})")
