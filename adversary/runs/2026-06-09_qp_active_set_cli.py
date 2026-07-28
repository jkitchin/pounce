"""Adversary cross-check: qp-active-set CLI routing on a convex QP
Family: qp-active-set   Class: small convex QP (active-set slot)
Source: Nocedal & Wright, Numerical Optimization (2nd ed.), Example 16.4.
  min q(x) = (x0-1)^2 + (x1-2.5)^2
  s.t.  x0 - 2 x1 + 2 >= 0
       -x0 - 2 x1 + 6 >= 0
       -x0 + 2 x1 + 2 >= 0
        x0, x1 >= 0
  Known optimum x* = (1.4, 1.7), q* = 0.8.

This family is CLI-only: the Python `minimize` facade does not expose
`solver_selection="qp-active-set"`. The CLI accepts `solver_selection=qp-active-set`,
validates it is a convex QP, then (per the documented behavior) falls through to
the NLP filter-IPM — the dedicated active-set algorithm is reachable only from
Rust unit tests, not the public surface. So this script checks two contracts:

  (1) CLI `solver_selection=qp-active-set` on the convex QP returns the correct
      optimum (cross-checked vs cvxpy and the known optimum);
  (2) forcing `solver_selection=qp-active-set` on a NONLINEAR .nl produces a
      clean class-mismatch error (non-zero exit, no panic / backtrace).

Known optimal: 0.8
"""
import json
import subprocess
import time

import numpy as np

CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
KNOWN_OPTIMAL = 0.8
X_STAR = np.array([1.4, 1.7])


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# ----------------------------------------------------------------------------
# (1) Build the convex QP as an .nl via pyomo and solve it through the CLI with
#     solver_selection=qp-active-set.
# ----------------------------------------------------------------------------
import pyomo.environ as pyo

m = pyo.ConcreteModel()
m.x0 = pyo.Var(domain=pyo.NonNegativeReals, initialize=2.0)
m.x1 = pyo.Var(domain=pyo.NonNegativeReals, initialize=0.0)
m.obj = pyo.Objective(expr=(m.x0 - 1.0) ** 2 + (m.x1 - 2.5) ** 2)
m.c1 = pyo.Constraint(expr=m.x0 - 2 * m.x1 + 2.0 >= 0)
m.c2 = pyo.Constraint(expr=-m.x0 - 2 * m.x1 + 6.0 >= 0)
m.c3 = pyo.Constraint(expr=-m.x0 + 2 * m.x1 + 2.0 >= 0)

nl_path = "/tmp/adv_qpas_nw1604.nl"
m.write(nl_path, format="nl")
json_path = "/tmp/adv_qpas_nw1604.json"
sol_path = "/tmp/adv_qpas_nw1604.sol"

t0 = time.perf_counter()
proc = subprocess.run(
    [CLI, nl_path, sol_path, "solver_selection=qp-active-set",
     "--json-output", json_path],
    capture_output=True, text=True, timeout=30,
)
t_cli = time.perf_counter() - t0

d = json.load(open(json_path))
obj_cli = d["solution"]["objective"]
x_cli = np.asarray(d["solution"]["x"])
status_cli = d["solution"]["status"]

# ----------------------------------------------------------------------------
# oracle: cvxpy
# ----------------------------------------------------------------------------
import cvxpy as cp

x = cp.Variable(2)
prob = cp.Problem(
    cp.Minimize((x[0] - 1.0) ** 2 + (x[1] - 2.5) ** 2),
    [x[0] - 2 * x[1] + 2.0 >= 0,
     -x[0] - 2 * x[1] + 6.0 >= 0,
     -x[0] + 2 * x[1] + 2.0 >= 0,
     x >= 0],
)
prob.solve(solver=cp.CLARABEL)
obj_oracle = prob.value

obj_err = rel(obj_cli, obj_oracle)
x_err = float(np.linalg.norm(x_cli - X_STAR, np.inf))

print("=== pounce CLI (solver_selection=qp-active-set) ===")
print(f"exit={proc.returncode} status={status_cli} obj={obj_cli:.10e} x={x_cli} t={t_cli:.4f}s")
print("=== oracle (cvxpy/CLARABEL) ===")
print(f"obj={obj_oracle:.10e}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_cli, KNOWN_OPTIMAL):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err_vs_known={x_err:.2e}")

part1_ok = (proc.returncode == 0
            and status_cli == "SolveSucceeded"
            and rel(obj_cli, KNOWN_OPTIMAL) < 1e-4
            and obj_err < 1e-4
            and x_err < 1e-4)

# ----------------------------------------------------------------------------
# (2) Mismatch case: force qp-active-set on a genuinely NONLINEAR program.
#     Expect a clean non-zero-exit error, NOT a panic / rust backtrace.
# ----------------------------------------------------------------------------
mn = pyo.ConcreteModel()
mn.x = pyo.Var(initialize=1.5)
mn.y = pyo.Var(initialize=1.5)
# Rosenbrock — nonconvex, non-QP; valid NLP but not a convex QP.
mn.obj = pyo.Objective(expr=100 * (mn.y - mn.x ** 2) ** 2 + (1 - mn.x) ** 2)
nl_nonlin = "/tmp/adv_qpas_rosen.nl"
mn.write(nl_nonlin, format="nl")
sol_nonlin = "/tmp/adv_qpas_rosen.sol"

proc2 = subprocess.run(
    [CLI, nl_nonlin, sol_nonlin, "solver_selection=qp-active-set"],
    capture_output=True, text=True, timeout=30,
)
combined = (proc2.stdout + proc2.stderr).lower()
panicked = ("panic" in combined or "backtrace" in combined
            or "RUST_BACKTRACE" in (proc2.stdout + proc2.stderr))
# A clean rejection: non-zero exit and a legible mismatch message, no panic.
clean_reject = (proc2.returncode != 0) and not panicked

print("=== mismatch: qp-active-set on a nonlinear (Rosenbrock) .nl ===")
print(f"exit={proc2.returncode} panicked={panicked}")
print("stderr/stdout tail:")
tail = (proc2.stdout + proc2.stderr).strip().splitlines()[-6:]
for line in tail:
    print("  " + line)

part2_ok = clean_reject

ok = part1_ok and part2_ok
print(f"part1(convex QP correct)={part1_ok}  part2(clean mismatch)={part2_ok}")
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (part1={part1_ok} part2={part2_ok})")
