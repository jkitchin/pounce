"""Adversary cross-check: qp-active-set on an equality+inequality convex QP.

Family: qp-active-set    Class: small convex QP (active-set slot)

Problem (one equality + one binding inequality at the optimum):
    min  q(x) = 1/2 (x0^2 + x1^2)
    s.t. x0 + x1 = 1            (equality, always active)
         x0 - x1 >= 0.5         (inequality, active at optimum)

  P = I (SPD), q = 0. On the equality line x0 + x1 = 1, the unconstrained
  minimizer of 1/2||x||^2 is x0 = x1 = 0.5, which gives x0 - x1 = 0 < 0.5 and
  therefore VIOLATES the inequality. Hence the inequality is binding at the
  optimum: solve x0 + x1 = 1 and x0 - x1 = 0.5 simultaneously ->

      x* = (0.75, 0.25),  active set = { eq, x0 - x1 = 0.5 },
      q* = 1/2 (0.75^2 + 0.25^2) = 1/2 (0.5625 + 0.0625) = 0.3125.

  KKT check: grad q = (x0, x1) = (0.75, 0.25). Active constraint gradients:
  eq a_eq = (1, 1); ineq (as g>=0) a_ineq = (1, -1). Need
  (0.75, 0.25) = mu*(1,1) + lam*(1,-1) with lam >= 0.
  => mu + lam = 0.75, mu - lam = 0.25 => mu = 0.5, lam = 0.25 >= 0. KKT holds,
  so x* = (0.75, 0.25) is the unique (strictly convex) global minimizer.

SOURCE: constructed; optimum derived + verified by KKT, cross-checked vs cvxpy
  CLARABEL and the convex IPM.
KNOWN_OPTIMAL: 0.3125   X_STAR = (0.75, 0.25)

HOW the active-set path is invoked: identical to Problem A -- the CLI only.
We emit the QP as a pyomo .nl and run

    pounce <model>.nl <out>.sol solver_selection=qp-active-set --json-output <json>

which dispatch.rs maps to SolverChoice::QpActiveSet and main.rs realizes by
setting `algorithm active-set-sqp` (active-set SQP engine, step QPs via
pounce-qp). IPM cross-check: solver_selection=qp-ipm. Oracle 2: cvxpy/CLARABEL.
"""
import json
import subprocess
import time

import numpy as np
import cvxpy as cp
import pyomo.environ as pyo

CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
KNOWN_OPTIMAL = 0.3125
X_STAR = np.array([0.75, 0.25])


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def build_nl(path):
    m = pyo.ConcreteModel()
    m.x0 = pyo.Var(initialize=0.0)
    m.x1 = pyo.Var(initialize=0.0)
    m.obj = pyo.Objective(expr=0.5 * (m.x0 ** 2 + m.x1 ** 2))
    m.eq = pyo.Constraint(expr=m.x0 + m.x1 == 1.0)
    m.ineq = pyo.Constraint(expr=m.x0 - m.x1 >= 0.5)
    m.write(path, format="nl")


def run_cli(selection, tag):
    nl = f"/tmp/adv_qpas_eqineq_{tag}.nl"
    sol = f"/tmp/adv_qpas_eqineq_{tag}.sol"
    js = f"/tmp/adv_qpas_eqineq_{tag}.json"
    build_nl(nl)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [CLI, nl, sol, f"solver_selection={selection}", "--json-output", js],
        capture_output=True, text=True, timeout=60,
    )
    dt = time.perf_counter() - t0
    d = json.load(open(js))
    s = d["solution"]
    return {
        "exit": proc.returncode,
        "status": s["status"],
        "obj": float(s["objective"]),
        "x": np.asarray(s["x"], dtype=float),
        "t": dt,
        "stderr": proc.stderr,
    }


# --- active-set path (the system under test) ---
AS = run_cli("qp-active-set", "as")
# --- convex IPM cross-check (oracle 1, internal) ---
IPM = run_cli("qp-ipm", "ipm")

# --- cvxpy / CLARABEL cross-check (oracle 2, external) ---
x = cp.Variable(2)
P = np.eye(2)
prob = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(x, P)),
    [x[0] + x[1] == 1.0, x[0] - x[1] >= 0.5],
)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_cvx = time.perf_counter() - t0
obj_cvx = float(prob.value)
x_cvx = np.asarray(x.value, dtype=float)

# --- errors ---
as_vs_known = rel(AS["obj"], KNOWN_OPTIMAL)
as_vs_ipm = rel(AS["obj"], IPM["obj"])
as_vs_cvx = rel(AS["obj"], obj_cvx)
x_inf_err = float(np.linalg.norm(AS["x"] - X_STAR, np.inf))

print("=== Problem B: equality + active-inequality convex QP ===")
print(f"KNOWN_OPTIMAL = {KNOWN_OPTIMAL}   X_STAR = {X_STAR}")
print("-- pounce active-set (solver_selection=qp-active-set) --")
print(f"   exit={AS['exit']} status={AS['status']} obj={AS['obj']:.12e}")
print(f"   x={AS['x']}  t={AS['t']:.4f}s")
print("-- pounce convex IPM (solver_selection=qp-ipm) --")
print(f"   exit={IPM['exit']} status={IPM['status']} obj={IPM['obj']:.12e}")
print(f"   x={IPM['x']}  t={IPM['t']:.4f}s")
print("-- cvxpy / CLARABEL --")
print(f"   obj={obj_cvx:.12e}  x={x_cvx}  t={t_cvx:.4f}s")
print("-- relative errors --")
print(f"   active-set vs known : {as_vs_known:.3e}")
print(f"   active-set vs IPM   : {as_vs_ipm:.3e}")
print(f"   active-set vs cvxpy : {as_vs_cvx:.3e}")
print(f"   x_inf_err vs X_STAR : {x_inf_err:.3e}")

TOL = 1e-6
ok = (
    AS["exit"] == 0
    and AS["status"] == "SolveSucceeded"
    and as_vs_known < TOL
    and as_vs_ipm < TOL
    and as_vs_cvx < TOL
    and x_inf_err < 1e-5
)
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
