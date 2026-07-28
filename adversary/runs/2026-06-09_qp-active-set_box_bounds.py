"""Adversary cross-check: qp-active-set on a purely bound-constrained convex QP.

Family: qp-active-set    Class: small convex QP (active-set slot)

Problem (box / bound-constrained QP, active set = one upper bound):
    min  q(x) = 1/2 (x0^2 + x1^2) - x0 - 4 x1
    s.t. 0 <= x0 <= 2
         0 <= x1 <= 2

  P = I (identity, SPD), q = (-1, -4).  The UNCONSTRAINED minimizer is
  x = -P^{-1} q = (1, 4). Clipped to the box, the bound x1 <= 2 is the only
  binding constraint, so the KKT-optimal point is

      x* = (1, 2),  active set = { x1 = 2 (upper) },
      q* = 1/2 (1 + 4) - 1 - 8 = 2.5 - 9 = -6.5.

  This is a separable strictly-convex QP whose optimum is verifiable by hand;
  the active-set solver must identify the single binding upper bound on x1.

SOURCE: constructed; optimum verified analytically by clip-of-unconstrained-min
  + KKT (separable, P=I), cross-checked vs cvxpy CLARABEL and the convex IPM.
KNOWN_OPTIMAL: -6.5   X_STAR = (1, 2)

HOW the active-set path is invoked: the Python `minimize` facade does NOT expose
solver_selection="qp-active-set". It is reachable only via the CLI. We emit the
QP as an AMPL .nl (via pyomo) and run:

    pounce <model>.nl <out>.sol solver_selection=qp-active-set --json-output <json>

dispatch.rs maps "qp-active-set" -> SolverChoice::QpActiveSet, and main.rs then
sets `algorithm active-set-sqp`, routing the convex QP through the active-set SQP
engine (step QPs solved by pounce-qp) -- a DISTINCT code path from the IPM.
The IPM cross-check uses solver_selection=qp-ipm; cvxpy/CLARABEL is oracle 2.
"""
import json
import subprocess
import time

import numpy as np
import cvxpy as cp
import pyomo.environ as pyo

CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
KNOWN_OPTIMAL = -6.5
X_STAR = np.array([1.0, 2.0])


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def build_nl(path):
    m = pyo.ConcreteModel()
    m.x0 = pyo.Var(bounds=(0.0, 2.0), initialize=0.0)
    m.x1 = pyo.Var(bounds=(0.0, 2.0), initialize=0.0)
    m.obj = pyo.Objective(
        expr=0.5 * (m.x0 ** 2 + m.x1 ** 2) - m.x0 - 4.0 * m.x1
    )
    m.write(path, format="nl")


def run_cli(selection, tag):
    nl = f"/tmp/adv_qpas_box_{tag}.nl"
    sol = f"/tmp/adv_qpas_box_{tag}.sol"
    js = f"/tmp/adv_qpas_box_{tag}.json"
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
q = np.array([-1.0, -4.0])
prob = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x),
    [x >= 0, x <= 2],
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

print("=== Problem A: box/bound-constrained convex QP ===")
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
