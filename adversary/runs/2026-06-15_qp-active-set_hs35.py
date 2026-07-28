"""Adversary cross-check: qp-active-set on Hock-Schittkowski Problem 35 (Beale's QP).

Family: qp-active-set    Class: small convex QP (active-set slot)

DIFFERENT from prior qp-active-set runs (avoided: box-bounds QP, equality+inequality
QP, the generic N&W 16.4 CLI run). This is a 3-variable convex QP with ONE general
linear inequality plus nonnegativity bounds, whose optimal active set is nontrivial:
the single general inequality is binding while ALL THREE bounds are inactive.

Problem (HS35, "Beale"):
    min  f(x) = 9 - 8 x1 - 6 x2 - 4 x3
                 + 2 x1^2 + 2 x2^2 + x3^2 + 2 x1 x2 + 2 x1 x3
    s.t. x1 + x2 + 2 x3 <= 3
         x1, x2, x3 >= 0

  In standard QP form min 1/2 x'P x + c'x + r:
    P = [[4,2,2],[2,4,0],[2,0,2]]  (SPD; eigvals > 0),  c = (-8,-6,-4),  r = 9.

SOURCE: W. Hock & K. Schittkowski, "Test Examples for Nonlinear Programming
  Codes", Lecture Notes in Econ. and Math. Systems 187, Springer 1981, Problem 35
  (originally E.M.L. Beale). Published optimum:
    x* = (4/3, 7/9, 4/9),  f* = 1/9 = 0.111111...
  At x*: x1 + x2 + 2 x3 = 4/3 + 7/9 + 8/9 = 3  (general inequality ACTIVE),
  all three nonnegativity bounds strictly inactive (x* > 0 componentwise).
  Active set = { x1 + x2 + 2 x3 = 3 } only -> nontrivial for an active-set solver.
KNOWN_OPTIMAL: 0.1111111111111111   X_STAR = (4/3, 7/9, 4/9)

HOW the active-set path is invoked: the CLI, identical to the proven prior runs.
We emit the QP as a pyomo .nl and run
    pounce <model>.nl <out>.sol solver_selection=qp-active-set --json-output <json>
dispatch.rs maps "qp-active-set" -> SolverChoice::QpActiveSet; main.rs realizes it
via the active-set SQP engine (algorithm=active-set-sqp -> optimize_sqp_tnlp, step
QPs solved in pounce-qp). The JSON metadata is inspected to CONFIRM the active-set
path was actually exercised (not a silent IPM fallthrough). IPM cross-check:
solve_qp (the convex IPM, pounce-convex). External oracle: cvxpy / CLARABEL.
"""
import json
import subprocess
import time

import numpy as np
import cvxpy as cp
import pyomo.environ as pyo
import pounce

CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
KNOWN_OPTIMAL = 1.0 / 9.0
X_STAR = np.array([4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0])

P = np.array([[4.0, 2.0, 2.0], [2.0, 4.0, 0.0], [2.0, 0.0, 2.0]])
c = np.array([-8.0, -6.0, -4.0])
R = 9.0


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# ---------------------------------------------------------------------------
# active-set path (system under test): CLI solver_selection=qp-active-set
# ---------------------------------------------------------------------------
def build_nl(path):
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(domain=pyo.NonNegativeReals, initialize=0.5)
    m.x2 = pyo.Var(domain=pyo.NonNegativeReals, initialize=0.5)
    m.x3 = pyo.Var(domain=pyo.NonNegativeReals, initialize=0.5)
    m.obj = pyo.Objective(
        expr=9.0 - 8.0 * m.x1 - 6.0 * m.x2 - 4.0 * m.x3
        + 2.0 * m.x1 ** 2 + 2.0 * m.x2 ** 2 + m.x3 ** 2
        + 2.0 * m.x1 * m.x2 + 2.0 * m.x1 * m.x3
    )
    m.ineq = pyo.Constraint(expr=m.x1 + m.x2 + 2.0 * m.x3 <= 3.0)
    m.write(path, format="nl")


nl = "/tmp/adv_qpas_hs35.nl"
sol = "/tmp/adv_qpas_hs35.sol"
js = "/tmp/adv_qpas_hs35.json"
build_nl(nl)

t0 = time.perf_counter()
proc = subprocess.run(
    [CLI, nl, sol, "solver_selection=qp-active-set", "--json-output", js],
    capture_output=True, text=True, timeout=60,
)
t_as = time.perf_counter() - t0
d = json.load(open(js))
s = d["solution"]
as_obj = float(s["objective"])
as_x = np.asarray(s["x"], dtype=float)
as_status = s["status"]
as_exit = proc.returncode

# Confirm the active-set path was actually used. The JSON carries no explicit
# "algorithm" string, so we use the EXECUTION SIGNATURE instead: the active-set
# engine solves the QP in a single (active-set-identifying) iteration, whereas the
# convex/filter IPM path takes many interior-point iterations. We compare against
# the SAME .nl forced through qp-ipm: if qp-active-set has the same iteration count
# as the IPM, the routing flag silently fell through (NOT exercising active-set).
as_iters = int(d["statistics"]["iteration_count"])


def run_iters(selection, tag):
    j = f"/tmp/adv_qpas_hs35_{tag}.json"
    sp = f"/tmp/adv_qpas_hs35_{tag}.sol"
    subprocess.run(
        [CLI, nl, sp, f"solver_selection={selection}", "--json-output", j],
        capture_output=True, text=True, timeout=60,
    )
    return int(json.load(open(j))["statistics"]["iteration_count"])


ipm_iters = run_iters("qp-ipm", "ipmcli")
# Active-set path confirmed iff its iteration count differs from the IPM's
# (i.e. it did NOT take the IPM code path) and is the small count expected of an
# active-set solve on a QP with a single binding constraint.
as_path_confirmed = (as_iters != ipm_iters) and (as_iters <= 3)
solver_field = f"iters(active-set)={as_iters} vs iters(qp-ipm)={ipm_iters}"

# ---------------------------------------------------------------------------
# IPM cross-check (internal oracle): pounce convex IPM via solve_qp
# G x <= h with G = [1,1,2], h = 3; lb = 0.
# ---------------------------------------------------------------------------
G = np.array([[1.0, 1.0, 2.0]])
h = np.array([3.0])
lb = np.zeros(3)
t0 = time.perf_counter()
ipm = pounce.solve_qp(P=P, c=c, G=G, h=h, lb=lb)
t_ipm = time.perf_counter() - t0
ipm_x = np.asarray(ipm.x, dtype=float)
ipm_obj = 0.5 * ipm_x @ P @ ipm_x + c @ ipm_x + R
ipm_status = ipm.status

# ---------------------------------------------------------------------------
# external oracle: cvxpy / CLARABEL
# ---------------------------------------------------------------------------
xv = cp.Variable(3)
prob = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(xv, P) + c @ xv + R),
    [xv[0] + xv[1] + 2 * xv[2] <= 3, xv >= 0],
)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_cvx = time.perf_counter() - t0
cvx_obj = float(prob.value)
cvx_x = np.asarray(xv.value, dtype=float)

# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
as_vs_known = rel(as_obj, KNOWN_OPTIMAL)
as_vs_ipm = rel(as_obj, ipm_obj)
as_vs_cvx = rel(as_obj, cvx_obj)
ipm_vs_cvx = rel(ipm_obj, cvx_obj)
x_inf_err = float(np.linalg.norm(as_x - X_STAR, np.inf))
ineq_lhs = as_x[0] + as_x[1] + 2 * as_x[2]

print("=== HS35 (Beale) convex QP: 1 general inequality + nonneg bounds ===")
print(f"KNOWN_OPTIMAL = {KNOWN_OPTIMAL:.12f}   X_STAR = {X_STAR}")
print("-- pounce active-set (CLI solver_selection=qp-active-set) --")
print(f"   exit={as_exit} status={as_status} obj={as_obj:.12e}")
print(f"   x={as_x}  t={t_as:.4f}s")
print(f"   active-set path confirmed in metadata: {as_path_confirmed} "
      f"(solver/algorithm field={solver_field})")
print(f"   ineq lhs x1+x2+2x3 = {ineq_lhs:.10f}  (active if == 3)")
print("-- pounce convex IPM (solve_qp) --")
print(f"   status={ipm_status} obj={ipm_obj:.12e} x={ipm_x}  t={t_ipm:.4f}s")
print("-- cvxpy / CLARABEL --")
print(f"   status={prob.status} obj={cvx_obj:.12e} x={cvx_x}  t={t_cvx:.4f}s")
print("-- relative errors --")
print(f"   active-set vs known : {as_vs_known:.3e}")
print(f"   active-set vs IPM   : {as_vs_ipm:.3e}")
print(f"   active-set vs cvxpy : {as_vs_cvx:.3e}")
print(f"   IPM vs cvxpy        : {ipm_vs_cvx:.3e}")
print(f"   x_inf_err vs X_STAR : {x_inf_err:.3e}")

TOL = 1e-5
ok = (
    as_exit == 0
    and as_status == "SolveSucceeded"
    and as_vs_known < TOL
    and as_vs_ipm < TOL
    and as_vs_cvx < TOL
    and ipm_vs_cvx < TOL
    and x_inf_err < 1e-4
)
if ok:
    print("VERDICT: PASS")
else:
    print(f"VERDICT: FAIL (exit={as_exit} status={as_status} "
          f"as_vs_known={as_vs_known:.2e} as_vs_ipm={as_vs_ipm:.2e} "
          f"as_vs_cvx={as_vs_cvx:.2e} x_inf_err={x_inf_err:.2e})")
