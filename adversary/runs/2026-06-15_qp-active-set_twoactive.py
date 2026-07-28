"""Adversary cross-check: qp-active-set on a 2-var convex QP with TWO active
inequalities at the optimum (a constraint-defined vertex).

Family: qp-active-set    Class: small convex QP, inequality-only, vertex optimum

DIFFERENT from prior qp-active-set runs (avoided: box-bounds QP, equality+single-
active-inequality QP, the generic N&W 16.4 CLI run, HS35/Beale which has only ONE
general inequality binding). This problem's optimum is a VERTEX where TWO general
linear inequalities are simultaneously active, so the optimal active set has size 2
and the active-set solver must identify the correct pair of binding rows.

Problem:
    min  f(x) = 1/2[(x1 - 2)^2 + (x2 - 2)^2]
    s.t. g1:  x1 + x2 <= 1
         g2:  x1      <= 0.3

  In standard QP form min 1/2 x'P x + c'x + r:
    P = [[1,0],[0,1]] (SPD),  c = (-2,-2),  r = (1/2)(4+4) = 4.

KNOWN_OPTIMAL (hand-derived, KKT-verified vertex):
  The unconstrained minimizer is (2,2), which violates both constraints, so both
  must be active. Their intersection: x1 = 0.3 (from g2 tight) and x1 + x2 = 1
  (g1 tight) => x2 = 0.7.  X_STAR = (0.3, 0.7).
  KKT stationarity  -grad f(x*) = lam1 * (1,1) + lam2 * (1,0), lam >= 0:
     grad f(x*) = (0.3-2, 0.7-2) = (-1.7, -1.3)
     (1.7, 1.3) = lam1 (1,1) + lam2 (1,0)  =>  lam1 = 1.3, lam2 = 0.4  (both >= 0)
  => x* is the constrained optimum; both inequalities active; duals strictly > 0.
  f* = 1/2[(0.3-2)^2 + (0.7-2)^2] = 1/2[2.89 + 1.69] = 2.29.
KNOWN_OPTIMAL: 2.29   X_STAR = (0.3, 0.7)   LAM* = (1.3, 0.4)

HOW the active-set path is invoked: the CLI, identical to the proven prior runs.
We emit the QP as a pyomo .nl and run
    pounce <model>.nl <out>.sol solver_selection=qp-active-set --json-output <json>
dispatch.rs maps "qp-active-set" -> SolverChoice::QpActiveSet; main.rs realizes it
via the active-set SQP engine. The JSON metadata is inspected (iteration-count
signature vs the SAME .nl forced through qp-ipm) to CONFIRM the active-set path was
actually exercised (not a silent IPM fallthrough). IPM cross-check: solve_qp (the
convex IPM, pounce-convex). External oracle: cvxpy / CLARABEL.
"""
import json
import subprocess
import time

import numpy as np
import cvxpy as cp
import pyomo.environ as pyo
import pounce

CLI = "/Users/jkitchin/projects/pounce/target/release/pounce"
KNOWN_OPTIMAL = 2.29
X_STAR = np.array([0.3, 0.7])
LAM_STAR = np.array([1.3, 0.4])

P = np.array([[1.0, 0.0], [0.0, 1.0]])
c = np.array([-2.0, -2.0])
R = 4.0


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# ---------------------------------------------------------------------------
# active-set path (system under test): CLI solver_selection=qp-active-set
# ---------------------------------------------------------------------------
def build_nl(path):
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var(initialize=0.0)
    m.x2 = pyo.Var(initialize=0.0)
    m.obj = pyo.Objective(
        expr=0.5 * ((m.x1 - 2.0) ** 2 + (m.x2 - 2.0) ** 2)
    )
    m.g1 = pyo.Constraint(expr=m.x1 + m.x2 <= 1.0)
    m.g2 = pyo.Constraint(expr=m.x1 <= 0.3)
    m.write(path, format="nl")


nl = "/tmp/adv_qpas_twoactive.nl"
sol = "/tmp/adv_qpas_twoactive.sol"
js = "/tmp/adv_qpas_twoactive.json"
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
as_iters = int(d["statistics"]["iteration_count"])


def run_iters(selection, tag):
    j = f"/tmp/adv_qpas_twoactive_{tag}.json"
    sp = f"/tmp/adv_qpas_twoactive_{tag}.sol"
    subprocess.run(
        [CLI, nl, sp, f"solver_selection={selection}", "--json-output", j],
        capture_output=True, text=True, timeout=60,
    )
    return int(json.load(open(j))["statistics"]["iteration_count"])


ipm_iters = run_iters("qp-ipm", "ipmcli")
# Active-set path confirmed iff its iteration count differs from the IPM's
# (i.e. it did NOT take the IPM code path).
as_path_confirmed = (as_iters != ipm_iters)
solver_field = f"iters(active-set)={as_iters} vs iters(qp-ipm)={ipm_iters}"

# ---------------------------------------------------------------------------
# IPM cross-check (internal oracle): pounce convex IPM via solve_qp
# G x <= h with G = [[1,1],[1,0]], h = [1, 0.3].
# ---------------------------------------------------------------------------
G = np.array([[1.0, 1.0], [1.0, 0.0]])
h = np.array([1.0, 0.3])
t0 = time.perf_counter()
ipm = pounce.solve_qp(P=P, c=c, G=G, h=h)
t_ipm = time.perf_counter() - t0
ipm_x = np.asarray(ipm.x, dtype=float)
ipm_obj = 0.5 * ipm_x @ P @ ipm_x + c @ ipm_x + R
ipm_status = ipm.status

# ---------------------------------------------------------------------------
# external oracle: cvxpy / CLARABEL
# ---------------------------------------------------------------------------
xv = cp.Variable(2)
prob = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(xv, P) + c @ xv + R),
    [xv[0] + xv[1] <= 1, xv[0] <= 0.3],
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
g1_lhs = as_x[0] + as_x[1]
g2_lhs = as_x[0]

print("=== 2-var convex QP, TWO active inequalities (vertex optimum) ===")
print(f"KNOWN_OPTIMAL = {KNOWN_OPTIMAL:.12f}   X_STAR = {X_STAR}   LAM* = {LAM_STAR}")
print("-- pounce active-set (CLI solver_selection=qp-active-set) --")
print(f"   exit={as_exit} status={as_status} obj={as_obj:.12e}")
print(f"   x={as_x}  t={t_as:.4f}s")
print(f"   active-set path confirmed: {as_path_confirmed} ({solver_field})")
print(f"   g1 x1+x2 = {g1_lhs:.10f} (active if ==1)   g2 x1 = {g2_lhs:.10f} (active if ==0.3)")
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
