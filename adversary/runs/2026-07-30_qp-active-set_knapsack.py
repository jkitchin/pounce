"""Adversary cross-check: qp-active-set on a continuous quadratic-knapsack QP
Family: qp-active-set   Class: separable QP, ONE equality (budget) + box bounds,
SPARSE active set (only a handful of the 10 bounds active at the optimum).
DIFFERENT from prior qp-active-set runs (box-only 8-var with several active
bounds and no equality; HS35's single general inequality with bounds all
inactive; two-active-inequality vertex; rank-deficient equality battery;
LICQ-degenerate vertex; feasible high-m/n battery; cycling battery) -- this is
a continuous quadratic knapsack / resource-allocation problem (separable
Hessian + ONE coupling equality), whose KKT structure is a classic 1-D
"water-filling" root-find, giving an independent closed-form oracle that is
NOT cvxpy and NOT pounce's own IPM.

Problem:
    minimize   sum_i 0.5 q_i x_i^2 - c_i x_i
    subject to sum_i x_i = s                  (single coupling equality)
               l_i <= x_i <= u_i               (box bounds, i=1..10)

KKT: stationarity gives x_i(lambda) = clip((c_i + lambda) / q_i, l_i, u_i) for
the equality multiplier lambda; g(lambda) = sum_i x_i(lambda) - s is
continuous and monotonically non-decreasing in lambda (each term is a clipped
non-decreasing function of lambda), so its root is found by a 1-D bisection
(scipy.optimize.brentq) -- an approach with no linear algebra and no relation
to pounce's own solver internals (classic water-filling / continuous
quadratic-knapsack algorithm, e.g. P. Brucker 1984, "An O(n log n) algorithm
for quadratic knapsack problems", Operations Research Letters 3(3):163-166).

SOURCE: Brucker (1984) knapsack-QP structure; Boyd & Vandenberghe Ch.5 water-
filling discussion for the KKT form. Data below is randomly generated with a
fixed seed and checked (in the script) to bracket a feasible sum.

KNOWN_OPTIMAL: computed at runtime via the 1-D root-find (see brentq_root_obj
below) -- independent of both pounce and cvxpy.
"""
import time
import numpy as np
from scipy.optimize import brentq

rng = np.random.default_rng(20260730)
N = 10
q = rng.uniform(0.5, 5.0, size=N)
c = rng.uniform(-8.0, 8.0, size=N)
L = rng.uniform(-3.0, -1.0, size=N)
U = L + rng.uniform(2.0, 5.0, size=N)          # ensure U_i > L_i
S = 0.4 * np.sum(L) + 0.6 * np.sum(U)          # feasible interior target sum


def x_of_lambda(lam):
    return np.clip((c + lam) / q, L, U)


def g(lam):
    return float(np.sum(x_of_lambda(lam)) - S)


# bracket: g is monotone non-decreasing in lambda
lam_lo, lam_hi = -1e4, 1e4
assert g(lam_lo) < 0 < g(lam_hi), f"bad bracket: g(lo)={g(lam_lo)}, g(hi)={g(lam_hi)}"
lam_star = brentq(g, lam_lo, lam_hi, xtol=1e-13, rtol=1e-13)
X_STAR = x_of_lambda(lam_star)
KNOWN_OPTIMAL = float(np.sum(0.5 * q * X_STAR ** 2 - c * X_STAR))
n_active = int(np.sum((X_STAR <= L + 1e-8) | (X_STAR >= U - 1e-8)))

# --- pounce active-set path via CLI on a hand-written .nl ---
import subprocess, json
import pyomo.environ as pyo

CLI = "/home/user/pounce/target/release/pounce"

m = pyo.ConcreteModel()
m.I = pyo.RangeSet(0, N - 1)
m.x = pyo.Var(m.I, initialize=0.0)
for i in range(N):
    m.x[i].setlb(float(L[i]))
    m.x[i].setub(float(U[i]))
m.obj = pyo.Objective(expr=sum(0.5 * q[i] * m.x[i] ** 2 - c[i] * m.x[i] for i in range(N)))
m.eq = pyo.Constraint(expr=sum(m.x[i] for i in range(N)) == float(S))

nl = "/tmp/adv_qpas_knapsack.nl"
m.write(nl, format="nl", io_options={"symbolic_solver_labels": True})


def run_cli(selection, tag):
    js = f"/tmp/adv_qpas_knapsack_{tag}.json"
    sol = f"/tmp/adv_qpas_knapsack_{tag}.sol"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [CLI, nl, sol, f"solver_selection={selection}", "--json-output", js],
        capture_output=True, text=True, timeout=60,
    )
    dt = time.perf_counter() - t0
    d = json.load(open(js))
    return d, proc.returncode, dt


d_as, exit_as, t_as = run_cli("qp-active-set", "as")
d_ipm, exit_ipm, t_ipm = run_cli("qp-ipm", "ipm")

as_x = np.asarray(d_as["solution"]["x"], dtype=float)
as_obj = float(d_as["solution"]["objective"])
as_status = d_as["solution"]["status"]
as_iters = int(d_as["statistics"]["iteration_count"])
ipm_x = np.asarray(d_ipm["solution"]["x"], dtype=float)
ipm_obj = float(d_ipm["solution"]["objective"])
ipm_iters = int(d_ipm["statistics"]["iteration_count"])
as_path_confirmed = as_iters != ipm_iters

# --- oracle: cvxpy (CLARABEL) ---
import cvxpy as cp

xv = cp.Variable(N)
prob = cp.Problem(
    cp.Minimize(0.5 * cp.sum(cp.multiply(q, cp.square(xv))) - c @ xv),
    [cp.sum(xv) == S, xv >= L, xv <= U],
)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_cvx = time.perf_counter() - t0
cvx_obj = float(prob.value)
cvx_x = np.asarray(xv.value)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


as_vs_known = rel(as_obj, KNOWN_OPTIMAL)
as_vs_ipm = rel(as_obj, ipm_obj)
as_vs_cvx = rel(as_obj, cvx_obj)
ipm_vs_cvx = rel(ipm_obj, cvx_obj)
x_err = float(np.linalg.norm(as_x - X_STAR, np.inf))

print("=== continuous quadratic-knapsack QP, N=10, sparse active set ===")
print(f"KNOWN_OPTIMAL={KNOWN_OPTIMAL:.10e}  n_active_bounds={n_active}/{N}  lambda*={lam_star:.6f}")
print("-- pounce active-set (CLI solver_selection=qp-active-set) --")
print(f"exit={exit_as} status={as_status} obj={as_obj:.10e} t={t_as:.4f}s iters={as_iters}")
print(f"active-set path confirmed (iters differ from qp-ipm {ipm_iters}): {as_path_confirmed}")
print("-- pounce qp-ipm (CLI forced) --")
print(f"exit={exit_ipm} obj={ipm_obj:.10e} iters={ipm_iters}")
print("-- cvxpy / CLARABEL --")
print(f"status={prob.status} obj={cvx_obj:.10e} t={t_cvx:.4f}s")
print(f"as_vs_known={as_vs_known:.2e} as_vs_ipm={as_vs_ipm:.2e} as_vs_cvx={as_vs_cvx:.2e} ipm_vs_cvx={ipm_vs_cvx:.2e}")
print(f"x_inf_err (active-set vs known) = {x_err:.2e}")

TOL = 1e-6
ok = (
    exit_as == 0
    and as_status in ("SolveSucceeded", "optimal")
    and as_vs_known < TOL
    and as_vs_ipm < TOL
    and as_vs_cvx < TOL
    and ipm_vs_cvx < TOL
    and x_err < 1e-4
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (exit={exit_as} status={as_status} as_vs_known={as_vs_known:.2e})")
