"""Adversary cross-check: Robust Linear Programming with ellipsoidal
(spherical) constraint uncertainty
Family: socp   Class: SOC-constrained LP (robust optimization -> SOCP)
Source: Boyd & Vandenberghe, Convex Optimization, Sec 4.4.2 "robust linear
    programming" (worst-case-over-ellipsoid reformulation
    a'x <= b for all a in {abar + P u : ||u||<=1}  <=>  abar'x + ||P'x|| <= b).
Known optimal: none published for this exact instance; certified instead by
    verifying the KKT stationarity/complementary-slackness conditions of the
    convex SOCP directly (see "Cross-check" below) -- for a convex problem
    with Slater's condition holding (x=0 is strictly feasible here, since
    b1,b2>0), KKT satisfaction is necessary AND sufficient for global
    optimality, independent of any solver's internal machinery.

Problem: a firm chooses production levels x=(x1,x2)>=0 to maximize profit
c'x=x1+2x2 subject to two resource constraints whose consumption
coefficients are only known to lie in a Euclidean ball around a nominal
value (robust/worst-case formulation):

    maximize   x1 + 2 x2
    subject to (a1 + P1 u)'x <= b1  for all ||u||<=1,   a1=[1,1],   P1=0.3*I, b1=10
               (a2 + P2 u)'x <= b2  for all ||u||<=1,   a2=[1,3],   P2=0.2*I, b2=15
               x >= 0

Worst case over u gives the robust (SOCP) constraints:
    a1'x + 0.3*||x||_2 <= 10
    a2'x + 0.2*||x||_2 <= 15
    x >= 0

pounce encoding (solve_socp): minimize c'x s.t. Gx<=h partitioned by cones.
Variables v=[x1,x2].  cones = [("nonneg",2), ("soc",3), ("soc",3)]:
  - nonneg block: s = x >= 0            -> G=-I(2), h=0
  - soc block i (dim 3): s=(t_i, rho_i*x1, rho_i*x2), t_i=b_i-a_i'x >= ||rho_i*x||
      row0: s0 = b_i - a_i'x  -> G_row = a_i,        h = b_i
      row1: s1 = rho_i*x1     -> G_row = [-rho_i,0], h = 0
      row2: s2 = rho_i*x2     -> G_row = [0,-rho_i], h = 0
"""
import time
import numpy as np

a1, rho1, b1 = np.array([1.0, 1.0]), 0.3, 10.0
a2, rho2, b2 = np.array([1.0, 3.0]), 0.2, 15.0
c_max = np.array([1.0, 2.0])  # maximize c_max'x  <=>  minimize -c_max'x
c = -c_max

G_nonneg = -np.eye(2)
h_nonneg = np.zeros(2)

G_soc1 = np.array([a1, [-rho1, 0.0], [0.0, -rho1]])
h_soc1 = np.array([b1, 0.0, 0.0])

G_soc2 = np.array([a2, [-rho2, 0.0], [0.0, -rho2]])
h_soc2 = np.array([b2, 0.0, 0.0])

G = np.vstack([G_nonneg, G_soc1, G_soc2])
h = np.concatenate([h_nonneg, h_soc1, h_soc2])
cones = [("nonneg", 2), ("soc", 3), ("soc", 3)]

import pounce

t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x_p = np.asarray(r.x, dtype=float)
obj_pounce = -float(r.obj)  # back to maximize convention
status = str(r.status)

# --- oracle: cvxpy, same SOCP, two independent conic solvers ---
import cvxpy as cp

x = cp.Variable(2)
constraints = [
    x >= 0,
    a1 @ x + rho1 * cp.norm(x, 2) <= b1,
    a2 @ x + rho2 * cp.norm(x, 2) <= b2,
]
prob = cp.Problem(cp.Maximize(c_max @ x), constraints)

t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_cla = time.perf_counter() - t0
val_cla, x_cla = float(prob.value), np.asarray(x.value, dtype=float)

prob.solve(solver=cp.SCS, eps=1e-9)
val_ecos, x_ecos = float(prob.value), np.asarray(x.value, dtype=float)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err = rel(obj_pounce, val_cla)
x_err = float(np.linalg.norm(x_p - x_cla, np.inf))

# --- KKT stationarity cross-check (independent of any solver's own duals) ---
# Active constraints at pounce's x_p (tolerance-based):
g1_val = a1 @ x_p + rho1 * np.linalg.norm(x_p) - b1
g2_val = a2 @ x_p + rho2 * np.linalg.norm(x_p) - b2
bound_active = x_p < 1e-6
active_tol = 1e-5
g1_active = g1_val > -active_tol
g2_active = g2_val > -active_tol

# gradient of g_i(x) = a_i'x + rho_i*||x|| is a_i + rho_i*x/||x||
nx = np.linalg.norm(x_p)
grad_g1 = a1 + rho1 * x_p / nx if nx > 1e-12 else a1
grad_g2 = a2 + rho2 * x_p / nx if nx > 1e-12 else a2

# Stationarity: c_max = mu1*grad_g1 + mu2*grad_g2 + nu  (nu>=0 on active bounds,
# nu=0 elsewhere), mu_i>=0 and mu_i=0 unless g_i active. Solve the active
# subsystem by least squares over the active multipliers only, then check the
# full stationarity residual and dual feasibility/complementary slackness.
active_grads = []
labels = []
if g1_active:
    active_grads.append(grad_g1)
    labels.append("g1")
if g2_active:
    active_grads.append(grad_g2)
    labels.append("g2")
free_idx = [i for i in range(2) if not bound_active[i]]

# Solve for (mu over active g's) using only the free (non-bound-active) rows
# of stationarity, since nu is unconstrained-sign-free-to-absorb on bound rows.
if active_grads and free_idx:
    A_stat = np.array([[g[i] for g in active_grads] for i in free_idx])
    mu, *_ = np.linalg.lstsq(A_stat, c_max[free_idx], rcond=None)
else:
    mu = np.zeros(len(active_grads))

nu = c_max - sum(m * g for m, g in zip(mu, active_grads)) if active_grads else c_max.copy()
stat_resid = float(np.linalg.norm(nu[bound_active] if bound_active.any() else np.array([0.0])
                                   if False else (nu * (~bound_active)), np.inf))
# nu should be ~0 on free (non-bound-active) coordinates; on bound-active
# coordinates nu plays the role of the (>=0) bound multiplier.
dual_feas = bool(np.all(mu >= -1e-6)) and bool(np.all(nu[bound_active] >= -1e-6) if bound_active.any() else True)

print("=== pounce (SOCP, robust LP via ellipsoidal uncertainty) ===")
print(f"status={status} obj(maximize)={obj_pounce:.10e} x={x_p} t={t_pounce:.4f}s")
print(f"g1(x)={g1_val:.3e} (active={g1_active})  g2(x)={g2_val:.3e} (active={g2_active})  bound_active={bound_active.tolist()}")
print("=== oracle: cvxpy CLARABEL / SCS (identical SOCP) ===")
print(f"CLARABEL obj={val_cla:.10e} x={x_cla} t={t_cla:.4f}s")
print(f"SCS      obj={val_ecos:.10e} x={x_ecos}")
print(f"obj_err_vs_CLARABEL={obj_err:.2e} x_inf_err_vs_CLARABEL={x_err:.2e}")
print("=== KKT stationarity re-derivation (independent of solver internals) ===")
print(f"active multipliers mu={mu} (labels={labels}) stationarity_resid(free coords)={stat_resid:.2e} dual_feasible={dual_feas}")

ok = (
    status in ("optimal", "optimal_inaccurate")
    and obj_err < 1e-4
    and x_err < 1e-3
    and stat_resid < 1e-4
    and dual_feas
)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, stat_resid={stat_resid:.2e}, dual_feasible={dual_feas})")
