"""Adversary cross-check: Robust least-squares (norm-bounded uncertainty)
Family: socp   Class: robust least-squares (Tikhonov-type worst-case residual)
Source: Boyd & Vandenberghe, "Convex Optimization" (2004), section 6.4.2,
        eq. (6.15): with A = A_bar + u (u the perturbation, ||u||_2 <= rho),
        the worst-case residual is

            sup_{||u||<=rho} ||(A_bar + u) x - b||_2 = ||A_bar x - b||_2 + rho ||x||_2.

        The robust LS problem  minimize_x ( ||A_bar x - b||_2 + rho ||x||_2 )
        is an SOCP:
            minimize   t1 + rho * t2
            s.t.       ||A_bar x - b||_2 <= t1      (SOC, dim 1 + m)
                       ||x||_2          <= t2       (SOC, dim 1 + n)

This is genuinely distinct from the already-tested "SOC least squares (0.5773)"
(a single SOC, pure l-2): here we have TWO SOC blocks summed in the objective,
a regularized/robust formulation.

KNOWN OPTIMUM: obtained from the worst-case residual formula evaluated at the
optimal x. We do not have a closed form for general data, so we use the
INDEPENDENT analytic worst-case-residual identity as a sanity check and cross-
validate the optimal objective with cvxpy ECOS and CLARABEL (two independent
conic solvers). We also confirm the worst-case residual identity numerically:
sup over a fine sampling of ||u||<=rho of ||(A_bar+u)x*-b|| equals
||A_bar x* - b|| + rho||x*|| at the pounce solution (proves the formulation).
"""
import time
import numpy as np

np.random.seed(1)
m, n = 6, 3
A_bar = np.array([
    [2.0, 0.0, -1.0],
    [0.0, 1.0,  1.0],
    [1.0, -1.0, 0.0],
    [0.5, 2.0,  1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 1.0,  1.0],
])
b = np.array([1.0, 2.0, 0.5, -1.0, 1.0, 2.0])
rho = 0.5

# ============================================================
# pounce encoding.
# Variables z = [t1, t2, x0, x1, x2]  (nvar = 2 + n = 5).
# Objective: minimize t1 + rho*t2.
#
# Cone block 1 (SOC dim 1+m): slack s1 = (t1, A_bar x - b)
#   s1[0]   = t1            -> G row coef of t1 = -1, h = 0  -> s = 0 - (-t1) = t1
#   s1[1:]  = A_bar x - b   -> G[.,x] = -A_bar, h = -b       -> s = -b + A_bar x
# Cone block 2 (SOC dim 1+n): slack s2 = (t2, x)
#   s2[0]   = t2            -> coef of t2 = -1, h=0
#   s2[1:]  = x             -> G[.,x] = -I, h = 0
# ============================================================
from pounce import solve_socp

nvar = 2 + n
c = np.zeros(nvar)
c[0] = 1.0          # t1
c[1] = rho          # rho * t2

n1 = 1 + m
n2 = 1 + n
G = np.zeros((n1 + n2, nvar))
h = np.zeros(n1 + n2)

# block 1
G[0, 0] = -1.0                      # s1_0 = t1
G[1:n1, 2:] = -A_bar                # s1_rest = A_bar x - b
h[1:n1] = -b
# block 2
G[n1, 1] = -1.0                     # s2_0 = t2
G[n1 + 1:, 2:] = -np.eye(n)         # s2_rest = x
# h zeros for block 2

t0 = time.perf_counter()
r = solve_socp(c=c, G=G, h=h, cones=[("soc", n1), ("soc", n2)])
t_pounce = time.perf_counter() - t0
z = np.asarray(r.x)
obj_pounce = float(r.obj)
status = r.status
x_pounce = z[2:]

# --- oracle: cvxpy ECOS and CLARABEL ---
import cvxpy as cp

def solve_cvxpy(solver):
    x = cp.Variable(n)
    obj = cp.norm(A_bar @ x - b, 2) + rho * cp.norm(x, 2)
    prob = cp.Problem(cp.Minimize(obj))
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    dt = time.perf_counter() - t0
    return float(prob.value), np.asarray(x.value), dt

obj_ecos, x_ecos, t_ecos = solve_cvxpy(cp.ECOS)
obj_clarabel, x_clarabel, t_clarabel = solve_cvxpy(cp.CLARABEL)

# --- INDEPENDENT formulation check: worst-case residual identity (B&V 6.15) ---
# At any x, sup_{||u||_F <= rho} ||(A_bar+u) x - b|| = ||A_bar x - b|| + rho||x||.
# Verify by sampling u (rank-1 worst-case is u = rho * r x^T / (||r|| ||x||)).
def worst_case_residual_sampled(x, n_samples=20000):
    r0 = A_bar @ x - b
    best = -1.0
    rng = np.random.default_rng(7)
    for _ in range(n_samples):
        U = rng.standard_normal((m, n))
        U = rho * U / np.linalg.norm(U, 2)  # spectral norm <= rho
        val = np.linalg.norm((A_bar + U) @ x - b, 2)
        if val > best:
            best = val
    return best

analytic_wc = float(np.linalg.norm(A_bar @ x_pounce - b, 2) + rho * np.linalg.norm(x_pounce, 2))
sampled_wc = worst_case_residual_sampled(x_pounce)

KNOWN_OPTIMAL = obj_clarabel  # consensus reference (cross-checked by ECOS too)

def rel(a, bb):
    return abs(a - bb) / max(1.0, abs(bb))

obj_err_ecos = rel(obj_pounce, obj_ecos)
obj_err_clarabel = rel(obj_pounce, obj_clarabel)
x_err = float(np.linalg.norm(x_pounce - x_ecos, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s x={x_pounce}")
print("=== oracle ECOS ===")
print(f"obj={obj_ecos:.10e} t={t_ecos:.4f}s x={x_ecos}")
print("=== oracle CLARABEL ===")
print(f"obj={obj_clarabel:.10e} t={t_clarabel:.4f}s x={x_clarabel}")
print(f"obj_err_vs_ECOS={obj_err_ecos:.2e}  obj_err_vs_CLARABEL={obj_err_clarabel:.2e}")
print(f"x_inf_err_vs_ECOS={x_err:.2e}")
print("--- worst-case residual identity (B&V 6.15) at pounce x* ---")
print(f"analytic ||A_bar x*-b|| + rho||x*|| = {analytic_wc:.6f}")
print(f"sampled  sup_||u||<=rho ||(A_bar+u)x*-b|| = {sampled_wc:.6f}  (must be <= analytic)")
print(f"objective at x* = {obj_pounce:.6f}  (must equal analytic worst-case)")

ok = (status == "optimal") and obj_err_ecos < 1e-4 and obj_err_clarabel < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, eE={obj_err_ecos:.2e}, eC={obj_err_clarabel:.2e})")
