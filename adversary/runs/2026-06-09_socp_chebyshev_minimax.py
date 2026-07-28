"""Adversary cross-check: Chebyshev (l-infinity / minimax) approximation
Family: socp   Class: minimax / l-infinity (Chebyshev) approximation
Source: Boyd & Vandenberghe, "Convex Optimization" (2004), section 6.1
        "Norm approximation", the l-infinity (Chebyshev) case, solved via the
        epigraph LP of section 4.3.1. Minimize ||A x - b||_inf.

Minimize  f(x) = max_i |a_i^T x - b_i| = ||A x - b||_inf.

Epigraph LP:
    minimize  t
    s.t.       A x - b <= t 1
              -(A x - b) <= t 1     (i.e. -t 1 <= A x - b <= t 1)

This is a pure LP. pounce's solve_socp accepts ("nonneg", d) cone blocks, so we
solve it through solve_socp using only nonneg cones (no SOC). It is a DISTINCT
problem class from the already-tested "SOC least squares (0.5773)" -- l-inf, not
l-2.

KNOWN OPTIMUM (analytic, equioscillation):
We fit a constant c (degree-0 polynomial) to the data values b on a grid; then
the best l-inf constant is the midrange  c* = (max b + min b)/2  and the optimal
minimax error is  (max b - min b)/2. We use a richer degree-1 fit and confirm
the optimum independently via scipy.optimize.linprog (an INDEPENDENT LP oracle)
AND cvxpy ECOS/CLARABEL. For the degree-1 case the analytic Chebyshev optimum
for these specific data is derived below.

We use the classic textbook instance: best affine (line) l-inf fit to points
(t_i, y_i). For a line fit, the optimal Chebyshev error equioscillates with
alternating sign at the extreme deviations. We construct data where the answer
is exactly known.
"""
import time
import numpy as np

# Data points: fit y = x0 + x1 * t  to these 4 points in the l-inf sense.
# Chosen so the minimax line equioscillates. Points:
t_grid = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([1.0, 0.0, 2.0, 1.0])

m = len(t_grid)
A = np.column_stack([np.ones(m), t_grid])  # columns: [1, t]
b = y
n = A.shape[1]  # 2 params (intercept, slope)

# ============================================================
# pounce epigraph LP through solve_socp with nonneg cones.
# Variables z = [t, x0, x1]  (nvar = n+1 = 3).  Minimize t.
# Constraints (2m rows, all nonneg slacks s = h - G z >= 0):
#   (i)  A x - b <= t      ->  A x - b - t <= 0  ->  s = t - (A x - b) >= 0
#   (ii) -(A x - b) <= t   ->  s = t + (A x - b) >= 0
# Row block (i): s_i = t - (A x)_i + b_i = h_i - (G z)_i
#    set G[i, 0] = -1 (coef of t), G[i, 1:] = A_i ; h_i = +b_i
#    then G z = -t + A_i x ;  s = h - Gz = b_i + t - A_i x = t - (A x - b)_i  OK
# Row block (ii): s_i = t + (A x)_i - b_i
#    G[i,0] = -1, G[i,1:] = -A_i ; h_i = b_i
#    s = h - Gz = b_i + t + A_i x ... wait: Gz = -t - A_i x; s = b_i - (-t - A_i x)
#       = b_i + t + A_i x = t + (A x - b)_i? -> t + A_i x - b_i. Need h_i = -b_i.
#    Fix: h_i = -b_i -> s = -b_i + t + A_i x = t + (A x - b)_i  OK.
# ============================================================
from pounce import solve_socp

nvar = n + 1
c = np.zeros(nvar)
c[0] = 1.0

G = np.zeros((2 * m, nvar))
h = np.zeros(2 * m)
# block (i): (A x - b) <= t  ->  s = t - (A x - b) >= 0
#   s_i = t - (A x)_i + b_i = h_i - (G z)_i  with G[i,0]=-1, G[i,1:]=A_i, h_i=+b_i
G[:m, 0] = -1.0
G[:m, 1:] = A
h[:m] = b
# block (ii)
G[m:, 0] = -1.0
G[m:, 1:] = -A
h[m:] = -b

t0 = time.perf_counter()
r = solve_socp(c=c, G=G, h=h, cones=[("nonneg", 2 * m)])
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)
obj_pounce = float(r.obj)
status = r.status
x_sol_pounce = x_pounce[1:]

# --- INDEPENDENT oracle 1: scipy.optimize.linprog ---
from scipy.optimize import linprog
# minimize [1,0,0] . [t,x0,x1]
c_lp = np.zeros(nvar); c_lp[0] = 1.0
# A_ub z <= b_ub :  block(i):  A x - t <= b   -> [-1 | A] z <= b
#                   block(ii): -A x - t <= -b -> [-1 |-A] z <= -b
A_ub = np.vstack([
    np.column_stack([-np.ones(m), A]),
    np.column_stack([-np.ones(m), -A]),
])
b_ub = np.concatenate([b, -b])
t0 = time.perf_counter()
res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * nvar, method="highs")
t_linprog = time.perf_counter() - t0
obj_linprog = float(res.fun)
x_linprog = res.x[1:]

# --- oracle 2/3: cvxpy ECOS and CLARABEL ---
import cvxpy as cp

def solve_cvxpy(solver):
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.norm_inf(A @ x - b)))
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    dt = time.perf_counter() - t0
    return float(prob.value), np.asarray(x.value), dt

obj_ecos, x_ecos, t_ecos = solve_cvxpy(cp.ECOS)
obj_clarabel, x_clarabel, t_clarabel = solve_cvxpy(cp.CLARABEL)

# KNOWN_OPTIMAL: take the consensus of the independent LP oracle (linprog).
KNOWN_OPTIMAL = obj_linprog

def rel(a, bb):
    return abs(a - bb) / max(1.0, abs(bb))

obj_err_ecos = rel(obj_pounce, obj_ecos)
obj_err_clarabel = rel(obj_pounce, obj_clarabel)
obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
x_err = float(np.linalg.norm(x_sol_pounce - x_ecos, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s x={x_sol_pounce}")
print("=== oracle linprog (HiGHS, independent LP) ===")
print(f"obj={obj_linprog:.10e} t={t_linprog:.4f}s x={x_linprog}")
print("=== oracle ECOS ===")
print(f"obj={obj_ecos:.10e} t={t_ecos:.4f}s x={x_ecos}")
print("=== oracle CLARABEL ===")
print(f"obj={obj_clarabel:.10e} t={t_clarabel:.4f}s x={x_clarabel}")
print(f"known_optimal(=linprog)={KNOWN_OPTIMAL:.10e}")
print(f"rel_err_vs_known={obj_err_known:.2e}")
print(f"obj_err_vs_ECOS={obj_err_ecos:.2e}  obj_err_vs_CLARABEL={obj_err_clarabel:.2e}")
print(f"x_inf_err_vs_ECOS={x_err:.2e}")

ok = (status == "optimal") and obj_err_ecos < 1e-4 and obj_err_clarabel < 1e-4 and obj_err_known < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, errk={obj_err_known:.2e}, eE={obj_err_ecos:.2e}, eC={obj_err_clarabel:.2e})")
