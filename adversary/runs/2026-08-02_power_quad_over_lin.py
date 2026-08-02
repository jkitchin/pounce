"""Adversary cross-check: quadratic-over-linear (Cauchy-Schwarz) minimization
Family: power   Class: power-cone, quadratic-over-linear / hyperbolic constraint

Problem:
    minimize   sum_i x_i^2 / y_i          (y_i fixed positive constants)
    subject to sum_i x_i = C

This is the classical Cauchy-Schwarz / weighted-harmonic-mean minimization
(e.g. Boyd & Vandenberghe Sec 3.1.5 "quadratic-over-linear" is convex; the
constrained minimization here is solved by Lagrange multipliers / weighted
Cauchy-Schwarz -- distinct from the previously-tested Holder-dual p-norm and
sum-of-p-th-power problems: it uses the power cone at alpha=1/2 with one
coordinate held at a FIXED constant (y_i), not three free variables).

CLOSED FORM (Lagrange multipliers):
    d/dx_i [x_i^2/y_i] = 2x_i/y_i = lambda  for all i  =>  x_i = lambda*y_i/2
    sum x_i = C  =>  lambda/2 * sum(y_i) = C  =>  x_i* = C*y_i / sum(y_j)
    optimal value = sum (x_i*)^2/y_i = C^2 / sum(y_j)   (weighted Cauchy-Schwarz
    equality case: sum(x_i^2/y_i) * sum(y_i) >= (sum x_i)^2 = C^2, equality iff
    x_i/y_i constant.)

Here y = (1, 2, 3), C = 12  =>  sum(y) = 6, f* = 144/6 = 24,
x* = 12*(1,2,3)/6 = (2, 4, 6).

CONE LAYOUT (pounce, solve_socp): variables x = [x_1..x_n, t_1..t_n]
(length 2n). objective: minimize sum t_i.
Per i, a power cone triple (x_i, y_i, t_i) with alpha=0.5:
    Kpow = { (u,v,w): |u| <= v^0.5 w^0.5, v,w >= 0 }
    y_i is a FIXED CONSTANT plugged into h (zero row of G), not a variable.
    |x_i| <= sqrt(y_i * t_i)  <=>  x_i^2 <= y_i*t_i  <=>  t_i >= x_i^2/y_i.
Equality sum x_i = C goes in A,b.
"""
import time
import numpy as np
import pounce
import cvxpy as cp

y = np.array([1.0, 2.0, 3.0])
n = len(y)
C = 12.0

KNOWN_OPTIMAL = float(C ** 2 / y.sum())
X_STAR = C * y / y.sum()
print(f"closed-form: x*={X_STAR} obj={KNOWN_OPTIMAL}")

# ---- pounce power-cone encoding --------------------------------------------
nv = 2 * n
c = np.zeros(nv)
c[n:] = 1.0  # minimize sum t_i

rows = 3 * n
G = np.zeros((rows, nv))
h = np.zeros(rows)
for i in range(n):
    r = 3 * i
    G[r, i] = -1.0        # s_r   = x_i
    h[r + 1] = y[i]        # s_{r+1} = y_i (constant)
    G[r + 2, n + i] = -1.0  # s_{r+2} = t_i

A = np.zeros((1, nv))
A[0, :n] = 1.0
b = np.array([C])

cones = [("pow", 0.5)] * n

t0 = time.perf_counter()
res = pounce.solve_socp(c=c, A=A, b=b, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x = np.asarray(res.x)
x_pounce = x[:n]
obj_pounce = res.obj
status = res.status

# ---- oracle: cvxpy (quad_over_lin atom), CLARABEL + SCS --------------------
def solve_cvxpy(solver):
    xv = cp.Variable(n)
    expr = cp.sum([cp.quad_over_lin(xv[i], y[i]) for i in range(n)])
    prob = cp.Problem(cp.Minimize(expr), [cp.sum(xv) == C])
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    return prob.value, time.perf_counter() - t0, np.asarray(xv.value)

obj_cl, t_cl, x_cl = solve_cvxpy(cp.CLARABEL)
obj_scs, t_scs, x_scs = solve_cvxpy(cp.SCS)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


err_known = rel(obj_pounce, KNOWN_OPTIMAL)
err_cl = rel(obj_pounce, obj_cl)
err_scs = rel(obj_pounce, obj_scs)
x_err_known = float(np.linalg.norm(x_pounce - X_STAR, np.inf))

print(f"=== pounce (n={n}) ===")
print(f"status={status} obj={obj_pounce:.10f} x={np.round(x_pounce, 6)} t={t_pounce:.4f}s")
print("=== oracle cvxpy ===")
print(f"CLARABEL obj={obj_cl:.10f} x={np.round(x_cl,6)} t={t_cl:.4f}s")
print(f"SCS      obj={obj_scs:.10f} t={t_scs:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10f}")
print(f"rel_err vs known={err_known:.2e}  vs CLARABEL={err_cl:.2e}  vs SCS={err_scs:.2e}  x_inf_err_vs_known={x_err_known:.2e}")

# NOTE: x_inf_err is ~1.5e-4 here even though obj_err is ~6.5e-8 -- see the
# .org report for a follow-up probe showing this is a precision-floor effect
# specific to power cones whose "y" coordinate is a FIXED CONSTANT baked into
# h (as opposed to a free variable pinned by an equality constraint, which
# gets ~6e-6 x-precision at a similar kkt_error). Objective-level agreement
# (the adversary.md PASS criterion) is what gates the verdict here.
ok = (status == "optimal" or res.success) and err_known < 1e-4 and err_cl < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, err_known={err_known:.2e})")
