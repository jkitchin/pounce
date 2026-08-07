"""Adversary cross-check: GP  min x1+x2+x3  s.t. x1*x2*x3 >= 8, x_i > 0
Family: exp   Class: geometric program (AM-GM constrained, sum objective / product constraint)
Source: AM-GM inequality (Hardy, Littlewood & Polya, "Inequalities" 2nd ed., Thm 9);
        x1+x2+x3 >= 3*(x1*x2*x3)^(1/3) >= 3*8^(1/3) = 6, equality at x1=x2=x3=2.
        Log-transform to exponential-cone epigraph per pounce.solve_socp docstring
        example (min x+1/x via Kexp).
Known optimal: 6.0 at x=(2,2,2)

This is the "sum objective / product-lower-bound constraint" GP shape -- distinct
from the previously-logged "sum x_i + a_i/x_i" (reciprocal) and "sum 1/x_i s.t.
sum x_i <= c" (Cauchy-Schwarz) GP probes; here the constraint itself becomes a
*linear* inequality after the log transform (sum y_i >= ln 8), so the only
nonlinearity is in the objective's exponential epigraphs -- a fresh combination
of an affine feasible region with 3 chained Kexp epigraphs.
"""
import time
import numpy as np

KNOWN_OPTIMAL = 6.0
KNOWN_X = np.array([2.0, 2.0, 2.0])

# --- pounce: variables (y1,y2,y3,t1,t2,t3); minimize sum t_i
# epigraph t_i >= exp(y_i) via Kexp triple (y_i, 1, t_i) since  1*exp(y_i/1) <= t_i
# constraint sum y_i >= ln(8)  as a "nonneg" cone row
from pounce import solve_socp

n = 3
nv = 2 * n  # y_0..y_2, t_0..t_2
rows = []
h = []
for i in range(n):
    # row a: s = y_i   -> G[a, i] = -1, h[a] = 0
    ra = np.zeros(nv); ra[i] = -1.0
    rows.append(ra); h.append(0.0)
    # row b: s = 1 (constant)
    rb = np.zeros(nv)
    rows.append(rb); h.append(1.0)
    # row c: s = t_i   -> G[c, n+i] = -1, h[c] = 0
    rc = np.zeros(nv); rc[n + i] = -1.0
    rows.append(rc); h.append(0.0)
# nonneg row: s = y1+y2+y3 - ln(8) >= 0  -> G_row = [-1,-1,-1,0,0,0], h_row = -ln(8)
rn = np.zeros(nv)
rn[0:n] = -1.0
rows.append(rn); h.append(-np.log(8.0))

G = np.vstack(rows)
h = np.array(h)
c = np.zeros(nv)
c[n:] = 1.0  # minimize sum t_i

cones = [("exp", 3)] * n + [("nonneg", 1)]

t0 = time.perf_counter()
r = solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
y_pounce = r.x[:n]
x_pounce = np.exp(y_pounce)
obj_pounce = float(np.sum(x_pounce))
status = r.status

# --- oracle: cvxpy, modeled directly in x-space (log-convex GP via DCP exp) ---
import cvxpy as cp

y = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.sum(cp.exp(y))), [cp.sum(y) >= np.log(8.0)])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle = np.exp(y.value)
obj_oracle = float(prob.value)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))
known_err = rel(obj_pounce, KNOWN_OPTIMAL)
known_x_err = float(np.linalg.norm(np.sort(x_pounce) - np.sort(KNOWN_X), np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s x={x_pounce}")
print("=== oracle (cvxpy/Clarabel) ===")
print(f"obj={obj_oracle:.10e} t={t_oracle:.4f}s x={x_oracle}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={known_err:.2e} x_err_vs_known={known_x_err:.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err={x_err:.2e}")

ok = status in ("optimal",) and obj_err < 1e-4 and known_err < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, known_err={known_err:.2e})")
