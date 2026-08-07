"""Adversary cross-check: min sum x_i^3  s.t.  sum x_i = 10, x_i >= 1
Family: power   Class: p-norm epigraph (p=3) with an equality budget AND an
                inactive box lower bound (nonneg-cone row, not solve_qp's lb=)
Source: convexity of t^3 on t>=0 + symmetric Lagrange stationarity; the
        equal-split point x_i = 10/3 satisfies the bound 10/3 > 1 so the bound
        is provably inactive -- KKT: nabla(sum x_i^3) = 3x_i^2 = lambda for all
        i (unconstrained-by-bounds stationarity) forces x_i equal.
Known optimal: 3*(10/3)^3 = 3000/27 = 111.111111... at x=(10/3,10/3,10/3)

Distinct from prior "power" log entries (which use only an equality/no bound,
or a pure Hoelder dual-norm identity): this one adds an explicit inequality
(box) cone row alongside the pow-cone epigraph and an equality block, and the
bound is inactive by construction -- a probe that the pow-cone epigraph and a
plain nonneg row compose correctly without the inactive row perturbing the
active-set / multiplier bookkeeping.
"""
import time
import numpy as np

n = 3
KNOWN_OPTIMAL = 3000.0 / 27.0
KNOWN_X = np.full(n, 10.0 / 3.0)

# --- pounce: variables (x1,x2,x3,t1,t2,t3); minimize sum t_i
# pow-cone triple (x_i, t_i, 1) with alpha=1/3:  |x_i| <= t_i^{1/3} * 1^{2/3}  <=>  x_i^3 <= t_i
# plus nonneg row x_i - 1 >= 0 (box lower bound)
# plus equality A x = b:  x1+x2+x3 = 10
from pounce import solve_socp

nv = 2 * n
rows = []
h = []
for i in range(n):
    # row a: s = x_i  -> G[a,i] = -1
    ra = np.zeros(nv); ra[i] = -1.0
    rows.append(ra); h.append(0.0)
    # row b: s = t_i  -> G[b,n+i] = -1
    rb = np.zeros(nv); rb[n + i] = -1.0
    rows.append(rb); h.append(0.0)
    # row c: s = 1 (constant)
    rc = np.zeros(nv)
    rows.append(rc); h.append(1.0)
for i in range(n):
    # nonneg row: s = x_i - 1 >= 0
    rn = np.zeros(nv); rn[i] = -1.0
    rows.append(rn); h.append(-1.0)

G = np.vstack(rows)
h = np.array(h)
c = np.zeros(nv)
c[n:] = 1.0

cones = [("pow", 1.0 / 3.0)] * n + [("nonneg", 1)] * n

A = np.zeros((1, nv))
A[0, 0:n] = 1.0
b = np.array([10.0])

t0 = time.perf_counter()
r = solve_socp(c=c, A=A, b=b, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x_pounce = r.x[:n]
obj_pounce = float(np.sum(x_pounce ** 3))
status = r.status

# --- oracle: cvxpy ---
import cvxpy as cp

x = cp.Variable(n)
prob = cp.Problem(
    cp.Minimize(cp.sum(cp.power(x, 3))),
    [cp.sum(x) == 10.0, x >= 1.0],
)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle = x.value
obj_oracle = float(prob.value)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))
known_err = rel(obj_pounce, KNOWN_OPTIMAL)
known_x_err = float(np.linalg.norm(x_pounce - KNOWN_X, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s x={x_pounce}")
print("=== oracle (cvxpy/Clarabel) ===")
print(f"obj={obj_oracle:.10e} t={t_oracle:.4f}s x={x_oracle}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={known_err:.2e} x_err_vs_known={known_x_err:.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err={x_err:.2e}")

ok = status in ("optimal",) and obj_err < 1e-4 and known_err < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, known_err={known_err:.2e})")
