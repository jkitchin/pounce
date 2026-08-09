"""Adversary cross-check: weighted min sum x_i^4 s.t. a^T x = b (Hoelder duality)
Family: power   Class: p-norm epigraph (p=4) with a NON-uniform linear
                equality (Hoelder/Lagrange dual-norm closed form)
Source: Boyd & Vandenberghe, "Convex Optimization", Sec 3.1.2 / the standard
        Hoelder-duality result: min ||x||_p s.t. a^T x = b (over x in R^n,
        p>1) has closed form via Lagrange stationarity nabla(sum|x_i|^p) =
        lambda*a  =>  p*x_i^3 = lambda*a_i (for p=4) => x_i = c*a_i^{1/3}
        (taking a_i>0, lambda>0). Constraint sum a_i x_i=b fixes
        c = b / sum(a_i^{4/3}); optimum value = c^4 * sum(a_i^{4/3}).
        Since t -> t^4 is monotonic on t>=0 and the p=4-norm minimizer set
        coincides with the sum-of-4th-powers minimizer set, minimizing
        sum x_i^4 subject to a^Tx=b gives the same x* as minimizing ||x||_4.
Known optimal: a=(1,2,3), b=12, n=3.
        c = b / sum(a_i^{4/3}) ;  x_i = c*a_i^{1/3} ;  obj = c^4*sum(a_i^{4/3})
        (computed exactly below with numpy -- a closed-form Lagrange
        stationarity result, not fit to any solver's output).

Distinct from the previously-logged power probe (uniform-budget symmetric
sum x_i^3 with an inactive box bound, alpha=1/3, equal-coefficient equality):
this one uses alpha=1/4 (p=4 not p=3), a genuinely NON-uniform weight vector
a in the linear constraint (so the minimizer is non-symmetric, x_i != x_j),
and no box constraint at all -- a fresh combination that stresses the
pow-cone epigraph's interaction with a weighted (not all-ones) equality row.
"""
import time
import numpy as np

n = 3
a_vec = np.array([1.0, 2.0, 3.0])
b_val = 12.0

# closed-form via Lagrange stationarity (independent of both pounce and cvxpy)
a43 = a_vec ** (4.0 / 3.0)
c_lag = b_val / np.sum(a43)
KNOWN_X = c_lag * a_vec ** (1.0 / 3.0)
KNOWN_OPTIMAL = float(c_lag ** 4 * np.sum(a43))

# sanity: constraint satisfied by construction
assert abs(np.dot(a_vec, KNOWN_X) - b_val) < 1e-10

# --- pounce: variables (x_0..x_2, t_0..t_2); minimize sum t_i
# pow-cone triple (x_i, t_i, 1) with alpha=1/4: |x_i| <= t_i^{1/4} * 1^{3/4}
# <=> x_i^4 <= t_i (t_i >= 0 enforced by the cone itself)
# equality: a^T x = b
from pounce import solve_socp

nv = 2 * n
rows = []
h = []
for i in range(n):
    ra = np.zeros(nv); ra[i] = -1.0       # slot a: s = x_i
    rows.append(ra); h.append(0.0)
    rb = np.zeros(nv); rb[n + i] = -1.0   # slot b: s = t_i
    rows.append(rb); h.append(0.0)
    rc = np.zeros(nv)                     # slot c: s = 1
    rows.append(rc); h.append(1.0)

G = np.vstack(rows)
h = np.array(h)
c = np.zeros(nv)
c[n:] = 1.0

cones = [("pow", 0.25)] * n

A = np.zeros((1, nv))
A[0, 0:n] = a_vec
b = np.array([b_val])

t0 = time.perf_counter()
r = solve_socp(c=c, A=A, b=b, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x_pounce = r.x[:n]
obj_pounce = float(np.sum(x_pounce ** 4))
status = r.status

# --- oracle: cvxpy ---
import cvxpy as cp

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.sum(cp.power(x, 4))), [a_vec @ x == b_val])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle = x.value
obj_oracle = float(prob.value)


def rel(u, v):
    return abs(u - v) / max(1.0, abs(v))


obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))
known_err = rel(obj_pounce, KNOWN_OPTIMAL)
known_x_err = float(np.linalg.norm(x_pounce - KNOWN_X, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s x={x_pounce}")
print("=== oracle (cvxpy/Clarabel) ===")
print(f"obj={obj_oracle:.10e} t={t_oracle:.4f}s x={x_oracle}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} (x*={KNOWN_X}) rel_err_vs_known={known_err:.2e} x_err_vs_known={known_x_err:.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err={x_err:.2e}")

ok = status in ("optimal",) and obj_err < 1e-4 and known_err < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, known_err={known_err:.2e})")
