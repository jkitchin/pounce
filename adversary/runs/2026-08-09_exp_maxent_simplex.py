"""Adversary cross-check: maximum-entropy distribution on the simplex
Family: exp   Class: entropy maximization (relative-entropy epigraph via Kexp)
Source: classic result (Cover & Thomas, "Elements of Information Theory",
        Thm 2.6.4 / Boyd & Vandenberghe, "Convex Optimization" Ex. 5.2 style):
        among all distributions on {1,...,n}, the uniform distribution
        maximizes entropy H(x) = -sum x_i log(x_i). Equivalently, minimizing
        sum x_i log(x_i) subject to sum x_i = 1, x >= 0 is solved by
        x_i = 1/n, with objective value -log(n).
Known optimal: n=4 -> -log(4) = -1.3862943611198906, at x = (0.25,0.25,0.25,0.25)

Distinct from the previously-logged exp probes (AM-GM constrained GP: sum
objective / product constraint, using the (y_i, 1, t_i) Kexp epigraph-of-exp
triple): this one is a genuine entropy/relative-entropy problem on the
simplex, using the OTHER standard Kexp embedding -- the (t_i, x_i, 1) triple
representing t_i <= entr(x_i) = -x_i*log(x_i) directly (derived below), plus
an equality constraint (sum x_i = 1) rather than a pure inequality.

Cone derivation: pounce's Kexp is {(x,y,z): y*exp(x/y) <= z, y>0}. Set
(x,y,z) = (t_i, x_i, 1): x_i*exp(t_i/x_i) <= 1  <=>  t_i/x_i <= log(1/x_i)
<=>  t_i <= x_i*log(1/x_i) = -x_i*log(x_i) = entr(x_i). So the triple
(t_i, x_i, 1) in Kexp enforces t_i <= entr(x_i). Minimizing sum(-t_i)
(equivalently maximizing sum t_i) drives each t_i up to equality with
entr(x_i) at the optimum, so min(-sum t_i) = min(sum x_i*log(x_i)) = -H(x).
"""
import time
import numpy as np

n = 4
KNOWN_OPTIMAL = -np.log(n)
KNOWN_X = np.full(n, 1.0 / n)

# --- pounce: variables (x_0..x_{n-1}, t_0..t_{n-1}); minimize -sum t_i
# cone triple i: (t_i, x_i, 1) in Kexp  =>  t_i <= entr(x_i)
# equality: sum x_i = 1
from pounce import solve_socp

nv = 2 * n
rows = []
h = []
for i in range(n):
    # slot a: s = t_i  -> G[a, n+i] = -1
    ra = np.zeros(nv); ra[n + i] = -1.0
    rows.append(ra); h.append(0.0)
    # slot b: s = x_i  -> G[b, i] = -1
    rb = np.zeros(nv); rb[i] = -1.0
    rows.append(rb); h.append(0.0)
    # slot c: s = 1 (constant)
    rc = np.zeros(nv)
    rows.append(rc); h.append(1.0)

G = np.vstack(rows)
h = np.array(h)
c = np.zeros(nv)
c[n:] = -1.0  # minimize -sum t_i

cones = [("exp", 3)] * n

A = np.zeros((1, nv))
A[0, 0:n] = 1.0
b = np.array([1.0])

t0 = time.perf_counter()
r = solve_socp(c=c, A=A, b=b, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x_pounce = r.x[:n]
obj_pounce = float(np.sum(x_pounce * np.log(x_pounce)))
status = r.status

# --- oracle: cvxpy (entr()) ---
import cvxpy as cp

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(-cp.sum(cp.entr(x))), [cp.sum(x) == 1.0, x >= 0])
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
