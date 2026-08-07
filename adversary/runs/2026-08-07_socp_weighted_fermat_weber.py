"""Adversary cross-check: weighted Fermat-Weber facility location
Family: socp   Class: weighted sum-of-norms (SOC epigraph per term)
Source: Boyd & Vandenberghe, Convex Optimization (2004) Sec. 8.7 (unweighted
        case) generalized to weights w_i > 0: minimize sum_i w_i*||x - p_i||_2.
        No closed form for 4 non-collinear weighted points; cross-checked
        against (a) cvxpy/Clarabel (independent conic oracle, DCP cp.norm)
        and (b) Weiszfeld's algorithm (Weiszfeld 1937 / Kuhn 1973 fixed-point
        iteration), an entirely different (non-conic, non-cvxpy) numerical
        method converged to ~1e-12 -- so pounce is checked against two
        *independent* oracles that don't trust each other either.
Known optimal: none published (no closed form) -- Weiszfeld fixed point used
               as the ground truth, itself cross-validated by cvxpy.

Distinct from the already-logged unweighted Fermat-Weber SOCP probe: unequal
weights break the symmetry that made the earlier instance's cvxpy-only check
comfortable, and adds the Weiszfeld oracle as a second, non-cvxpy witness.
"""
import time
import numpy as np

pts = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.0], [-1.0, 2.0]])
w = np.array([1.0, 2.0, 1.0, 3.0])
m = len(pts)

# --- Weiszfeld's algorithm (independent fixed-point oracle, no cvxpy/pounce) ---
def weiszfeld(pts, w, tol=1e-14, max_iter=10000):
    x = np.average(pts, axis=0, weights=w)
    for _ in range(max_iter):
        d = np.linalg.norm(pts - x, axis=1)
        d = np.maximum(d, 1e-16)
        wts = w / d
        x_new = (wts[:, None] * pts).sum(axis=0) / wts.sum()
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x


x_weiszfeld = weiszfeld(pts, w)
obj_weiszfeld = float(np.sum(w * np.linalg.norm(pts - x_weiszfeld, axis=1)))

# --- pounce: variables (x1,x2,t1,t2,t3,t4); minimize sum w_i t_i
# soc cone triple per point i: (t_i, x1-p_i1, x2-p_i2), dim 3, {(t,d):t>=||d||2}
from pounce import solve_socp

nv = 2 + m
rows = []
h = []
for i in range(m):
    # row a: s = t_i -> G[a, 2+i] = -1
    ra = np.zeros(nv); ra[2 + i] = -1.0
    rows.append(ra); h.append(0.0)
    # row b: s = x1 - p_i1 -> G[b,0] = -1, h[b] = -p_i1... check: s=h-Gx
    # want s = x1 - p_i1  => G[b,0] = -1 (s0 contrib = x1), h[b] = -p_i1
    rb = np.zeros(nv); rb[0] = -1.0
    rows.append(rb); h.append(-pts[i, 0])
    # row c: s = x2 - p_i2
    rc = np.zeros(nv); rc[1] = -1.0
    rows.append(rc); h.append(-pts[i, 1])

G = np.vstack(rows)
h = np.array(h)
c = np.zeros(nv)
c[2:] = w

cones = [("soc", 3)] * m

t0 = time.perf_counter()
r = solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x_pounce = r.x[:2]
obj_pounce = float(np.sum(w * np.linalg.norm(pts - x_pounce, axis=1)))
status = r.status

# --- oracle: cvxpy ---
import cvxpy as cp

xv = cp.Variable(2)
terms = [w[i] * cp.norm(xv - pts[i]) for i in range(m)]
prob = cp.Problem(cp.Minimize(cp.sum(terms)))
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle = xv.value
obj_oracle = float(prob.value)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))
wobj_err = rel(obj_pounce, obj_weiszfeld)
wx_err = float(np.linalg.norm(x_pounce - x_weiszfeld, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s x={x_pounce}")
print("=== oracle (cvxpy/Clarabel) ===")
print(f"obj={obj_oracle:.10e} t={t_oracle:.4f}s x={x_oracle}")
print("=== oracle (Weiszfeld fixed point) ===")
print(f"obj={obj_weiszfeld:.10e} x={x_weiszfeld}")
print(f"obj_err_vs_cvxpy={obj_err:.2e} x_inf_err_vs_cvxpy={x_err:.2e}")
print(f"obj_err_vs_weiszfeld={wobj_err:.2e} x_inf_err_vs_weiszfeld={wx_err:.2e}")

ok = status in ("optimal",) and obj_err < 1e-4 and wobj_err < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, wobj_err={wobj_err:.2e})")
