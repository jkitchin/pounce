"""Adversary cross-check: maximum-entropy distribution on the probability simplex.
Family: exp   Class: exponential-cone (entropy maximization)

Problem:  maximize  H(p) = - sum_i p_i log p_i
          s.t.       sum_i p_i = 1,   p_i >= 0      (probability simplex)

SOURCE: Boyd & Vandenberghe, "Convex Optimization", Sec 3.5 / Example 4.x;
  the maximum-entropy distribution subject to ONLY the normalization
  constraint is the uniform distribution, with H = log(n). This is an exact,
  closed-form known optimum (no numerical reference needed).

KNOWN_OPTIMAL: log(n).  Here n = 5  ->  log(5) = 1.6094379124341003.
  Optimal p* = (1/5, ..., 1/5).

N_VARIABLES (pounce): n + n = 10 decision vars (p_i and epigraph z_i).

CONE LAYOUT:
  We MAXIMIZE sum z_i  (== minimize -sum z_i) with the per-coordinate
  exp-cone relation  z_i <= -p_i log p_i  encoded as
        (x=z_i, y=p_i, z=1) in Kexp
  because Kexp = {(x,y,z): y*exp(x/y) <= z, y>0} gives
        p_i * exp(z_i/p_i) <= 1  <=>  log p_i + z_i/p_i <= 0  <=> z_i <= -p_i log p_i.
  Equality constraint sum_i p_i = 1 goes in A,b.
"""
import time
import numpy as np
import pounce
import cvxpy as cp

n = 5
KNOWN_OPTIMAL = float(np.log(n))  # uniform distribution entropy

# ---- pounce encoding -------------------------------------------------------
# variables: [p_0..p_{n-1}, z_0..z_{n-1}]   (length 2n)
# objective: maximize sum z_i  ->  minimize  c.x with c = [0..0, -1..-1]
N = 2 * n
c = np.zeros(N)
c[n:] = -1.0  # minimize -sum z  == maximize sum z

# exp cones: per i a triple (z_i, p_i, 1).  slack s = h - G x must equal that.
# Build G,h with 3 rows per cone.
G = np.zeros((3 * n, N))
h = np.zeros(3 * n)
for i in range(n):
    r = 3 * i
    # s_r   = z_i        -> G[r, z_i] = -1
    G[r, n + i] = -1.0
    # s_{r+1} = p_i      -> G[r+1, p_i] = -1
    G[r + 1, i] = -1.0
    # s_{r+2} = 1        -> h[r+2] = 1 (G row zero)
    h[r + 2] = 1.0

# equality: sum p_i = 1
A = np.zeros((1, N))
A[0, :n] = 1.0
b = np.array([1.0])

cones = [("exp", 3)] * n

t0 = time.perf_counter()
res = pounce.solve_socp(c=c, A=A, b=b, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x = np.asarray(res.x)
p_pounce = x[:n]
# pounce minimized -sum z = -H  ->  entropy = -res.obj
obj_pounce = -res.obj
status = res.status

# ---- oracle: cvxpy with TWO solvers ---------------------------------------
def solve_cvxpy(solver):
    p = cp.Variable(n, nonneg=True)
    prob = cp.Problem(cp.Maximize(cp.sum(cp.entr(p))), [cp.sum(p) == 1])
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    return prob.value, time.perf_counter() - t0, np.asarray(p.value)

obj_ecos, t_ecos, p_ecos = solve_cvxpy(cp.ECOS)
obj_scs, t_scs, p_scs = solve_cvxpy(cp.SCS)

def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

err_known = rel(obj_pounce, KNOWN_OPTIMAL)
err_ecos = rel(obj_pounce, obj_ecos)
err_scs = rel(obj_pounce, obj_scs)

print(f"=== pounce (n={n}) ===")
print(f"status={status} entropy={obj_pounce:.10f} t={t_pounce:.4f}s")
print(f"p*={np.round(p_pounce,5)}")
print("=== oracle cvxpy ===")
print(f"ECOS entropy={obj_ecos:.10f} t={t_ecos:.4f}s")
print(f"SCS  entropy={obj_scs:.10f} t={t_scs:.4f}s")
print(f"known_optimal=log({n})={KNOWN_OPTIMAL:.10f}")
print(f"rel_err vs known={err_known:.2e}  vs ECOS={err_ecos:.2e}  vs SCS={err_scs:.2e}")

ok = (status == "optimal" or res.success) and err_known < 1e-5 and err_ecos < 1e-5 and err_scs < 1e-5
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, err_known={err_known:.2e})")
