"""Adversary cross-check: max-entropy Gibbs distribution with a near-zero tail.
Family: exp   Class: exponential-cone, ILL-CONDITIONED / BAD SCALING

Problem (n = 5):
    maximize   H(p) = - sum_i p_i log p_i
    s.t.       sum_i p_i = 1
               sum_i a_i p_i = b
               p_i >= 0

with a = (0, 5, 10, 15, 20) and b chosen (below) so that the optimal
distribution is the Gibbs/exponential family p_i* = exp(-lam a_i)/Z with
lam = 1.4.  The exp-cone arguments therefore span 0 .. -28, and the smallest
optimal probability is p_4* = e^{-28}/Z ~ 6.9e-13 -- twelve orders of magnitude
below p_0* ~ 1, i.e. sitting essentially ON the boundary of the domain, which
is exactly the numerically dangerous regime for an exp-cone IPM.

KNOWN OPTIMUM (closed form, no numerical reference needed):
  For max-entropy with constraints sum p = 1 and sum a p = b, the unique
  maximizer is the Gibbs distribution p_i = exp(-lam a_i)/Z(lam), and
      H* = lam * b + log Z(lam).
  We CONSTRUCT the instance by fixing lam = 1.4 and setting
      Z  = sum_i exp(-lam a_i),  p* = exp(-lam a)/Z,  b = sum_i a_i p_i*.
  Then p* is feasible and of Gibbs form => it is the optimum, and H* is exact.
  (Boyd & Vandenberghe, "Convex Optimization" (2004), Sec. 3.5 / Example 5.x;
   Cover & Thomas, "Elements of Information Theory", Thm 12.1.1 -- maximum
   entropy under moment constraints is the exponential family.)

CONE ENCODING (verified against the pounce/Clarabel/MOSEK orientation
Kexp = {(x,y,z): y*exp(x/y) <= z, y > 0}):
    z_i <= -p_i log p_i   <=>   p_i exp(z_i/p_i) <= 1   <=>  (z_i, p_i, 1) in Kexp
Variables: [p_0..p_4, z_0..z_4]; minimize -sum z_i.
Slack s = h - G x must land in the cone, so each cone block is
    s = (z_i, p_i, 1).
"""
import time
import numpy as np
import pounce
import cvxpy as cp

n = 5
a = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
lam = 1.4

# ---- construct the instance so the optimum is closed-form ------------------
w = np.exp(-lam * a)          # 1, e^-7, e^-14, e^-21, e^-28
Z = w.sum()
p_star = w / Z
b = float(a @ p_star)
KNOWN_OPTIMAL = float(lam * b + np.log(Z))     # H* = lam*b + log Z
H_direct = float(-(p_star * np.log(p_star)).sum())
assert abs(KNOWN_OPTIMAL - H_direct) < 1e-13, (KNOWN_OPTIMAL, H_direct)

# ---- pounce encoding -------------------------------------------------------
N = 2 * n
c = np.zeros(N)
c[n:] = -1.0                                   # minimize -sum z == maximize sum z

G = np.zeros((3 * n, N))
h = np.zeros(3 * n)
for i in range(n):
    r = 3 * i
    G[r, n + i] = -1.0                         # s_r     = z_i
    G[r + 1, i] = -1.0                         # s_{r+1} = p_i
    h[r + 2] = 1.0                             # s_{r+2} = 1

A = np.zeros((2, N))
A[0, :n] = 1.0                                 # sum p = 1
A[1, :n] = a                                   # sum a p = b
bvec = np.array([1.0, b])

cones = [("exp", 3)] * n

t0 = time.perf_counter()
res = pounce.solve_socp(c=c, A=A, b=bvec, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x = np.asarray(res.x)
p_pounce = x[:n]
obj_pounce = -res.obj                           # entropy
status = res.status

# ---- oracle: cvxpy with TWO solvers ---------------------------------------
def solve_cvxpy(solver, **kw):
    p = cp.Variable(n, nonneg=True)
    prob = cp.Problem(cp.Maximize(cp.sum(cp.entr(p))),
                      [cp.sum(p) == 1, a @ p == b])
    t0 = time.perf_counter()
    prob.solve(solver=solver, **kw)
    return prob.value, time.perf_counter() - t0, np.asarray(p.value)

obj_ecos, t_ecos, p_ecos = solve_cvxpy(cp.ECOS)
obj_clar, t_clar, p_clar = solve_cvxpy(cp.CLARABEL)


def rel(u, v):
    return abs(u - v) / max(1.0, abs(v))


print(f"=== construction (n={n}, lam={lam}) ===")
print(f"a={a}  b={b:.15f}")
print(f"p*        = {p_star}")
print(f"min p*    = {p_star.min():.3e}   (exp-cone args span 0 .. {-lam*a.max():.0f})")
print(f"known H*  = {KNOWN_OPTIMAL:.15f}")

print("=== pounce ===")
print(f"status={status} entropy={obj_pounce:.15f} t={t_pounce:.4f}s")
print(f"p_pounce={p_pounce}")
print("=== oracle cvxpy (two solvers must agree first) ===")
print(f"ECOS     entropy={obj_ecos:.15f} t={t_ecos:.4f}s  p={p_ecos}")
print(f"CLARABEL entropy={obj_clar:.15f} t={t_clar:.4f}s  p={p_clar}")

oracle_gap = rel(obj_ecos, obj_clar)
print(f"oracle ECOS-vs-CLARABEL rel gap = {oracle_gap:.2e}")
print(f"ECOS     vs known = {rel(obj_ecos, KNOWN_OPTIMAL):.2e}")
print(f"CLARABEL vs known = {rel(obj_clar, KNOWN_OPTIMAL):.2e}")

err_known = rel(obj_pounce, KNOWN_OPTIMAL)
err_ecos = rel(obj_pounce, obj_ecos)
err_clar = rel(obj_pounce, obj_clar)
print(f"pounce rel_err vs known={err_known:.2e}  vs ECOS={err_ecos:.2e}  vs CLARABEL={err_clar:.2e}")

# feasibility / primal quality of the pounce point
feas_sum = abs(p_pounce.sum() - 1.0)
feas_mom = abs(float(a @ p_pounce) - b)
neg = float(min(0.0, p_pounce.min()))
print(f"pounce feas: |sum p - 1|={feas_sum:.2e}  |a.p - b|={feas_mom:.2e}  min p={p_pounce.min():.3e}")
print(f"pounce p abs err vs p*: {np.abs(p_pounce - p_star)}")

ok = ((status == "optimal" or getattr(res, "success", False))
      and err_known < 1e-5 and err_ecos < 1e-5 and err_clar < 1e-5
      and feas_sum < 1e-7 and feas_mom < 1e-6 and neg > -1e-9)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err_known={err_known:.2e}, "
      f"feas_sum={feas_sum:.2e}, feas_mom={feas_mom:.2e}, min_p={p_pounce.min():.3e})")
