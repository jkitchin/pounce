"""Adversary cross-check: max-entropy on a DEGENERATE face -- optimal variables
driven EXACTLY to zero, Slater's condition fails.

Family: exp   Class: exp-cone, NON-SMOOTH CONE BOUNDARY / CQ FAILURE /
                     non-unique optimal face (dual)

Problem (n = 4):
    maximize   H(p) = - sum_i p_i log p_i
    s.t.       sum_i p_i = 1
               a . p = 0,     a = (0, 0, 1, 2)
               p >= 0

DEGENERACY / CQ FAILURE (the point of this test):
  Because a >= 0 and b = 0, feasibility FORCES p_2 = p_3 = 0 exactly.  The
  feasible set therefore has EMPTY relative interior with respect to the
  exponential cone: there is NO strictly feasible point, i.e. **Slater's
  condition fails**.  The optimum sits on the non-smooth part of the exp cone
  boundary -- the ray {(x, 0, z) : x <= 0, z >= 0} in
  cl(Kexp) = cl{(x,y,z) : y exp(x/y) <= z, y > 0} -- which is precisely the
  piece of the exp cone that is not facially exposed in the usual way and where
  the barrier/Nesterov-Todd scaling of an IPM degenerates (y -> 0).
  Consequences: the central path has no interior to follow in those two blocks,
  and the dual multiplier on the moment constraint is NOT unique (any
  sufficiently large multiplier certifies p_2 = p_3 = 0), so the optimal DUAL
  face is a whole unbounded ray.  Classic exp-cone pathology.

KNOWN OPTIMUM (exact, closed form -- no numerical reference needed):
  p_2 = p_3 = 0 is forced; the remaining mass 1 splits over the two
  zero-moment atoms, and max entropy over a 2-point support with fixed total
  mass 1 is the uniform distribution.  Hence
      p* = (1/2, 1/2, 0, 0)  and  H* = log 2 = 0.6931471805599453.
  (Cover & Thomas, "Elements of Information Theory" 2ed, Thm 2.6.4 -- entropy
   is maximized by the uniform distribution on the support; Boyd &
   Vandenberghe, "Convex Optimization" (2004), Sec. 3.5 / 5.2.3 for the
   Slater/CQ discussion.)

CONE ENCODING (pounce / Clarabel / MOSEK orientation
Kexp = {(x, y, z) : y exp(x/y) <= z, y > 0}):
    z_i <= -p_i log p_i  <=>  p_i exp(z_i / p_i) <= 1  <=>  (z_i, p_i, 1) in Kexp
  Verified below by a direct membership check on the analytic optimum.
Variables: [p_0..p_3, z_0..z_3];  minimize -sum z_i  (== maximize entropy).
Slack s = h - G x must land in the cone, so each cone block is s = (z_i, p_i, 1).
"""
import time
import numpy as np
import pounce
import cvxpy as cp

n = 4
a = np.array([0.0, 0.0, 1.0, 2.0])
b = 0.0

p_star = np.array([0.5, 0.5, 0.0, 0.0])
KNOWN_OPTIMAL = float(np.log(2.0))

# ---- sanity: the analytic optimum is feasible and hits the cone boundary ----
assert abs(p_star.sum() - 1.0) < 1e-15
assert abs(float(a @ p_star) - b) < 1e-15
# entr at the optimum, with the 0 log 0 = 0 convention
z_star = np.array([-0.5 * np.log(0.5), -0.5 * np.log(0.5), 0.0, 0.0])
assert abs(z_star.sum() - KNOWN_OPTIMAL) < 1e-15


def in_exp_cone(x, y, z, tol=1e-12):
    """Membership in cl(Kexp) = cl{(x,y,z): y exp(x/y) <= z, y > 0}."""
    if y > tol:
        return y * np.exp(x / y) <= z + tol
    return x <= tol and z >= -tol          # the NON-SMOOTH ray


for i in range(n):
    assert in_exp_cone(z_star[i], p_star[i], 1.0), i
# blocks 2 and 3 are on the non-smooth ray (y == 0) -- that is the whole point
assert p_star[2] == 0.0 and p_star[3] == 0.0

# ---- pounce encoding -------------------------------------------------------
N = 2 * n
c = np.zeros(N)
c[n:] = -1.0

G = np.zeros((3 * n, N))
h = np.zeros(3 * n)
for i in range(n):
    r = 3 * i
    G[r, n + i] = -1.0          # s_r     = z_i
    G[r + 1, i] = -1.0          # s_{r+1} = p_i
    h[r + 2] = 1.0              # s_{r+2} = 1

A = np.zeros((2, N))
A[0, :n] = 1.0
A[1, :n] = a
bvec = np.array([1.0, b])

cones = [("exp", 3)] * n

t0 = time.perf_counter()
res = pounce.solve_socp(c=c, A=A, b=bvec, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x = np.asarray(res.x)
p_pounce = x[:n]
obj_pounce = -res.obj
status = res.status

# ---- oracle: cvxpy with TWO solvers (must agree with each other first) ------
def solve_cvxpy(solver, **kw):
    p = cp.Variable(n, nonneg=True)
    prob = cp.Problem(cp.Maximize(cp.sum(cp.entr(p))),
                      [cp.sum(p) == 1, a @ p == b])
    t0 = time.perf_counter()
    try:
        prob.solve(solver=solver, **kw)
    except Exception as e:                      # noqa: BLE001
        return None, time.perf_counter() - t0, None, f"EXC:{type(e).__name__}"
    return prob.value, time.perf_counter() - t0, (
        None if p.value is None else np.asarray(p.value)), prob.status


obj_ecos, t_ecos, p_ecos, st_ecos = solve_cvxpy(cp.ECOS)
obj_clar, t_clar, p_clar, st_clar = solve_cvxpy(cp.CLARABEL)
obj_scs, t_scs, p_scs, st_scs = solve_cvxpy(cp.SCS, eps=1e-9, max_iters=100000)


def rel(u, v):
    if u is None or v is None:
        return float("nan")
    return abs(u - v) / max(1.0, abs(v))


print(f"=== construction (n={n}) ===")
print(f"a={a}  b={b}   (a >= 0, b = 0  =>  p_2 = p_3 = 0 forced; NO Slater point)")
print(f"p*        = {p_star}")
print(f"known H*  = log 2 = {KNOWN_OPTIMAL:.15f}")
print("optimal exp-cone blocks 2,3 = (0, 0, 1): the NON-SMOOTH ray of cl(Kexp)")

print("=== pounce ===")
print(f"status={status} entropy={obj_pounce:.15f} t={t_pounce:.4f}s")
print(f"p_pounce={p_pounce}")

print("=== oracle cvxpy (ECOS + CLARABEL must agree first; SCS as third) ===")
print(f"ECOS     status={st_ecos} entropy={obj_ecos} t={t_ecos:.4f}s p={p_ecos}")
print(f"CLARABEL status={st_clar} entropy={obj_clar} t={t_clar:.4f}s p={p_clar}")
print(f"SCS      status={st_scs} entropy={obj_scs} t={t_scs:.4f}s p={p_scs}")

print(f"oracle ECOS-vs-CLARABEL rel gap = {rel(obj_ecos, obj_clar):.2e}")
print(f"ECOS     vs known = {rel(obj_ecos, KNOWN_OPTIMAL):.2e}")
print(f"CLARABEL vs known = {rel(obj_clar, KNOWN_OPTIMAL):.2e}")
print(f"SCS      vs known = {rel(obj_scs, KNOWN_OPTIMAL):.2e}")

err_known = rel(obj_pounce, KNOWN_OPTIMAL)
print(f"pounce rel_err vs known={err_known:.2e}  vs ECOS={rel(obj_pounce, obj_ecos):.2e}"
      f"  vs CLARABEL={rel(obj_pounce, obj_clar):.2e}")

feas_sum = abs(p_pounce.sum() - 1.0)
feas_mom = abs(float(a @ p_pounce) - b)
neg = float(min(0.0, p_pounce.min()))
print(f"pounce feas: |sum p - 1|={feas_sum:.2e}  |a.p - b|={feas_mom:.2e}"
      f"  min p={p_pounce.min():.3e}")
print(f"pounce |p - p*|_inf = {np.abs(p_pounce - p_star).max():.3e}")
print(f"  driven-to-zero coords: p_2={p_pounce[2]:.3e}  p_3={p_pounce[3]:.3e}")

# oracles must agree with each other before pounce can be accused
oracle_ok = (rel(obj_ecos, obj_clar) < 1e-6
             and rel(obj_ecos, KNOWN_OPTIMAL) < 1e-6)
print(f"ORACLES_AGREE: {oracle_ok}")

# NOTE on the x tolerance: the objective is FLAT to second order along the
# free direction, H(t, 1-t, 0, 0) = log 2 - 2 (t - 1/2)^2 + O((t-1/2)^4), so an
# objective error eps only pins x to sqrt(eps/2).  With eps ~ 1.8e-7 that bound
# is ~3e-4; any |p - p*| below that is consistent with an exactly-converged
# objective and is NOT evidence of a solver defect.  Likewise the O(1e-9)
# negative entries in the forced-zero coordinates are the usual closure
# slop of an IPM on the non-smooth ray (CLARABEL leaves 1.3e-9 there too).
x_err = float(np.abs(p_pounce - p_star).max())
x_bound = float(np.sqrt(max(err_known, 1e-16) / 2.0))
print(f"flatness bound on x given obj err: {x_bound:.2e}  (actual {x_err:.2e})")

ok = ((status == "optimal" or getattr(res, "success", False))
      and err_known < 1e-4
      and feas_sum < 1e-7 and feas_mom < 1e-7 and neg > -1e-8
      and x_err < max(1e-4, x_bound))
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err_known={err_known:.2e}, "
      f"feas_sum={feas_sum:.2e}, feas_mom={feas_mom:.2e}, "
      f"x_err={x_err:.3e})")
