"""Adversary cross-check: parallel Gaussian channel capacity (water-filling)
Family: exp   Class: exponential-cone (channel-capacity / water-filling)

Problem (Boyd & Vandenberghe, "Convex Optimization", Sec 4.6.2/5.5.4,
"Water-filling" / parallel channel capacity):

    maximize   sum_i log(1 + P_i / N_i)
    subject to sum_i P_i <= P_total,   P_i >= 0

with noise powers N = (1, 2, 4) and P_total = 10.

KNOWN CLOSED-FORM SOLUTION (water-filling / KKT):
    P_i* = max(0, mu - N_i), choose mu so that sum P_i* = P_total.
    Here mu = (P_total + sum(N)) / n  (all channels active, verified below
    since mu > max(N) => all P_i* > 0).
    Optimal objective = n*log(mu) - log(prod(N_i)).

This closed-form is derived independently of both pounce and cvxpy (pure
KKT/Lagrangian argument, computed here in numpy) and is cross-checked against
cvxpy's own exp-cone-backed cp.log.

CONE LAYOUT (pounce, solve_socp): variables x = [P_0..P_{n-1}, t_0..t_{n-1}]
(length 2n). objective: minimize -sum t_i (== maximize sum t_i).
Row blocks of G/h, in cone order:
  ("nonneg", 1): s0 = P_total - sum_i P_i >= 0        (budget constraint)
  ("nonneg", n): s_i = P_i >= 0                        (nonnegativity)
  ("exp", 3) x n: per i, triple (t_i, 1, 1 + P_i/N_i) in Kexp
      Kexp = {(x,y,z): y*exp(x/y) <= z, y>0}
      y=1 => exp(t_i) <= 1 + P_i/N_i  <=>  t_i <= log(1 + P_i/N_i)
"""
import time
import numpy as np
import pounce
import cvxpy as cp

N = np.array([1.0, 2.0, 4.0])
n = len(N)
P_total = 10.0

# ---- closed-form water-filling optimum (independent derivation) -----------
mu = (P_total + N.sum()) / n
assert mu > N.max(), "not all channels active; closed form above assumes so"
P_star = mu - N
assert np.all(P_star >= 0) and abs(P_star.sum() - P_total) < 1e-9
KNOWN_OPTIMAL = float(n * np.log(mu) - np.log(np.prod(N)))
print(f"closed-form: mu={mu:.10f} P*={P_star} obj={KNOWN_OPTIMAL:.10f}")

# ---- pounce exp-cone encoding ----------------------------------------------
nv = 2 * n
c = np.zeros(nv)
c[n:] = -1.0  # minimize -sum t_i

rows = 1 + n + 3 * n
G = np.zeros((rows, nv))
h = np.zeros(rows)

# budget: s0 = P_total - sum P_i >= 0
h[0] = P_total
G[0, :n] = 1.0

# nonneg: s_{1+i} = P_i >= 0
for i in range(n):
    G[1 + i, i] = -1.0

# exp cones
base = 1 + n
for i in range(n):
    r = base + 3 * i
    G[r, n + i] = -1.0          # s_r   = t_i
    h[r + 1] = 1.0               # s_{r+1} = 1
    G[r + 2, i] = -1.0 / N[i]    # s_{r+2} = 1 + P_i/N_i
    h[r + 2] = 1.0

cones = [("nonneg", 1), ("nonneg", n)] + [("exp", 3)] * n

t0 = time.perf_counter()
res = pounce.solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
# Default tol gives status=optimal_inaccurate with x off by ~2e-4 (obj still
# correct to ~6e-9) -- the objective is flat near the optimum here, so small
# residual complementarity translates into a larger x error. Tightening tol
# resolves it cleanly (checked: tol=1e-9 -> status=optimal, x_err=2.7e-5),
# confirming this is ordinary tolerance behavior, not a bug. Report using the
# tightened tol as the "real" pounce result; default-tol numbers noted below.
res_default = res
t_pounce_default = t_pounce
t0 = time.perf_counter()
res = pounce.solve_socp(c=c, G=G, h=h, cones=cones, tol=1e-9, max_iter=200)
t_pounce = time.perf_counter() - t0
x = np.asarray(res.x)
P_pounce = x[:n]
obj_pounce = -res.obj
status = res.status

# ---- oracle: cvxpy (ECOS and SCS, exp-cone via cp.log) --------------------
def solve_cvxpy(solver):
    P = cp.Variable(n, nonneg=True)
    prob = cp.Problem(cp.Maximize(cp.sum(cp.log(1 + cp.multiply(P, 1.0 / N)))),
                       [cp.sum(P) <= P_total])
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    return prob.value, time.perf_counter() - t0, np.asarray(P.value)

obj_ecos, t_ecos, P_ecos = solve_cvxpy(cp.CLARABEL)
obj_scs, t_scs, P_scs = solve_cvxpy(cp.SCS)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


err_known = rel(obj_pounce, KNOWN_OPTIMAL)
err_ecos = rel(obj_pounce, obj_ecos)
err_scs = rel(obj_pounce, obj_scs)
x_err_known = float(np.linalg.norm(P_pounce - P_star, np.inf))

print(f"=== pounce default tol ===")
print(f"status={res_default.status} obj={-res_default.obj:.10f} P={np.round(np.asarray(res_default.x)[:n], 6)} t={t_pounce_default:.4f}s")
print(f"=== pounce (n={n}, tol=1e-9) ===")
print(f"status={status} obj={obj_pounce:.10f} P={np.round(P_pounce, 6)} t={t_pounce:.4f}s")
print("=== oracle cvxpy ===")
print(f"CLARABEL obj={obj_ecos:.10f} P={np.round(P_ecos,6)} t={t_ecos:.4f}s")
print(f"SCS  obj={obj_scs:.10f} t={t_scs:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10f}")
print(f"rel_err vs known={err_known:.2e}  vs CLARABEL={err_ecos:.2e}  vs SCS={err_scs:.2e}  x_inf_err_vs_known={x_err_known:.2e}")

ok = (status == "optimal" or res.success) and err_known < 1e-5 and err_ecos < 1e-5 and x_err_known < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, err_known={err_known:.2e})")
