"""Adversary cross-check: Markowitz portfolio with a hard risk (SOC) budget.
Family: socp   Class: SOC constraint ACTIVE/TIGHT at the optimum, plus an equality

Problem (MOSEK Modeling Cookbook v3.3 sec. 3.3.3 "Markowitz portfolio
optimization"; Boyd & Vandenberghe, *Convex Optimization*, sec. 4.4.1 / ex. 4.52
-- the "maximize return subject to a risk bound" SOCP form):

    maximize    mu' x
    subject to  1' x = 1                     (fully invested)
                || Sigma^{1/2} x ||_2 <= s   (risk budget; SOC)

pounce minimizes, so we solve  min -mu' x  with the same constraints.

KNOWN OPTIMAL -- exact closed form (derived below, no solver involved):
    Let L = chol(Sigma) (lower), so Sigma = L L'.  Substitute y = L' x, i.e.
    x = L^{-T} y.  Then
        mu' x   = (L^{-1} mu)' y  =: mt' y
        1' x    = (L^{-1} 1)' y   =: a' y
        ||Sigma^{1/2} x||_2 = ||y||_2
    (note L^{-1}mu is the solve with L, since x = L^{-T}y => mu'x = mu'L^{-T}y
     = (L^{-1}mu)'y.)
    So the problem is
        max mt'y   s.t.  a'y = 1,  ||y|| <= s.
    Split y = y_p + y_o with y_p = a/||a||^2 (the min-norm point on the
    hyperplane a'y=1) and y_o ⟂ a.  Then ||y||^2 = 1/||a||^2 + ||y_o||^2, so
    the budget allows ||y_o|| <= r := sqrt(s^2 - 1/||a||^2), and
        mt'y = mt'a/||a||^2 + mt'y_o
    is maximized by y_o = r * P a-perp mt / ||P a-perp mt||.  Hence

        OPT = mt'a/||a||^2 + r * ||(I - aa'/||a||^2) mt||,
        x*  = L^{-T} ( a/||a||^2 + r * P_perp mt / ||P_perp mt|| ).

    Note 1/||a||^2 = 1/(1' Sigma^{-1} 1) is exactly the classic minimum-variance
    portfolio variance, so r > 0 iff s exceeds the minimum achievable risk --
    which is how we picked s.  The SOC constraint is TIGHT at the optimum
    (||Sigma^{1/2}x*|| = s exactly), which is the point of this test: prior socp
    runs (least squares, Chebyshev, robust LS, min-enclosing-ball, Fermat-Weber)
    all had the cone as an epigraph of the objective; here it is a hard active
    constraint on the boundary with a coupled equality.

Data: Sigma PD (eigs ~0.059/0.097/0.144), minimum-variance std = 0.214217...,
      s = 0.30 > that, so the problem is strictly feasible and the bound binds.

Cone encoding (re-derived, pounce convention s = h - G x in cone):
    ("soc", 4): s = (s0, s1..s3) with s0 >= ||s1..s3||.
      s0 = s_budget      -> h[0] = s_budget, G[0,:] = 0
      s1..3 = L' x       -> h[1:] = 0,       G[1:,:] = -L'
    Equality 1'x = 1 goes in (A, b), not in a cone.
"""
import time
import numpy as np

np.set_printoptions(precision=8, suppress=True)

# ---------------- data ----------------
Sigma = np.array([[0.10, 0.02, 0.01],
                  [0.02, 0.08, 0.03],
                  [0.01, 0.03, 0.12]])
mu = np.array([0.10, 0.20, 0.15])
s_budget = 0.30
n = 3
assert np.all(np.linalg.eigvalsh(Sigma) > 0)

L = np.linalg.cholesky(Sigma)          # Sigma = L L', L lower

# ---------------- closed-form reference ----------------
mt = np.linalg.solve(L, mu)            # L^{-1} mu
a = np.linalg.solve(L, np.ones(n))     # L^{-1} 1
na2 = float(a @ a)
min_var = 1.0 / na2                    # = 1/(1' Sigma^{-1} 1)
assert s_budget ** 2 > min_var, "risk budget below minimum-variance portfolio"
r_rad = np.sqrt(s_budget ** 2 - min_var)
P_perp_mt = mt - (a @ mt) / na2 * a
KNOWN_OPTIMAL_MAX = float((mt @ a) / na2 + r_rad * np.linalg.norm(P_perp_mt))
y_star = a / na2 + r_rad * P_perp_mt / np.linalg.norm(P_perp_mt)
x_star = np.linalg.solve(L.T, y_star)  # x = L^{-T} y
# pounce minimizes -mu'x:
KNOWN_OPTIMAL = -KNOWN_OPTIMAL_MAX

# independent sanity checks on the closed form
assert abs(np.sum(x_star) - 1.0) < 1e-12
assert abs(np.linalg.norm(L.T @ x_star) - s_budget) < 1e-12   # SOC TIGHT
assert abs(-mu @ x_star - KNOWN_OPTIMAL) < 1e-12

# ---------------- pounce ----------------
c = -mu
A_eq = np.ones((1, n))
b_eq = np.array([1.0])
G = np.zeros((4, n))
G[1:, :] = -L.T
h = np.array([s_budget, 0.0, 0.0, 0.0])
cones = [("soc", 4)]

import pounce
t0 = time.perf_counter()
res = pounce.solve_socp(c=c, A=A_eq, b=b_eq, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
x_p = np.asarray(res.x, dtype=float)
obj_pounce = float(c @ x_p)
status = res.status
risk_p = float(np.linalg.norm(L.T @ x_p))
sum_p = float(x_p.sum())

# ---------------- oracle: cvxpy, two solvers ----------------
import cvxpy as cp


def solve_cvxpy(solver):
    x = cp.Variable(n)
    cons = [cp.sum(x) == 1, cp.norm(L.T @ x, 2) <= s_budget]
    prob = cp.Problem(cp.Minimize(-mu @ x), cons)
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    return float(prob.value), time.perf_counter() - t0, np.asarray(x.value)


obj_cla, t_cla, x_cla = solve_cvxpy(cp.CLARABEL)
obj_scs, t_scs, x_scs = solve_cvxpy(cp.SCS)
obj_ecos, t_ecos, x_ecos = solve_cvxpy(cp.ECOS)


def rel(u, v):
    return abs(u - v) / max(1.0, abs(v))


print("=== closed-form reference ===")
print(f"x* = {x_star}")
print(f"min-variance std = {np.sqrt(min_var):.10f}  budget s = {s_budget}")
print(f"known_optimal (min -mu'x) = {KNOWN_OPTIMAL:.10e}   (max return {KNOWN_OPTIMAL_MAX:.10e})")
print("=== pounce (solve_socp, 1 SOC dim 4 + 1 equality) ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"x = {x_p}")
print(f"sum(x) = {sum_p:.12f}   ||L'x|| = {risk_p:.12f} (budget {s_budget}, tight?)")
print("=== oracle cvxpy/CLARABEL ===")
print(f"obj={obj_cla:.10e} t={t_cla:.4f}s x={x_cla}")
print("=== oracle cvxpy/SCS ===")
print(f"obj={obj_scs:.10e} t={t_scs:.4f}s x={x_scs}")
print("=== oracle cvxpy/ECOS ===")
print(f"obj={obj_ecos:.10e} t={t_ecos:.4f}s x={x_ecos}")
print(f"rel_err pounce vs known    = {rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"rel_err pounce vs CLARABEL = {rel(obj_pounce, obj_cla):.2e}")
print(f"rel_err pounce vs SCS      = {rel(obj_pounce, obj_scs):.2e}")
print(f"rel_err pounce vs ECOS     = {rel(obj_pounce, obj_ecos):.2e}")
print(f"x_inf_err vs closed form   = {np.max(np.abs(x_p - x_star)):.2e}")

ok = ((status == "optimal") or getattr(res, "success", False)) \
    and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 \
    and rel(obj_pounce, obj_cla) < 1e-4 \
    and rel(obj_pounce, obj_scs) < 1e-4 \
    and abs(sum_p - 1.0) < 1e-6 \
    and risk_p <= s_budget + 1e-6
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}, "
      f"sum={sum_p:.6f}, risk={risk_p:.6f})")
