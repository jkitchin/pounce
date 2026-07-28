"""Adversary cross-check: Markowitz mean-variance portfolio QP.
Family: qp   Class: equality-constrained convex QP (+ no-short box bounds)
Source: Markowitz mean-variance portfolio optimization (Markowitz 1952).
Closed-form KKT used for the EQUALITY-ONLY relaxation (no bounds binding),
verified against cvxpy/CLARABEL for the full bounded problem.

  min  0.5 * x' Sigma x - gamma * mu' x
  s.t. sum(x) = 1            (fully invested)
       x >= 0                (no short selling)

We construct Sigma SPD and mu so that the *equality-only* optimum is
strictly positive in every asset (no bound binds). Then the no-short
constraint is inactive and the closed-form analytic optimum applies:

KKT for min 0.5 x'Sigma x - gamma mu'x  s.t. 1'x = 1 :
   Sigma x - gamma mu + lam*1 = 0,   1'x = 1
=> x = Sigma^{-1}(gamma mu - lam 1),  with lam chosen so 1'x = 1.
Let a = Sigma^{-1} 1, b = Sigma^{-1} mu.
   1'x = gamma (1'b) - lam (1'a) = 1
   lam = (gamma (1'b) - 1) / (1'a)
   x* = gamma b - lam a
This is exact; objective at x* is the analytic KNOWN_OPTIMAL.

pounce form: P = Sigma, c = -gamma*mu, A = ones(1,n), b_eq = [1], lb = 0.
"""
import time
import numpy as np

np.random.seed(3)
n = 8
gamma = 0.1

# Build SPD covariance Sigma = D + L L'  (positive definite).
F = np.random.randn(n, 3) * 0.15           # 3 factors
diag = np.random.uniform(0.05, 0.15, n)    # idiosyncratic variance
Sigma = F @ F.T + np.diag(diag)
Sigma = 0.5 * (Sigma + Sigma.T)            # symmetrize exactly

mu = np.random.uniform(0.05, 0.20, n)      # expected returns

# Closed-form (equality-only) optimum
Sinv = np.linalg.inv(Sigma)
a = Sinv @ np.ones(n)
bvec = Sinv @ mu
lam = (gamma * np.ones(n) @ bvec - 1.0) / (np.ones(n) @ a)
X_STAR = gamma * bvec - lam * a
KNOWN_OPTIMAL = 0.5 * X_STAR @ Sigma @ X_STAR - gamma * mu @ X_STAR

assert np.all(X_STAR > 0), f"closed-form optimum has nonpositive weights, bounds would bind: {X_STAR}"

P = Sigma
c = -gamma * mu
A = np.ones((1, n))
b_eq = np.array([1.0])
lb = np.zeros(n)

# --- pounce convex QP IPM ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_qp(P=P, c=c, A=A, b=b_eq, lb=lb)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)
obj_pounce = r.obj   # no missing constant in this objective
status = r.status

# --- oracle: cvxpy/CLARABEL ---
import cvxpy as cp
xv = cp.Variable(n)
prob = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(Sigma)) - gamma * mu @ xv),
    [cp.sum(xv) == 1, xv >= 0])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle, obj_oracle = np.asarray(xv.value), prob.value


def rel(a_, b_):
    return abs(a_ - b_) / max(1.0, abs(b_))


obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))
x_err_known = float(np.linalg.norm(x_pounce - X_STAR, np.inf))
budget_resid = abs(float(np.sum(x_pounce)) - 1.0)
min_w = float(np.min(x_pounce))
eig_min = float(np.min(np.linalg.eigvalsh(Sigma)))

print("=== problem ===")
print(f"n={n} Sigma_min_eig={eig_min:.3e} (PD if >0)")
print(f"closed-form x* (all>0? {np.all(X_STAR > 0)}): {np.array2string(X_STAR, precision=4)}")
print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s iters={r.iters}")
print(f"budget_resid|sum-1|={budget_resid:.2e} min_weight={min_w:.3e}")
print("=== oracle (cvxpy/CLARABEL) ===")
print(f"obj={obj_oracle:.10e} t={t_oracle:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"x_err_vs_known(inf)={x_err_known:.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err_vs_oracle={x_err:.2e}")

ok = ((status == "optimal" or r.success)
      and obj_err < 1e-4
      and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4
      and x_err_known < 1e-4
      and budget_resid < 1e-6)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, x_err_known={x_err_known:.2e}, budget={budget_resid:.2e})")
