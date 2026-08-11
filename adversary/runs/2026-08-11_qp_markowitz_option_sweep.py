"""Adversary cross-check: option-space sweep on a known-PASS QP.
Family: qp   Class: option-space attack (tolerance monotonicity + engine
cross-check), reusing a previously-logged PASS problem rather than a new
formulation, per adversary.md's "attack the option space" arm.

Base problem: the Markowitz mean-variance portfolio QP already logged
2026-06-09 (adversary/log.org, PASS, rel obj err 6.5e-12):

  min  0.5 * x' Sigma x - gamma * mu' x
  s.t. sum(x) = 1, x >= 0

Sigma/mu are built (seed=3, n=8, gamma=0.1) so the equality-only KKT
optimum is strictly interior (all x_i > 0, no bound active) -- this is
copied verbatim from adversary/runs/2026-06-09_qp_markowitz_portfolio.py
(fetched from the adversary-runs branch) so the closed-form KNOWN_OPTIMAL
is exactly reproducible.

This probe does NOT re-adjudicate the formulation. It sweeps:
  (1) tol over 6 orders of magnitude (1e-4 .. 1e-12) on the default IPM
      engine, and asserts the verdict never degrades from "optimal" (no
      manufactured infeasibility/failure on a model that is feasible and
      strictly complementary at every tolerance);
  (2) method="active-set" vs the default method="ipm" (the two QP engines
      pounce ships) on the SAME data, and asserts they agree with each
      other and with the closed-form optimum -- an internal cross-path
      contradiction here needs no external oracle at all.
"""
import time
import numpy as np

np.random.seed(3)
n = 8
gamma = 0.1

F = np.random.randn(n, 3) * 0.15
diag = np.random.uniform(0.05, 0.15, n)
Sigma = F @ F.T + np.diag(diag)
Sigma = 0.5 * (Sigma + Sigma.T)

mu = np.random.uniform(0.05, 0.20, n)

Sinv = np.linalg.inv(Sigma)
a = Sinv @ np.ones(n)
bvec = Sinv @ mu
lam = (gamma * np.ones(n) @ bvec - 1.0) / (np.ones(n) @ a)
X_STAR = gamma * bvec - lam * a
KNOWN_OPTIMAL = 0.5 * X_STAR @ Sigma @ X_STAR - gamma * mu @ X_STAR
assert np.all(X_STAR > 0), "closed-form optimum has nonpositive weights"

P = Sigma
c = -gamma * mu
A = np.ones((1, n))
b_eq = np.array([1.0])
lb = np.zeros(n)

from pounce import solve_qp


def rel(a_, b_):
    return abs(a_ - b_) / max(1.0, abs(b_))


print(f"KNOWN_OPTIMAL={KNOWN_OPTIMAL:.10e}")

# --- (1) tolerance monotonicity sweep, default IPM engine ---
print("=== tol sweep (method=ipm) ===")
tol_results = []
for tol in [1e-4, 1e-6, 1e-8, 1e-9, 1e-10, 1e-12]:
    t0 = time.perf_counter()
    r = solve_qp(P=P, c=c, A=A, b=b_eq, lb=lb, tol=tol)
    dt = time.perf_counter() - t0
    err = rel(r.obj, KNOWN_OPTIMAL)
    tol_results.append((tol, r.status, err, dt))
    print(f"  tol={tol:.0e}  status={r.status:12s}  obj={r.obj:.10e}  rel_err={err:.2e}  t={dt:.4f}s")

tol_ok = all(status == "optimal" and err < 1e-3 for (_, status, err, _) in tol_results)

# --- (2) engine cross-check: ipm vs active-set ---
print("=== engine cross-check ===")
t0 = time.perf_counter()
r_ipm = solve_qp(P=P, c=c, A=A, b=b_eq, lb=lb, method="ipm")
t_ipm = time.perf_counter() - t0
t0 = time.perf_counter()
r_as = solve_qp(P=P, c=c, A=A, b=b_eq, lb=lb, method="active-set")
t_as = time.perf_counter() - t0

err_ipm = rel(r_ipm.obj, KNOWN_OPTIMAL)
err_as = rel(r_as.obj, KNOWN_OPTIMAL)
x_disagree = float(np.linalg.norm(np.asarray(r_ipm.x) - np.asarray(r_as.x), np.inf))

print(f"  ipm:         status={r_ipm.status:12s} obj={r_ipm.obj:.10e} rel_err={err_ipm:.2e} t={t_ipm:.4f}s")
print(f"  active-set:  status={r_as.status:12s} obj={r_as.obj:.10e} rel_err={err_as:.2e} t={t_as:.4f}s")
print(f"  x_inf_disagreement(ipm vs active-set)={x_disagree:.2e}")

engine_ok = (
    r_ipm.status == "optimal" and r_as.status == "optimal"
    and err_ipm < 1e-4 and err_as < 1e-4 and x_disagree < 1e-4
)

ok = tol_ok and engine_ok
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (tol_ok={tol_ok}, engine_ok={engine_ok})")
