"""Adversary cross-check: Hock-Schittkowski problem 28 (HS028), convex QP.
Family: qp   Class: pure equality-constrained convex QP (rank-deficient PSD P)

Source: W. Hock and K. Schittkowski, "Test Examples for Nonlinear Programming
Codes", Lecture Notes in Economics and Mathematical Systems 187, Springer 1981,
Problem 28 (p. 51).

  min f(x) = (x1 + x2)^2 + (x2 + x3)^2
  s.t.  x1 + 2 x2 + 3 x3 = 1

Known optimal (HS book):
  x* = (0.5, -0.5, 0.5)
  f* = 0.0
No bounds, single equality constraint. CONST = 0.

Adversarial features for the QP IPM:
  - P is symmetric PSD but RANK-DEFICIENT (rank 2 in R^3): the quadratic has a
    null direction, so the Hessian is singular along the feasible manifold's
    complement. A good stress test for the KKT factorization.
  - Optimal objective is exactly 0, so any positive obj is a clean error signal.
"""
import time
import numpy as np

KNOWN_OPTIMAL = 0.0
X_STAR = np.array([0.5, -0.5, 0.5])
CONST = 0.0

# f(x) = (x1+x2)^2 + (x2+x3)^2
#      = x1^2 + 2 x1 x2 + 2 x2^2 + 2 x2 x3 + x3^2
# f = 0.5 x'P x + c'x  with c = 0.
#   x1^2     -> P11 = 2
#   2 x2^2   -> P22 = 4
#   x3^2     -> P33 = 2
#   2 x1 x2  -> P12 = P21 = 2
#   2 x2 x3  -> P23 = P32 = 2
P = np.array([
    [2.0, 2.0, 0.0],
    [2.0, 4.0, 2.0],
    [0.0, 2.0, 2.0],
])
c = np.zeros(3)

# Equality: x1 + 2 x2 + 3 x3 = 1
A = np.array([[1.0, 2.0, 3.0]])
b = np.array([1.0])

# --- sanity checks on the model (do this BEFORE blaming the solver) ---
assert np.allclose(P, P.T), "P not symmetric"
eig = np.linalg.eigvalsh(P)
assert eig.min() > -1e-9, f"P not PSD: {eig}"
rank = np.linalg.matrix_rank(P)
print(f"# P eigenvalues={eig}  rank(P)={rank} (rank-deficient PSD as expected)")
# verify the documented optimum: feasible and f=0
assert abs(A @ X_STAR - b).max() < 1e-12, "X_STAR infeasible"
f_star_check = 0.5 * X_STAR @ P @ X_STAR + c @ X_STAR
assert abs(f_star_check - KNOWN_OPTIMAL) < 1e-12, f_star_check

# --- pounce convex QP IPM ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_qp(P=P, c=c, A=A, b=b)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)
obj_pounce = r.obj + CONST
status = r.status

# --- oracle 1: cvxpy / CLARABEL ---
import cvxpy as cp
xv = cp.Variable(3)
obj = 0.5 * cp.quad_form(xv, cp.psd_wrap(P)) + c @ xv
p1 = cp.Problem(cp.Minimize(obj), [A @ xv == b])
t0 = time.perf_counter()
p1.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle, obj_oracle = np.asarray(xv.value), p1.value

# --- oracle 2: closed-form KKT (eq-constrained QP) ---
# [P A'; A 0] [x; lam] = [-c; b]
n = 3
m = 1
KKT = np.block([[P, A.T], [A, np.zeros((m, m))]])
rhs = np.concatenate([-c, b])
sol = np.linalg.solve(KKT, rhs)
x_kkt = sol[:n]
obj_kkt = 0.5 * x_kkt @ P @ x_kkt + c @ x_kkt


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))
eq_resid = float(np.linalg.norm(A @ x_pounce - b, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} x={x_pounce} t={t_pounce:.4f}s iters={r.iters}")
print(f"eq_residual(|Ax-b|inf)={eq_resid:.2e}")
print("=== oracle1 (cvxpy/CLARABEL) ===")
print(f"obj={obj_oracle:.10e} x={x_oracle} t={t_oracle:.4f}s")
print("=== oracle2 (closed-form KKT) ===")
print(f"obj={obj_kkt:.10e} x={x_kkt}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e}  |obj_pounce - known|={abs(obj_pounce - KNOWN_OPTIMAL):.2e}")
print(f"x*_known={X_STAR} x_err_vs_known={np.linalg.norm(x_pounce - X_STAR, np.inf):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err_vs_oracle={x_err:.2e}")
print(f"KKT vs CLARABEL: obj diff={abs(obj_kkt - obj_oracle):.2e} x diff={np.linalg.norm(x_kkt - x_oracle, np.inf):.2e}")

# Objective optimum is exactly 0, so use absolute tolerance for the obj.
ok = ((status == "optimal" or getattr(r, "success", False))
      and abs(obj_pounce - KNOWN_OPTIMAL) < 1e-6
      and x_err < 1e-5
      and eq_resid < 1e-7
      and abs(obj_kkt - obj_oracle) < 1e-7)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, |obj-known|={abs(obj_pounce - KNOWN_OPTIMAL):.2e}, "
      f"x_err={x_err:.2e}, eq_resid={eq_resid:.2e})")
