"""Adversary cross-check: Hock-Schittkowski problem 76 (HS076), convex QP.
Family: qp   Class: inequality + bound constrained convex QP

Source: W. Hock and K. Schittkowski, "Test Examples for Nonlinear Programming
Codes", Lecture Notes in Economics and Mathematical Systems 187, Springer 1981,
Problem 76 (p. 95).

  min f(x) = x1^2 + 0.5 x2^2 + x3^2 + 0.5 x4^2
             - x1 x3 + x3 x4 - x1 - 3 x2 + x3 - x4
  s.t.  x1 + 2 x2 +   x3 +   x4 <= 5
        3 x1 +   x2 + 2 x3 -   x4 <= 4
              x2 + 4 x3        >= 1.5
        x >= 0

Known optimal (HS book):
  x* = (0.2727273, 2.090909, 0.0, 0.5454545)
  f* = -4.681818...
There is no additive constant in f, so CONST = 0.
"""
import time
import numpy as np

KNOWN_OPTIMAL = -4.681818181818182
X_STAR = np.array([0.2727273, 2.090909, 0.0, 0.5454545])
CONST = 0.0

# f(x) = 0.5 x'P x + c'x
# Quadratic part:
#   x1^2          -> P11 = 2
#   0.5 x2^2      -> P22 = 1
#   x3^2          -> P33 = 2
#   0.5 x4^2      -> P44 = 1
#   - x1 x3       -> P13 = P31 = -1
#   + x3 x4       -> P34 = P43 = +1
P = np.array([
    [2.0,  0.0, -1.0, 0.0],
    [0.0,  1.0,  0.0, 0.0],
    [-1.0, 0.0,  2.0, 1.0],
    [0.0,  0.0,  1.0, 1.0],
])
# Linear part: -x1 -3 x2 + x3 - x4
c = np.array([-1.0, -3.0, 1.0, -1.0])

# Inequalities as Gx <= h:
#   x1 + 2 x2 + x3 + x4 <= 5
#   3 x1 + x2 + 2 x3 - x4 <= 4
#   x2 + 4 x3 >= 1.5  ->  -x2 - 4 x3 <= -1.5
G = np.array([
    [1.0, 2.0, 1.0, 1.0],
    [3.0, 1.0, 2.0, -1.0],
    [0.0, -1.0, -4.0, 0.0],
])
h = np.array([5.0, 4.0, -1.5])
lb = np.zeros(4)

# sanity: P symmetric PSD?
assert np.allclose(P, P.T)
eig = np.linalg.eigvalsh(P)
assert eig.min() > -1e-9, eig

# --- pounce convex QP IPM ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_qp(P=P, c=c, G=G, h=h, lb=lb)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)
obj_pounce = r.obj + CONST
status = r.status

# --- oracle: cvxpy with two solvers ---
import cvxpy as cp
xv = cp.Variable(4)
obj = 0.5 * cp.quad_form(xv, cp.psd_wrap(P)) + c @ xv
cons = [G @ xv <= h, xv >= 0]

t0 = time.perf_counter()
p1 = cp.Problem(cp.Minimize(obj), cons)
p1.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle, obj_oracle = np.asarray(xv.value), p1.value

xv2 = cp.Variable(4)
obj2 = 0.5 * cp.quad_form(xv2, cp.psd_wrap(P)) + c @ xv2
p2 = cp.Problem(cp.Minimize(obj2), [G @ xv2 <= h, xv2 >= 0])
p2.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=100000)
obj_osqp = p2.value


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} x={x_pounce} t={t_pounce:.4f}s iters={r.iters}")
print("=== oracle (cvxpy/CLARABEL) ===")
print(f"obj={obj_oracle:.10e} x={x_oracle} t={t_oracle:.4f}s")
print(f"oracle2 (OSQP) obj={obj_osqp:.10e}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"x*_known={X_STAR} x_err_vs_known={np.linalg.norm(x_pounce-X_STAR, np.inf):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err={x_err:.2e}")

ok = ((status == "optimal" or getattr(r, "success", False))
      and obj_err < 1e-4
      and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4
      and rel(obj_oracle, obj_osqp) < 1e-5)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, known_err={rel(obj_pounce, KNOWN_OPTIMAL):.2e})")
