"""Adversary cross-check: least-Euclidean-norm solution of Ax=b (SOCP).
Family: socp   Class: min-norm / minimum-energy solution of an underdetermined
                       linear system, epigraph SOC + equality constraints.

Problem:  minimize  ||x||_2   s.t.  A x = b,   A in R^{m x n}, m < n, full row rank.
CLOSED FORM (unique):  x* = A^T (A A^T)^{-1} b,  optimal value = ||x*||_2.
SOURCE: Boyd & Vandenberghe, "Convex Optimization" (2004), least-norm problem
(sec 4.4.1 / A.5.5); pseudoinverse least-norm solution.

Encoding:  vars z = [t, x] (nvar = 1+n).  minimize t.
  SOC (dim 1+n): slack s = (t, x) must satisfy t >= ||x||.
     s0 = t -> G[0, t] = -1, h0 = 0 ; s[1:] = x -> G[.,x] = -I, h=0.
  Equality A x = b via the A/b block of solve_socp: A_eq z = b with A_eq = [0 | A].

Distinct from logged socp tests (SOC least-squares residual, Chebyshev minimax,
robust LS, min enclosing ball, Fermat-Weber, Markowitz, eccentric ellipsoid,
no-Slater boundary): here the SOC bounds the DECISION norm under hard linear
EQUALITY constraints, with an exact pseudoinverse reference.
"""
import time
import numpy as np
import pounce
import cvxpy as cp

rng = np.random.default_rng(3)
m, n = 3, 6
A = rng.standard_normal((m, n))
b = rng.standard_normal(m)

# closed-form least-norm reference
x_ref = A.T @ np.linalg.solve(A @ A.T, b)
KNOWN_OPTIMAL = float(np.linalg.norm(x_ref))

nvar = 1 + n
c = np.zeros(nvar); c[0] = 1.0                # minimize t
G = np.zeros((1 + n, nvar)); h = np.zeros(1 + n)
G[0, 0] = -1.0                                # s0 = t
G[1:, 1:] = -np.eye(n)                        # s[1:] = x
A_eq = np.zeros((m, nvar)); A_eq[:, 1:] = A   # A x = b
cones = [("soc", 1 + n)]

t0 = time.perf_counter()
r = pounce.solve_socp(c=c, A=A_eq, b=b, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
z = np.asarray(r.x)
x_pounce = z[1:]
obj_pounce = float(r.obj)
status = r.status
feas = float(np.linalg.norm(A @ x_pounce - b, np.inf))


def rel(a, bb):
    return abs(a - bb) / max(1.0, abs(bb))


# oracle: cvxpy ECOS + CLARABEL
def cvx(solver):
    xv = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.norm(xv, 2)), [A @ xv == b])
    t0 = time.perf_counter(); prob.solve(solver=solver); dt = time.perf_counter() - t0
    return float(prob.value), np.asarray(xv.value), dt


obj_ecos, x_ecos, t_ecos = cvx(cp.ECOS)
obj_clar, x_clar, t_clar = cvx(cp.CLARABEL)

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} feas_inf={feas:.2e} t={t_pounce:.4f}s")
print(f"x_pounce={x_pounce}")
print("=== closed-form reference (pseudoinverse) ===")
print(f"||x*||={KNOWN_OPTIMAL:.10e}  x*={x_ref}")
print("=== oracle cvxpy ===")
print(f"ECOS obj={obj_ecos:.10e} t={t_ecos:.4f}s ; CLARABEL obj={obj_clar:.10e} t={t_clar:.4f}s")
print(f"rel_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e} rel_vs_ECOS={rel(obj_pounce, obj_ecos):.2e} "
      f"rel_vs_CLARABEL={rel(obj_pounce, obj_clar):.2e}")
print(f"x_inf_err_vs_ref={np.max(np.abs(x_pounce - x_ref)):.2e}")

ok = (status == "optimal" or r.success) and feas < 1e-6 \
    and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 and rel(obj_pounce, obj_ecos) < 1e-4 \
    and np.max(np.abs(x_pounce - x_ref)) < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, "
      f"rel_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}, feas={feas:.2e})")
