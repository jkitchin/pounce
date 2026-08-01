"""Adversary cross-check: LASSO via variable-splitting QP reformulation
Family: qp   Class: bound-constrained QP (L1-regularized least squares,
    variable-split into u,v>=0), new class for qp (prior qp runs used
    Markowitz eq+box, HS76 ineq+bounded, HS28 equality-only, Hilbert
    near-singular equality, ill-conditioned Vandermonde box, singular-PSD
    status battery, N&W16.4 closed-form-KKT eq+ineq, duals/multipliers
    battery, SVM dual Gram-matrix, box-simplex water-filling projection --
    none used the L1/soft-threshold structure or a variable-splitting QP
    reformulation).
Source: standard LASSO-as-QP reformulation (see e.g. Boyd & Vandenberghe,
    "Convex Optimization" Sec 6.1 / EE364a notes on L1 regularization via
    variable splitting). For an ORTHOGONAL design (A = I here), the exact
    minimizer of  0.5||x-b||_2^2 + lambda*||x||_1  is the closed-form
    soft-threshold operator:  x_i* = sign(b_i) * max(|b_i| - lambda, 0)
    (this is the proximal operator of the L1 norm; see Boyd & Vandenberghe
    or Beck, "First-Order Methods in Optimization", 2017, Sec 6.3-6.5).

Reformulation: x = u - v, u,v >= 0. Then
    ||x||_1 = 1'u + 1'v  (exact at optimum since not both u_i,v_i > 0 there)
    minimize 0.5||u-v-b||_2^2 + lambda*(1'u+1'v)  s.t. u,v >= 0
is a QP in z=[u;v] with P = M'M (M=[I,-I]), c = [-b+lambda*1; b+lambda*1].
Known optimal (b, lambda given below): x* = (2.0, 0.0, 0.0, -3.0, 0.5),
f* = 7.145 exactly (hand-computed below in exact rational arithmetic).
"""
import time
from fractions import Fraction
import numpy as np

b = np.array([3.0, -0.5, 0.2, -4.0, 1.5])
lam = 1.0
n = len(b)

# --- closed-form soft-threshold oracle (exact rational arithmetic) ---
bf = [Fraction(3, 1), Fraction(-1, 2), Fraction(1, 5), Fraction(-4, 1), Fraction(3, 2)]
lamf = Fraction(1, 1)


def softf(bi, l):
    m = abs(bi) - l
    if m <= 0:
        return Fraction(0)
    return (1 if bi > 0 else -1) * m


x_soft = [softf(bi, lamf) for bi in bf]


def obj_exact(xs):
    q = sum((xi - bi) ** 2 for xi, bi in zip(xs, bf)) * Fraction(1, 2)
    l1 = sum(abs(xi) for xi in xs) * lamf
    return q + l1


f_exact = obj_exact(x_soft)
KNOWN_X = np.array([float(v) for v in x_soft])
KNOWN_OPTIMAL = float(f_exact)
assert abs(KNOWN_OPTIMAL - 7.145) < 1e-12, KNOWN_OPTIMAL
assert np.allclose(KNOWN_X, [2.0, 0.0, 0.0, -3.0, 0.5]), KNOWN_X


def full_obj(x):
    return 0.5 * float(np.sum((np.asarray(x) - b) ** 2)) + lam * float(np.sum(np.abs(x)))


# --- pounce QP (variable-split reformulation) ---
I = np.eye(n)
M = np.hstack([I, -I])          # 5 x 10, x = M z
P = M.T @ M                      # 10 x 10, PSD (rank 5)
c = np.concatenate([-b + lam * np.ones(n), b + lam * np.ones(n)])
lb = np.zeros(2 * n)

import pounce
t0 = time.perf_counter()
r = pounce.solve_qp(P=P, c=c, lb=lb)
t_pounce = time.perf_counter() - t0
z = np.asarray(r.x, float)
u, v = z[:n], z[n:]
x_pounce = u - v
status = str(r.status)
f_pounce = full_obj(x_pounce)

# --- oracle: cvxpy, native L1 atom (DCP, no variable splitting) ---
import cvxpy as cp

xv = cp.Variable(n)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(xv - b) + lam * cp.norm(xv, 1)))
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_cvx = time.perf_counter() - t0
x_cvx = np.asarray(xv.value, float)
f_cvx = full_obj(x_cvx)

def rel(a, ref):
    return abs(a - ref) / max(1.0, abs(ref))

x_err_known = float(np.linalg.norm(x_pounce - KNOWN_X, np.inf))
x_err_cvx = float(np.linalg.norm(x_pounce - x_cvx, np.inf))
uv_complementarity = float(np.max(np.minimum(u, v)))   # should be ~0 (not both active)

print("=== pounce (solve_qp, variable-split LASSO) ===")
print(f"status={status} x={x_pounce} f={f_pounce:.10e} t={t_pounce:.4f}s")
print(f"  min(u,v) complementarity max={uv_complementarity:.2e} (want ~0)")
print("=== oracle: cvxpy/CLARABEL (native L1 atom) ===")
print(f"x={x_cvx} f={f_cvx:.10e} t={t_cvx:.4f}s")
print(f"known (soft-threshold, exact rational) x*={KNOWN_X} f*={KNOWN_OPTIMAL:.10e}")
print(f"x_inf_err_vs_known={x_err_known:.2e} x_inf_err_vs_cvx={x_err_cvx:.2e} "
      f"f_err_vs_known={rel(f_pounce, KNOWN_OPTIMAL):.2e} f_err_vs_cvx={rel(f_pounce, f_cvx):.2e}")

ok = (status in ("optimal",) or getattr(r, "success", False)) \
    and x_err_known < 1e-5 and x_err_cvx < 1e-5 and uv_complementarity < 1e-6
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, x_err_known={x_err_known:.2e}, x_err_cvx={x_err_cvx:.2e}, "
      f"complementarity={uv_complementarity:.2e})")
