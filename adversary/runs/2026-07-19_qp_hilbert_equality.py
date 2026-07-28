"""Adversary cross-check: equality-constrained QP with a near-singular (Hilbert) Hessian
Family: qp   Class: equality-heavy convex QP, ill-conditioned PD Hessian, exact rational KKT

Problem:
    minimize   1/2 x' H x + c' x
    subject to A x = b

with H = the 8x8 Hilbert matrix, H_ij = 1/(i+j-1), which is symmetric positive
definite but notoriously ill-conditioned: cond_2(H_8) ~ 1.5e10.
(Reference for the Hilbert matrix as the canonical ill-conditioned SPD test
matrix: N. J. Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed.,
SIAM 2002, S28.1 "The Hilbert matrix"; also Golub & Van Loan, "Matrix
Computations" 4e, S3.5.)

c = -(1,1,...,1)
A = [[1,1,1,1,1,1,1,1],
     [1,-1,1,-1,1,-1,1,-1],
     [1,2,3,4,5,6,7,8]]
b = [1, 0, 10]

KNOWN OPTIMAL: no published value -- it is derived in CLOSED FORM here.  With
only equality constraints and H positive definite, the KKT conditions

    [ H   A' ] [ x ]   [ -c ]
    [ A   0  ] [ l ] = [  b ]

are necessary AND sufficient, and the KKT matrix is nonsingular (A has full row
rank 3, H is PD).  Every entry of H, A, b, c is RATIONAL, so the KKT system is
solved EXACTLY in rational arithmetic with sympy -- an oracle that involves no
floating point and no iterative solver at all.  This is the strongest possible
independent check: it cannot inherit any conditioning error.

Second oracle: cvxpy / CLARABEL.
"""
import time
import numpy as np
from fractions import Fraction
import sympy as sp

N = 8

# --- exact rational data ---
H_exact = sp.Matrix(N, N, lambda i, j: sp.Rational(1, i + j + 1))
c_exact = sp.Matrix([-1] * N)
A_exact = sp.Matrix([[1] * N,
                     [(-1) ** j for j in range(N)],
                     [j + 1 for j in range(N)]])
b_exact = sp.Matrix([1, 0, 10])
M = A_exact.rows

# --- float versions handed to the solvers ---
H = np.array(H_exact.tolist(), dtype=float)
c = np.array(c_exact.tolist(), dtype=float).ravel()
A = np.array(A_exact.tolist(), dtype=float)
b = np.array(b_exact.tolist(), dtype=float).ravel()

print(f"cond_2(H) = {np.linalg.cond(H):.6e}   rank(A) = {np.linalg.matrix_rank(A)}/{M}")
print(f"eig_min(H) = {np.linalg.eigvalsh(H).min():.6e}  (PD => KKT sufficient)")

# --- exact closed-form oracle: solve the KKT system in rational arithmetic ---
t0 = time.perf_counter()
KKT = sp.Matrix(sp.BlockMatrix([[H_exact, A_exact.T],
                                [A_exact, sp.zeros(M, M)]]))
rhs = sp.Matrix.vstack(-c_exact, b_exact)
sol = KKT.solve(rhs)                      # exact rational solve
t_exact = time.perf_counter() - t0
x_exact = sol[:N, 0]
obj_exact_rat = (sp.Rational(1, 2) * (x_exact.T * H_exact * x_exact)[0]
                 + (c_exact.T * x_exact)[0])
x_star = np.array([float(v) for v in x_exact])
KNOWN_OPTIMAL = float(obj_exact_rat)
print(f"exact rational optimum = {sp.nsimplify(obj_exact_rat)}  = {KNOWN_OPTIMAL!r}")
print(f"exact x* = {x_star}")

# --- pounce ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_qp(P=H, c=c, A=A, b=b)
t_pounce = time.perf_counter() - t0
x_p, obj_p, st_p = np.asarray(r.x, dtype=float), float(r.obj), r.status

# --- oracle 2: cvxpy / CLARABEL ---
import cvxpy as cp
xv = cp.Variable(N)
prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(H)) + c @ xv),
                  [A @ xv == b])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_cvx = time.perf_counter() - t0
x_c, obj_c = np.asarray(xv.value, dtype=float), float(prob.value)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


e_known = rel(obj_p, KNOWN_OPTIMAL)
e_cvx = rel(obj_p, obj_c)
x_err_p = float(np.linalg.norm(x_p - x_star, np.inf))
x_err_c = float(np.linalg.norm(x_c - x_star, np.inf))
feas_p = float(np.linalg.norm(A @ x_p - b, np.inf))
feas_c = float(np.linalg.norm(A @ x_c - b, np.inf))
# stationarity residual H x + c + A' l = 0, l recovered by least squares
lam_p = np.linalg.lstsq(A.T, -(H @ x_p + c), rcond=None)[0]
stat_p = float(np.linalg.norm(H @ x_p + c + A.T @ lam_p, np.inf))

print()
print("=== pounce ===")
print(f"status={st_p} obj={obj_p:.12e} t={t_pounce:.4f}s")
print(f"x={x_p}")
print(f"eq_feas_inf={feas_p:.3e}  stationarity_inf={stat_p:.3e}  x_inf_err_vs_exact={x_err_p:.3e}")
print("=== oracle: exact rational KKT (sympy) ===")
print(f"obj={KNOWN_OPTIMAL:.12e} t={t_exact:.4f}s")
print("=== oracle: cvxpy/CLARABEL ===")
print(f"status={prob.status} obj={obj_c:.12e} t={t_cvx:.4f}s")
print(f"eq_feas_inf={feas_c:.3e}  x_inf_err_vs_exact={x_err_c:.3e}")
print()
print(f"known_optimal={KNOWN_OPTIMAL:.12e} rel_err_vs_known={e_known:.3e}")
print(f"obj_err_vs_clarabel={e_cvx:.3e}")
print(f"x_inf_err: pounce={x_err_p:.3e}  clarabel={x_err_c:.3e}")

ok = (st_p == "optimal") and e_known < 1e-4 and feas_p < 1e-6
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={st_p}, rel_err_vs_known={e_known:.3e}, feas={feas_p:.3e})")
