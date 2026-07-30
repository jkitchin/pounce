"""Adversary cross-check: hard-margin SVM dual QP
Family: qp   Class: kernel Gram-matrix QP, box + single equality (new class for
qp; prior qp runs used Markowitz eq+box, HS76 ineq+bounded, HS28 equality-only,
Hilbert near-singular equality, ill-conditioned Vandermonde box, singular-PSD
status battery, closed-form-KKT eq+ineq, duals/multipliers battery -- none
used a dense (non-diagonal) Gram-matrix P or the SVM margin duality identity).

Problem: 4-point, 2-class, linearly-separable dataset, symmetric about the
origin so the dual optimum is derivable in closed form.
    X = [(-2,-1), (-2,1), (2,-1), (2,1)],  y = (-1, -1, +1, +1)

Hard-margin SVM dual (linear kernel K = X X^T):
    minimize   0.5 alpha^T P alpha - sum(alpha_i)      P_ij = y_i y_j K_ij
    subject to y^T alpha = 0,   0 <= alpha_i <= C   (C = 1, non-binding)

By symmetry the maximum-margin separator is w = (0.5, 0), b = 0 (separating
plane x = 0, margin width 4, so ||w|| = 2/4 = 0.5); ALL FOUR points are
support vectors with equal alpha_i = a. Stationarity (P alpha)_i = 1 for all i
(worked out by hand below) gives a = 1/16 = 0.0625, and checking the KKT
gradient explicitly: (P alpha)_i - 1 = mu * y_i for every i requires mu = 0
(each row of P alpha sums to 16a = 1 for all four rows by the matrix's
symmetry), which is satisfied exactly at a = 0.0625 -- confirming this is the
true stationary point, not just a plausible guess.

Known dual objective (minimize form): 0.5*alpha^T P alpha - sum(alpha)
    = 0.5*(4 * a * 1) - 4*a = 0.5*0.25 - 0.25 = -0.125
Strong duality check: primal optimum = 0.5*||w||^2 = 0.5*0.25 = 0.125 = -KNOWN_OPTIMAL.

SOURCE: standard hard-margin SVM dual (Boser, Guyon & Vapnik 1992; Cortes &
Vapnik 1995; see also Boyd & Vandenberghe Ch.8 for the max-margin/QP
correspondence). Closed-form solution is this run's own hand derivation from
the dataset's symmetry (verified below by an explicit KKT-gradient check, not
just asserted), cross-checked independently against cvxpy.

KNOWN_OPTIMAL: -0.125   ALPHA_STAR: (0.0625, 0.0625, 0.0625, 0.0625)
"""
import time
import numpy as np

X = np.array([[-2.0, -1.0], [-2.0, 1.0], [2.0, -1.0], [2.0, 1.0]])
y = np.array([-1.0, -1.0, 1.0, 1.0])
C = 1.0
KNOWN_OPTIMAL = -0.125
ALPHA_STAR = np.full(4, 0.0625)

K = X @ X.T
P = np.outer(y, y) * K
c = -np.ones(4)
A = y.reshape(1, -1)
b = np.array([0.0])
lb = np.zeros(4)
ub = np.full(4, C)

# Sanity: confirm the hand-derived stationary point actually zeroes the KKT
# gradient (P alpha - 1 == mu*y with mu==0), i.e. the "known optimal" isn't
# a typo -- this is a check on OUR formulation, not on pounce.
grad = P @ ALPHA_STAR - 1.0
assert np.allclose(grad, 0.0, atol=1e-10), f"hand derivation is wrong: grad={grad}"

# --- pounce ---
from pounce import solve_qp

t0 = time.perf_counter()
r = solve_qp(P=P, c=c, A=A, b=b, lb=lb, ub=ub)
t_pounce = time.perf_counter() - t0
alpha_pounce = np.asarray(r.x)
obj_pounce = float(r.obj)
w_pounce = (alpha_pounce * y) @ X

# --- oracle: cvxpy (CLARABEL) ---
import cvxpy as cp

av = cp.Variable(4)
prob = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(av, P) - cp.sum(av)),
    [y @ av == 0, av >= 0, av <= C],
)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_cvx = time.perf_counter() - t0
obj_cvx = float(prob.value)
alpha_cvx = np.asarray(av.value)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_cvx = rel(obj_pounce, obj_cvx)
cvx_vs_known = rel(obj_cvx, KNOWN_OPTIMAL)
alpha_err = float(np.linalg.norm(alpha_pounce - ALPHA_STAR, np.inf))
w_err = float(np.linalg.norm(w_pounce - np.array([0.5, 0.0]), np.inf))

print("=== hard-margin SVM dual QP, 4 points, 2 classes ===")
print(f"KNOWN_OPTIMAL={KNOWN_OPTIMAL:.10f}  ALPHA_STAR={ALPHA_STAR}")
print("-- pounce (solve_qp) --")
print(f"status={r.status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"alpha={alpha_pounce}  w=(w_x,w_y)={w_pounce}")
print("-- cvxpy / CLARABEL --")
print(f"status={prob.status} obj={obj_cvx:.10e} t={t_cvx:.4f}s alpha={alpha_cvx}")
print(f"obj_err vs known    = {obj_err_known:.2e}")
print(f"obj_err vs cvxpy    = {obj_err_cvx:.2e}")
print(f"cvxpy vs known      = {cvx_vs_known:.2e}")
print(f"alpha_inf_err vs known = {alpha_err:.2e}")
print(f"w_inf_err vs (0.5,0)   = {w_err:.2e}")

TOL = 1e-6
ok = (
    r.status == "optimal"
    and obj_err_known < TOL
    and obj_err_cvx < TOL
    and alpha_err < 1e-5
    and w_err < 1e-5
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={r.status}, obj_err_known={obj_err_known:.2e})")
