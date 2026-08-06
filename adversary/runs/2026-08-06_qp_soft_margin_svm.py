"""Adversary cross-check: Soft-margin (C-SVM) dual QP
Family: qp   Class: box+equality convex QP (kernel Gram matrix, bounded box)
Source: Cortes & Vapnik (1995), "Support-Vector Networks"; standard C-SVM
        dual, e.g. Boyd & Vandenberghe / Bishop PRML Sec. 7.1.1, Eq. (7.10)-(7.11):
            max_a  sum_i a_i - 1/2 sum_ij a_i a_j y_i y_j K(x_i,x_j)
            s.t.   sum_i a_i y_i = 0,   0 <= a_i <= C
Known optimal: none published in closed form for this instance (small,
non-separable 2D dataset requiring slack) -- validated purely against the
independent cvxpy oracle, per the family table (qp oracle = cvxpy).
"""
import time
import numpy as np

# --- small 2D, non-separable dataset (needs C-box / slack) ---
X = np.array([
    [1.0, 1.0],
    [2.0, 1.0],
    [1.5, 3.0],
    [-1.0, -1.0],
    [-2.0, -1.0],
    [-0.5, 1.2],   # near the margin, on the wrong side -> forces slack
], dtype=float)
y = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
C = 1.0
n = len(y)

K = X @ X.T                      # linear kernel Gram matrix
P = np.outer(y, y) * K           # dual QP Hessian: (y y^T) o K
P = 0.5 * (P + P.T)              # symmetrize (roundoff)
c = -np.ones(n)                  # maximize sum a_i  ->  minimize -sum a_i
A = y.reshape(1, n)
b = np.array([0.0])
lb = np.zeros(n)
ub = np.full(n, C)

# --- pounce ---
from pounce import solve_qp
t0 = time.perf_counter()
r = solve_qp(P=P, c=c, A=A, b=b, lb=lb, ub=ub)
t_pounce = time.perf_counter() - t0
x_pounce, obj_pounce, status = r.x, r.obj, r.status

# --- oracle: cvxpy ---
import cvxpy as cp
a = cp.Variable(n)
objective = cp.Minimize(0.5 * cp.quad_form(a, cp.psd_wrap(P)) + c @ a)
constraints = [A @ a == b, a >= lb, a <= ub]
prob = cp.Problem(objective, constraints)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle, obj_oracle = a.value, prob.value


def rel(u, v):
    return abs(u - v) / max(1.0, abs(v))


obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(np.asarray(x_pounce) - np.asarray(x_oracle), np.inf))

# Recover primal SVM solution (w, b) from alphas and cross-check margin/slack
# structure is consistent (w computed both ways must agree).
w_pounce = (x_pounce * y) @ X
w_oracle = (x_oracle * y) @ X
w_err = float(np.linalg.norm(w_pounce - w_oracle, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"alpha={np.array2string(x_pounce, precision=6)}")
print("=== oracle (cvxpy/CLARABEL) ===")
print(f"obj={obj_oracle:.10e} t={t_oracle:.4f}s")
print(f"alpha={np.array2string(np.asarray(x_oracle), precision=6)}")
print(f"obj_err_vs_oracle={obj_err:.2e} alpha_inf_err={x_err:.2e} w_inf_err={w_err:.2e}")

# box/equality feasibility of pounce's own solution, independent check
feas_box = np.all(x_pounce >= lb - 1e-6) and np.all(x_pounce <= ub + 1e-6)
feas_eq = abs(float((A @ x_pounce - b)[0])) < 1e-6
print(f"feasible: box={feas_box} eq={feas_eq}")

ok = (status == "optimal") and obj_err < 1e-4 and x_err < 1e-4 and feas_box and feas_eq
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, x_err={x_err:.2e})")
