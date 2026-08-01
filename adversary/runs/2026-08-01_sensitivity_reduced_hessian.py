"""Adversary cross-check: QpSensitivity.reduced_hessian() vs independent
null-space projection.
Family: sensitivity   Class: reduced-Hessian / active-manifold curvature.
    Fresh QpSensitivity surface -- prior sensitivity runs (active_inequality,
    parametric_qp, sipopt_cli/shared_param, active_bound, active_ineq_rhs,
    degenerate_weakly_active, duals_vs_sensitivity, illcond_hilbert,
    nearparallel_licq_fd, barrier_curvature) all probed parametric_step
    (dx/db); none exercised reduced_hessian() (Z'PZ + eigendecomposition
    on the active manifold).
Source: reduced-Hessian second-order theory is standard NLP/QP background
    (Nocedal & Wright, "Numerical Optimization" 2nd ed., Sec 16.3, "Direct
    solution of the linear system" / Sec 18.2 the reduced Hessian and SOSC).
    The independent oracle below builds the active-constraint Jacobian,
    takes an orthonormal null-space basis Z via numpy SVD, and forms
    Z'^T P Z' directly -- a completely different code path from pounce's
    Rust QR/null-space implementation. Eigenvalues of Z^T P Z are invariant
    to the choice of orthonormal basis Z for a fixed subspace (any two such
    bases are related by an orthogonal Q, and Z2 = Z1 Q => Z2'PZ2 =
    Q'(Z1'PZ1)Q, an orthogonal similarity transform, which preserves the
    spectrum exactly) -- so comparing sorted eigenvalues is basis-free and
    exact, not an approximate cross-check.
Known optimal: none published (custom instance); validated by 3 independent
    signals: (1) eigenvalue match vs numpy null-space projection, (2) all
    reduced-Hessian eigenvalues > 0 confirms strict SOSC at a point that is
    independently confirmed optimal by cvxpy/CLARABEL, (3) n_dof matches
    n - rank(active Jacobian) computed independently via numpy.
"""
import time
import numpy as np

n = 4
P = np.array([
    [6.0, 2.0, 1.0, 0.0],
    [2.0, 5.0, 2.0, 1.0],
    [1.0, 2.0, 4.0, 1.0],
    [0.0, 1.0, 1.0, 3.0],
])
assert np.allclose(P, P.T)
eig_P = np.linalg.eigvalsh(P)
assert eig_P[0] > 0, f"P not PD: eig={eig_P}"   # independent PD check

c = np.array([-2.0, -3.0, -1.0, -4.0])

A = np.array([[1.0, 1.0, 1.0, 1.0]])
b = np.array([3.0])
G = np.array([[1.0, 0.0, 0.0, -1.0]])   # x0 - x3 <= h
h = np.array([0.2])
lb = np.array([0.0, 0.0, 0.0, 0.0])
ub = np.array([np.inf, 1.0, np.inf, np.inf])   # x1 <= 1 (likely active)

from pounce.qp import QpSensitivity
t0 = time.perf_counter()
s = QpSensitivity(P=P, c=c, A=A, b=b, G=G, h=h, lb=lb, ub=ub)
t_pounce = time.perf_counter() - t0
x_star = s.x
act = s.active_indices
print(f"x*={x_star} obj={s.obj:.10e} t={t_pounce:.4f}s")
print(f"active inequalities={act.inequalities} active bounds={act.bounds}")

rh = s.reduced_hessian()
eig_pounce = np.sort(rh.eigenvalues)
print(f"pounce reduced_hessian: n_dof={rh.n_dof} eigenvalues={eig_pounce}")

# --- independent oracle: build active-constraint Jacobian by hand, project ---
rows = [A[0]]                      # equality always active
for i in act.inequalities:
    rows.append(G[i])
for j in act.bounds:
    e = np.zeros(n)
    e[j] = 1.0
    rows.append(e)
J = np.array(rows)
rank_J = np.linalg.matrix_rank(J, tol=1e-9)

# orthonormal null-space basis via SVD: null(J) = columns of V corresponding
# to (numerically) zero singular values.
U, sv, Vt = np.linalg.svd(J)
tol = 1e-9 * max(J.shape) * (sv[0] if len(sv) else 1.0)
n_dof_oracle = n - int(np.sum(sv > tol))
Z = Vt[int(np.sum(sv > tol)):, :].T   # n x n_dof, orthonormal columns

H_R = Z.T @ P @ Z
eig_oracle = np.sort(np.linalg.eigvalsh(H_R))

# --- second oracle: confirm x* is the true optimum via cvxpy ---
import cvxpy as cp

xv = cp.Variable(n)
cons = [A @ xv == b, G @ xv <= h, xv >= lb, xv <= ub]
prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(xv, P) + c @ xv), cons)
prob.solve(solver=cp.CLARABEL)
x_cvx = np.asarray(xv.value, float)

def rel(a, ref):
    return abs(a - ref) / max(1.0, abs(ref))

x_err_cvx = float(np.linalg.norm(x_star - x_cvx, np.inf))
n_dof_match = (rh.n_dof == n_dof_oracle) and (n - rank_J == n_dof_oracle)
eig_err = float(np.max(np.abs(eig_pounce - eig_oracle))) if len(eig_pounce) == len(eig_oracle) else np.inf
all_pos = bool(np.all(eig_pounce > -1e-8))

print(f"oracle (numpy SVD null-space projection): n_dof={n_dof_oracle} rank(J)={rank_J} "
      f"eigenvalues={eig_oracle}")
print(f"oracle cvxpy/CLARABEL x={x_cvx} obj={prob.value:.10e}")
print(f"x_err_vs_cvx={x_err_cvx:.2e} n_dof_match={n_dof_match} eig_max_abs_err={eig_err:.2e} "
      f"all_eig_positive={all_pos}")

ok = x_err_cvx < 1e-6 and n_dof_match and eig_err < 1e-6 and all_pos
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (x_err_cvx={x_err_cvx:.2e}, n_dof_match={n_dof_match}, "
      f"eig_err={eig_err:.2e}, all_pos={all_pos})")
