"""Adversary cross-check: box-constrained least squares (diagonal QP).
Family: qp   Class: box-constrained least squares (separable diagonal QP)
Source: Closed-form KKT (derivable). For a SEPARABLE diagonal QP the
solution decouples coordinate-wise into a clamp of the unconstrained
minimizer onto its box.

  min  0.5 * sum_i w_i (x_i - t_i)^2   s.t.  lb_i <= x_i <= ub_i

For a separable objective each coordinate is independent, so the optimum
is the projection of the unconstrained minimizer t_i onto [lb_i, ub_i]:
    x_i* = clamp(t_i, lb_i, ub_i)
This is exact (closed form), giving an analytic KNOWN_OPTIMAL.

We pick data where some coordinates are interior, some hit lb, some hit ub,
so the active set is nontrivial.

In pounce form: 0.5 x'Px + c'x with P = diag(w), c = -w*t.
Objective reported by pounce omits the constant 0.5*sum w_i t_i^2; we add it.
"""
import time
import numpy as np

np.random.seed(7)
n = 50
w = np.random.uniform(0.5, 3.0, n)          # positive weights -> P PD
t = np.random.uniform(-5.0, 5.0, n)          # unconstrained targets
lb = np.full(n, -2.0)
ub = np.full(n, 2.0)

# Closed-form optimum: clamp targets onto box.
X_STAR = np.clip(t, lb, ub)
CONST = 0.5 * np.sum(w * t * t)
# obj value (full quadratic incl const) at x*:
KNOWN_OPTIMAL = 0.5 * np.sum(w * (X_STAR - t) ** 2)

P = np.diag(w)
c = -w * t

# --- pounce convex QP IPM ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_qp(P=P, c=c, lb=lb, ub=ub)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)
obj_pounce = r.obj + CONST
status = r.status

# --- oracle: cvxpy/CLARABEL ---
import cvxpy as cp
xv = cp.Variable(n)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum(cp.multiply(w, cp.square(xv - t)))),
                  [xv >= lb, xv <= ub])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle, obj_oracle = np.asarray(xv.value), prob.value


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))
x_err_known = float(np.linalg.norm(x_pounce - X_STAR, np.inf))
n_at_lb = int(np.sum(np.isclose(X_STAR, lb)))
n_at_ub = int(np.sum(np.isclose(X_STAR, ub)))
n_interior = n - n_at_lb - n_at_ub

# PSD sanity check
eig_min = float(np.min(np.linalg.eigvalsh(P)))

print("=== problem ===")
print(f"n={n} P_min_eig={eig_min:.3e} (PD if >0); active: lb={n_at_lb} ub={n_at_ub} interior={n_interior}")
print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s iters={r.iters}")
print("=== oracle (cvxpy/CLARABEL) ===")
print(f"obj={obj_oracle:.10e} t={t_oracle:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"x_err_vs_known(inf)={x_err_known:.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err_vs_oracle={x_err:.2e}")

ok = ((status == "optimal" or r.success)
      and obj_err < 1e-4
      and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4
      and x_err_known < 1e-4)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, x_err_known={x_err_known:.2e})")
