"""Adversary cross-check: pounce.jax differentiable QP layer, dx/dtheta
Family: diff   Class: QP OptNet-style implicit differentiation (equality-only)
Source: Amos & Kolter, "OptNet: Differentiable Optimization as a Layer in
        Neural Networks", ICML 2017, sec 3 (implicit-function KKT gradient).
        The pounce implementation is pounce.jax._qp.solve_qp
        (python/pounce/jax/_qp.py); oracle is a central finite difference of
        an independent re-solve via the plain host API pounce.qp.solve_qp
        (NOT the jax layer under test), i.e. the oracle never touches the
        differentiable machinery being tested.
Known optimal: none published; this is a gradient-correctness check, not an
        objective-value check (per adversary.md's `diff` family notes: check
        the layer's dx/dtheta against a central finite difference of a
        re-solve, float64).

Problem: min 1/2 x^T P x + c(theta)^T x  s.t.  1^T x = 1,  c(theta) = c0 + theta*e0
(a parametrized minimum-variance-style QP; theta perturbs the linear cost on x0).
"""
import time
import numpy as np

P = np.array([
    [4.0, 0.4, 0.2],
    [0.4, 3.0, 0.3],
    [0.2, 0.3, 2.0],
])
c0 = np.array([0.1, -0.2, 0.05])
A_eq = np.ones((1, 3))
b_eq = np.array([1.0])
e0 = np.array([1.0, 0.0, 0.0])
THETA0 = 0.3

# --- pounce.jax differentiable layer: dx/dtheta via autodiff (analytic KKT) ---
import jax
import jax.numpy as jnp
from pounce.jax import solve_qp as jax_solve_qp


def x_of_theta(theta):
    c = jnp.asarray(c0) + theta * jnp.asarray(e0)
    return jax_solve_qp(P=P, c=c, A=A_eq, b=b_eq)


t0 = time.perf_counter()
x_at_theta0 = np.asarray(x_of_theta(THETA0))
dx_dtheta_ad = np.asarray(jax.jacobian(x_of_theta)(THETA0))
t_pounce = time.perf_counter() - t0
print("=== pounce.jax layer ===")
print(f"x(theta0)={x_at_theta0}")
print(f"dx/dtheta (autodiff)={dx_dtheta_ad}  t={t_pounce:.4f}s")

# --- oracle: central finite difference via the plain (non-diff) host API ---
from pounce import solve_qp as host_solve_qp

DELTA = 1e-6


def x_host(theta):
    c = c0 + theta * e0
    r = host_solve_qp(P=P, c=c, A=A_eq, b=b_eq)
    assert r.status == "optimal", r.status
    return np.asarray(r.x)


t0 = time.perf_counter()
x_plus = x_host(THETA0 + DELTA)
x_minus = x_host(THETA0 - DELTA)
dx_dtheta_fd = (x_plus - x_minus) / (2 * DELTA)
t_oracle = time.perf_counter() - t0
print("=== oracle (central finite difference, host solve_qp) ===")
print(f"dx/dtheta (FD)={dx_dtheta_fd}  t={t_oracle:.4f}s")

# --- closed-form cross-check: equality-only QP has an exact KKT solution ---
# [P A^T; A 0] [x; y] = [-c; b]  =>  dx/dtheta = -Kinv[:n,:n] @ e0 (since only c depends on theta)
n = 3
K = np.block([[P, A_eq.T], [A_eq, np.zeros((1, 1))]])
Kinv = np.linalg.inv(K)
dx_dtheta_closed = (-Kinv[:n, :n] @ e0)
print(f"dx/dtheta (closed-form KKT)={dx_dtheta_closed}")

err_ad_vs_fd = float(np.max(np.abs(dx_dtheta_ad - dx_dtheta_fd)))
err_ad_vs_closed = float(np.max(np.abs(dx_dtheta_ad - dx_dtheta_closed)))
err_fd_vs_closed = float(np.max(np.abs(dx_dtheta_fd - dx_dtheta_closed)))
x_host_check = x_host(THETA0)
err_x_fwd = float(np.max(np.abs(x_at_theta0 - x_host_check)))

print(f"max|ad-fd|={err_ad_vs_fd:.2e} max|ad-closed|={err_ad_vs_closed:.2e} max|fd-closed|={err_fd_vs_closed:.2e}")
print(f"forward x layer-vs-host max_err={err_x_fwd:.2e}")

ok = err_ad_vs_fd < 1e-5 and err_ad_vs_closed < 1e-8 and err_x_fwd < 1e-8
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (err_ad_vs_fd={err_ad_vs_fd:.2e}, err_ad_vs_closed={err_ad_vs_closed:.2e})")
