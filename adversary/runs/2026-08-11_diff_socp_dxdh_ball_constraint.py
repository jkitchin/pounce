"""Adversary cross-check: differentiable SOCP layer, gradient w.r.t. h
(cone right-hand side), via a linear-cost/fixed-radius-ball problem with an
EXACT closed-form Jacobian -- not just a finite-difference oracle.
Family: diff   Class: SOCP layer dx*/dh (prior diff/SOCP probes covered
dP, dc, dG, dA for the QP layer and dG for the SOCP layer; dh for the SOCP
layer is untested in the log so far).
Source: closed-form solution of "minimize c'x s.t. ||x-a||_2 <= r" is the
        projection of the unconstrained linear-cost direction onto the
        ball boundary:
            x*(a) = a - r * c / ||c||_2         (independent of a's value)
            obj*(a) = c'x*(a) = c'a - r*||c||_2
        so the EXACT Jacobians are known in closed form:
            dx*/da = I_n              (identity -- translating the ball
                                        center translates the optimum by
                                        exactly the same amount)
            dObj/da = c                (constant vector)
        The parameter 'a' enters only through h (h = [r; a], with G's
        first row all-zero and remaining rows = I_n so that
        s = h - Gx = (r, a - x)), so this isolates dx*/dh specifically.

Checked three ways: (1) JAX autodiff vs the exact closed-form Jacobian,
(2) JAX autodiff vs central finite differences of a fresh re-solve, (3)
forward solve x* vs the closed-form point directly.
"""
import numpy as np
import jax
import jax.numpy as jnp

import pounce.jax as pjax

np.random.seed(21)
n = 4
r = 2.5
c_np = np.random.uniform(-1.5, 1.5, n)
c = jnp.asarray(c_np, dtype=jnp.float64)
a0 = jnp.asarray(np.random.uniform(-1.0, 1.0, n), dtype=jnp.float64)

P = jnp.zeros((n, n), dtype=jnp.float64)
G = jnp.concatenate([jnp.zeros((1, n), dtype=jnp.float64), jnp.eye(n, dtype=jnp.float64)], axis=0)


def x_star(a):
    h = jnp.concatenate([jnp.array([r], dtype=jnp.float64), a])
    return pjax.solve_socp(P=P, c=c, G=G, h=h, cones=[("soc", n + 1)])


def obj(a):
    return c @ x_star(a)


# --- forward solve vs closed form ---
c_norm = float(np.linalg.norm(c_np))
x_closed = np.asarray(a0) - r * c_np / c_norm
obj_closed = float(c_np @ x_closed)

x_pounce = np.asarray(x_star(a0))
obj_pounce = float(obj(a0))

fwd_err = float(np.linalg.norm(x_pounce - x_closed, np.inf))

# --- JAX autodiff ---
grad_obj_autodiff = np.asarray(jax.grad(obj)(a0))
jac_x_autodiff = np.asarray(jax.jacobian(x_star)(a0))

grad_obj_closed = c_np                       # dObj/da = c
jac_x_closed = np.eye(n)                     # dx*/da = I

grad_err_closed = float(np.linalg.norm(grad_obj_autodiff - grad_obj_closed, np.inf))
jac_err_closed = float(np.linalg.norm(jac_x_autodiff - jac_x_closed, np.inf))

# --- central finite-difference cross-check of the gradient ---
eps = 1e-5
grad_fd = np.zeros(n)
for i in range(n):
    ap = np.asarray(a0).copy(); ap[i] += eps
    am = np.asarray(a0).copy(); am[i] -= eps
    grad_fd[i] = (float(obj(jnp.asarray(ap))) - float(obj(jnp.asarray(am)))) / (2 * eps)

grad_err_fd = float(np.linalg.norm(grad_obj_autodiff - grad_fd, np.inf))

print("=== forward solve ===")
print(f"x_pounce={x_pounce}")
print(f"x_closed_form={x_closed}")
print(f"obj_pounce={obj_pounce:.10e} obj_closed_form={obj_closed:.10e}")
print(f"forward_x_inf_err={fwd_err:.2e}")
print("=== gradient dObj/da ===")
print(f"autodiff={grad_obj_autodiff}")
print(f"closed_form(=c)={grad_obj_closed}")
print(f"finite_diff={grad_fd}")
print(f"grad_err_vs_closed_form={grad_err_closed:.2e} grad_err_vs_finite_diff={grad_err_fd:.2e}")
print("=== jacobian dx*/da ===")
print(f"jac_err_vs_identity={jac_err_closed:.2e}")

ok = (
    fwd_err < 1e-6
    and grad_err_closed < 1e-6
    and grad_err_fd < 1e-4
    and jac_err_closed < 1e-5
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (fwd_err={fwd_err:.2e}, grad_err_closed={grad_err_closed:.2e}, grad_err_fd={grad_err_fd:.2e}, jac_err_closed={jac_err_closed:.2e})")
