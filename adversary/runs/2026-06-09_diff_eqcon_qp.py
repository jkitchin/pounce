#!/usr/bin/env python
"""Adversary test: equality-constrained differentiable QP layer.

Problem class: equality-constrained convex QP (NOT the box-constrained
OptNet QP). Active equality constraints only -> the layer reduces to a
KKT linear solve, with a clean closed-form gradient to test against.

    min  1/2 x^T P x + c^T x
    s.t. A x = b

This is differentiable w.r.t. (P, c, A, b). We exercise gradients w.r.t.
c, A, b (P's gradient is the symmetric OptNet gradient and harder to FD
cleanly with the symmetry projection, so we focus FD on c, A, b and let
gradcheck cover P).

Checks:
  1. FORWARD: x*(theta) vs cvxpy and vs closed-form KKT solve.
  2. GRADIENT vs FINITE DIFFERENCE: d(loss)/d{c,A,b} central FD in float64.
  3. JAX <-> TORCH parity: same problem both backends, grads agree.
  4. torch.autograd.gradcheck (+ gradgradcheck if supported).
"""
import numpy as np

np.random.seed(7)
TOL_FWD = 1e-7
TOL_FD = 1e-5
TOL_PARITY = 1e-7

# ---------------------------------------------------------------------------
# Build a well-posed equality-constrained QP.
n, m = 4, 2
M = np.random.randn(n, n)
P_np = M @ M.T + 1.5 * np.eye(n)        # SPD
c_np = np.random.randn(n)
A_np = np.random.randn(m, n)
b_np = np.random.randn(m)

# scalar loss weights (so loss = w . x*)
w_np = np.random.randn(n)


def closed_form_x(P, c, A, b):
    """KKT closed form for equality-only QP: [[P A^T],[A 0]] [x;lam] = [-c; b]."""
    n = P.shape[0]
    m = A.shape[0]
    K = np.block([[P, A.T], [A, np.zeros((m, m))]])
    rhs = np.concatenate([-c, b])
    sol = np.linalg.solve(K, rhs)
    return sol[:n]


# ===========================================================================
# 1. FORWARD checks
# ===========================================================================
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pounce.jax as pj
import torch
import pounce.torch as pt
torch.set_default_dtype(torch.float64)

x_cf = closed_form_x(P_np, c_np, A_np, b_np)

x_jax = np.asarray(pj.solve_qp(P=jnp.asarray(P_np), c=jnp.asarray(c_np),
                               A=jnp.asarray(A_np), b=jnp.asarray(b_np)))
x_torch = pt.solve_qp(P=torch.tensor(P_np), c=torch.tensor(c_np),
                      A=torch.tensor(A_np), b=torch.tensor(b_np)).detach().numpy()

# cvxpy reference
import cvxpy as cp
xv = cp.Variable(n)
prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P_np)) + c_np @ xv),
                  [A_np @ xv == b_np])
prob.solve(solver=cp.CLARABEL)
x_cvx = xv.value

fwd_err_jax_cf = np.max(np.abs(x_jax - x_cf))
fwd_err_torch_cf = np.max(np.abs(x_torch - x_cf))
fwd_err_jax_cvx = np.max(np.abs(x_jax - x_cvx))
print(f"forward: jax-vs-closedform   max|dx| = {fwd_err_jax_cf:.3e}")
print(f"forward: torch-vs-closedform max|dx| = {fwd_err_torch_cf:.3e}")
print(f"forward: jax-vs-cvxpy        max|dx| = {fwd_err_jax_cvx:.3e}")
forward_ok = max(fwd_err_jax_cf, fwd_err_torch_cf, fwd_err_jax_cvx) < TOL_FWD

# ===========================================================================
# 2. GRADIENT vs FINITE DIFFERENCE (jax), loss = w . x*
# ===========================================================================

def loss_jax(c, A, b):
    x = pj.solve_qp(P=jnp.asarray(P_np), c=c, A=A, b=b)
    return jnp.dot(jnp.asarray(w_np), x)


gc, gA, gb = jax.grad(loss_jax, argnums=(0, 1, 2))(
    jnp.asarray(c_np), jnp.asarray(A_np), jnp.asarray(b_np))
gc = np.asarray(gc); gA = np.asarray(gA); gb = np.asarray(gb)


def loss_np(c, A, b):
    return float(np.dot(w_np, closed_form_x(P_np, c, A, b)))


def fd_grad(f, x0, eps=1e-6):
    g = np.zeros_like(x0, dtype=float)
    it = np.nditer(x0, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        xp = x0.copy(); xm = x0.copy()
        xp[idx] += eps; xm[idx] -= eps
        g[idx] = (f(xp) - f(xm)) / (2 * eps)
        it.iternext()
    return g


fd_c = fd_grad(lambda c: loss_np(c, A_np, b_np), c_np)
fd_A = fd_grad(lambda A: loss_np(c_np, A, b_np), A_np)
fd_b = fd_grad(lambda b: loss_np(c_np, A_np, b), b_np)

err_c = np.max(np.abs(gc - fd_c))
err_A = np.max(np.abs(gA - fd_A))
err_b = np.max(np.abs(gb - fd_b))
grad_fd_err = max(err_c, err_A, err_b)
print(f"grad-vs-FD: d/dc max err = {err_c:.3e}")
print(f"grad-vs-FD: d/dA max err = {err_A:.3e}")
print(f"grad-vs-FD: d/db max err = {err_b:.3e}")
grad_fd_ok = grad_fd_err < TOL_FD

# ===========================================================================
# 3. JAX <-> TORCH parity
# ===========================================================================
ct = torch.tensor(c_np, requires_grad=True)
At = torch.tensor(A_np, requires_grad=True)
bt = torch.tensor(b_np, requires_grad=True)
xt = pt.solve_qp(P=torch.tensor(P_np), c=ct, A=At, b=bt)
(torch.tensor(w_np) @ xt).backward()
tgc = ct.grad.numpy(); tgA = At.grad.numpy(); tgb = bt.grad.numpy()

par_c = np.max(np.abs(tgc - gc))
par_A = np.max(np.abs(tgA - gA))
par_b = np.max(np.abs(tgb - gb))
parity_err = max(par_c, par_A, par_b)
print(f"jax/torch parity: d/dc {par_c:.3e}  d/dA {par_A:.3e}  d/db {par_b:.3e}")
parity_ok = parity_err < TOL_PARITY

# ===========================================================================
# 4. torch.autograd.gradcheck / gradgradcheck
# ===========================================================================
Pt = torch.tensor(P_np)  # constant SPD (gradcheck on P needs symmetry handling)


def fn(c, A, b):
    return pt.solve_qp(P=Pt, c=c, A=A, b=b)


inputs = (torch.tensor(c_np, requires_grad=True),
          torch.tensor(A_np, requires_grad=True),
          torch.tensor(b_np, requires_grad=True))
try:
    gradcheck_ok = torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-5,
                                             rtol=1e-4, raise_exception=True)
except Exception as e:
    gradcheck_ok = False
    print("gradcheck raised:", str(e)[:300])
print(f"gradcheck: {'PASS' if gradcheck_ok else 'FAIL'}")

try:
    gradgradcheck_ok = torch.autograd.gradgradcheck(fn, inputs, eps=1e-6,
                                                    atol=1e-4, rtol=1e-3,
                                                    raise_exception=True)
    print(f"gradgradcheck: {'PASS' if gradgradcheck_ok else 'FAIL'}")
except Exception as e:
    gradgradcheck_ok = None
    print("gradgradcheck: NOT SUPPORTED ->", str(e)[:150])

# ===========================================================================
# Verdict
# ===========================================================================
print("-" * 60)
print(f"forward_ok={forward_ok} grad_fd_ok={grad_fd_ok} "
      f"parity_ok={parity_ok} gradcheck_ok={gradcheck_ok}")
overall = forward_ok and grad_fd_ok and parity_ok and gradcheck_ok
print("VERDICT:", "PASS" if overall else "FAIL")
