#!/usr/bin/env python
"""Adversary test: inequality-constrained differentiable QP layer, d/dh.

Problem class: general inequality-constrained convex QP (G x <= h), NOT a
plain box QP, NOT equality-only, NOT a SOC. We differentiate the solution
w.r.t. the RHS h (and the cost c), with a PARTIALLY ACTIVE active set so
the gradient genuinely depends on which constraints bind (the active-set
structure matters):

    min  1/2 x^T P x + c^T x
    s.t. G x <= h                          (n = 3, 5 inequalities)

The arrow/KKT path differs from the box-only layer because constraints are
general half-spaces, several of which are slack and at least one active.

Reference gradient: cvxpy re-solve under central finite differences in
float64 (no clean closed form once the active set is what we're probing),
plus JAX<->Torch parity and torch.autograd.gradcheck on a float64 layer.

Checks: forward vs cvxpy AND vs pounce.solve_qp; analytic d(loss)/d{c,h}
vs FD; JAX<->Torch parity; gradcheck.
"""
import time
import numpy as np

np.random.seed(3)
TOL_FWD = 1e-6
TOL_FD = 5e-5
TOL_PARITY = 1e-6

n = 3
P_np = np.array([[2.0, 0.2, 0.0],
                 [0.2, 1.5, 0.1],
                 [0.0, 0.1, 1.0]])
c_np = np.array([-1.0, 0.5, 0.8])

# 5 general half-space constraints G x <= h. Designed so the unconstrained
# minimizer x_u = -P^{-1} c violates a couple of them -> active set nontrivial.
G_np = np.array([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [-1.0, 0.5, 0.0]])
h_np = np.array([0.3, 0.4, 1.0, 0.6, -0.85])

w_np = np.array([1.0, -0.5, 0.7])   # scalar loss = w . x*

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pounce
import pounce.jax as pj
import torch
import pounce.torch as pt
torch.set_default_dtype(torch.float64)
import cvxpy as cp


# ---------------------------------------------------------------------------
def cvx_solve(c, h):
    xv = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P_np)) + c @ xv)
    prob = cp.Problem(obj, [G_np @ xv <= h])
    prob.solve(solver=cp.CLARABEL)
    return np.asarray(xv.value)


# ===========================================================================
# 1. FORWARD
# ===========================================================================
x_np = np.asarray(pounce.solve_qp(P=P_np, c=c_np, G=G_np, h=h_np).x)
x_jax = np.asarray(pj.solve_qp(P=jnp.asarray(P_np), c=jnp.asarray(c_np),
                               G=jnp.asarray(G_np), h=jnp.asarray(h_np)))
x_torch = pt.solve_qp(P=torch.tensor(P_np), c=torch.tensor(c_np),
                      G=torch.tensor(G_np), h=torch.tensor(h_np)).detach().numpy()
x_cvx = cvx_solve(c_np, h_np)

fwd_jc = np.max(np.abs(x_jax - x_cvx))
fwd_tc = np.max(np.abs(x_torch - x_cvx))
fwd_jn = np.max(np.abs(x_jax - x_np))
print(f"x_np   = {x_np}")
print(f"x_jax  = {x_jax}")
print(f"x_torch= {x_torch}")
print(f"x_cvx  = {x_cvx}")
print(f"forward: jax-vs-cvxpy={fwd_jc:.3e}  torch-vs-cvxpy={fwd_tc:.3e}  jax-vs-pounce={fwd_jn:.3e}")
slack = h_np - G_np @ x_jax
active = np.abs(slack) < 1e-6
print(f"constraint slack = {slack}")
print(f"active set (slack~0) = {np.where(active)[0].tolist()}  ({active.sum()} active)")
forward_ok = max(fwd_jc, fwd_tc, fwd_jn) < TOL_FWD and active.sum() >= 1

# ===========================================================================
# 2. GRADIENT vs FINITE DIFFERENCE (jax), loss = w . x*
# ===========================================================================
def loss_jax(c, h):
    x = pj.solve_qp(P=jnp.asarray(P_np), c=c, G=jnp.asarray(G_np), h=h)
    return jnp.dot(jnp.asarray(w_np), x)


t0 = time.perf_counter()
gc, gh = jax.grad(loss_jax, argnums=(0, 1))(jnp.asarray(c_np), jnp.asarray(h_np))
t_jax = time.perf_counter() - t0
gc = np.asarray(gc); gh = np.asarray(gh)


def loss_cvx(c, h):
    return float(np.dot(w_np, cvx_solve(c, h)))


def fd_grad(f, x0, eps=1e-6):
    g = np.zeros_like(x0, dtype=float)
    for i in range(x0.size):
        xp = x0.astype(float).copy(); xm = x0.astype(float).copy()
        xp[i] += eps; xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


t0 = time.perf_counter()
fd_c = fd_grad(lambda c: loss_cvx(c, h_np), c_np)
fd_h = fd_grad(lambda h: loss_cvx(c_np, h), h_np)
t_fd = time.perf_counter() - t0
err_c = np.max(np.abs(gc - fd_c))
err_h = np.max(np.abs(gh - fd_h))
grad_fd_err = max(err_c, err_h)
print("=== gradient dL/d{c,h} ===")
print(f"jax  gc={gc}")
print(f"fd   gc={fd_c}")
print(f"jax  gh={gh}")
print(f"fd   gh={fd_h}")
print(f"grad-vs-FD: d/dc max err={err_c:.3e}  d/dh max err={err_h:.3e}")
fd_mag = max(np.max(np.abs(fd_c)), np.max(np.abs(fd_h)))
grad_fd_ok = grad_fd_err < TOL_FD

# ===========================================================================
# 3. JAX <-> TORCH parity
# ===========================================================================
ct = torch.tensor(c_np, requires_grad=True)
ht = torch.tensor(h_np, requires_grad=True)
xt = pt.solve_qp(P=torch.tensor(P_np), c=ct, G=torch.tensor(G_np), h=ht)
(torch.tensor(w_np) @ xt).backward()
tgc = ct.grad.numpy(); tgh = ht.grad.numpy()
par_c = np.max(np.abs(tgc - gc))
par_h = np.max(np.abs(tgh - gh))
parity_err = max(par_c, par_h)
print(f"jax/torch parity: d/dc {par_c:.3e}  d/dh {par_h:.3e}")
parity_ok = parity_err < TOL_PARITY

# ===========================================================================
# 4. torch.autograd.gradcheck
# ===========================================================================
Gt = torch.tensor(G_np)
Pt = torch.tensor(P_np)


def fn(c, h):
    return pt.solve_qp(P=Pt, c=c, G=Gt, h=h)


inputs = (torch.tensor(c_np, requires_grad=True),
          torch.tensor(h_np, requires_grad=True))
try:
    gradcheck_ok = torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-4,
                                             rtol=1e-3, raise_exception=True)
except Exception as e:
    gradcheck_ok = False
    print("gradcheck raised:", str(e)[:300])
print(f"gradcheck: {'PASS' if gradcheck_ok else 'FAIL'}")

# ===========================================================================
print("-" * 60)
print(f"forward_ok={forward_ok} grad_fd_ok={grad_fd_ok} "
      f"parity_ok={parity_ok} gradcheck_ok={gradcheck_ok}")
overall = forward_ok and grad_fd_ok and parity_ok and gradcheck_ok
if overall:
    print("VERDICT: PASS")
else:
    print(f"VERDICT: FAIL (fwd={max(fwd_jc,fwd_tc,fwd_jn):.2e} "
          f"grad_fd={grad_fd_err:.2e} parity={parity_err:.2e} gc={gradcheck_ok})")

print(f"# timings: t_jax_grad={t_jax:.3f}s t_fd={t_fd:.3f}s "
      f"max_fd_vs_analytic={grad_fd_err:.2e} max_parity={parity_err:.2e} fd_mag={fd_mag:.3e}")
