#!/usr/bin/env python
"""Adversary test: differentiable SOCP layer (norm-ball constrained QP).

Problem class: second-order-cone program with a single ACTIVE SOC
constraint (NOT a box QP, NOT an equality QP). Projection of a point onto
a ball, expressed as an SOCP:

    min  1/2 ||x||^2 + c^T x
    s.t. || x - center || <= r            (one SOC of dim n+1)

In the layer's standard form  min 1/2 x^T P x + c^T x  s.t.  Gx <=_K h
with P = I and the single cone slack s = h - G x in SOC(n+1):
    s0   = r                 -> G row 0 = 0,  h0 = r
    s1:  = center - x        -> G rows  = I,  h1: = center

Differentiable w.r.t. (c, h) where h carries r and center. We check
d(loss)/d{c,h} so the cone-active gradient (the arrow operator path) is
exercised, not a trivial inactive constraint.

Reference gradient: cvxpy projection re-solve under central finite
differences in float64 (no clean closed form once c shifts the center of
projection, so FD on an independent cvxpy solve is the oracle).

Checks: forward vs cvxpy; analytic grad vs FD; JAX<->Torch parity;
torch.autograd.gradcheck.
"""
import numpy as np

np.random.seed(0)
TOL_FWD = 1e-6
TOL_FD = 5e-5
TOL_PARITY = 1e-6

n = 3
center = np.array([2.0, 1.0, -1.5])
r = 0.8
c_np = np.array([0.3, -0.2, 0.1])

P_np = np.eye(n)
G_np = np.vstack([np.zeros((1, n)), np.eye(n)])   # constant
h_np = np.concatenate([[r], center])              # differentiable
cones = [("soc", n + 1)]
w_np = np.array([1.0, -0.5, 0.7])                 # loss = w . x*
MAXIT = 200

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pounce.jax as pj
import torch
import pounce.torch as pt
torch.set_default_dtype(torch.float64)
import cvxpy as cp


# ---------------------------------------------------------------------------
# cvxpy reference solve as a function of (c, h) -> x*
# h = [r, center]
def cvx_solve(c, h):
    rr = float(h[0])
    cen = np.asarray(h[1:])
    xv = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.sum_squares(xv) + c @ xv)
    prob = cp.Problem(obj, [cp.SOC(cp.Constant(rr), xv - cen)])
    prob.solve(solver=cp.CLARABEL)
    return np.asarray(xv.value)


# ===========================================================================
# 1. FORWARD
# ===========================================================================
x_jax = np.asarray(pj.solve_socp(P=jnp.asarray(P_np), c=jnp.asarray(c_np),
                                 G=jnp.asarray(G_np), h=jnp.asarray(h_np),
                                 cones=cones, max_iter=MAXIT))
x_torch = pt.solve_socp(P=torch.tensor(P_np), c=torch.tensor(c_np),
                        G=torch.tensor(G_np), h=torch.tensor(h_np),
                        cones=cones, max_iter=MAXIT).detach().numpy()
x_cvx = cvx_solve(c_np, h_np)

fwd_j = np.max(np.abs(x_jax - x_cvx))
fwd_t = np.max(np.abs(x_torch - x_cvx))
print(f"forward: jax-vs-cvxpy   max|dx| = {fwd_j:.3e}")
print(f"forward: torch-vs-cvxpy max|dx| = {fwd_t:.3e}")
# confirm cone is active
s = h_np - G_np @ x_jax
print(f"cone slack: s0={s[0]:.4f}  ||s_rest||={np.linalg.norm(s[1:]):.4f} (active if equal)")
forward_ok = max(fwd_j, fwd_t) < TOL_FWD

# ===========================================================================
# 2. GRADIENT vs FINITE DIFFERENCE (jax), loss = w . x*
# ===========================================================================

def loss_jax(c, h):
    x = pj.solve_socp(P=jnp.asarray(P_np), c=c, G=jnp.asarray(G_np), h=h,
                      cones=cones, max_iter=MAXIT)
    return jnp.dot(jnp.asarray(w_np), x)


gc, gh = jax.grad(loss_jax, argnums=(0, 1))(jnp.asarray(c_np), jnp.asarray(h_np))
gc = np.asarray(gc); gh = np.asarray(gh)


def loss_cvx(c, h):
    return float(np.dot(w_np, cvx_solve(c, h)))


def fd_grad(f, x0, eps=1e-6):
    g = np.zeros_like(x0, dtype=float)
    for i in range(x0.size):
        xp = x0.astype(float).copy().ravel(); xm = xp.copy()
        xp[i] += eps; xm[i] -= eps
        g.ravel()[i] = (f(xp.reshape(x0.shape)) - f(xm.reshape(x0.shape))) / (2 * eps)
    return g


fd_c = fd_grad(lambda c: loss_cvx(c, h_np), c_np)
fd_h = fd_grad(lambda h: loss_cvx(c_np, h), h_np)
err_c = np.max(np.abs(gc - fd_c))
err_h = np.max(np.abs(gh - fd_h))
grad_fd_err = max(err_c, err_h)
print(f"grad-vs-FD: d/dc max err = {err_c:.3e}")
print(f"grad-vs-FD: d/dh max err = {err_h:.3e}")
grad_fd_ok = grad_fd_err < TOL_FD

# ===========================================================================
# 3. JAX <-> TORCH parity
# ===========================================================================
ct = torch.tensor(c_np, requires_grad=True)
ht = torch.tensor(h_np, requires_grad=True)
xt = pt.solve_socp(P=torch.tensor(P_np), c=ct, G=torch.tensor(G_np), h=ht,
                   cones=cones, max_iter=MAXIT)
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
    return pt.solve_socp(P=Pt, c=c, G=Gt, h=h, cones=cones, max_iter=MAXIT)


inputs = (torch.tensor(c_np, requires_grad=True),
          torch.tensor(h_np, requires_grad=True))
try:
    gradcheck_ok = torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-4,
                                             rtol=1e-3, raise_exception=True)
except Exception as e:
    gradcheck_ok = False
    print("gradcheck raised:", str(e)[:300])
print(f"gradcheck: {'PASS' if gradcheck_ok else 'FAIL'}")

try:
    ggc = torch.autograd.gradgradcheck(fn, inputs, eps=1e-6, atol=1e-3,
                                       rtol=1e-2, raise_exception=True)
    print(f"gradgradcheck: {'PASS' if ggc else 'FAIL'}")
except Exception as e:
    print("gradgradcheck: NOT SUPPORTED ->", str(e)[:120])

# ===========================================================================
print("-" * 60)
print(f"forward_ok={forward_ok} grad_fd_ok={grad_fd_ok} "
      f"parity_ok={parity_ok} gradcheck_ok={gradcheck_ok}")
overall = forward_ok and grad_fd_ok and parity_ok and gradcheck_ok
print("VERDICT:", "PASS" if overall else "FAIL")
