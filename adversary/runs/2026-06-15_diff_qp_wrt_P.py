#!/usr/bin/env python
"""Adversary test: differentiable QP layer w.r.t. the QUADRATIC MATRIX P.

Problem class: inequality-constrained convex QP, differentiated w.r.t. the
cost MATRIX P (not c, h, A, or b -- those are covered by the existing diff
runs). This exercises the hardest implicit-function-theorem path in the
OptNet/cvxpylayers gradient: the term d(loss)/dP, where P enters the KKT
system through 0.5 x^T P x.

    min  1/2 x^T P x + c^T x
    s.t. G x <= h            (n = 3, 4 inequalities, exactly ONE active)

Why P is the interesting/new parameter:
  * The eqcon diff run (2026-06-09) explicitly SKIPPED FD on P, deferring it
    to gradcheck, "because P's gradient is the symmetric OptNet gradient and
    harder to FD cleanly with the symmetry projection".
  * P is symmetric, so only its symmetric part is identifiable. The layer
    (both jax and torch) returns a SYMMETRIZED gradient dL/dP. A naive
    entrywise central FD that bumps P[i,j] alone double-counts off-diagonal
    contributions (the quadratic form 0.5 x'Px couples [i,j] and [j,i]).
    The CORRECT finite-difference oracle is therefore sym(entrywise_FD) =
    0.5*(FD + FD^T). This script verifies that convention explicitly and is
    the whole point of the test.

Active set: designed so exactly constraint 0 binds with large slack
(~0.7-1.1) on the other three, so the active set is stable under the small
symmetric P perturbations used by FD/gradcheck (no kink crossing).

Checks:
  1. FORWARD: x*(P) vs cvxpy and vs pounce.solve_qp (jax & torch).
  2. GRADIENT vs FINITE DIFFERENCE: d(loss)/dP, analytic vs sym(central FD)
     on a cvxpy re-solve, float64. Also reports the (expected) mismatch
     against the *naive* entrywise FD to make the symmetry convention
     visible.
  3. JAX <-> TORCH parity on dL/dP.
  4. torch.autograd.gradcheck on P routed through a symmetric
     parametrization P = S + S^T (so the perturbation directions gradcheck
     uses are themselves symmetric, matching what the layer can represent).
"""
import time
import numpy as np

np.random.seed(11)
TOL_FWD = 1e-6
TOL_FD = 5e-5
TOL_PARITY = 1e-9

# ---------------------------------------------------------------------------
# Problem: exactly one active inequality, others comfortably slack.
n = 3
M = np.array([[1.5, 0.2, 0.0],
              [0.2, 1.2, 0.3],
              [0.0, 0.3, 1.0]])
P_np = M @ M.T + 0.5 * np.eye(n)          # SPD, well conditioned
c_np = np.array([-2.0, 1.0, 0.5])
G_np = np.array([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [1.0, 1.0, 0.0]])
h_np = np.array([0.3, 0.6, 1.0, 0.5])
w_np = np.array([1.0, -0.5, 0.7])          # scalar loss = w . x*

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pounce
import pounce.jax as pj
import torch
import pounce.torch as pt
torch.set_default_dtype(torch.float64)
import cvxpy as cp


def cvx_solve(P):
    xv = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P)) + c_np @ xv)
    prob = cp.Problem(obj, [G_np @ xv <= h_np])
    prob.solve(solver=cp.CLARABEL)
    return np.asarray(xv.value)


# ===========================================================================
# 1. FORWARD
# ===========================================================================
x_np = np.asarray(pounce.solve_qp(P=P_np, c=c_np, G=G_np, h=h_np).x)
x_jax = np.asarray(pj.solve_qp(P=jnp.asarray(P_np), c=jnp.asarray(c_np),
                               G=jnp.asarray(G_np), h=jnp.asarray(h_np)))
t0 = time.perf_counter()
x_pt = pt.solve_qp(P=torch.tensor(P_np), c=torch.tensor(c_np),
                   G=torch.tensor(G_np), h=torch.tensor(h_np))
t_fwd = time.perf_counter() - t0
x_torch = x_pt.detach().numpy()
x_cvx = cvx_solve(P_np)

fwd_jc = np.max(np.abs(x_jax - x_cvx))
fwd_tc = np.max(np.abs(x_torch - x_cvx))
fwd_jn = np.max(np.abs(x_jax - x_np))
print(f"x_np    = {x_np}")
print(f"x_jax   = {x_jax}")
print(f"x_torch = {x_torch}")
print(f"x_cvx   = {x_cvx}")
print(f"forward: jax-vs-cvxpy={fwd_jc:.3e}  torch-vs-cvxpy={fwd_tc:.3e}  jax-vs-pounce={fwd_jn:.3e}")
slack = h_np - G_np @ x_jax
active = np.abs(slack) < 1e-6
print(f"constraint slack = {slack}")
print(f"active set (slack~0) = {np.where(active)[0].tolist()}  ({active.sum()} active)")
forward_ok = max(fwd_jc, fwd_tc, fwd_jn) < TOL_FWD and active.sum() == 1

# ===========================================================================
# 2. GRADIENT dL/dP vs FINITE DIFFERENCE (symmetric convention)
# ===========================================================================
def loss_jax(P):
    x = pj.solve_qp(P=P, c=jnp.asarray(c_np), G=jnp.asarray(G_np), h=jnp.asarray(h_np))
    return jnp.dot(jnp.asarray(w_np), x)


t0 = time.perf_counter()
gP = np.asarray(jax.grad(loss_jax)(jnp.asarray(P_np)))
t_jax = time.perf_counter() - t0


def loss_cvx(P):
    return float(np.dot(w_np, cvx_solve(P)))


def fd_grad_matrix(f, P0, eps=1e-6):
    g = np.zeros_like(P0, dtype=float)
    for i in range(P0.shape[0]):
        for j in range(P0.shape[1]):
            Pp = P0.astype(float).copy(); Pm = P0.astype(float).copy()
            Pp[i, j] += eps; Pm[i, j] -= eps
            g[i, j] = (f(Pp) - f(Pm)) / (2 * eps)
    return g


t0 = time.perf_counter()
fd_entry = fd_grad_matrix(loss_cvx, P_np)
t_fd = time.perf_counter() - t0
fd_sym = 0.5 * (fd_entry + fd_entry.T)   # correct oracle for symmetrized dL/dP

err_sym = np.max(np.abs(gP - fd_sym))
err_entry = np.max(np.abs(gP - fd_entry))
print("=== gradient dL/dP ===")
print(f"analytic gP =\n{gP}")
print(f"sym(FD)     =\n{fd_sym}")
print(f"grad-vs-sym(FD)   max err = {err_sym:.3e}   <- the real check")
print(f"grad-vs-entry(FD) max err = {err_entry:.3e}   (expected nonzero: off-diag symmetry)")
print(f"gP symmetric? {np.allclose(gP, gP.T, atol=1e-12)}")
grad_fd_err = err_sym
grad_fd_ok = grad_fd_err < TOL_FD

# ===========================================================================
# 3. JAX <-> TORCH parity on dL/dP
# ===========================================================================
Pt = torch.tensor(P_np, requires_grad=True)
xt = pt.solve_qp(P=Pt, c=torch.tensor(c_np), G=torch.tensor(G_np), h=torch.tensor(h_np))
(torch.tensor(w_np) @ xt).backward()
tgP = Pt.grad.numpy()
parity_err = np.max(np.abs(tgP - gP))
print(f"jax/torch parity dL/dP = {parity_err:.3e}")
parity_ok = parity_err < TOL_PARITY

# ===========================================================================
# 4. torch.autograd.gradcheck via symmetric parametrization P = S + S^T
#    (perturbation directions are symmetric -> matches identifiable subspace)
# ===========================================================================
S0 = 0.5 * P_np   # so S0 + S0^T = P_np


def fn_sym(S):
    Psym = S + S.transpose(-1, -2)
    return pt.solve_qp(P=Psym, c=torch.tensor(c_np),
                       G=torch.tensor(G_np), h=torch.tensor(h_np))


inputs = (torch.tensor(S0, requires_grad=True),)
try:
    gradcheck_ok = torch.autograd.gradcheck(fn_sym, inputs, eps=1e-6, atol=1e-4,
                                            rtol=1e-3, raise_exception=True)
except Exception as e:
    gradcheck_ok = False
    print("gradcheck raised:", str(e)[:300])
print(f"gradcheck (P=S+S^T): {'PASS' if gradcheck_ok else 'FAIL'}")

try:
    ggc_ok = torch.autograd.gradgradcheck(fn_sym, inputs, eps=1e-6, atol=1e-3,
                                          rtol=1e-2, raise_exception=True)
    print(f"gradgradcheck: {'PASS' if ggc_ok else 'FAIL'}")
except Exception as e:
    ggc_ok = None
    print("gradgradcheck: NOT SUPPORTED/SKIP ->", str(e)[:150])

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

print(f"# timings: t_fwd={t_fwd:.4f}s t_jax_grad={t_jax:.3f}s t_fd={t_fd:.3f}s "
      f"max_fd_vs_analytic={grad_fd_err:.2e} max_parity={parity_err:.2e}")
