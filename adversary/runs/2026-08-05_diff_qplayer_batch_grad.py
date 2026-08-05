"""Adversary cross-check: differentiable QpLayer.batch() gradient correctness
Family: diff   Class: fixed-structure batched QP layer (rayon-parallel), gradient

Prior diff-family probes exercise pj.solve_qp / pt.solve_qp directly, jax.vmap
of a single-item solve, and pounce.solve_qp_batch -- but never the reusable
QpLayer's own `.batch()` method (a distinct entry point: fixed P/G/A captured
once, batch of `c` vectors solved via the rayon-parallel batch driver, and
required to be differentiable w.r.t. the batched `c`). This probe targets
exactly that path.

Problem: fixed n=3 box+inequality-constrained convex QP structure,
    min  1/2 x^T P x + c^T x   s.t.  Gx <= h,  lb <= x <= ub
shared across a batch of B=6 different `c` vectors (loss = sum_i w_i . x_i*).

Checks:
  1. FORWARD: QpLayer.batch(cs) == loop of QpLayer(c_i) for each item (bit-close).
  2. GRADIENT vs per-item loop: d(sum loss)/d(cs) from one vmapped/batched
     jax.grad call, vs summing independent jax.grad calls through the
     single-item __call__ (which does NOT share the batch driver) -- these
     exercise different Rust code paths and must agree.
  3. GRADIENT vs central finite difference (float64) on two batch items.
  4. JAX <-> Torch parity (pounce.torch.QpLayer.batch has the same contract).
"""
import numpy as np

np.random.seed(20260805)
n, mI = 3, 2
B = 6

M = np.random.randn(n, n)
P0 = M @ M.T + 1.5 * np.eye(n)          # SPD
G0 = np.random.randn(mI, n)
h0 = np.abs(np.random.randn(mI)) + 1.0  # loose enough to keep some slack
lb0 = -3.0 * np.ones(n)
ub0 = 3.0 * np.ones(n)

C = np.random.randn(B, n)               # batch of B different c vectors
W = np.random.randn(B, n)               # loss weights per item

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pounce.jax as pj
import torch
import pounce.torch as pt
torch.set_default_dtype(torch.float64)

layer_jax = pj.QpLayer(P=jnp.asarray(P0), G=jnp.asarray(G0), lb=jnp.asarray(lb0),
                        ub=jnp.asarray(ub0))
layer_torch = pt.QpLayer(P=torch.tensor(P0), G=torch.tensor(G0),
                          lb=torch.tensor(lb0), ub=torch.tensor(ub0))

h_jax = jnp.asarray(h0)
h_torch = torch.tensor(h0)

# ===========================================================================
# 1. FORWARD: batch call vs loop of single calls
# ===========================================================================
Cj = jnp.asarray(C)
x_batch = np.asarray(layer_jax.batch(Cj, h=h_jax))
x_loop = np.stack([np.asarray(layer_jax(Cj[i], h=h_jax)) for i in range(B)])
fwd_err = float(np.max(np.abs(x_batch - x_loop)))
print(f"forward: batch-vs-loop max|dx| = {fwd_err:.3e}")

# cvxpy reference for item 0 as an outside sanity check
import cvxpy as cp
xv = cp.Variable(n)
prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P0)) + C[0] @ xv),
                   [G0 @ xv <= h0, xv >= lb0, xv <= ub0])
prob.solve(solver=cp.CLARABEL)
fwd_err_cvx = float(np.max(np.abs(x_batch[0] - xv.value)))
print(f"forward: batch[0]-vs-cvxpy max|dx| = {fwd_err_cvx:.3e}")

# ===========================================================================
# 2. GRADIENT: batched jax.grad vs summed per-item jax.grad
# ===========================================================================
Wj = jnp.asarray(W)


def loss_batch(cs):
    xs = layer_jax.batch(cs, h=h_jax)
    return jnp.sum(xs * Wj)


g_batch = np.asarray(jax.grad(loss_batch)(Cj))


def loss_item(c, w):
    x = layer_jax(c, h=h_jax)
    return jnp.dot(w, x)


g_loop = np.stack([
    np.asarray(jax.grad(loss_item)(Cj[i], Wj[i])) for i in range(B)
])
grad_path_err = float(np.max(np.abs(g_batch - g_loop)))
print(f"gradient: batch-vs-perloop max|dg| = {grad_path_err:.3e}")

# ===========================================================================
# 3. GRADIENT vs central finite difference (two items)
# ===========================================================================
eps = 1e-6
fd_errs = []
for i in (0, B - 1):
    for k in range(n):
        Cp = C.copy(); Cp[i, k] += eps
        Cm = C.copy(); Cm[i, k] -= eps
        xp = np.asarray(layer_jax.batch(jnp.asarray(Cp), h=h_jax))
        xm = np.asarray(layer_jax.batch(jnp.asarray(Cm), h=h_jax))
        lp = float(np.sum(xp * W))
        lm = float(np.sum(xm * W))
        fd = (lp - lm) / (2 * eps)
        fd_errs.append(abs(fd - g_batch[i, k]))
fd_err = max(fd_errs)
print(f"gradient: batch-vs-FD max|dg| = {fd_err:.3e}")

# ===========================================================================
# 4. JAX <-> Torch parity
# ===========================================================================
Ct = torch.tensor(C, requires_grad=True)
Wt = torch.tensor(W)
xs_t = layer_torch.batch(Ct, h=h_torch)
loss_t = torch.sum(xs_t * Wt)
loss_t.backward()
g_torch = Ct.grad.detach().numpy()

x_fwd_parity = float(np.max(np.abs(np.asarray(xs_t.detach().numpy()) - x_batch)))
g_parity = float(np.max(np.abs(g_torch - g_batch)))
print(f"parity: jax-vs-torch forward max|dx| = {x_fwd_parity:.3e}")
print(f"parity: jax-vs-torch gradient max|dg| = {g_parity:.3e}")

TOL_FWD = 1e-9
TOL_GRAD_PATH = 1e-8
TOL_FD = 5e-5
TOL_PARITY = 1e-7

ok = (
    fwd_err < TOL_FWD
    and fwd_err_cvx < 1e-6
    and grad_path_err < TOL_GRAD_PATH
    and fd_err < TOL_FD
    and x_fwd_parity < TOL_PARITY
    and g_parity < TOL_PARITY
)
print("VERDICT: PASS" if ok else "VERDICT: FAIL "
      f"(fwd={fwd_err:.2e} cvx={fwd_err_cvx:.2e} gpath={grad_path_err:.2e} "
      f"fd={fd_err:.2e} parity_x={x_fwd_parity:.2e} parity_g={g_parity:.2e})")
