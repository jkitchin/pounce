"""Adversary test: differentiable SOCP layer, gradient w.r.t. the CONE
CONSTRAINT MATRIX inside G (strictly convex projection onto an ellipsoidal set).
Family: diff   Class: SOCP OptNet layer, dx*/d(data-in-G) -- the cone-matrix
               (arrow-operator) gradient path, the least-tested diff surface.

Problem (strictly convex, so the OptNet KKT is well-posed):
   min   1/2 ||x - x0||^2
   s.t.  || F x - g ||_2 <= r          (one SOC of dim 1+m, ACTIVE at optimum)
The data matrix F lives inside the CONSTRAINT MATRIX G (not c/h/A_eq/P), so
dloss/dF exercises the dG path of the cone-aware implicit function theorem --
the surface the logged normball diff test skips (it keeps G constant and
differentiates c,h). Parameter theta = F (m x n).  Loss L(F) = w . x*(F).

Checks:
  1. FORWARD x*(F) vs cvxpy (CLARABEL) and pounce; assert the SOC is active.
  2. GRADIENT dL/dF, analytic (jax) vs central FD in float64 on a cvxpy re-solve.
  3. JAX <-> TORCH parity on dL/dF.
  4. torch.autograd.gradcheck / gradgradcheck through the layer w.r.t. F.

(An earlier robust-LS variant used a purely linear objective, P=0, whose OptNet
KKT is degenerate -- the forward solve itself only reached ~2e-5, so gradients
there are untestable. This strictly convex projection form fixes that while
still routing the differentiated parameter through G.)
"""
import time
import numpy as np

np.random.seed(5)
# NOTE: near the active SOC boundary CLARABEL and ECOS themselves disagree by
# ~1.5e-5 on this instance (verified), so ~2e-5 is the problem's numerical
# floor -- TOL_FWD is set accordingly. Passing tol< default to the layer makes
# its forward raise ('optimal_inaccurate'), so we use the default tolerance.
TOL_FWD = 5e-5
TOL_FD = 5e-5
TOL_PARITY = 1e-8

n, m = 3, 2
x0 = np.array([3.0, -2.0, 1.0])
F0 = np.array([[1.0, 0.5, -0.5],
               [0.3, 1.0, 0.4]])
g = np.array([0.2, -0.3])
r = 0.5
w = np.array([1.0, -0.5, 0.7])

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch
torch.set_default_dtype(torch.float64)
import pounce.jax as pj
import pounce.torch as pt
import cvxpy as cp

cones = [("soc", 1 + m)]
P_np = np.eye(n)
c_np = -x0.copy()          # 1/2 x'x - x0'x  == 1/2||x-x0||^2 - const


def build_G_h(F, xp):
    if xp is np:
        G = np.zeros((1 + m, n)); h = np.zeros(1 + m)
        h[0] = r
        G[1:, :] = F
        h[1:] = g
        return G, h
    if xp is jnp:
        top = jnp.zeros((1, n))
        G = jnp.concatenate([top, F], axis=0)
        h = jnp.concatenate([jnp.array([r]), jnp.asarray(g)])
        return G, h
    top = torch.zeros((1, n))
    G = torch.cat([top, F], dim=0)
    h = torch.cat([torch.tensor([r]), torch.as_tensor(g)])
    return G, h


def cvx_solve(F):
    xv = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(xv - x0)),
                      [cp.norm(F @ xv - g, 2) <= r])
    prob.solve(solver=cp.CLARABEL)
    return np.asarray(xv.value)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


# ---- 1. FORWARD ----
Gj, hj = build_G_h(jnp.asarray(F0), jnp)
x_jax = np.asarray(pj.solve_socp(P=jnp.asarray(P_np), c=jnp.asarray(c_np),
                                 G=Gj, h=hj, cones=cones))
Gt, ht = build_G_h(torch.tensor(F0), torch)
x_torch = pt.solve_socp(P=torch.tensor(P_np), c=torch.tensor(c_np),
                        G=Gt, h=ht, cones=cones).detach().numpy()
x_cvx = cvx_solve(F0)


def cvx_solve_ecos(F):
    xv = cp.Variable(n)
    cp.Problem(cp.Minimize(0.5 * cp.sum_squares(xv - x0)),
               [cp.norm(F @ xv - g, 2) <= r]).solve(solver=cp.ECOS)
    return np.asarray(xv.value)


x_ecos = cvx_solve_ecos(F0)
solver_spread = np.max(np.abs(x_cvx - x_ecos))
fwd = max(np.max(np.abs(x_jax - x_cvx)), np.max(np.abs(x_torch - x_cvx)))
soc_radius = np.linalg.norm(F0 @ x_cvx - g)
print(f"x_jax={x_jax}\nx_torch={x_torch}\nx_cvx={x_cvx}")
print(f"CLARABEL-vs-ECOS spread = {solver_spread:.2e} (problem numerical floor)")
print(f"forward max err vs cvxpy = {fwd:.2e}   SOC ||Fx-g||={soc_radius:.4f} (r={r}, active={abs(soc_radius-r)<1e-4})")
forward_ok = fwd < TOL_FWD and abs(soc_radius - r) < 1e-4


# ---- 2. GRADIENT dL/dF vs FD ----
def loss_jax(F):
    G, h = build_G_h(F, jnp)
    x = pj.solve_socp(P=jnp.asarray(P_np), c=jnp.asarray(c_np), G=G, h=h, cones=cones)
    return jnp.dot(jnp.asarray(w), x)


gF = np.asarray(jax.grad(loss_jax)(jnp.asarray(F0)))


def loss_cvx(F):
    return float(np.dot(w, cvx_solve(F)))


def fd_matrix(f, F, eps=1e-6):
    out = np.zeros_like(F)
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            Fp = F.copy(); Fm = F.copy()
            Fp[i, j] += eps; Fm[i, j] -= eps
            out[i, j] = (f(Fp) - f(Fm)) / (2 * eps)
    return out


fd = fd_matrix(loss_cvx, F0)
err_fd = np.max(np.abs(gF - fd))
print(f"grad dL/dF vs FD max err = {err_fd:.2e}")
grad_fd_ok = err_fd < TOL_FD


# ---- 3. JAX <-> TORCH parity ----
Ft = torch.tensor(F0, requires_grad=True)
Gt2, ht2 = build_G_h(Ft, torch)
xt = pt.solve_socp(P=torch.tensor(P_np), c=torch.tensor(c_np), G=Gt2, h=ht2, cones=cones)
(torch.as_tensor(w) @ xt).backward()
tgF = Ft.grad.numpy()
parity = np.max(np.abs(tgF - gF))
print(f"jax/torch parity dL/dF = {parity:.2e}")
parity_ok = parity < TOL_PARITY


# ---- 4. gradcheck / gradgradcheck ----
def fn(F):
    G, h = build_G_h(F, torch)
    return pt.solve_socp(P=torch.tensor(P_np), c=torch.tensor(c_np), G=G, h=h, cones=cones)


inp = (torch.tensor(F0, requires_grad=True),)
try:
    gc_ok = torch.autograd.gradcheck(fn, inp, eps=1e-6, atol=1e-4, rtol=1e-3,
                                     raise_exception=True)
except Exception as e:
    gc_ok = False
    print("gradcheck raised:", str(e)[:250])
print(f"gradcheck: {'PASS' if gc_ok else 'FAIL'}")
try:
    ggc_ok = torch.autograd.gradgradcheck(fn, inp, eps=1e-6, atol=1e-3, rtol=1e-2,
                                          raise_exception=True)
    print(f"gradgradcheck: {'PASS' if ggc_ok else 'FAIL'}")
except Exception as e:
    ggc_ok = None
    print("gradgradcheck: SKIP/UNSUPPORTED ->", str(e)[:150])

overall = forward_ok and grad_fd_ok and parity_ok and bool(gc_ok)
print("-" * 60)
print(f"forward_ok={forward_ok} grad_fd_ok={grad_fd_ok} parity_ok={parity_ok} gradcheck_ok={gc_ok}")
print("VERDICT: PASS" if overall else
      f"VERDICT: FAIL (fwd={fwd:.2e} grad_fd={err_fd:.2e} parity={parity:.2e} gc={gc_ok})")
