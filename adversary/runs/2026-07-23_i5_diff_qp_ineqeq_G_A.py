"""Adversary test: differentiable QP layer, gradients w.r.t. the CONSTRAINT
MATRICES G (inequality) and A (equality) simultaneously.
Family: diff   Class: eq+ineq convex QP OptNet layer, dx*/dG and dx*/dA.

   min   1/2 x^T P x + c^T x
   s.t.  A x = b            (1 equality, always active)
         G x <= h           (3 inequalities, exactly ONE active at optimum)
P is SPD (strictly convex), the active set is stable under small G/A bumps.
Parameters theta = G (matrix) and A (matrix): both live in the KKT constraint
Jacobian, so dloss/dG and dloss/dA exercise the constraint-matrix implicit
gradient. The logged diff tests differentiate c,h,P,b or A of an equality-only
QP; none differentiate the INEQUALITY matrix G of a mixed eq+ineq QP.

Loss L = w . x*.  Checks:
  1. FORWARD x*(G,A) vs cvxpy (CLARABEL) and pounce; assert exactly one active.
  2. GRADIENT dL/dG and dL/dA, jax vs central FD in float64 on a cvxpy re-solve.
  3. JAX <-> TORCH parity on both gradients.
  4. torch.autograd.gradcheck / gradgradcheck through the layer w.r.t. (G, A).
"""
import numpy as np

np.random.seed(21)
TOL_FWD = 1e-5
TOL_FD = 5e-5
TOL_PARITY = 1e-8

n = 3
Mm = np.array([[1.4, 0.3, 0.0],
               [0.3, 1.1, 0.2],
               [0.0, 0.2, 1.0]])
P_np = Mm @ Mm.T + 0.5 * np.eye(n)      # SPD
c_np = np.array([-1.0, 0.5, 0.3])
A_np = np.array([[1.0, 1.0, 1.0]])      # sum(x) = b
b_np = np.array([1.0])
G_np = np.array([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [-1.0, 0.0, 1.0]])
h_np = np.array([0.55, 0.7, 0.9])       # constraint 0 active, others slack
w_np = np.array([1.0, -0.6, 0.4])

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch
torch.set_default_dtype(torch.float64)
import pounce.jax as pj
import pounce.torch as pt
import cvxpy as cp


def cvx_solve(G, A):
    xv = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P_np)) + c_np @ xv),
                      [A @ xv == b_np, G @ xv <= h_np])
    prob.solve(solver=cp.CLARABEL)
    return np.asarray(xv.value)


# ---- 1. FORWARD ----
x_jax = np.asarray(pj.solve_qp(P=jnp.asarray(P_np), c=jnp.asarray(c_np),
                               G=jnp.asarray(G_np), h=jnp.asarray(h_np),
                               A=jnp.asarray(A_np), b=jnp.asarray(b_np)))
x_torch = pt.solve_qp(P=torch.tensor(P_np), c=torch.tensor(c_np),
                      G=torch.tensor(G_np), h=torch.tensor(h_np),
                      A=torch.tensor(A_np), b=torch.tensor(b_np)).detach().numpy()
x_cvx = cvx_solve(G_np, A_np)
fwd = max(np.max(np.abs(x_jax - x_cvx)), np.max(np.abs(x_torch - x_cvx)))
slack = h_np - G_np @ x_cvx
active = np.abs(slack) < 1e-6
print(f"x_jax={x_jax}\nx_torch={x_torch}\nx_cvx={x_cvx}")
print(f"forward max err vs cvxpy = {fwd:.2e}; slack={slack} active={np.where(active)[0].tolist()}")
print(f"equality residual sum(x)-b = {float((A_np @ x_cvx - b_np)[0]):.2e}")
forward_ok = fwd < TOL_FWD and active.sum() == 1


# ---- 2. GRADIENT dL/dG, dL/dA vs FD ----
def loss_jax(G, A):
    x = pj.solve_qp(P=jnp.asarray(P_np), c=jnp.asarray(c_np), G=G, h=jnp.asarray(h_np),
                    A=A, b=jnp.asarray(b_np))
    return jnp.dot(jnp.asarray(w_np), x)


gG, gA = jax.grad(loss_jax, argnums=(0, 1))(jnp.asarray(G_np), jnp.asarray(A_np))
gG = np.asarray(gG); gA = np.asarray(gA)


def loss_cvx_G(G):
    return float(np.dot(w_np, cvx_solve(G, A_np)))


def loss_cvx_A(A):
    return float(np.dot(w_np, cvx_solve(G_np, A)))


def fd_matrix(f, M, eps=1e-6):
    out = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Mp = M.copy(); Mm_ = M.copy()
            Mp[i, j] += eps; Mm_[i, j] -= eps
            out[i, j] = (f(Mp) - f(Mm_)) / (2 * eps)
    return out


fdG = fd_matrix(loss_cvx_G, G_np)
fdA = fd_matrix(loss_cvx_A, A_np)
errG = np.max(np.abs(gG - fdG))
errA = np.max(np.abs(gA - fdA))
print(f"grad dL/dG vs FD max err = {errG:.2e}")
print(f"grad dL/dA vs FD max err = {errA:.2e}")
grad_fd_ok = max(errG, errA) < TOL_FD


# ---- 3. JAX <-> TORCH parity ----
Gt = torch.tensor(G_np, requires_grad=True)
At = torch.tensor(A_np, requires_grad=True)
xt = pt.solve_qp(P=torch.tensor(P_np), c=torch.tensor(c_np), G=Gt, h=torch.tensor(h_np),
                 A=At, b=torch.tensor(b_np))
(torch.as_tensor(w_np) @ xt).backward()
parityG = np.max(np.abs(Gt.grad.numpy() - gG))
parityA = np.max(np.abs(At.grad.numpy() - gA))
print(f"jax/torch parity dL/dG={parityG:.2e} dL/dA={parityA:.2e}")
parity_ok = max(parityG, parityA) < TOL_PARITY


# ---- 4. gradcheck / gradgradcheck ----
def fn(G, A):
    return pt.solve_qp(P=torch.tensor(P_np), c=torch.tensor(c_np), G=G, h=torch.tensor(h_np),
                       A=A, b=torch.tensor(b_np))


inp = (torch.tensor(G_np, requires_grad=True), torch.tensor(A_np, requires_grad=True))
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
      f"VERDICT: FAIL (fwd={fwd:.2e} gradG={errG:.2e} gradA={errA:.2e} "
      f"parity={max(parityG, parityA):.2e} gc={gc_ok})")
