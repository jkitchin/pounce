"""Adversary cross-check: differentiable QP layer dx/dtheta
Family: diff   Class: differentiable convex QP w.r.t. the linear term c
Source: implicit-function-theorem layer (OptNet). Forward solve checked vs
  pounce.solve_qp; gradient dL/dtheta checked vs central finite differences
  AND JAX<->Torch parity; plus torch.autograd.gradcheck (float64).
  Problem: min 0.5||x||^2 + theta'x  s.t.  -0.5 <= x <= 0.5  (n=2).
  Unconstrained x* = -theta, clamped to the box. With theta=(1.0, 0.2):
    x* = (-0.5 [pinned at lb], -0.2 [interior]); L=sum(x*) => dL/dtheta=(0,-1).
Known: analytic dL/dtheta = (0, -1).
"""
import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch

P = np.eye(2)
lb = np.array([-0.5, -0.5])
ub = np.array([0.5, 0.5])
theta0 = np.array([1.0, 0.2])
ANALYTIC_GRAD = np.array([0.0, -1.0])

import pounce
import pounce.jax as pj
import pounce.torch as pt

# --- forward solve, numpy reference ---
r_np = pounce.solve_qp(P=P, c=theta0, lb=lb, ub=ub)
x_ref = np.asarray(r_np.x)

# --- JAX: forward + grad of L=sum(x*) wrt theta ---
def loss_jax(theta):
    x = pj.solve_qp(P=jnp.asarray(P), c=theta, lb=jnp.asarray(lb), ub=jnp.asarray(ub))
    return jnp.sum(x)

t0 = time.perf_counter()
g_jax = np.asarray(jax.grad(loss_jax)(jnp.asarray(theta0)))
t_jax = time.perf_counter() - t0
x_jax = np.asarray(pj.solve_qp(P=jnp.asarray(P), c=jnp.asarray(theta0),
                               lb=jnp.asarray(lb), ub=jnp.asarray(ub)))

# --- Torch: forward + grad ---
Pt = torch.tensor(P, dtype=torch.float64)
lbt = torch.tensor(lb, dtype=torch.float64)
ubt = torch.tensor(ub, dtype=torch.float64)
th = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
t0 = time.perf_counter()
x_t = pt.solve_qp(P=Pt, c=th, lb=lbt, ub=ubt)
L = x_t.sum()
L.backward()
t_torch = time.perf_counter() - t0
g_torch = th.grad.detach().numpy()
x_torch = x_t.detach().numpy()

# --- central finite difference (independent oracle) ---
def L_np(theta):
    return float(np.sum(pounce.solve_qp(P=P, c=theta, lb=lb, ub=ub).x))

eps = 1e-6
g_fd = np.zeros(2)
for i in range(2):
    tp = theta0.copy(); tp[i] += eps
    tm = theta0.copy(); tm[i] -= eps
    g_fd[i] = (L_np(tp) - L_np(tm)) / (2 * eps)


def ninf(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), np.inf))

fwd_err = ninf(x_jax, x_ref)
parity_x = ninf(x_jax, x_torch)
parity_g = ninf(g_jax, g_torch)
g_jax_vs_fd = ninf(g_jax, g_fd)
g_vs_analytic = ninf(g_jax, ANALYTIC_GRAD)

print("=== forward ===")
print(f"x_ref(np)={x_ref}  x_jax={x_jax}  x_torch={x_torch}")
print(f"fwd_err(jax vs np)={fwd_err:.2e}  parity_x(jax vs torch)={parity_x:.2e}")
print("=== gradient dL/dtheta ===")
print(f"jax={g_jax}  torch={g_torch}  fd={g_fd}  analytic={ANALYTIC_GRAD}")
print(f"parity_grad(jax vs torch)={parity_g:.2e}  jax_vs_fd={g_jax_vs_fd:.2e}  "
      f"jax_vs_analytic={g_vs_analytic:.2e}  t_jax={t_jax:.3f}s t_torch={t_torch:.3f}s")

# --- torch.autograd.gradcheck (float64) on an interior-only perturbation ---
# Use theta where both coords stay interior so the map is smooth for gradcheck.
theta_int = torch.tensor([0.1, -0.2], dtype=torch.float64, requires_grad=True)
def fn(t):
    return pt.solve_qp(P=Pt, c=t, lb=lbt, ub=ubt).sum()
try:
    gc = torch.autograd.gradcheck(fn, (theta_int,), eps=1e-6, atol=1e-4, rtol=1e-3)
except Exception as e:
    gc = f"raised {type(e).__name__}: {e}"
print(f"gradcheck(interior)={gc}")

ok = (fwd_err < 1e-6 and parity_x < 1e-6 and parity_g < 1e-5
      and g_jax_vs_fd < 1e-4 and g_vs_analytic < 1e-5 and gc is True)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (fwd={fwd_err:.2e} parity_g={parity_g:.2e} fd={g_jax_vs_fd:.2e} gc={gc})")
