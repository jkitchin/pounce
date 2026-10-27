"""Adversary cross-check: torch differentiable QP layer, Euclidean projection onto
a scaled simplex {sum x = b, x >= -h}; full Jacobians dx/dc, dx/db, dx/dh
Family: diff   Class: torch QP layer (first torch-side probe in the log), eq+ineq,
        strictly complementary, closed-form Jacobians + gradcheck + JAX parity
Source: Held, Wolfe & Crowder, Math. Prog. 6 (1974) 62-88; Duchi et al., ICML
        2008 ("Efficient projections onto the l1-ball"): x* = max(y - tau, -h)
        with tau solving sum x* = b. On the support S (x_i > -h_i):
        dx_S/dy = I - 11^T/|S|; dx_S/db = 1/|S|; for j not in S:
        dx_j/dh_j = -1, dx_S/dh_j = +1/|S|; dx/dh_j = 0 for j in S.
        Layer form: P = I, c = -y  =>  dx/dc = -dx/dy.
Known optimal: closed form above.
"""
import time, numpy as np, torch
torch.set_default_dtype(torch.float64)
y = np.array([0.9, -0.4, 0.55, 1.3, -1.1, 0.2]); n = len(y); bval = 1.5; h0 = np.zeros(n)
# closed form
def proj(y, b, h):
    lo, hi = -10, 10
    for _ in range(200):
        tau = 0.5 * (lo + hi)
        s = np.maximum(y - tau, -h).sum()
        lo, hi = (tau, hi) if s > b else (lo, tau)
    return np.maximum(y - tau, -h), tau
xk, tau = proj(y, bval, h0)
S = xk > 1e-9; k = S.sum()
Jy = np.zeros((n, n)); Jy[np.ix_(S, S)] = np.eye(k) - 1.0 / k
Jc = -Jy
Jb = np.where(S, 1.0 / k, 0.0)
Jh = np.zeros((n, n))
for j in np.where(~S)[0]:
    Jh[j, j] = -1.0; Jh[S, j] = 1.0 / k
print(f"support={np.where(S)[0]} tau={tau:.6f} min margin={np.min(np.abs((y - tau)[~S])):.3f}")
import pounce.torch as pt
P = torch.eye(n); G = -torch.eye(n); A = torch.ones(1, n)
def layer(c, b, h):
    return pt.solve_qp(P=P, c=c, G=G, h=h, A=A, b=b)
c_t = torch.tensor(-y, requires_grad=True); b_t = torch.tensor([bval], requires_grad=True); h_t = torch.tensor(h0, requires_grad=True)
t0 = time.perf_counter(); x = layer(c_t, b_t, h_t); tp = time.perf_counter() - t0
fwd = np.abs(x.detach().numpy() - xk).max()
from torch.autograd.functional import jacobian
t0 = time.perf_counter()
Jc_t, Jb_t, Jh_t = [J.numpy() for J in jacobian(layer, (c_t, b_t, h_t))]
tj = time.perf_counter() - t0
errs = dict(c=np.abs(Jc_t - Jc).max(), b=np.abs(Jb_t[:, 0] - Jb).max(), h=np.abs(Jh_t - Jh).max())
# FD of re-solve (independent of the backward pass)
eps = 1e-6
def fd(which):
    base = dict(c=-y.copy(), b=np.array([bval]), h=h0.copy()); m = base[which].size; J = np.zeros((n, m))
    for i in range(m):
        p = {k_: v.copy() for k_, v in base.items()}; q = {k_: v.copy() for k_, v in base.items()}
        p[which][i] += eps; q[which][i] -= eps
        f = lambda d: layer(*(torch.tensor(d[k_]) for k_ in ("c", "b", "h"))).numpy()
        J[:, i] = (f(p) - f(q)) / (2 * eps)
    return J
fd_err = dict(c=np.abs(fd("c") - Jc).max(), b=np.abs(fd("b")[:, 0] - Jb).max(), h=np.abs(fd("h") - Jh).max())
gc = torch.autograd.gradcheck(lambda c, b, h: layer(c, b, h), (c_t, b_t, h_t), eps=1e-6, atol=1e-5, rtol=1e-4)
# JAX parity
import jax; jax.config.update("jax_enable_x64", True); import jax.numpy as jnp; import pounce.jax as pj
jl = lambda c, b, h: pj.solve_qp(P=jnp.eye(n), c=c, G=-jnp.eye(n), h=h, A=jnp.ones((1, n)), b=b)
Jj = jax.jacrev(jl, argnums=(0, 1, 2))(jnp.asarray(-y), jnp.asarray([bval]), jnp.asarray(h0))
par = max(np.abs(np.asarray(Jj[0]) - Jc_t).max(), np.abs(np.asarray(Jj[1]) - Jb_t).max(), np.abs(np.asarray(Jj[2]) - Jh_t).max())
print(f"forward x_err={fwd:.2e} t_fwd={tp:.4f}s t_jac={tj:.4f}s")
print("analytic-vs-closed:", {k_: f"{v:.2e}" for k_, v in errs.items()})
print("FD-vs-closed:      ", {k_: f"{v:.2e}" for k_, v in fd_err.items()})
print(f"gradcheck={gc} torch-vs-jax jacobian max diff={par:.2e}")
ok = fwd < 1e-6 and max(errs.values()) < 1e-5 and max(fd_err.values()) < 1e-4 and gc and par < 1e-6
print("VERDICT: PASS" if ok else "VERDICT: FAIL (GRADIENT_ERROR candidate)")
