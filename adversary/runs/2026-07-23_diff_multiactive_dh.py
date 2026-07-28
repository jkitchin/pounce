"""Adversary cross-check: dx*/dh with TWO simultaneously-active inequalities.
Family: diff   Class: multi-active QP layer, gradient w.r.t. RHS h
Oracle: central finite-difference re-solve (cvxpy CLARABEL) + JAX<->Torch parity.
Direction: at a vertex where 2 inequalities bind, dx*/dh couples both active rows.
"""
import numpy as np, torch
torch.set_default_dtype(torch.float64)
import pounce.torch as pt, pounce.jax as pj
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import cvxpy as cp

n = 2
P0 = np.array([[2.0,0.0],[0.0,2.0]])
c0 = np.array([-3.0,-2.0])                  # unconstrained min (1.5,1.0)
G0 = np.array([[1.0,0.0],[0.0,1.0]])        # x0<=h0, x1<=h1  -> both bind
h0 = np.array([0.5, 0.4])
w  = np.array([0.7,-0.4])

def cvx(h):
    x=cp.Variable(n); pr=cp.Problem(cp.Minimize(0.5*cp.quad_form(x,cp.psd_wrap(P0))+c0@x),[G0@x<=h]); pr.solve(solver=cp.CLARABEL); return np.asarray(x.value)
x0=cvx(h0); slack=h0-G0@x0
print(f"x0={x0} slack={slack} active={np.where(np.abs(slack)<1e-6)[0].tolist()}")

# jax grad of L=w.x wrt h
def loss(h):
    x=pj.solve_qp(P=jnp.asarray(P0),c=jnp.asarray(c0),G=jnp.asarray(G0),h=h)
    return jnp.dot(jnp.asarray(w),x)
g_jax=np.asarray(jax.grad(loss)(jnp.asarray(h0)))
# torch parity
ht=torch.tensor(h0,requires_grad=True)
xt=pt.solve_qp(P=torch.tensor(P0),c=torch.tensor(c0),G=torch.tensor(G0),h=ht)
torch.dot(torch.tensor(w),xt).backward()
g_torch=ht.grad.numpy()
# central FD on cvxpy
d=1e-6; g_fd=np.zeros(n)
for i in range(n):
    hp=h0.copy(); hp[i]+=d; hm=h0.copy(); hm[i]-=d
    g_fd[i]=(np.dot(w,cvx(hp))-np.dot(w,cvx(hm)))/(2*d)
err_fd=float(np.max(np.abs(g_jax-g_fd))); parity=float(np.max(np.abs(g_jax-g_torch)))
print(f"g_jax={g_jax}\ng_fd ={g_fd}\ng_tor={g_torch}")
print(f"grad err vs FD={err_fd:.2e}  JAX<->Torch parity={parity:.2e}")
ok = slack.max()<1e-6 and (np.abs(slack)<1e-6).sum()==2 and err_fd<5e-5 and parity<1e-7
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (err_fd={err_fd:.2e}, parity={parity:.2e})")
