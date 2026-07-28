"""Adversary cross-check: SOCP layer gradient dx*/dc near the cone boundary.
Family: diff   Class: differentiable SOCP, gradient w.r.t. objective c
Oracle: central FD re-solve (cvxpy CLARABEL) + JAX<->Torch parity.
Direction: solution sits on the SOC boundary where the solution map has high
curvature; gradient of L=w.x wrt c.
"""
import numpy as np, torch
torch.set_default_dtype(torch.float64)
import pounce.torch as pt, pounce.jax as pj
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import cvxpy as cp

# min c'x s.t. ||x[1:]||<=x[0], x[0]<=1  (ball of radius 1); vars x in R^3
# slack (s0,s1,s2) = (x0, x1, x2) in SOC(3)? we want x0>=||(x1,x2)||.
# Represent SOC on s=(x0,x1,x2): G=-I, h=0 -> s=x in SOC.  Plus x0<=1 nonneg.
n=3
c0=np.array([0.4,-0.9,-0.5])   # pulls (x1,x2) outward, x0 to boundary
G0=np.zeros((4,3)); h0=np.zeros(4)
G0[0:3,:]=-np.eye(3)           # s[0:3]=x in SOC(3)
G0[3,0]=1.0; h0[3]=1.0         # s3 = 1 - x0 >=0  (x0<=1)
cones=[("soc",3),("nonneg",1)]
w=np.array([0.6,0.2,-0.7])

def cvx(c):
    x=cp.Variable(n)
    pr=cp.Problem(cp.Minimize(c@x),[cp.norm(x[1:],2)<=x[0], x[0]<=1]); pr.solve(solver=cp.CLARABEL)
    return np.asarray(x.value)
x0=cvx(c0); nrm=np.linalg.norm(x0[1:])
print(f"x0={x0}  ||x[1:]||={nrm:.6f} x0[0]={x0[0]:.6f}  on-boundary={abs(nrm-x0[0])<1e-6}")

def loss(c):
    x=pj.solve_socp(P=jnp.zeros((n,n)),c=c,G=jnp.asarray(G0),h=jnp.asarray(h0),cones=cones)
    return jnp.dot(jnp.asarray(w),x)
g_jax=np.asarray(jax.grad(loss)(jnp.asarray(c0)))
ct=torch.tensor(c0,requires_grad=True)
xt=pt.solve_socp(P=torch.zeros((n,n)),c=ct,G=torch.tensor(G0),h=torch.tensor(h0),cones=cones)
torch.dot(torch.tensor(w),xt).backward()
g_torch=ct.grad.numpy()
d=1e-6; g_fd=np.zeros(n)
for i in range(n):
    cp_=c0.copy(); cp_[i]+=d; cm=c0.copy(); cm[i]-=d
    g_fd[i]=(np.dot(w,cvx(cp_))-np.dot(w,cvx(cm)))/(2*d)
err_fd=float(np.max(np.abs(g_jax-g_fd))); parity=float(np.max(np.abs(g_jax-g_torch)))
print(f"g_jax={g_jax}\ng_fd ={g_fd}\ng_tor={g_torch}")
print(f"grad err vs FD={err_fd:.2e} JAX<->Torch parity={parity:.2e}")
ok = abs(nrm-x0[0])<1e-5 and err_fd<5e-5 and parity<1e-7
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (err_fd={err_fd:.2e}, parity={parity:.2e})")
