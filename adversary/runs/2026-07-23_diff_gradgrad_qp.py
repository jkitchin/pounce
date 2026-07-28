"""Adversary cross-check: SECOND-ORDER correctness through a QP layer.
Family: diff   Class: gradcheck + gradgradcheck (double backward) w.r.t. P, c, h
Oracle: torch.autograd.gradcheck / gradgradcheck (float64) + JAX<->Torch parity.
Direction: logged diff tests check forward + first-order; this checks the Hessian
of the solution map (double backward). P is routed through the symmetric
parametrization P=S+S^T (established convention, see 2026-06-15_diff_qp_wrt_P.org)
so gradcheck perturbations live in the identifiable subspace.
"""
import numpy as np, torch
torch.set_default_dtype(torch.float64)
import pounce.torch as pt
import pounce.jax as pj
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

n = 3
S0 = np.array([[1.3,0.2,0.0],[0.2,1.0,0.1],[0.0,0.1,0.9]])     # P = S0 S0^T + 0.4 I via S param
c0 = np.array([-0.8,0.4,0.2])
G0 = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
h0 = np.array([0.3, 0.9])
Gt = torch.tensor(G0)

def P_of(S):                 # symmetric, SPD, identifiable directions only
    return S + S.transpose(-1,-2) + 0.8*torch.eye(n)

def layer_S(S): return pt.solve_qp(P=P_of(S), c=torch.tensor(c0), G=Gt, h=torch.tensor(h0))
def layer_c(c): return pt.solve_qp(P=P_of(torch.tensor(S0)), c=c, G=Gt, h=torch.tensor(h0))
def layer_h(h): return pt.solve_qp(P=P_of(torch.tensor(S0)), c=torch.tensor(c0), G=Gt, h=h)

x0 = layer_S(torch.tensor(S0)).detach().numpy(); slack = h0 - G0@x0
print(f"x0={x0} slack={slack} active={np.where(np.abs(slack)<1e-6)[0].tolist()}")

results = {}
for nm, fn, x in [("S(=P)",layer_S,S0),("c",layer_c,c0),("h",layer_h,h0)]:
    xv = torch.tensor(x, requires_grad=True)
    g1 = torch.autograd.gradcheck(fn,(xv,),eps=1e-6,atol=1e-4,rtol=1e-3,raise_exception=False)
    g2 = torch.autograd.gradgradcheck(fn,(xv,),eps=1e-6,atol=1e-4,rtol=1e-3,raise_exception=False)
    results[nm]=(g1,g2); print(f"  input {nm}: gradcheck={g1} gradgradcheck={g2}")

# JAX<->Torch parity on dL/dP (symmetric convention)
w = np.array([1.0,-0.5,0.3]); P0 = S0+S0.T+0.8*np.eye(n)
def loss_jax(P):
    x = pj.solve_qp(P=P, c=jnp.asarray(c0), G=jnp.asarray(G0), h=jnp.asarray(h0))
    return jnp.dot(jnp.asarray(w), x)
gP_jax = np.asarray(jax.grad(loss_jax)(jnp.asarray(P0)))
Pt = torch.tensor(P0, requires_grad=True)
torch.dot(torch.tensor(w), pt.solve_qp(P=Pt,c=torch.tensor(c0),G=Gt,h=torch.tensor(h0))).backward()
parity = float(np.max(np.abs(gP_jax - Pt.grad.numpy())))
print(f"JAX<->Torch dL/dP parity={parity:.2e}")

allok = all(g1 and g2 for g1,g2 in results.values()) and parity < 1e-7
print("VERDICT: PASS" if allok else f"VERDICT: FAIL ({results}, parity={parity:.2e})")
