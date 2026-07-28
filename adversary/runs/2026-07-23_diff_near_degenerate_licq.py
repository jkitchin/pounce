"""Adversary cross-check: gradient with NEAR-DEGENERATE active constraints (LICQ borderline).
Family: diff   Class: two nearly-parallel active inequalities, dx*/dh
Oracle: central FD re-solve (cvxpy CLARABEL) + JAX<->Torch parity.
Direction: as two active constraint normals become nearly parallel, the active
KKT matrix -> singular; the implicit-gradient linear solve is the fragile step.
Sweep the near-parallel angle eps and watch gradient accuracy vs FD.
"""
import numpy as np, torch
torch.set_default_dtype(torch.float64)
import pounce.torch as pt, pounce.jax as pj
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import cvxpy as cp

n = 2
P0 = np.array([[2.0,0.0],[0.0,2.0]])
c0 = np.array([-2.0,-2.0])          # pushes both constraints active at a vertex
w  = np.array([0.5,-0.3])

def make(eps):
    # two nearly-parallel active normals: [1,0] and [1,eps]
    G = np.array([[1.0,0.0],[1.0,eps]])
    h = np.array([0.5, 0.5])          # both bind near x0~0.5
    return G, h

def cvx(G,h):
    x=cp.Variable(n); pr=cp.Problem(cp.Minimize(0.5*cp.quad_form(x,cp.psd_wrap(P0))+c0@x),[G@x<=h]); pr.solve(solver=cp.CLARABEL); return np.asarray(x.value)

print(f"{'eps':>8} {'both_active':>11} {'grad_err_vs_FD':>15} {'parity':>10} {'condKKT':>10}")
worst=0.0
for eps in [1e-1, 1e-3, 1e-5, 1e-7]:
    G0,h0=make(eps)
    x0=cvx(G0,h0); slack=h0-G0@x0; both=int((np.abs(slack)<1e-6).sum()==2)
    # analytic grad dL/dh (jax)
    def loss(h):
        x=pj.solve_qp(P=jnp.asarray(P0),c=jnp.asarray(c0),G=jnp.asarray(G0),h=h)
        return jnp.dot(jnp.asarray(w),x)
    g_jax=np.asarray(jax.grad(loss)(jnp.asarray(h0)))
    ht=torch.tensor(h0,requires_grad=True)
    torch.dot(torch.tensor(w),pt.solve_qp(P=torch.tensor(P0),c=torch.tensor(c0),G=torch.tensor(G0),h=ht)).backward()
    g_torch=ht.grad.numpy()
    d=1e-6; g_fd=np.array([(np.dot(w,cvx(G0,h0+d*np.eye(2)[i]))-np.dot(w,cvx(G0,h0-d*np.eye(2)[i])))/(2*d) for i in range(2)])
    err=float(np.max(np.abs(g_jax-g_fd))); par=float(np.max(np.abs(g_jax-g_torch)))
    KKT=np.block([[P0, G0.T],[G0, np.zeros((2,2))]]); cond=np.linalg.cond(KKT)
    worst=max(worst,err if both else 0.0)
    print(f"{eps:>8.0e} {both:>11} {err:>15.2e} {par:>10.2e} {cond:>10.1e}  g_jax={g_jax} g_fd={g_fd}")

# PASS if, wherever both constraints are genuinely active, the analytic grad tracks FD.
ok = worst < 1e-3
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (worst grad_err where both-active={worst:.2e})")
