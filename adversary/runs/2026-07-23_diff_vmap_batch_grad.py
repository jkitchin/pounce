"""Adversary cross-check: differentiation through a VMAPPED batch of QP layers.
Family: diff   Class: vmap_solve gradient vs per-item gradient
Oracle: per-item jax.grad on single solve_qp (same math) + central FD re-solve.
Direction: batched/vectorized differentiation must equal stacking independent
per-item gradients; a batching bug would couple items or drop a term.
"""
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pounce.jax as pj

n = 2; B = 4
rng = np.random.RandomState(7)
P0 = np.array([[2.0,0.3],[0.3,1.5]])
G0 = np.array([[1.0,0.0],[0.0,1.0],[-1.0,-1.0]])
h0 = np.array([0.6,0.6,0.2])
# batch differs by c
C = rng.randn(B, n)
W = rng.randn(B, n)

def single_loss(c, w):
    x = pj.solve_qp(P=jnp.asarray(P0), c=c, G=jnp.asarray(G0), h=jnp.asarray(h0))
    return jnp.dot(w, x)

# per-item gradients
g_per = np.stack([np.asarray(jax.grad(single_loss)(jnp.asarray(C[i]), jnp.asarray(W[i]))) for i in range(B)])

# vmapped gradient: grad of sum over batch, via vmap of grad
gfun = jax.vmap(jax.grad(single_loss), in_axes=(0,0))
g_vmap = np.asarray(gfun(jnp.asarray(C), jnp.asarray(W)))

# central FD oracle on item 0
def solve_c(c):
    return np.asarray(pj.solve_qp(P=jnp.asarray(P0), c=jnp.asarray(c), G=jnp.asarray(G0), h=jnp.asarray(h0)))
d=1e-6; g_fd0=np.array([(np.dot(W[0],solve_c(C[0]+d*np.eye(n)[k]))-np.dot(W[0],solve_c(C[0]-d*np.eye(n)[k])))/(2*d) for k in range(n)])

err_vmap = float(np.max(np.abs(g_per - g_vmap)))
err_fd   = float(np.max(np.abs(g_per[0] - g_fd0)))
print(f"g_per[0]={g_per[0]}  g_vmap[0]={g_vmap[0]}  g_fd0={g_fd0}")
print(f"vmap-vs-per-item max err={err_vmap:.2e}  per-item-vs-FD (item0)={err_fd:.2e}")
ok = err_vmap < 1e-10 and err_fd < 5e-5
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (err_vmap={err_vmap:.2e}, err_fd={err_fd:.2e})")
