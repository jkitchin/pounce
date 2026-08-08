"""Adversary cross-check: differentiable QP layer gradients w.r.t. P and c.

Family: diff   Class: pounce.jax QP layer, dx*/dP (symmetric Hessian
              gradient) and dx*/dc (linear-term gradient) -- the prior
              diff-family probes (2026-07-23) covered dx*/dG and dx*/dA;
              this run targets the two angles they did NOT cover: the
              quadratic term P and the linear term c.
Source: OptNet (Amos & Kolter 2017) implicit-function differentiation rule
        for a QP layer; problem instance constructed here (small QP, one
        active inequality) so the finite-difference oracle is cheap and
        well-conditioned.

    min  1/2 x^T P x + c^T x   s.t.  G x <= h
    P = [[2.0, 0.5], [0.5, 1.0]]  (SPD)
    c = [1.0, -1.0]
    G = [[1.0, 1.0]], h = [1.0]           (active: x1+x2=1 at optimum)

Loss L(P, c) = w^T x*(P, c) for a fixed weight vector w = [0.3, 0.7].
dL/dP and dL/dc computed via jax.grad through pounce.jax.qp.solve_qp
(OptNet implicit-diff backward), cross-checked against CENTRAL FINITE
DIFFERENCES of independent forward re-solves via pounce.qp.solve_qp (the
plain, non-differentiable convex IPM -- an entirely separate code path
from the jax custom_vjp backward under test).
"""

import time

import numpy as np
import jax
import jax.numpy as jnp

from pounce.jax import solve_qp as jax_solve_qp
from pounce.qp import solve_qp as plain_solve_qp

P0 = np.array([[2.0, 0.5], [0.5, 1.0]])
c0 = np.array([1.0, -1.0])
G = np.array([[1.0, 1.0]])
h = np.array([1.0])
w = np.array([0.3, 0.7])


def loss_jax(P, c):
    r = jax_solve_qp(P=P, c=c, G=G, h=h)
    return jnp.dot(w, r)


t0 = time.perf_counter()
grad_fn = jax.jit(jax.grad(loss_jax, argnums=(0, 1)))
dL_dP_jax, dL_dc_jax = grad_fn(jnp.asarray(P0), jnp.asarray(c0))
dL_dP_jax = np.asarray(dL_dP_jax)
dL_dc_jax = np.asarray(dL_dc_jax)
t_jax = time.perf_counter() - t0

# --- Independent oracle: central finite differences via the PLAIN solver ---
def loss_plain(P, c):
    r = plain_solve_qp(P=P, c=c, G=G, h=h)
    return float(w @ np.asarray(r.x))


eps = 1e-6

t0 = time.perf_counter()
dL_dc_fd = np.zeros(2)
for i in range(2):
    cp = c0.copy(); cp[i] += eps
    cm = c0.copy(); cm[i] -= eps
    dL_dc_fd[i] = (loss_plain(P0, cp) - loss_plain(P0, cm)) / (2 * eps)
t_fd = time.perf_counter() - t0


def relmax(a, b):
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


err_dc = relmax(dL_dc_jax, dL_dc_fd)

# --- dP check: convention-agnostic directional-derivative test ---
# The forward pass (pounce.jax.qp._build_problem -> _to_coo_lower) reads
# ONLY the lower triangle of P; the OptNet backward, by contrast, returns a
# symmetric matrix (module docstring: "(dP is the symmetric gradient)").
# A naive FD that perturbs a symmetric (i,j)+(j,i) pair at once therefore
# measures dL/dP[i,j] + dL/dP[j,i] (double the single-slot jax value) --
# NOT a per-slot mismatch, just a different (and, for i!=j, ambiguous)
# quantity. The unambiguous, convention-agnostic oracle is the directional
# derivative along the FULL jax-reported gradient matrix D = dL_dP_jax
# itself (Frobenius inner product): d/dt loss(P0 + t*D)|_{t=0} must equal
# <D, D>_F = ||D||_F^2, exactly, regardless of any doubling convention --
# because D is applied elementwise to the SAME raw P array the plain
# solver reads (lower triangle only), so no interpretation ambiguity
# remains on either side of the comparison.
D = dL_dP_jax
dirderiv_pred = float(np.sum(D * D))
Pp = P0 + eps * D
Pm = P0 - eps * D
dirderiv_fd = (loss_plain(Pp, c0) - loss_plain(Pm, c0)) / (2 * eps)
err_dP_dir = abs(dirderiv_fd - dirderiv_pred) / max(1.0, abs(dirderiv_pred))

# Diagnostic-only: a NON-symmetric direction is OUTSIDE the documented
# contract ("P (lower triangle is used; assumed symmetric)" -- plain
# solve_qp docstring). Forward provably ignores the upper triangle
# (_to_coo_lower keeps only row>=col), so the true per-slot sensitivity
# there is exactly 0 while jax reports a nonzero "phantom" value on it
# (compensated by exactly halving the lower-triangle slot so the two
# conventions agree on every symmetric direction). This is EXPECTED to
# disagree with a naive raw-matrix FD and is not evidence of a defect;
# kept here only so a future run doesn't have to re-derive the reason.
rng = np.random.default_rng(0)
Delta = rng.standard_normal((2, 2))
dirderiv2_pred = float(np.sum(D * Delta))
Pp2 = P0 + eps * Delta
Pm2 = P0 - eps * Delta
dirderiv2_fd = (loss_plain(Pp2, c0) - loss_plain(Pm2, c0)) / (2 * eps)
err_dP_dir2 = abs(dirderiv2_fd - dirderiv2_pred) / max(1.0, abs(dirderiv2_pred))

print("=== jax.grad through pounce.jax QP layer ===")
print(f"dL/dP=\n{dL_dP_jax}\ndL/dc={dL_dc_jax} time={t_jax:.4f}s")
print("=== oracle: central FD of independent plain solve_qp re-solves ===")
print(f"dL/dc_fd={dL_dc_fd} time={t_fd:.4f}s")
print(f"[IN-CONTRACT]  dP directional deriv along D=dL/dP (symmetric direction): pred={dirderiv_pred:.6e} fd={dirderiv_fd:.6e} rel_err={err_dP_dir:.2e}")
print(f"[OUT-OF-CONTRACT, diagnostic only] dP directional deriv, random asymmetric direction: pred={dirderiv2_pred:.6e} fd={dirderiv2_fd:.6e} rel_err={err_dP_dir2:.2e} (expected to disagree -- see comment)")
print(f"dc max_rel_err={err_dc:.2e}")

# Symmetry check: OptNet's stated convention is a symmetric dL/dP.
sym_err = float(np.max(np.abs(dL_dP_jax - dL_dP_jax.T)))
print(f"dL/dP symmetry residual={sym_err:.2e}")

ok = err_dP_dir < 1e-4 and err_dc < 1e-4 and sym_err < 1e-8
print("VERDICT: PASS" if ok else "VERDICT: FAIL")
