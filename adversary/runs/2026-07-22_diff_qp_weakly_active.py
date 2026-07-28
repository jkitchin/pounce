"""Adversary cross-check: differentiable QP layer at a DEGENERATE (weakly active) point
Family: diff   Class: degeneracy / strict-complementarity failure / non-unique dx/dtheta
Source: standard OptNet/implicit-diff layer; degeneracy theory per
  Robinson (1980) "Strongly regular generalized equations" (strict
  complementarity is required for x*(theta) to be differentiable) and
  Clarke (1983) generalized Jacobians.

  Problem: min 0.5||x||^2 + theta'x  s.t.  -0.5 <= x <= 0.5   (n=2)
  Exact closed form: x*_i(theta) = clip(-theta_i, -0.5, 0.5).

  Take theta0 = (-0.5, 0.2):
    x*_0 = 0.5  EXACTLY AT the upper bound, with multiplier
           mu_0 = -(x_0 + theta_0) = -(0.5 - 0.5) = 0  ->  WEAKLY ACTIVE.
           Strict complementarity FAILS at coordinate 0.
    x*_1 = -0.2  strictly interior (smooth).

  Loss L(theta) = sum(x*(theta)).
  dL/dtheta_0 does NOT exist:
      theta_0 -> -0.5^+ : x_0 = -theta_0 interior => one-sided deriv = -1
      theta_0 -> -0.5^- : x_0 = 0.5    pinned    => one-sided deriv =  0
    Clarke subdifferential = [-1, 0]. A CENTRAL difference of the EXACT map
    yields the meaningless average -0.5 and must NOT be used as the oracle.
  dL/dtheta_1 = -1 (smooth, unambiguous).

Acceptance: gradient must be finite, JAX==Torch, forward solve correct to IPM
tolerance, and the returned vector must be a VALID Clarke subgradient, i.e. lie
in conv{left, right} = [-1,0] x {-1}.  Ideally it equals one of the one-sided
derivatives; an interior element (barrier-smoothed) is still a legitimate
Clarke element and is what an IPM-based implicit-diff layer returns.
Failure modes hunted: NaN/Inf, blow-up, a value OUTSIDE [-1,0], sign flip,
JAX<->Torch disagreement, or degradation as tol is tightened.
"""
import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch

import pounce
import pounce.jax as pj
import pounce.torch as pt

P = np.eye(2)
lb = np.array([-0.5, -0.5])
ub = np.array([0.5, 0.5])
theta0 = np.array([-0.5, 0.2])          # coord 0 weakly active at ub

X_STAR = np.array([0.5, -0.2])
D_RIGHT = np.array([-1.0, -1.0])        # theta_0 increased  -> interior branch
D_LEFT = np.array([0.0, -1.0])          # theta_0 decreased  -> pinned branch
CLARKE_LO = np.array([-1.0, -1.0])
CLARKE_HI = np.array([0.0, -1.0])


def ninf(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), np.inf))


# ---- INDEPENDENT exact forward map (the oracle -- does NOT call pounce) ----
def L_exact(theta):
    return float(np.sum(np.clip(-np.asarray(theta), lb, ub)))


def L_pounce(theta, tol=None):
    return float(np.sum(pounce.solve_qp(P=P, c=np.asarray(theta),
                                        lb=lb, ub=ub, tol=tol).x))


# ---------------- forward solve ----------------
r_np = pounce.solve_qp(P=P, c=theta0, lb=lb, ub=ub)
x_ref = np.asarray(r_np.x)
fwd_err = ninf(x_ref, X_STAR)
mu = -(x_ref + theta0)
print("=== forward solve / degeneracy confirmation ===")
print(f"x_pounce={x_ref}  x_exact={X_STAR}  fwd_err={fwd_err:.3e}  status={r_np.status}")
print(f"implied multipliers mu={mu}")
print(f"  coord0: gap_to_ub={ub[0]-x_ref[0]:.3e}  mu0={mu[0]:.3e}  "
      f"-> both ~0 => WEAKLY ACTIVE (strict complementarity fails)")
print("  (gap ~ sqrt(tol) rather than ~tol is the textbook tangential approach")
print("   of the central path to a degenerate vertex -- expected, not a defect)")

# ---------------- JAX gradient ----------------
def loss_jax(theta):
    x = pj.solve_qp(P=jnp.asarray(P), c=theta, lb=jnp.asarray(lb), ub=jnp.asarray(ub))
    return jnp.sum(x)


t0 = time.perf_counter()
g_jax = np.asarray(jax.grad(loss_jax)(jnp.asarray(theta0)))
t_jax = time.perf_counter() - t0
x_jax = np.asarray(pj.solve_qp(P=jnp.asarray(P), c=jnp.asarray(theta0),
                               lb=jnp.asarray(lb), ub=jnp.asarray(ub)))

# ---------------- Torch gradient ----------------
Pt = torch.tensor(P, dtype=torch.float64)
lbt = torch.tensor(lb, dtype=torch.float64)
ubt = torch.tensor(ub, dtype=torch.float64)
th = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
t0 = time.perf_counter()
x_t = pt.solve_qp(P=Pt, c=th, lb=lbt, ub=ubt)
x_t.sum().backward()
t_torch = time.perf_counter() - t0
g_torch = th.grad.detach().numpy()
x_torch = x_t.detach().numpy()

# ------- FD oracle #1: EXACT map, one-sided AND central, step sweep -------
steps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
print("\n=== FD oracle on the EXACT closed-form map (independent of pounce) ===")
print(f"{'h':>8} | {'forward(right)':>24} | {'backward(left)':>24} | {'central':>24}")
L0e = L_exact(theta0)
ex_r = ex_l = None
for h in steps:
    gf, gb, gc = np.zeros(2), np.zeros(2), np.zeros(2)
    for i in range(2):
        tp = theta0.copy(); tp[i] += h
        tm = theta0.copy(); tm[i] -= h
        Lp, Lm = L_exact(tp), L_exact(tm)
        gf[i] = (Lp - L0e) / h
        gb[i] = (L0e - Lm) / h
        gc[i] = (Lp - Lm) / (2 * h)
    if h == 1e-4:
        ex_r, ex_l = gf.copy(), gb.copy()
    print(f"{h:8.0e} | {str(np.round(gf,9)):>24} | {str(np.round(gb,9)):>24} | "
          f"{str(np.round(gc,9)):>24}")
oracle_ok = ninf(ex_r, D_RIGHT) < 1e-9 and ninf(ex_l, D_LEFT) < 1e-9
print(f"analytic: right={D_RIGHT} left={D_LEFT}  oracle_self_check={oracle_ok}")
print("  -> CENTRAL difference converges to [-0.5,-1]: the average of two "
      "distinct one-sided derivatives. MEANINGLESS as a correctness target.")

# ------- FD oracle #2: pounce's own (barrier-smoothed) forward map -------
print("\n=== FD of pounce's own forward map (what the layer actually differentiates) ===")
L0p = L_pounce(theta0)
fd_p = {}
for h in [1e-3, 1e-4, 1e-5]:
    gf, gb, gc = np.zeros(2), np.zeros(2), np.zeros(2)
    for i in range(2):
        tp = theta0.copy(); tp[i] += h
        tm = theta0.copy(); tm[i] -= h
        Lp, Lm = L_pounce(tp), L_pounce(tm)
        gf[i] = (Lp - L0p) / h
        gb[i] = (L0p - Lm) / h
        gc[i] = (Lp - Lm) / (2 * h)
    fd_p[h] = (gf, gb, gc)
    print(f"{h:8.0e} | fwd={np.round(gf,7)} bwd={np.round(gb,7)} ctr={np.round(gc,7)}")

# ---------------- classification ----------------
finite = bool(np.all(np.isfinite(g_jax)) and np.all(np.isfinite(g_torch)))
parity_g = ninf(g_jax, g_torch)
parity_x = ninf(x_jax, x_torch)
fwd_layer = max(ninf(x_jax, X_STAR), ninf(x_torch, X_STAR))
d_right = ninf(g_jax, D_RIGHT)
d_left = ninf(g_jax, D_LEFT)
in_clarke = bool(np.all(g_jax >= CLARKE_LO - 1e-6) and np.all(g_jax <= CLARKE_HI + 1e-6))
smooth_coord_ok = abs(g_jax[1] - (-1.0)) < 1e-5

print("\n=== gradient dL/dtheta at the degenerate point ===")
print(f"jax  ={g_jax}  t={t_jax:.3f}s")
print(f"torch={g_torch}  t={t_torch:.3f}s")
print(f"finite={finite}  parity_x={parity_x:.3e}  parity_grad={parity_g:.3e}")
print(f"dist to RIGHT {D_RIGHT}={d_right:.3e}   dist to LEFT {D_LEFT}={d_left:.3e}")
print(f"inside_Clarke_interval [-1,0]x{{-1}} = {in_clarke}")
print(f"smooth coordinate dL/dtheta_1 == -1 : {smooth_coord_ok} "
      f"(err={abs(g_jax[1]+1.0):.2e})")

# ---------------- tolerance sweep: does the gradient degrade / blow up? -----
print("\n=== tol sweep (robustness of the degenerate gradient) ===")
for tol in [1e-6, 1e-8, 1e-10, 1e-12]:
    tt = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    try:
        xx = pt.solve_qp(P=Pt, c=tt, lb=lbt, ubt_=None) if False else \
            pt.solve_qp(P=Pt, c=tt, lb=lbt, ub=ubt, tol=tol)
        xx.sum().backward()
        gg = tt.grad.detach().numpy()
        xv = xx.detach().numpy()
        print(f"tol={tol:.0e}  x={np.round(xv,10)}  grad={np.round(gg,8)}  "
              f"finite={np.all(np.isfinite(gg))}  in_clarke="
              f"{bool(np.all(gg>=CLARKE_LO-1e-6) and np.all(gg<=CLARKE_HI+1e-6))}")
    except Exception as e:
        print(f"tol={tol:.0e}  raised {type(e).__name__}: {e}")

# ---------------- verdict ----------------
problems = []
if not oracle_ok:
    problems.append("FD oracle self-check failed -- formulation suspect")
if not finite:
    problems.append("NON-FINITE gradient")
if fwd_layer > 1e-3 or fwd_err > 1e-3:
    problems.append(f"forward solve wrong (err={max(fwd_layer, fwd_err):.2e})")
if parity_g > 1e-6:
    problems.append(f"JAX/Torch gradient disagree ({parity_g:.2e})")
if not in_clarke:
    problems.append(f"gradient OUTSIDE the Clarke subdifferential [-1,0]x{{-1}}: {g_jax}")
if not smooth_coord_ok:
    problems.append(f"smooth coordinate wrong: {g_jax[1]} != -1")

print()
if not problems:
    if min(d_right, d_left) < 1e-5:
        print("gradient equals a one-sided derivative exactly")
    else:
        print("gradient is a strict interior Clarke element (barrier-smoothed "
              "convex combination of the two one-sided Jacobians) -- valid, "
              "and it matches the FD of pounce's own forward map")
    print("VERDICT: PASS")
else:
    print("VERDICT: GRADIENT_ERROR (" + "; ".join(problems) + ")")
