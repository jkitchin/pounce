"""Compare pounce's stiff ODE / DAE solver to scipy.integrate.solve_ivp.

Three parts:

1. **Stiff accuracy & speed** of ``pounce.ode.solve_ivp`` (adaptive Radau
   IIA order-5 collocation, the same RADAU5 method SciPy implements) against
   ``scipy.integrate.solve_ivp(method="Radau")`` on the Van der Pol
   oscillator at mu=1000 and a stiff scalar problem with a known solution.

2. **Index-1 DAE** via a singular mass matrix ``M y' = f`` — Robertson
   kinetics with the conservation law as an algebraic constraint. SciPy's
   ``solve_ivp`` cannot do this; we check the algebraic constraint is held.

3. **Differentiability** of ``pounce.jax.odeint`` (and the PyTorch mirror):
   gradients of the trajectory w.r.t. ODE parameters and the initial
   condition, via the implicit-function theorem on the collocation system,
   each checked against a finite-difference reference and the analytic value.

Run: ``python examples/ode_scipy_compare.py``
"""

import time

import numpy as np
from scipy.integrate import solve_ivp as scipy_solve_ivp

import pounce.ode as po


def _banner(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# --------------------------------------------------------------------------
# 1. Stiff accuracy & speed vs SciPy
# --------------------------------------------------------------------------
def stiff_comparison():
    _banner("1. Stiff problems: pounce vs scipy Radau")

    # (a) stiff scalar with a known forced solution shape
    def f_scalar(t, y):
        return [-2000.0 * (y[0] - np.cos(t))]

    kw = dict(method="Radau", rtol=1e-8, atol=1e-10, dense_output=True)
    sp = scipy_solve_ivp(f_scalar, (0, 1.5), [0.0], **kw)
    po_ = po.solve_ivp(f_scalar, (0, 1.5), [0.0], **kw)
    tt = np.linspace(0, 1.5, 200)
    print("stiff scalar  y' = -2000 (y - cos t):")
    print(f"  scipy : {sp.t.size:5d} steps, nfev={sp.nfev}")
    print(f"  pounce: {po_.nstep:5d} steps, nfev={po_.nfev}, nrej={po_.nrej}")
    print(f"  max |pounce - scipy| = {np.max(np.abs(sp.sol(tt) - po_.sol(tt))):.2e}")

    # (b) Van der Pol, mu = 1000 (a classic stiff benchmark)
    mu = 1000.0

    def f_vdp(t, y):
        return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]

    def jac_vdp(t, y):
        return [[0.0, 1.0],
                [-2 * mu * y[0] * y[1] - 1.0, mu * (1 - y[0] ** 2)]]

    kw = dict(method="Radau", jac=jac_vdp, rtol=1e-6, atol=1e-8,
              dense_output=True)
    t0 = time.perf_counter()
    sp = scipy_solve_ivp(f_vdp, (0, 3000.0), [2.0, 0.0], **kw)
    t_sp = time.perf_counter() - t0
    t0 = time.perf_counter()
    po_ = po.solve_ivp(f_vdp, (0, 3000.0), [2.0, 0.0], **kw)
    t_po = time.perf_counter() - t0
    tt = np.linspace(0, 3000, 500)
    print("\nVan der Pol, mu=1000, [0, 3000]:")
    print(f"  scipy : {sp.t.size:5d} steps, nlu={sp.nlu:5d}, {t_sp:.3f}s")
    print(f"  pounce: {po_.nstep:5d} steps, nlu={po_.nlu:5d}, {t_po:.3f}s "
          f"(nrej={po_.nrej})")
    print(f"  max |pounce - scipy| over 500 pts = "
          f"{np.max(np.abs(sp.sol(tt) - po_.sol(tt))):.2e}")


# --------------------------------------------------------------------------
# 2. Index-1 DAE (singular mass matrix) — beyond SciPy's solve_ivp
# --------------------------------------------------------------------------
def dae_demo():
    _banner("2. Index-1 DAE: Robertson kinetics (M y' = f, M singular)")

    k1, k2, k3 = 0.04, 3e7, 1e4

    def f(t, y):
        return [-k1 * y[0] + k3 * y[1] * y[2],
                 k1 * y[0] - k3 * y[1] * y[2] - k2 * y[1] ** 2,
                 y[0] + y[1] + y[2] - 1.0]      # algebraic: mass conservation

    M = np.diag([1.0, 1.0, 0.0])               # third row is algebraic
    res = po.solve_ivp(f, (0, 1e4), [1.0, 0.0, 0.0], mass=M,
                       rtol=1e-6, atol=1e-8)
    yend = res.y[:, -1]
    print("scipy.integrate.solve_ivp cannot integrate a DAE; pounce can.")
    print(f"  y(1e4)            = {np.array2string(yend, precision=6)}")
    print(f"  conservation resid = {abs(yend.sum() - 1.0):.2e} (should be ~0)")
    print(f"  steps={res.nstep}, nrej={res.nrej}")


# --------------------------------------------------------------------------
# 3. Differentiability (JAX + Torch), checked vs analytic & finite diff
# --------------------------------------------------------------------------
def differentiability_demo():
    _banner("3. Differentiable integration: pounce.jax.odeint / torch.odeint")

    try:
        import jax
        import jax.numpy as jnp
        from jax import config
        config.update("jax_enable_x64", True)
        import pounce.jax as pj
    except Exception as e:                       # pragma: no cover
        print(f"  (JAX unavailable: {e})")
        return

    # Damped harmonic oscillator y'' + 2 c w y' + w^2 y = 0, theta = [w, c].
    def f(t, y, th):
        w, c = th[0], th[1]
        return jnp.array([y[1], -w * w * y[0] - 2 * c * w * y[1]])

    t = jnp.linspace(0.0, 5.0, 201)
    y0 = jnp.array([1.0, 0.0])
    th0 = jnp.array([1.3, 0.1])

    def end_state(th):
        return pj.odeint(f, y0, t, th).y[:, -1]   # final [y, y']

    # Full Jacobian d(final state)/d(theta) and an FD check.
    J = np.asarray(jax.jacobian(end_state)(th0))
    h = 1e-6
    Jfd = np.stack([
        (np.asarray(end_state(th0.at[i].add(h)))
         - np.asarray(end_state(th0.at[i].add(-h)))) / (2 * h)
        for i in range(2)
    ], axis=1)
    print("d(final state)/d(theta):")
    print(f"  jax.jacobian vs finite-diff max diff = {np.max(np.abs(J - Jfd)):.2e}")

    # Gradient w.r.t. the initial condition.
    def y_end_scalar(y0v):
        return pj.odeint(f, y0v, t, th0).y[0, -1]

    g = np.asarray(jax.grad(y_end_scalar)(y0))
    gfd = np.array([
        (float(y_end_scalar(y0.at[i].add(h)))
         - float(y_end_scalar(y0.at[i].add(-h)))) / (2 * h)
        for i in range(2)
    ])
    print("d(y(T))/d(y0):")
    print(f"  jax.grad vs finite-diff max diff = {np.max(np.abs(g - gfd)):.2e}")

    # Analytic check on the exponential-decay problem.
    def fdecay(t, y, th):
        return jnp.array([-th[0] * y[0]])

    td = jnp.linspace(0.0, 2.0, 81)
    k0, T = 0.7, 2.0
    val = float(pj.odeint(fdecay, jnp.array([1.0]), td, jnp.array([k0])).y[0, -1])
    dk = float(jax.grad(
        lambda k: pj.odeint(fdecay, jnp.array([1.0]), td, jnp.array([k])).y[0, -1]
    )(k0))
    print("exp-decay y' = -k y, y(T):")
    print(f"  value  pounce={val:.8f}  exact={np.exp(-k0 * T):.8f}")
    print(f"  d/dk   pounce={dk:.6e}  exact={-T * np.exp(-k0 * T):.6e}")

    # PyTorch mirror.
    try:
        import torch
        import pounce.torch as pt
        torch.set_default_dtype(torch.float64)
        tt = torch.linspace(0.0, 2.0, 81, dtype=torch.float64)
        k = torch.tensor([k0], dtype=torch.float64, requires_grad=True)
        yT = pt.odeint(lambda t, y, th: torch.stack([-th[0] * y[0]]),
                       torch.tensor([1.0], dtype=torch.float64), tt, k).y[0, -1]
        yT.backward()
        print("torch.odeint d/dk:")
        print(f"  value={yT.item():.8f}  d/dk={k.grad.item():.6e} "
              f"(exact {-T * np.exp(-k0 * T):.6e})")
    except Exception as e:                       # pragma: no cover
        print(f"  (PyTorch unavailable: {e})")


if __name__ == "__main__":
    stiff_comparison()
    dae_demo()
    differentiability_demo()
