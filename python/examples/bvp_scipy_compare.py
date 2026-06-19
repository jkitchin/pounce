"""Compare pounce's BVP solver to scipy.integrate.solve_bvp.

Two halves:

1. **Accuracy & speed** of ``pounce.bvp.solve_bvp`` (a fixed-mesh
   Hermite--Simpson collocation solved as a pounce feasibility NLP) against
   ``scipy.integrate.solve_bvp`` on problems with closed-form solutions.

2. **Differentiability** of the pounce solution. pounce poses the
   collocation root-find ``R(z, theta) = 0`` as an NLP and differentiates
   ``z*(theta)`` with the implicit-function theorem on the KKT system, so
   the solution is differentiable w.r.t. *anything* ``fun`` / ``bc`` close
   over. We exercise every flavour of derivative and check each against a
   finite-difference reference through the same solver:

   * gradient w.r.t. an ODE parameter,
   * gradient w.r.t. a boundary value,
   * sensitivity of a solved-for unknown parameter ``p*``,
   * the full Jacobian ``dy/dtheta`` of the solution,
   * a second derivative (Hessian) through the solver,
   * the PyTorch frontend (mirror of the JAX one).

Run: ``python examples/bvp_scipy_compare.py``
"""

import time

import numpy as np
from scipy.integrate import solve_bvp as scipy_solve_bvp

import pounce


def _hr(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def _time_it(fn, repeats=5):
    fn()  # warm up
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


# --------------------------------------------------------------------------
# 1. Accuracy & speed vs scipy
# --------------------------------------------------------------------------

def accuracy_and_speed():
    _hr("1. Accuracy & speed vs scipy.integrate.solve_bvp")

    # (a) y'' + y = 0, y(0)=0, y(pi/2)=1  ->  exact y = sin(x).
    def fun(x, y):
        return np.vstack((y[1], -y[0]))

    def bc(ya, yb):
        return np.array([ya[0], yb[0] - 1.0])

    a, b = 0.0, np.pi / 2
    exact = lambda x: np.sin(x)

    print("\nProblem: y'' + y = 0, y(0)=0, y(pi/2)=1  (exact: sin x)")
    print(f"{'nodes':>6} | {'scipy err':>11} {'pounce err':>11} | "
          f"{'scipy ms':>9} {'pounce ms':>10}")
    print("-" * 60)
    for m in (11, 21, 41, 81):
        x = np.linspace(a, b, m)
        y0 = np.zeros((2, m))
        y0[0] = x / (np.pi / 2)

        r_sp = scipy_solve_bvp(fun, bc, x, y0)
        r_pc = pounce.solve_bvp(fun, bc, x, y0)

        xt = np.linspace(a, b, 201)
        err_sp = np.max(np.abs(r_sp.sol(xt)[0] - exact(xt)))
        err_pc = np.max(np.abs(r_pc.sol(xt)[0] - exact(xt)))

        t_sp = _time_it(lambda: scipy_solve_bvp(fun, bc, x, y0))
        t_pc = _time_it(lambda: pounce.solve_bvp(fun, bc, x, y0))

        print(f"{m:>6} | {err_sp:>11.2e} {err_pc:>11.2e} | "
              f"{t_sp * 1e3:>9.2f} {t_pc * 1e3:>10.2f}")

    print("\nNote: pounce matches scipy's accuracy exactly (same 4th-order "
          "Hermite--\nSimpson collocation). It is slower here because the "
          "NumPy path forms the\ncollocation Jacobian by dense finite "
          "differences (O(N) residual evals per\nIPM iteration); an exact "
          "sparse-Jacobian assembly is the obvious next step.\npounce uses "
          "the fixed mesh as given (no adaptive refinement); scipy refines.\n"
          "The payoff for the collocation-as-NLP formulation is "
          "differentiability ↓.")


# --------------------------------------------------------------------------
# Bratu problem (used for the differentiability demos)
#
#   y'' + lambda * exp(y) = 0,  y(0) = y(1) = 0.
# Lower-branch analytic solution:
#   y(x) = -2 ln( cosh((x - 1/2) c / 2) / cosh(c / 4) ),
# where c solves  c = sqrt(2 lambda) cosh(c / 4).
# --------------------------------------------------------------------------

def _bratu_c(lam):
    from scipy.optimize import brentq
    return brentq(lambda c: c - np.sqrt(2 * lam) * np.cosh(c / 4), 1e-6, 4.0)

def bratu_exact(x, lam):
    c = _bratu_c(lam)
    return -2.0 * np.log(np.cosh((x - 0.5) * c / 2.0) / np.cosh(c / 4.0))


# --------------------------------------------------------------------------
# 2. Differentiability (JAX frontend)
# --------------------------------------------------------------------------

def differentiability_jax():
    try:
        import jax
        import jax.numpy as jnp
        import pounce.jax as pj
    except ImportError:
        print("\n[JAX not installed — skipping JAX differentiability demos]")
        return

    _hr("2. Differentiability (pounce.jax.solve_bvp) vs finite differences")

    x = jnp.linspace(0, 1, 51)
    y0 = jnp.zeros((2, x.size))

    # ---- (a) gradient w.r.t. an ODE parameter (Bratu's lambda) ----
    def fun(xx, y, lam):
        return jnp.vstack((y[1], -lam * jnp.exp(y[0])))

    def bc(ya, yb, lam):
        return jnp.array([ya[0], yb[0]])

    def midpoint(lam):
        sol = pj.solve_bvp(fun, bc, x, y0, theta=lam)
        return sol.y[0, sol.y.shape[1] // 2]  # y at the domain midpoint

    lam = 1.0
    print("\n(a) d/dlambda of y(0.5) for Bratu  y'' + lambda e^y = 0")
    g = float(jax.grad(midpoint)(lam))
    fd = float((midpoint(lam + 1e-5) - midpoint(lam - 1e-5)) / 2e-5)
    print(f"    implicit grad : {g:.10f}")
    print(f"    finite diff   : {fd:.10f}")
    print(f"    rel error     : {abs(g - fd) / abs(fd):.2e}")
    # cross-check the forward solution against the analytic Bratu solution
    sol = pj.solve_bvp(fun, bc, x, y0, theta=lam)
    err = np.max(np.abs(np.asarray(sol.y[0]) - bratu_exact(np.asarray(x), lam)))
    print(f"    fwd soln max-err vs analytic Bratu: {err:.2e}")

    # ---- (b) gradient w.r.t. a boundary value ----
    def fun_h(xx, y, theta):
        return jnp.vstack((y[1], -y[0]))

    def bc_h(ya, yb, theta):
        return jnp.array([ya[0] - theta, yb[0]])  # y(0) = theta

    def energy(theta):
        sol = pj.solve_bvp(fun_h, bc_h, x, y0, theta=theta)
        return jnp.trapezoid(sol.y[0] ** 2, x)

    th = 0.8
    print("\n(b) d/dtheta of ∫ y^2 dx with boundary value y(0)=theta")
    g = float(jax.grad(energy)(th))
    fd = float((energy(th + 1e-5) - energy(th - 1e-5)) / 2e-5)
    print(f"    implicit grad : {g:.10f}")
    print(f"    finite diff   : {fd:.10f}")
    print(f"    rel error     : {abs(g - fd) / abs(fd):.2e}")

    # ---- (c) sensitivity of a solved-for unknown parameter p* ----
    # y'' + p^2 y = 0, y(0)=0, y(1)=0, y'(0)=theta (normalisation).
    # p* is the eigenvalue (~pi); the demo differentiates p* w.r.t theta.
    def fun_e(xx, y, p, theta):
        return jnp.vstack((y[1], -(p[0] ** 2) * y[0]))

    def bc_e(ya, yb, p, theta):
        return jnp.array([ya[0], yb[0], ya[1] - theta])

    xe = jnp.linspace(0, 1, 41)
    ye = jnp.zeros((2, xe.size))
    ye = ye.at[0].set(jnp.sin(jnp.pi * xe)).at[1].set(jnp.pi * jnp.cos(jnp.pi * xe))

    def pstar(theta):
        return pj.solve_bvp(fun_e, bc_e, xe, ye, p=[3.0], theta=theta).p[0]

    th = 1.5
    print("\n(c) eigenvalue p* and dp*/dtheta for y'' + p^2 y = 0")
    print(f"    p*(theta={th}) : {float(pstar(th)):.8f}  (exact pi = {np.pi:.8f})")
    g = float(jax.grad(pstar)(th))
    fd = float((pstar(th + 1e-5) - pstar(th - 1e-5)) / 2e-5)
    print(f"    dp*/dtheta implicit : {g:.3e}")
    print(f"    dp*/dtheta finite   : {fd:.3e}  (expect ~0: eigenvalue is "
          "scale-invariant)")

    # ---- (d) full Jacobian dy/dtheta of the whole solution ----
    def y_of_lambda(lam):
        return pj.solve_bvp(fun, bc, x, y0, theta=lam).y[0]  # (m,)

    print("\n(d) full Jacobian dy(x)/dlambda over the mesh (Bratu)")
    J = jax.jacobian(y_of_lambda)(1.0)            # (m,)
    base = np.asarray(y_of_lambda(1.0))
    fd_J = (np.asarray(y_of_lambda(1.0 + 1e-5)) - base) / 1e-5
    print(f"    max |implicit - finite diff| over mesh : "
          f"{np.max(np.abs(np.asarray(J) - fd_J)):.2e}")

    # ---- (e) gradient of a scalar loss w.r.t. a *vector* theta ----
    # theta = (lambda, boundary_value): both an ODE coefficient and a BC,
    # differentiated together in one reverse pass.
    def fun_v(xx, y, theta):
        return jnp.vstack((y[1], -theta[0] * jnp.exp(y[0])))

    def bc_v(ya, yb, theta):
        return jnp.array([ya[0] - theta[1], yb[0]])

    def loss_v(theta):
        sol = pj.solve_bvp(fun_v, bc_v, x, y0, theta=theta)
        return jnp.trapezoid(sol.y[0] ** 2, x)

    th_v = jnp.array([1.0, 0.2])
    print("\n(e) gradient w.r.t. a vector theta=(lambda, y(0)) in one pass")
    g_v = np.asarray(jax.grad(loss_v)(th_v))
    fd_v = np.array([
        float((loss_v(th_v.at[i].add(1e-5)) - loss_v(th_v.at[i].add(-1e-5))) / 2e-5)
        for i in range(2)
    ])
    print(f"    implicit grad : [{g_v[0]:.8f}, {g_v[1]:.8f}]")
    print(f"    finite diff   : [{fd_v[0]:.8f}, {fd_v[1]:.8f}]")
    print(f"    max rel error : {np.max(np.abs(g_v - fd_v) / np.abs(fd_v)):.2e}")

    # ---- (f) second derivative through the solver (second_order=True) ----
    def y_mid_so(lam):
        sol = pj.solve_bvp(fun, bc, x, y0, theta=lam, second_order=True)
        return sol.y[0, sol.y.shape[1] // 2]

    print("\n(f) second derivative d^2 y(0.5)/dlambda^2 (second_order=True)")
    h2 = float(jax.grad(jax.grad(y_mid_so))(1.0))
    dd = 1e-4
    fd2 = float((jax.grad(y_mid_so)(1.0 + dd) - jax.grad(y_mid_so)(1.0 - dd))
                / (2 * dd))
    print(f"    implicit Hessian : {h2:.8f}")
    print(f"    finite diff      : {fd2:.8f}")
    print(f"    rel error        : {abs(h2 - fd2) / abs(fd2):.2e}")

    print("\nNote: first-order gradients/Jacobians w.r.t. any theta work by "
          "default.\nPass second_order=True to also take jax.grad(jax.grad) / "
          "jax.hessian\nthrough the solve (a custom_jvp re-applies the "
          "implicit function theorem).")


# --------------------------------------------------------------------------
# 3. Differentiability (PyTorch frontend)
# --------------------------------------------------------------------------

def differentiability_torch():
    try:
        import torch
        torch.set_default_dtype(torch.float64)
        import pounce.torch as pt
    except ImportError:
        print("\n[PyTorch not installed — skipping Torch differentiability demo]")
        return

    _hr("3. Differentiability (pounce.torch.solve_bvp) vs finite differences")

    x = torch.linspace(0, 1, 51, dtype=torch.float64)
    y0 = torch.zeros((2, x.shape[0]), dtype=torch.float64)

    def fun(xx, y, lam):
        return torch.vstack((y[1], -lam * torch.exp(y[0])))

    def bc(ya, yb, lam):
        return torch.stack([ya[0], yb[0]])

    def midpoint(lam):
        sol = pt.solve_bvp(fun, bc, x, y0, theta=lam)
        return sol.y[0, sol.y.shape[1] // 2]

    print("\nd/dlambda of y(0.5) for Bratu  y'' + lambda e^y = 0")
    lam = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    out = midpoint(lam)
    out.backward()
    g = float(lam.grad)
    with torch.no_grad():
        fp = float(midpoint(torch.tensor(1.0 + 1e-5, dtype=torch.float64)))
        fm = float(midpoint(torch.tensor(1.0 - 1e-5, dtype=torch.float64)))
    fd = (fp - fm) / 2e-5
    print(f"    implicit grad : {g:.10f}")
    print(f"    finite diff   : {fd:.10f}")
    print(f"    rel error     : {abs(g - fd) / abs(fd):.2e}")


if __name__ == "__main__":
    accuracy_and_speed()
    differentiability_jax()
    differentiability_torch()
    print()
