# ODE / DAE Initial Value Problems

`pounce.ode.solve_ivp` integrates stiff initial value problems

```text
M y' = f(t, y),    y(t0) = y0
```

as a **drop-in** for [`scipy.integrate.solve_ivp`][scipy-ivp] with the
implicit `Radau` method. It implements the 3-stage **Radau IIA** collocation
scheme (order 5, L-stable) — the same method SciPy's `Radau` uses, and the
classic `RADAU5` of Hairer & Wanner. Each step's coupled stage system is
solved by a simplified Newton iteration whose Jacobian is factored with
[FERAL][feral]'s sparse LU.

Two things set it apart from SciPy:

* **Mass matrix / DAEs.** Pass `mass=M` to integrate `M y' = f`. When `M` is
  **singular** this is an **index-1 differential-algebraic equation** —
  something `scipy.integrate.solve_ivp` cannot do at all.
* **Differentiability.** `pounce.jax.odeint` and `pounce.torch.odeint`
  integrate on a fixed mesh and return the trajectory differentiably with
  respect to the ODE parameters **and** the initial condition, via the
  implicit-function theorem on the collocation system (no per-step adjoint,
  no unrolled tape).

`solve_ivp` only implements `method="Radau"` — the implicit, stiff/DAE
capable method that is pounce's niche. For non-stiff explicit integration,
SciPy or [diffrax][diffrax] are the right tools, and `solve_ivp` raises for
those methods rather than silently substituting.

[scipy-ivp]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
[feral]: https://github.com/jkitchin/feral
[diffrax]: https://github.com/patrick-kidger/diffrax

> A SciPy speed/accuracy comparison, a DAE example, and a differentiability
> demo are in `python/examples/ode_scipy_compare.py`.

## Drop-in stiff solve

```python
import numpy as np
import pounce.ode as po

# Van der Pol, mu = 1000 (very stiff)
mu = 1000.0
def f(t, y):
    return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

res = po.solve_ivp(f, (0.0, 3000.0), [2.0, 0.0],
                   method="Radau", rtol=1e-6, atol=1e-8, dense_output=True)

print(res.t.shape, res.y.shape)   # (nsteps,) (2, nsteps)
ys = res.sol(np.linspace(0, 3000, 1000))   # continuous extension
```

The call signature and the returned object match SciPy: `res.t`, `res.y`
`(n, n_points)`, `res.sol` (when `dense_output=True`), `res.nfev` /
`res.njev` / `res.nlu`, `res.status` / `res.message` / `res.success`. The
result is also dict-subscriptable like SciPy's `Bunch`, so `res["y"]` and
`"success" in res` work too.

Provide an analytic Jacobian with `jac=...` (else it is estimated by finite
differences), and the usual `t_eval`, `args`, `first_step`, `max_step`,
`rtol`, `atol` controls.

## Index-1 DAE via a mass matrix

A singular mass matrix turns the same solver into a DAE integrator. Robertson
kinetics, written with the conservation law as an algebraic constraint:

```python
import numpy as np
import pounce.ode as po

k1, k2, k3 = 0.04, 3e7, 1e4
def f(t, y):
    return [-k1*y[0] + k3*y[1]*y[2],
             k1*y[0] - k3*y[1]*y[2] - k2*y[1]**2,
             y[0] + y[1] + y[2] - 1.0]      # 0 = ...  (algebraic)

M = np.diag([1.0, 1.0, 0.0])                # third equation is algebraic
res = po.solve_ivp(f, (0, 1e4), [1.0, 0.0, 0.0], mass=M,
                   rtol=1e-6, atol=1e-8)
```

The algebraic constraint is satisfied to round-off at every accepted step.

## Differentiable integration (JAX / PyTorch)

For gradient-based work — fitting ODE parameters, neural ODEs, optimal
control — use the autodiff frontends. They integrate on a **fixed mesh**
`t` (make it fine enough to resolve the dynamics) and return the trajectory
differentiably w.r.t. the parameters `theta` *and* the initial condition
`y0`:

```python
import jax, jax.numpy as jnp
import pounce.jax as pj

def f(t, y, theta):           # dy/dt, JAX-traceable
    k = theta[0]
    return jnp.array([-k * y[0]])

t = jnp.linspace(0.0, 2.0, 81)

def y_final(k):
    sol = pj.odeint(f, jnp.array([1.0]), t, jnp.array([k]))
    return sol.y[0, -1]

val  = y_final(0.7)            # = exp(-0.7 * 2)
grad = jax.grad(y_final)(0.7)  # exact d/dk via the implicit-function theorem
```

The PyTorch mirror is `pounce.torch.odeint`, with `theta`/`y0` as tensors and
`.backward()` filling `theta.grad` / `y0.grad`. Both return a solution whose
`y` is `(n, m)` in SciPy layout and carries the autodiff graph; `sol` is a
(detached) cubic-Hermite interpolant for plotting.

Under the hood an IVP on a fixed mesh is just a boundary value problem with
`bc(ya, yb) = ya - y0`, so the differentiable path reuses pounce's
Hermite–Simpson collocation and the same FERAL sparse-LU implicit-diff
back-solve as [`pounce.jax.solve_bvp`](bvp.md). The result is the collocation
solution on the mesh you pass, and its gradients are **exact** for that
discretisation.

## What it is and isn't

* It is a faithful, L-stable Radau IIA(5) implementation that tracks SciPy's
  `Radau` step-for-step on stiff problems and adds DAE and differentiability
  support SciPy lacks.
* It is **not** a general non-stiff integrator: only `method="Radau"` is
  implemented. Event detection (`events=`) is not yet supported.
* The differentiable layer is **fixed-mesh** (the mesh keeps `theta → y`
  smooth); the adaptive solver is the non-differentiable `solve_ivp`.
