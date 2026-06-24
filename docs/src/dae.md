# Fully-implicit DAEs

`pounce.ode.solve_dae` integrates a **fully-implicit, index-1**
differential-algebraic equation

\\[ F(t, y, y') = 0 \\]

with the same Radau IIA(5) collocation as [`solve_ivp`](./ode.md), written in
residual form. This is a pounce extension: `scipy.integrate.solve_ivp` has no
fully-implicit DAE solver (its closest relative, the mass-matrix form
`M y' = f`, is also available via `solve_ivp(..., mass=M)`).

```python
import numpy as np
from pounce.ode import solve_dae

# Robertson kinetics as an index-1 DAE: two rate equations + a conservation law.
k1, k2, k3 = 0.04, 3.0e7, 1.0e4
def F(t, y, yp):
    return np.array([
        yp[0] - (-k1*y[0] + k3*y[1]*y[2]),
        yp[1] - ( k1*y[0] - k3*y[1]*y[2] - k2*y[1]**2),
        y[0] + y[1] + y[2] - 1.0,            # algebraic constraint (no y')
    ])

res = solve_dae(F, (0.0, 1e4), y0=[1.0, 0.0, 0.0], rtol=1e-8, atol=1e-10)
print(res.y[:, -1], res.y[:, -1].sum())     # constraint held to round-off
```

## Consistent initial conditions

A DAE solve needs `(y0, y'0)` with `F(t0, y0, y'0) = 0`. By default
(`consistent="project"`) `solve_dae` computes them for you: it detects which
variables are **algebraic** (those that `F` does not depend on `y'` for — a
structurally-zero column of `∂F/∂y'`) and Newton-projects onto the constraint
manifold, holding the differential `y` and algebraic `y'` fixed and solving for
the differential `y'` and algebraic `y` (the IDA `IDA_YA_YDP_INIT` computation).
So a rough `y0` (even one off the constraint) and `yp0=None` are fine:

```python
# y0 violates the constraint (sum = 1.5) and no derivative guess is given —
# both are projected to a consistent state before integrating.
solve_dae(F, (0.0, 1e4), y0=[1.0, 0.0, 0.5], yp0=None)
```

Pass `consistent="assume"` with an explicit `yp0` to skip the projection (you
guarantee `F(t0, y0, yp0) == 0`).

## Jacobians

`jac(t, y, yp) -> (∂F/∂y, ∂F/∂y')` is optional; both blocks are
finite-differenced (`2n` evaluations) when omitted. Supplying them avoids the
FD cost and improves robustness on stiff problems.

## Scope

- **Index-1 only.** The stage matrix `I₃⊗∂F/∂y' + h(A⊗∂F/∂y)` stays nonsingular
  for index-1 problems; higher index needs index reduction (not done here).
- Same adaptive Radau engine as `solve_ivp` — stiff-capable, sparse-LU stage
  solve, dense output (`dense_output=True` / `t_eval=`), `args=`.
- Events are not supported.

## Differentiable integration (JAX / PyTorch)

`pounce.jax.daeint` / `pounce.torch.daeint` integrate `F(t, y, y', theta) = 0`
on a **fixed mesh** and return the node trajectory differentiable w.r.t. the
parameters `theta` **and** the initial condition `y0`, via the
implicit-function theorem on the collocation system. As with
`pounce.jax.odeint`, the mesh is fixed (keeping the solution map smooth);
accuracy is controlled by the mesh, and the scheme is backward Euler (L-stable,
order 1 — refine the mesh for accuracy). `F` must be framework-traceable.

```python
import jax, jax.numpy as jnp
from pounce.jax import daeint

def F(t, y, yp, theta):                 # y0' + theta*y0 - y1 = 0 ; y0 + y1 = 1
    return jnp.array([yp[0] + theta*y[0] - y[1], y[0] + y[1] - 1.0])

t  = jnp.linspace(0.0, 2.0, 81)
y0 = jnp.array([0.5, 0.5])
loss = lambda th: daeint(F, y0, t, th)[0, -1] ** 2
g = jax.grad(loss)(1.3)                  # exact for the discretisation
```

The forward solve and the `R_yᵀ` back-solve run on the host (FERAL sparse LU);
the parameter VJP is taken by framework autodiff of the collocation residual at
the converged nodes. Gradients are validated against finite differences in the
test suite (`python/tests/test_dae.py`).
