# Boundary Value Problems

`pounce.bvp.solve_bvp` solves two-point boundary value problems

```text
dy/dx = f(x, y, p),    a ≤ x ≤ b
bc(y(a), y(b), p) = 0
```

with a **drop-in** for [`scipy.integrate.solve_bvp`][scipy-bvp]. It
discretises the problem with the 4th-order Lobatto IIIA (Hermite–Simpson)
collocation scheme — the same one SciPy uses — and solves the resulting
square root-find as a **pounce feasibility NLP** (`min 0` subject to the
collocation residual `R(z) = 0`).

The motivation is **differentiability**: because the discretised problem is
an NLP, the converged solution `z*(θ)` is differentiable with respect to
any parameter `θ` baked into `f` or `bc`, via the implicit-function theorem
on the collocation KKT system. The differentiable entry points live in the
autodiff frontends, `pounce.jax.solve_bvp` and `pounce.torch.solve_bvp`.

[scipy-bvp]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html

## Drop-in NumPy solve

```python
import numpy as np
import pounce

# y'' = -|y|, y(0) = 0, y(4) = -2
def fun(x, y):
    return np.vstack((y[1], -np.abs(y[0])))

def bc(ya, yb):
    return np.array([ya[0], yb[0] + 2.0])

x = np.linspace(0, 4, 41)
y0 = np.zeros((2, x.size)); y0[0] = 1.0

res = pounce.solve_bvp(fun, bc, x, y0)
print(res.success, res.rms_residuals.max())
res.sol(np.linspace(0, 4, 9))   # cubic-Hermite interpolant, shape (n, 9)
```

The call signature and the returned bunch (`sol`, `x`, `y`, `yp`, `p`,
`rms_residuals`, `niter`, `status`, `message`, `success`) match SciPy, so
existing code consumes the result unchanged. Unknown parameters work the
same way — pass `p=[...]` and a `fun(x, y, p)` / `bc(ya, yb, p)`:

```python
# Eigenvalue: y'' + k² y = 0, y(0)=y(1)=0, y'(0)=k
def fun(x, y, p): return np.vstack((y[1], -p[0]**2 * y[0]))
def bc(ya, yb, p): return np.array([ya[0], yb[0], ya[1] - p[0]])

res = pounce.solve_bvp(fun, bc, x, y0, p=[3.0])
res.p           # ≈ [π]
```

### Differences from SciPy

- **Fixed mesh.** The mesh you pass is used as-is — there is no adaptive
  refinement. This keeps the solution map `θ ↦ y` smooth, which is what the
  differentiable frontends exploit. Refine by passing a denser `x`;
  `max_nodes` is accepted for signature compatibility.
- **Derivatives.** The collocation Jacobian handed to the interior-point
  solver is formed by forward finite differences of the residual, and the
  Hessian uses pounce's limited-memory quasi-Newton approximation.
  `fun_jac` / `bc_jac` are accepted for signature compatibility (an exact
  sparse-Jacobian assembly is a planned optimisation). Accuracy is
  identical to SciPy (same collocation); the NumPy path is currently
  slower because of the dense finite-difference Jacobian.
- **Singular term `S`** is not yet supported.

## Differentiable solves (JAX / PyTorch)

The differentiable frontends take `fun(x, y, p, theta)` / `bc(ya, yb, p,
theta)` (drop `p` when there are no unknown parameters), where `theta` is
the autodiff knob, and return a solution whose `y` / `p` participate in the
autodiff graph. Everything `fun` / `bc` close over is differentiable: a
physical coefficient, a boundary value, or the sensitivity of a solved-for
unknown parameter.

```python
import jax, jax.numpy as jnp
import pounce.jax as pj

# Bratu: y'' + λ e^y = 0, y(0)=y(1)=0
def fun(x, y, lam): return jnp.vstack((y[1], -lam * jnp.exp(y[0])))
def bc(ya, yb, lam): return jnp.array([ya[0], yb[0]])

x = jnp.linspace(0, 1, 51)
y0 = jnp.zeros((2, x.size))

def y_mid(lam):
    sol = pj.solve_bvp(fun, bc, x, y0, theta=lam)
    return sol.y[0, sol.y.shape[1] // 2]

grad = jax.grad(y_mid)(1.0)          # d y(0.5) / d λ
J = jax.jacobian(lambda l: pj.solve_bvp(fun, bc, x, y0, theta=l).y[0])(1.0)
```

The PyTorch frontend mirrors this exactly:

```python
import torch
import pounce.torch as pt
torch.set_default_dtype(torch.float64)

lam = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
sol = pt.solve_bvp(fun, bc, x, y0, theta=lam)  # fun/bc written with torch ops
sol.y[0, 25].backward()
lam.grad
```

### What's differentiable

| Target | How | Demo |
| --- | --- | --- |
| ODE/BC coefficient `θ` | `jax.grad` / `.backward()` through `sol.y` | `examples/bvp_scipy_compare.py` (a) |
| Boundary value | put it in `bc` and differentiate `θ` | (b) |
| Solved-for unknown `p*` | differentiate `sol.p` | (c) |
| Full solution `dy/dθ` | `jax.jacobian` over `sol.y` | (d) |
| Vector `θ` | one reverse pass | (e) |
| Second derivative / Hessian | `second_order=True` | (f) |

All of these are validated against finite differences to ~1e-11 in
`python/examples/bvp_scipy_compare.py`, which also benchmarks accuracy and
speed against SciPy.

### Second-order derivatives

By default `solve_bvp` routes through `pounce.jax.solve`'s first-order
`custom_vjp`, whose forward crosses a `pure_callback` (no JVP rule) — so
`jax.grad(jax.grad(...))` / `jax.hessian` raise. Pass **`second_order=True`**
to wrap the solve in a `custom_jvp` whose tangent rule re-applies the
implicit-function theorem to the square collocation root-find,

```text
dz/dθ = -(∂R/∂z)⁻¹ (∂R/∂θ),
```

and recovers `z*` through the *same* custom-ruled primitive, so JAX
recurses to arbitrary order:

```python
def y_mid(lam):
    sol = pj.solve_bvp(fun, bc, x, y0, theta=lam, second_order=True)
    return sol.y[0, sol.y.shape[1] // 2]

jax.grad(jax.grad(y_mid))(1.0)     # d²y(0.5)/dλ²  — works
```

The cost is one extra forward solve per differentiation level (the rule
re-solves to recover `z*`); the opaque forward is still only evaluated for
primal values. Leave it off for plain gradient-based training; turn it on
for Hessians / Newton-type outer loops.

## How it works

For a mesh `x₀ < … < x_{m-1}`, each interval contributes the Hermite–Simpson
collocation residual

```text
y_mid = (y_i + y_{i+1})/2 - h/8 (f_{i+1} - f_i)
r_i   = y_{i+1} - y_i - h/6 (f_i + 4 f(x_mid, y_mid) + f_{i+1})  = 0
```

Stacking the `n·(m-1)` collocation residuals with the `n + k` boundary
residuals gives a **square** system `R(z) = 0` in the unknowns
`z = [vec(Y); p]` of size `N = n·m + k`. pounce solves it as `min 0` s.t.
`R(z) = 0`. At the solution the interior-point method holds the KKT factor
of `[[H, Jᵀ], [J, 0]]` with `J = ∂R/∂z`; for this all-equality, no-bounds,
zero-objective problem the generic
[implicit-diff backward](differentiable-solves.md) collapses to the Newton
sensitivity

```text
dz*/dθ = -(∂R/∂z)⁻¹ (∂R/∂θ),
```

which is exactly what `jax.grad` / autograd return — no BVP-specific
backward code. The collocation residual itself is shared verbatim across
the NumPy, JAX, and PyTorch paths (`pounce/bvp/_core.py`).
