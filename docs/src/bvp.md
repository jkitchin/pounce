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

> A runnable tour of every feature is in
> [`python/notebooks/24_boundary_value_problems.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/24_boundary_value_problems.ipynb),
> and a SciPy speed/accuracy comparison in
> `python/examples/bvp_scipy_compare.py` (plus the GLC tritium-column case in
> `python/examples/glc_feral_vs_scipy.py`).

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

- **Mesh.** `adaptive=False` (default) solves on the mesh you pass — fast and
  predictable, and what the differentiable frontends rely on (a fixed mesh
  keeps `θ ↦ y` smooth). `adaptive=True` turns on SciPy-style residual-driven
  refinement (below).
- **Solver (`method`).** `method="newton"` (default) runs a **modified
  (frozen-Jacobian) Newton** on the square collocation system, factorising
  the `N×N` Jacobian with FERAL's unsymmetric sparse LU
  (`pounce._pounce.SparseLU`) and reusing that factor across steps
  (refactoring only when progress stalls — the same trick SciPy's
  `solve_newton` uses). Both scale linearly in the mesh; at equal mesh
  pounce is **typically faster than SciPy** (≈0.6–1.0×), including large
  nonlinear problems, because the factorisation dominates and it does far
  fewer of them. The Jacobian is the exact **sparse**
  collocation Jacobian (analytic per-node `∂f/∂y` blocks from
  `fun_jac`/`bc_jac` if supplied, else a vectorised finite difference that
  perturbs each state across the whole mesh — `O(n)` `fun` calls, not
  `O(n·m)`). `method="ipm"` instead poses the system as a pounce feasibility
  NLP and solves with the interior-point method (factoring the `2N` saddle
  KKT each iteration — slower, but the basis for the constrained solver
  below). Accuracy is identical to SciPy either way.
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

### Differentiable solver backends (`method`)

The differentiable `solve_bvp` (both `pounce.jax` and `pounce.torch`) takes
the same `method` switch:

- **`method="newton"` (default)** — the fast path. Forward is the FERAL
  sparse-LU Newton solve; the backward is the implicit-function-theorem VJP
  `dz/dθ = −R_z⁻¹ R_θ`, solving `R_zᵀu = v` with the same sparse LU
  (`SparseLU.solve_transpose`). Both directions stay on the `N` system — no
  `2N` saddle — so it is fast *and* differentiable. **First-order only**
  (the forward is an opaque callback).
- **`method="ipm"`** — routes the forward through `pounce.jax.solve` /
  `pounce.torch.solve` (the interior-point feasibility NLP). Needed for
  second-order derivatives (below).

### Second-order derivatives

With `method="ipm"`, pass **`second_order=True`** to wrap the solve in a
`custom_jvp` whose tangent rule re-applies the implicit-function theorem to
the square collocation root-find,

```text
dz/dθ = -(∂R/∂z)⁻¹ (∂R/∂θ),
```

and recovers `z*` through the *same* custom-ruled primitive, so JAX
recurses to arbitrary order:

```python
def y_mid(lam):
    sol = pj.solve_bvp(fun, bc, x, y0, theta=lam,
                       method="ipm", second_order=True)
    return sol.y[0, sol.y.shape[1] // 2]

jax.grad(jax.grad(y_mid))(1.0)     # d²y(0.5)/dλ²  — works
```

The cost is one extra forward solve per differentiation level (the rule
re-solves to recover `z*`); the opaque forward is still only evaluated for
primal values. Leave it off for plain gradient-based training; turn it on
for Hessians / Newton-type outer loops.

## Adaptive mesh refinement

By default the solve is **fixed-mesh**. Pass `adaptive=True` for SciPy-style
refinement driven by `tol` / `max_nodes`:

```python
res = pounce.solve_bvp(fun, bc, x, y0, tol=1e-6, adaptive=True, max_nodes=2000)
```

Each round: solve on the current mesh (to round-off), estimate the relative
RMS residual of the continuous solution per interval with a 5-point Lobatto
quadrature at the *superconvergent* Gauss points `x_mid ± ½h√(3/7)`, insert
nodes where it exceeds `tol` (one node, or two if it's >100× over), and
re-solve warm-started off the previous solution. This is a faithful port of
SciPy's estimator and refinement rule, so it reproduces SciPy's mesh
sequence essentially node-for-node:

| problem | SciPy nodes | pounce nodes | solution agreement |
| --- | --- | --- | --- |
| `y''+y=0` | 6 → 31 | 6 → 31 | 1e-16 |
| Bratu | 5 → 29 | 5 → 29 | 6e-17 |
| `y''=-|y|` (kink) | 11 → 58 | 11 → 58 | 9e-16 |

Adaptive is **numpy-only** — the differentiable `pounce.jax` /
`pounce.torch` paths are always fixed-mesh, because a parameter-dependent
mesh would make `y(θ)` nonsmooth and break the gradients. Pick a fixed mesh
fine enough for your `θ` range, or run an adaptive solve once to size it.

## Constrained / optimal-control BVPs (pounce-unique)

`pounce.solve_bvp_constrained` solves a collocation BVP **subject to bounds
on the states/parameters and inequality path constraints**, optionally
minimising an objective:

```text
dy/dx = f(x, y, p),  bc(y(a), y(b), p) = 0
ylo <= y(x) <= yhi              (state bounds, every node)
clo <= c(x, y, p) <= chi        (path constraints, every node)
minimise  J(Y, p)               (optional)
```

This is a genuine NLP, so it goes through pounce's interior-point method
(not the Newton path), and **SciPy's `solve_bvp` cannot express any of it**.
A fully determined BVP (`n + k` boundary residuals) has a unique solution,
so constraints only bite when there is freedom — return *fewer* boundary
residuals and let the objective resolve the remainder (an optimal-control
collocation):

```python
import numpy as np, pounce

# minimise ∫(y-1)² s.t. y''=0, y(0)=0  (slope free) — optimal control.
def fun(x, y): return np.vstack((y[1], np.zeros_like(y[0])))
def bc(ya, yb): return np.array([ya[0]])          # one boundary residual → 1 DOF
x = np.linspace(0, 1, 41); y0 = np.zeros((2, x.size)); y0[0] = x

obj = lambda Y, p: np.trapezoid((Y[0] - 1.0) ** 2, x)

r  = pounce.solve_bvp_constrained(fun, bc, x, y0, objective=obj)        # y(1) ≈ 1.5
rc = pounce.solve_bvp_constrained(fun, bc, x, y0, objective=obj,
                                  y_bounds=([-np.inf, -np.inf], [1.2, np.inf]))
rc.y[0].max()    # ≤ 1.2 — the bound is active and respected
```

`path=path(x, Y, p) -> (q, m)` with `path_bounds=(clo, chi)` adds inequality
path constraints at every node (assembled with a sparse block-diagonal
Jacobian). The objective's gradient is finite-differenced; the Lagrangian
Hessian uses pounce's limited-memory quasi-Newton (the path constraints
make it nonzero in general).

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
