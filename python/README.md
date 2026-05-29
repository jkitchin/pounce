# pounce — Python interface

`pounce` is a Python wrapper around POUNCE, a pure-Rust port of the
[Ipopt](https://github.com/coin-or/Ipopt) interior-point nonlinear
programming solver. The Python surface area is intentionally
cyipopt-compatible: code written for cyipopt typically runs against
pounce by changing only the import.

## Install (development)

```sh
# from the repo root:
cd python
pip install maturin
maturin develop --release            # builds the native extension into your venv
# optional extras:
pip install -e .[jax]                # jax integration
pip install -e .[dev]                # tests + jax + scipy
```

## Quick start (cyipopt-style)

```python
import numpy as np
import pounce

class HS071:
    def objective(self, x):
        return x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]
    def gradient(self, x):
        return np.array([
            x[0]*x[3] + x[3]*(x[0]+x[1]+x[2]),
            x[0]*x[3],
            x[0]*x[3] + 1.0,
            x[0]*(x[0]+x[1]+x[2]),
        ])
    def constraints(self, x):
        return np.array([np.prod(x), np.dot(x, x)])
    def jacobianstructure(self):
        return (np.repeat([0,1], 4), np.tile([0,1,2,3], 2))
    def jacobian(self, x):
        return np.array([
            x[1]*x[2]*x[3], x[0]*x[2]*x[3], x[0]*x[1]*x[3], x[0]*x[1]*x[2],
            2*x[0], 2*x[1], 2*x[2], 2*x[3],
        ])

prob = pounce.Problem(
    n=4, m=2,
    problem_obj=HS071(),
    lb=[1]*4, ub=[5]*4,
    cl=[25, 40], cu=[2e19, 40],
)
prob.add_option('tol', 1e-8)
x, info = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))
print(info['status_msg'], info['obj_val'], x)
```

## Problem scaling

For problems whose natural variable and constraint magnitudes differ
by orders of magnitude, attach explicit scaling factors:

```python
prob.set_problem_scaling(
    obj_scaling=1.0,
    x_scaling=np.array([1e-3, 1.0, 1.0, 1e3]),  # optional
    g_scaling=np.array([1.0, 1e-2]),            # optional
)
prob.add_option('nlp_scaling_method', 'user-scaling')
```

See [`docs/src/scaling.md`](../docs/src/scaling.md) for the
gradient-based vs. user-scaling tradeoff.

## Sensitivity analysis (sIPOPT-compatible)

`solve_with_sens` runs the standard solve, then a post-optimal
sensitivity step on the converged KKT factor — no second solve:

```python
x_star, info, sens = prob.solve_with_sens(
    x0=x0,
    perturbed_indices=[2, 3],     # which constraints are parametric
    deltas=[0.05, 0.0],            # perturbation magnitudes
    rh_eigendecomp=True,           # also return reduced-Hessian eigendecomp
    sens_boundcheck=True,
)
# sens.dx, sens.reduced_hessian, sens.reduced_hessian_eigvals
```

## Factor-once / solve-many: `Solver`

For workflows that issue several follow-up operations against the
converged KKT factor (sensitivity sweeps, reduced Hessians over many
pin sets, raw back-solves), `pounce.Solver` keeps the factor alive
between calls:

```python
solver = pounce.Solver(prob)
x_star, info = solver.solve(x0=x0)

# Reuse the factor for downstream queries:
dx = solver.parametric_step(perturbed_indices=[2], deltas=[0.05])
rh = solver.reduced_hessian(pin_constraint_indices=[0, 1])
y  = solver.kkt_solve(rhs)             # raw KKT back-solve
print(solver.kkt_dim(), solver.converged())
```

The full walk-through is in
[`docs/src/sessions.md`](../docs/src/sessions.md).

## Warm-start working sets

For active-set or repeated-solve workflows, the working set (the
guess at which constraints are active) can be pinned across solves:

```python
prob.set_working_set(working_set)
x, info = prob.solve(x0=x0)
ws_out = prob.get_working_set()
# Or classify a fresh iterate:
ws = pounce.classify_working_set(...)
```

## scipy.optimize-style

```python
from pounce import minimize
res = minimize(lambda x: (x-1)**2 @ (x-1) + 1, x0=np.zeros(5))
print(res.fun, res.x)
```

## JAX integration

`pounce.jax` exposes five entry points: `from_jax`, `solve`,
`solve_with_warm`, `vmap_solve` / `vmap_solve_parallel`, and
`JaxProblem`.

```python
import jax, jax.numpy as jnp
from pounce.jax import from_jax

def f(x): return jnp.sum((x-1)**2)
def g(x): return jnp.stack([jnp.sum(x) - 5.0])

prob = from_jax(f, g, n=4, m=1, lb=jnp.zeros(4), ub=jnp.full(4, 10.0),
                cl=jnp.zeros(1), cu=jnp.zeros(1))
x, info = prob.solve(x0=jnp.ones(4))
```

Differentiate through the solver (the backward respects the active
set, so slack inequalities don't pollute the gradient — pounce#73):

```python
from pounce.jax import solve as psolve

def f_p(x, p): return jnp.sum((x - p) ** 2)
def g_p(x, p): return jnp.stack([x[0] + x[1] - 1.0])

def loss(p):
    x_star = psolve(p, f=f_p, g=g_p, x0=jnp.zeros(2), n=2, m=1,
                    lb=jnp.full(2, -10.0), ub=jnp.full(2, 10.0),
                    cl=jnp.zeros(1), cu=jnp.zeros(1),
                    options={"tol": 1e-10, "print_level": 0})
    return jnp.sum(x_star ** 2)

dloss_dp = jax.grad(loss)(jnp.array([0.3, 0.7]))
```

Warm-start across a parameter trajectory and batch solves in parallel
(pounce#74):

```python
from pounce.jax import solve_with_warm, vmap_solve_parallel

x, warm = solve_with_warm(p0, f=f_p, g=g_p, x0=jnp.zeros(2), n=2, m=1,
                          lb=..., ub=..., cl=..., cu=...,
                          warm_start=None)
x, warm = solve_with_warm(p1, f=f_p, g=g_p, x0=x, n=2, m=1,
                          lb=..., ub=..., cl=..., cu=...,
                          warm_start=warm)   # reuse λ, z

X = vmap_solve_parallel(p_batch, f=f_p, g=g_p, x0=jnp.zeros(2), n=2, m=1,
                        lb=..., ub=..., cl=..., cu=..., workers=8)
```

For iterative use, `JaxProblem` builds the JIT artefacts, sparsity
probe, and underlying `pounce.Problem` once and reuses them across
calls — ~14× per-solve speedup on small problems (pounce#75):

```python
from pounce.jax import JaxProblem

jp = JaxProblem(f=f_p, g=g_p, n=2, m=1, p_example=jnp.zeros(2),
                lb=..., ub=..., cl=..., cu=...,
                options={"tol": 1e-9, "print_level": 0})

x = jp.solve(p0, x0=jnp.zeros(2))                       # differentiable
x, warm = jp.solve_with_warm(p0, x0=x, warm_start=None) # trajectory
X = jp.vmap_solve_parallel(p_batch, x0=jnp.zeros(2), workers=8)
X = jp.batched_solve(p_batch, x0=jnp.zeros(2))          # one stacked IPM (pounce#76)
```

`batched_solve` runs *one* IPM over a stacked block-diagonal NLP
(variables `[x^(1); ...; x^(B)]`, constraints
`concat(g(x^(k), p^(k)))`, objective `Σ_k f(x^(k), p^(k))`). One
shared barrier homotopy and symbolic factorisation across the batch
— complementary to `vmap_solve_parallel`, which runs B independent
IPMs in worker threads. `jax.grad` through it works end-to-end (the
backward vmaps the per-element dense KKT back-solve, exact because
the block-diagonal coupling makes `∂x^(k)*/∂p^(j) = 0` for `k ≠ j`).

The `jp.solve` / `solve_with_warm` backward defaults to a k_aug-style
factor-reuse path: instead of assembling a dense `(n+m) × (n+m)` KKT
block in JAX and running `jnp.linalg.solve` on it, it reuses the IPM's
converged LDLᵀ factor via `pounce.Solver.kkt_solve` (pounce#76). The
compound block's barrier rows on `(z_l, z_u)` and `(v_l, v_u)`
encode active bounds and slack inequalities exactly, so no explicit
active-set masking is needed. Set `JaxProblem(..., factor_reuse=False)`
for the verbatim dense path (needed for higher-order differentiation).
