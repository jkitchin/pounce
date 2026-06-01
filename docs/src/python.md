# Python API

POUNCE ships a Python wrapper that is intentionally cyipopt-compatible:
code written for cyipopt typically runs against POUNCE by changing only
the import.

## Install

```sh
cd python
pip install maturin
maturin develop --release    # builds the native extension into your venv
```

Optional extras:

```sh
pip install -e .[jax]        # JAX integration
pip install -e .[dev]        # tests + jax + scipy
```

## cyipopt-style interface

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
        return (np.repeat([0, 1], 4), np.tile([0, 1, 2, 3], 2))
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

## scipy.optimize-style

```python
import numpy as np
from pounce import minimize

res = minimize(lambda x: (x - 1) @ (x - 1) + 1, x0=np.zeros(5))
print(res.fun, res.x)
```

## JAX integration

The `pounce.jax` subpackage provides five entry points:

| Surface | Use it for |
|---------|-----------|
| `from_jax(f, g, …)` | Build a one-shot `pounce.Problem` from JAX-traced `f(x)` and `g(x)`. |
| `solve(p, …)`       | `custom_vjp`-wrapped differentiable solve over a parameter `p`. |
| `solve_with_warm(p, …, warm_start=)` | `solve` + dual-triple (`x, λ, z`) warm-start hand-off across calls. |
| `vmap_solve(p_batch, …)` / `vmap_solve_parallel(…)` | Batched solve over a leading axis of `p`; the `_parallel` variant uses a `ThreadPoolExecutor` and releases the GIL inside each solve. |
| `JaxProblem(f, g, n, m, p_example=, …)` | Build-once / solve-many handle that caches JIT artefacts, the sparsity probe, and the underlying `pounce.Problem` across calls. |

### One-shot build with `from_jax`

```python
import jax.numpy as jnp
from pounce.jax import from_jax

def f(x): return jnp.sum((x - 1) ** 2)
def g(x): return jnp.stack([jnp.sum(x) - 5.0])

prob = from_jax(f, g, n=4, m=1, lb=jnp.zeros(4), ub=jnp.full(4, 10.0),
                cl=jnp.zeros(1), cu=jnp.zeros(1))
x, info = prob.solve(x0=jnp.ones(4))
```

### Differentiable solve

`pounce.jax.solve(p, f=, g=, …)` is a `custom_vjp`-wrapped solve that
differentiates `x*(p)` through the implicit function theorem on the
converged KKT system. Inequality rows that are *not* active at `x*`
are dropped from the KKT block before the implicit-diff back-solve, so
the gradient matches the analytic active-set sensitivity even on
slack-inequality problems (pounce#73).

```python
import jax, jax.numpy as jnp
from pounce.jax import solve as psolve

def f(x, p): return jnp.sum((x - p) ** 2)
def g(x, p): return jnp.stack([x[0] + x[1] - 1.0])   # equality

def x_star(p):
    return psolve(
        p, f=f, g=g, x0=jnp.zeros(2), n=2, m=1,
        lb=jnp.full(2, -10.0), ub=jnp.full(2, 10.0),
        cl=jnp.zeros(1),       cu=jnp.zeros(1),
        options={"tol": 1e-10, "print_level": 0},
    )

# Gradient of the L2 distance to the target as p moves:
loss = lambda p: jnp.sum(x_star(p) ** 2)
print(jax.grad(loss)(jnp.array([0.3, 0.7])))
```

### Warm-start across a parameter trajectory

`solve_with_warm` returns the full primal-dual triple alongside `x*`,
and consumes one on the next call. The warm-state is opaque from the
JAX side (`pytree of jnp arrays`) but maps directly onto the
`x0` / `λ0` / `z0` ports of the underlying solver — for a sequence of
nearby `p` values this often cuts solver iterations by an order of
magnitude (pounce#74).

```python
from pounce.jax import solve_with_warm

trajectory = [jnp.array([0.3 + 0.01 * k, 0.7 - 0.01 * k]) for k in range(50)]

x, warm = solve_with_warm(
    trajectory[0], f=f, g=g, x0=jnp.zeros(2), n=2, m=1,
    lb=jnp.full(2, -10.0), ub=jnp.full(2, 10.0),
    cl=jnp.zeros(1), cu=jnp.zeros(1),
    warm_start=None,                           # first call → cold start
    options={"tol": 1e-10, "print_level": 0},
)
xs = [x]
for p_k in trajectory[1:]:
    x, warm = solve_with_warm(
        p_k, f=f, g=g, x0=x, n=2, m=1,
        lb=jnp.full(2, -10.0), ub=jnp.full(2, 10.0),
        cl=jnp.zeros(1), cu=jnp.zeros(1),
        warm_start=warm,                       # reuse λ, z
        options={"tol": 1e-10, "print_level": 0},
    )
    xs.append(x)
```

### Batched solve (`vmap_solve` / `vmap_solve_parallel`)

`vmap_solve` runs one solve per row of `p_batch` sequentially.
`vmap_solve_parallel` is the same surface but dispatches each row to a
`ThreadPoolExecutor`; the underlying Rust solve releases the GIL via
`py.allow_threads`, so workers actually run in parallel on multi-core
CPUs (pounce#74).

```python
import numpy as np
from pounce.jax import vmap_solve_parallel

rng   = np.random.default_rng(0)
batch = jnp.asarray(rng.standard_normal((32, 2)))

X = vmap_solve_parallel(
    batch, f=f, g=g, x0=jnp.zeros(2), n=2, m=1,
    lb=jnp.full(2, -10.0), ub=jnp.full(2, 10.0),
    cl=jnp.zeros(1), cu=jnp.zeros(1),
    workers=8,                                 # ThreadPoolExecutor size
    options={"tol": 1e-9, "print_level": 0},
)
assert X.shape == (32, 2)
```

Both batched surfaces are `custom_vjp`-wrapped, so a downstream
`jax.grad`/`jax.jacobian` over a batched loss works end-to-end.

### Build once, solve many: `JaxProblem`

For iterative use — a parameter trajectory in a continuation loop, a
training step that calls the solver inside a batch, a notebook cell
that sweeps a knob — `from_jax`/`solve` rebuild the JIT artefacts, the
sparsity probe, and the underlying `pounce.Problem` on *every* call.
`JaxProblem` does that work once at construction and exposes the same
four method shapes against the cached state. On the
`pounce#75` microbench shape (`n=5, m=6`, 20 sequential solves at
different `p`) this is roughly a 14× speedup, taking per-solve time
from ~96 ms down to ~7 ms (pounce#75).

```python
from pounce.jax import JaxProblem

jp = JaxProblem(
    f=f, g=g, n=2, m=1, p_example=jnp.zeros(2),     # p_example fixes shape/dtype only
    lb=jnp.full(2, -10.0), ub=jnp.full(2, 10.0),
    cl=jnp.zeros(1),       cu=jnp.zeros(1),
    options={"tol": 1e-9, "print_level": 0},
)

# Sequential, differentiable:
x = jp.solve(jnp.array([0.3, 0.7]), x0=jnp.zeros(2))

# Dual-warm-start trajectory (composes warm-state hand-off with reuse):
x, warm = jp.solve_with_warm(trajectory[0], x0=jnp.zeros(2), warm_start=None)
for p_k in trajectory[1:]:
    x, warm = jp.solve_with_warm(p_k, x0=x, warm_start=warm)

# Batched parallel solve over a row-axis of p_batch:
X = jp.vmap_solve_parallel(batch, x0=jnp.zeros(2), workers=8)
```

Each worker thread in `vmap_solve_parallel` keeps its own cached
`pounce.Problem` via `threading.local`, so the per-thread build cost
is paid at most once per worker rather than once per batch row.

### Factor-reuse backward (`factor_reuse=`)

`JaxProblem.solve` and `solve_with_warm` default to a k_aug-style
backward that reuses the IPM's converged compound KKT factor
(`pounce.Solver.kkt_solve`) instead of assembling a dense
`(n+m) × (n+m)` block and running `jnp.linalg.solve` on it
(pounce#76). The held LDLᵀ factor turns the bwd back-solve from
O((n+m)³) into O(nnz(L)) and drops the explicit active-set masking
that the dense path does — the barrier rows on the bound multipliers
`(z_l, z_u)` already encode "active bounds force `Δx_i = 0`" exactly,
and the `(v_l, v_u)` rows do the same for slack inequalities. The
accuracy of the resulting gradient is `O(μ)` at the IPM barrier
parameter, which sits well below `tol` after convergence.

```python
jp = JaxProblem(..., factor_reuse=True)   # default; reuse the IPM factor
jp = JaxProblem(..., factor_reuse=False)  # dense JAX backward
```

Pick `factor_reuse=False` when you want higher-order differentiation
(`jax.grad(jax.grad(...))` through the solver) — the dense backward
stays JAX-traced and is itself differentiable, the factor-reuse one
crosses to the Rust host via `pure_callback` and is opaque to a
second-order trace.

#### When to pick which on `batched_solve` workloads (pounce#77)

`factor_reuse=False` is itself a form of factor reuse — it builds
the per-block `(n+m) × (n+m)` KKT at pounce's converged
`(x*, λ*, μ_l*, μ_u*)` (saved in the custom_vjp residual) and
solves it under `jax.vmap` with a JIT-fused per-block
`jnp.linalg.solve`. So both modes reuse pounce's converged solution;
they differ only in **what** they back-solve:

* `factor_reuse=True` — back-solves pounce's held LDLᵀ factor of the
  full stacked KKT (Rust-side, via FFI through a single-thread
  executor pin).
* `factor_reuse=False` — back-solves a freshly assembled per-block
  dense KKT in JAX, fused under `vmap`.

For `batched_solve` + `jax.jacrev` / `jax.vmap` minibatch projections
**`factor_reuse=False` is faster at every scale we measured**
(n = 3 through 48 per block, B = 64 stacked):

```
n=3   reuse bwd =  16.6 ms   dense bwd =  20.6 ms   reuse/dense = 0.80×
n=8   reuse bwd =  52.5 ms   dense bwd =  38.5 ms   reuse/dense = 1.36×
n=16  reuse bwd = 157.6 ms   dense bwd =  57.2 ms   reuse/dense = 2.76×
n=32  reuse bwd = 558.6 ms   dense bwd = 103.6 ms   reuse/dense = 5.39×
n=48  reuse bwd =1262.9 ms   dense bwd = 137.4 ms   reuse/dense = 9.19×
```

The dense path scales as `B · (n+m)³`; the factor-reuse path scales
as `N · kkt_dim ≈ B² · n · (n+m)` because `jax.jacrev` fans out
`N = B·n` cotangents and each triggers a back-solve of the **full**
stacked LDLᵀ even though only one block has nonzero signal.

**Guidance:**

* **Single solve + many sensitivities** —
  `jax.jacrev(jp.solve, argnums=0)(p, x0)` and friends — keep
  `factor_reuse=True`. One LDLᵀ back-solve per cotangent against
  the held factor beats JAX dense-solving a fresh `(n+m) × (n+m)`
  block.
* **Batched solve + jacrev / vmap** —
  `jax.jacrev(lambda P: jp.batched_solve(P, x0))(pb)` — set
  `factor_reuse=False`. Treat the dense path as the default for
  minibatch projections.

Each fwd registers its converged factor in a bounded LRU on the
`JaxProblem` (default capacity 128). For very long-running training
loops with many distinct forward solves you can drop the cache
explicitly:

```python
jp.clear_solver_cache()
```

#### Off-thread dispatch (training loops, `jit(value_and_grad(...))`)

`pounce.Solver` is a `!Send` PyO3 type (it holds an
`Rc<RefCell<dyn TNLP>>` interior), so any attempt to touch the held
factor from a thread other than the one that built it raises a PyO3
panic. JAX hits this whenever the bwd `pure_callback` lands on an XLA
worker thread — typical for `jax.jit(jax.value_and_grad(...))` inside
a training step.

`JaxProblem(factor_reuse=True)` defends against this by routing every
`pounce.Solver` interaction (fwd register, warm-start solve, batched
solve, bwd `kkt_solve`) through a dedicated single-thread
`ThreadPoolExecutor` owned by the `JaxProblem` (pounce#77). All solver
touches are pinned to that one worker thread regardless of which
thread JAX dispatches from. `vmap_solve_parallel` bypasses the pin
(it doesn't register with the factor cache), so its B-way thread
concurrency is preserved.

#### Pickle / distributed training

`JaxProblem` round-trips through `pickle.dumps` / `pickle.loads`, so
it works with the realistic distributed-training paths:

* `multiprocessing(start_method='spawn')` — the default on macOS and
  what `torch.utils.data.DataLoader(num_workers>0)` uses;
* Ray and Dask actors via `cloudpickle`;
* Naive checkpointing for resume.

The per-process runtime state (JIT'd closures, `threading.Lock`,
`threading.local`, the factor-reuse executor, the held LDLᵀ factor
registry) is dropped from the pickle and rebuilt on the receiving
side. The sparsity-pattern arrays survive the round trip, so the
worker doesn't redo the one-shot JAX probe. Held factors do not
survive — a fresh process has no history of fwd solves, so the
receiver's registry starts empty and the bwd factor-reuse path picks
up from the next solve.

User-side requirement: `f` and `g` must themselves be picklable.
Module-level functions work with stdlib pickle; lambdas / inner
functions need `cloudpickle` (which is what Ray, Dask, and
`torch.multiprocessing` use by default anyway).

`multiprocessing(start_method='fork')` is *not* supported — JAX
itself warns that `os.fork()` is incompatible with its threading;
use `spawn` instead.

### Stacked block-diagonal batched solve (`batched_solve`)

`JaxProblem.batched_solve(p_batch, x0)` runs *one* IPM solve over a
single NLP whose variables are `[x^(1); ...; x^(B)]`, constraints are
`concat(g(x^(k), p^(k)))`, and objective is `Σ_k f(x^(k), p^(k))`.
The Jacobian and Lagrangian Hessian are block-diagonal — each block-`k`
constraint touches only the block-`k` slice of `X`, and the objective
is a pure sum, so there's no cross-block coupling. The IPM sees one
big sparse problem but does only `B × (per-block factor cost)` work
on the linear system.

```python
p_batch = jnp.array([[0.3, 0.7], [0.5, 0.5], [-0.1, 0.4]])
x_batch = jp.batched_solve(p_batch, x0=jnp.zeros(2))    # (B, n)
```

`custom_vjp`-wrapped, so `jax.grad`/`jax.jacobian` through the
batched solve work end-to-end:

```python
def loss(P):
    return jnp.sum(jp.batched_solve(P, x0=jnp.zeros(2)) ** 2)

dloss_dP = jax.grad(loss)(p_batch)                       # (B, p_shape)
```

The backward path follows `factor_reuse=`:

* `factor_reuse=True` (default) — one `Solver.kkt_solve` against the
  *stacked* held LDLᵀ factor; the per-block `∂²L/∂x∂p` / `∂g/∂p` are
  `jax.vmap`'d autodiff over the user's `f` / `g`, then contracted
  with the per-block `u_x` / `u_g` slices of the single back-solve.
  Composes (A) and (B) — one factor for both forward and per-batch
  sensitivities (pounce#76).
* `factor_reuse=False` — `jax.vmap` of the per-element dense
  `(n+m) × (n+m)` JAX KKT solve. Exact for the same reason: block-
  diagonal coupling means `∂x^(k)*/∂p^(j) = 0` for `k ≠ j`.

When to pick `batched_solve` vs the existing batched surfaces:

| Surface | Wins when |
|---------|-----------|
| `vmap_solve` | Long batches, want one solve per iterate sequentially. |
| `vmap_solve_parallel` | Batch elements have very different convergence behaviour — slow blocks don't drag fast ones (B independent IPMs in worker threads, GIL released per solve). |
| `batched_solve` | Blocks have similar convergence behaviour (shared barrier homotopy and symbolic factorisation amortise) *and* B is large enough that the per-call Python overhead of B fwd dispatches becomes visible (one Rust crossing instead of B). |

Per-block `lb`/`ub`/`cl`/`cu` are tiled across the batch; the
parameter `p` is what varies, not the feasible region. Stacked
Problems are cached per (thread, B) in a tiny LRU (cap 4), so
calls in a loop with one or two batch sizes pay the build cost at
most once per worker.

### Post-solve Jacobian and sensitivities (`batched_solve_with_jacobian`)

When you need the explicit per-block Jacobian `J[k] = ∂x^(k)*/∂p^(k)`
as a first-class result — for validation, linear-update layers, or
diagnostics — `batched_solve_with_jacobian` returns it directly from
the held KKT factor instead of wrapping `batched_solve` in
`jax.jacrev`:

```python
x_star, (lam, zL, zU), J = jp.batched_solve_with_jacobian(p_batch, x0)
# x_star : (B, n)   J : (B, n, p_dim)   duals match batched_solve_with_warm
```

`J`'s row `i` is the reverse-mode VJP at cotangent `e_i` (the KKT
system is symmetric), so the whole Jacobian is one multi-RHS back-solve
against the held LDLᵀ factor — no NLP re-solve, no repeated public
`jax.vjp` calls. Pass `wrt_cols` (1-D `p` only) to keep just the
parameter columns you care about, e.g. `wrt_cols=slice(0, ny)` to drop
context columns; `J` then has trailing dim `len(wrt_cols)`.

For the linear-update pattern — anchor once, then apply several nearby
sensitivity products — pin the factor with an `AnchorState` and reuse it:

```python
with jp.anchor(p_batch, x0, wrt_cols=slice(0, ny)) as state:
    dx     = jp.batched_jvp_from_state(state, dp)      # J @ dp   (forward)
    dp_bar = jp.batched_vjp_from_state(state, x_bar)   # J^T @ x_bar (reverse)
```

`batched_jvp_from_state` is the cheap path for linear updates that only
need the directional sensitivity `delta_x = J @ delta_p` and never the
full `J`: it assembles the parameter-side RHS `[∂²L/∂x∂p · dp; ∂g/∂p · dp]`
and back-solves once against the held factor. When the state was anchored
with `wrt_cols`, pass the reduced `dp` (one entry per selected column);
otherwise pass a full `(B,) + p_shape` perturbation (zero out the columns
you don't want to move).

`anchor(...)` (and `batched_solve_with_jacobian(..., return_state=True)`)
return an `AnchorState` that holds the factor across calls. Prefer the
context-manager form; for handles that must outlive a single block
(e.g. stored on a projection layer), use explicit ownership:

```python
state = jp.anchor(p_batch, x0)
...                          # later calls reuse `state`
state.reanchor(p_new, x0)    # swap the solve in place (closes prior pin)
state.close()                # release the held factor
```

Pinned factors are exempt from the backward LRU but capped
(`_pinned_capacity`, default 16) so a missed `close()` fails loudly
rather than leaking; a `weakref` finalizer reclaims the factor if a
handle is garbage-collected without `close()`. A worked example —
projection layer, full Jacobian, JVP/VJP-from-state, and the lifetime
patterns — is in
[`notebooks/13_post_solve_jacobian.ipynb`](https://github.com/jkitchin/pounce/blob/main/python/notebooks/13_post_solve_jacobian.ipynb).

## Notebooks

The notebooks under
[`python/notebooks/`](https://github.com/jkitchin/pounce/tree/main/python/notebooks)
work through getting started, JAX autodiff, implicit differentiation,
sensitivity analysis, the Pyomo integration,
[NLP scaling](https://github.com/jkitchin/pounce/blob/main/python/notebooks/07_scaling.ipynb)
(`set_problem_scaling` + `nlp_scaling_method=user-scaling`), and
[FBBT](https://github.com/jkitchin/pounce/blob/main/python/notebooks/08_fbbt.ipynb)
(nonlinear bound tightening via `presolve_fbbt=yes` on Pyomo
models).
