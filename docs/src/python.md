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

`minimize` is a thin facade over `pounce.Problem` shaped after
`scipy.optimize.minimize`, so SciPy code ports with few changes. It returns a
SciPy-`OptimizeResult`-shaped object (`res.x`, `res.fun`, `res.success`,
`res.status`, `res.message`, `res.nit`, plus `res.info` and dict-style
`res["x"]`).

### Compatibility with `scipy.optimize.minimize`

```python
minimize(fun, x0, jac=None, hess=None, bounds=None,
         constraints=None, options=None)
```

| Argument | Status | Notes |
|---|---|---|
| `fun`, `x0` | ✅ | objective callable and start point |
| `jac` | ✅ | callable; **omitted → forward finite differences** (`√eps` step). Provide one for production. |
| `hess` | ⚠️ | used **only when there are no constraints**; with constraints the solver falls back to L-BFGS (`hessian_approximation=limited-memory`) |
| `bounds` | ✅ | a sequence of `(lo, hi)` pairs; a `None` element or a `None` endpoint means ±∞ |
| `constraints` | ✅ | SciPy **dict(s)** `{"type": "eq"\|"ineq", "fun": …, "jac": …}`; multiple are concatenated; `"jac"` optional (finite-diff fallback) |
| `options` | ⚠️ | forwarded to `Problem.add_option` — keys are **pounce/Ipopt option names** (`tol`, `max_iter`, `hessian_approximation`), **not** SciPy's (`maxiter`, `ftol`) |
| `args` | ❌ | not supported — close over extra arguments in `fun`/`jac` |
| `method` | ❌ | always the filter-IPM (see below for why there is no `method=`) |
| `hessp` | ❌ | no Hessian-vector-product mode |
| `tol` | ❌ | pass it via `options={"tol": …}` |
| `callback` | ❌ | not supported |

**Conventions that match SciPy** (so constraint dicts port directly):

- Inequalities use the SciPy sign convention **`g(x) ≥ 0`**; equalities are
  **`g(x) = 0`**.
- The result object is SciPy-`OptimizeResult`-shaped (subset of fields + an
  `info` map).

**Gaps worth knowing:**

- **Only the dict form of `constraints`** is accepted — a SciPy `Bounds`,
  `LinearConstraint`, or `NonlinearConstraint` *object* will not work, and
  `bounds` must be `(lo, hi)` pairs (not a `Bounds` object).
- The constraint **Jacobian is dense**; for large sparse Jacobians use the
  `Problem` class directly (it takes a sparse Jacobian and structure).
- The most common porting snag is `options`: `options={"maxiter": 100}` is a
  no-op — it is `options={"max_iter": 100}`.

### Solver routing in `minimize`

By default `minimize` **auto-routes** the same way the CLI's
`solver_selection=auto` does: a problem that is provably a **linear program**
or a **convex quadratic program** is dispatched to the specialized convex
interior-point solver (`pounce.solve_qp`, the HSDE driver), and a provably
**convex QCQP** (convex-quadratic objective and/or constraints) is reformulated
to a second-order cone program and dispatched to the conic solver
(`pounce.solve_socp`). Both reach a **global** optimum in materially fewer
iterations; everything else is solved by the general NLP filter line-search
interior-point method, exactly as before.

The catch is that `minimize` only sees **opaque callables** — it cannot read a
`.nl` expression tree the way the CLI can. So instead of *reading* the
structure it **probes** it: it evaluates `fun`/`jac`/`hess` at several points,
fits a linear/quadratic model, and then **validates that model against the
true callables at held-out points** before trusting it. The two
misclassification directions are not symmetric, and the validation gates the
dangerous one:

- A convex LP/QP/QCQP mistakenly sent to the NLP solver is merely *slower* —
  the filter-IPM still solves it correctly.
- A genuinely nonlinear or nonconvex problem sent to the convex solver would
  return a **silently wrong** answer.

So any probe that raises, any model mismatch beyond `route_tol`, a
non-constant Hessian/Jacobian, an indefinite objective Hessian (a nonconvex
QP), a quadratic *equality*, or a quadratic inequality whose feasible set is
nonconvex (a non-PSD constraint Hessian) all fall back to the NLP solver.
**You never get a wrong "optimum" from a misclassification.**

#### Forcing the solver

The `solver_selection` option (passed in `options=`) overrides the automatic
choice — mirroring the CLI option of the same name:

| `options={"solver_selection": …}` | Behavior |
|---|---|
| `"auto"` | **Default.** Probe-and-validate; route provable LP/convex-QP to `solve_qp`, a convex QCQP to `solve_socp`, else NLP. |
| `"nlp"` | Skip routing entirely; always use the NLP solver (the pre-routing behavior). |
| `"lp-ipm"` | Force the convex solver; raise `ValueError` if the problem is not detected as an LP. |
| `"qp-ipm"` | Force the convex solver; raise `ValueError` if it is not detected as a convex LP/QP. |
| `"socp"` | Force the conic solver; raise `ValueError` if it is not detected as a convex QCQP. |

```python
# Default: route a convex QP to the fast convex IPM automatically.
res = minimize(fun, x0, bounds=bounds)
print(res.info["solver"])          # 'qp-ipm' / 'socp' when routed; absent on the NLP path

# Keep the pre-routing behavior — always the NLP solver:
res = minimize(fun, x0, options={"solver_selection": "nlp"})

# Insist the problem is a convex QP; fail loudly if the probe disagrees:
res = minimize(fun, x0, options={"solver_selection": "qp-ipm"})

# A convex QCQP (e.g. a quadratic ball constraint) routes to the conic solver:
ball = {"type": "ineq", "fun": lambda x: 1.0 - x @ x}   # x·x ≤ 1
res = minimize(lambda x: -x[0] - x[1], [0.1, 0.1], constraints=[ball])
print(res.info["solver"])          # 'socp'
```

`route_tol` (default `1e-5`) sets the relative tolerance for the held-out
validation; raise it if a genuinely-linear problem with noisy finite-difference
Jacobians is being conservatively rejected, lower it to be stricter. The
routing keys are consumed by `minimize` and never forwarded to the backend, so
the rest of `options` still reaches the NLP solver unchanged.

#### When you still need a typed entry point

Auto-routing handles LP, convex QP, and convex QCQP from the
`minimize(fun, x0, …)` shape. The remaining specialized solvers need structure
that a callable cannot carry — an explicit cone list (exp/power/PSD cones), a
symbolic objective to relax and bound — so each keeps its own pounce-native
entry point:

| Want | Entry point | You provide | Optimum |
|---|---|---|---|
| General nonlinear, fast local solve | `minimize(fun, x0, …)` | callables (`fun`/`jac`/`hess`) | local |
| LP / convex QP | `minimize` (auto) or `solve_qp(P, c, A, b, G, h, lb, ub, …)` | callables / matrices | **global** |
| Convex QCQP | `minimize` (auto / `socp`) or `solve_socp(…, cones=…)` | callables / matrices + cone list | **global** |
| SOCP / exp / power / PSD cones | `solve_socp(P, c, A, b, G, h, *, cones, …)` | matrices + cone list | **global** |
| Polynomial, certified global | `sos_minimize(objective, *, inequalities, equalities, …)` | a polynomial | **global** |

The `solve_qp` / `solve_socp` / `sos_minimize` functions are pounce-native (not
SciPy-shaped) by necessity — e.g. `sos_minimize` takes a polynomial as a
coefficient dict and returns a certificate, *not* callables and SciPy dicts. See
[Choosing a Solver](choosing-a-solver.md) for the full map.

> A `minimize_global` entry point for factorable nonconvex problems (spatial
> branch-and-bound) is in development on the `feature/global` branch and is not
> exposed in this release; today the certified-global Python path is
> `sos_minimize`, for polynomials.

## Curve fitting

`pounce.curve_fit` is the data-fitting companion to `minimize` — a
`scipy.optimize.curve_fit`-style front end that adds parameter constraints,
robust losses, confidence intervals, and `∂params/∂data` sensitivity, with the
covariance read from the solver's reduced Hessian. See
[Curve Fitting](curve-fitting.md).

```python
from pounce import curve_fit

res = curve_fit(model, xdata, ydata, p0=[1, 1, 0])   # model written with jax.numpy
print(res.summary())
```

## Finding multiple minima

`pounce.find_minima` is the global-search companion to `minimize`: it drives
the same solver in a loop to discover *many* distinct minima (flooding,
deflation, tunneling, multistart, MLSL, basin-hopping). See
[Finding Multiple Minima](find-minima.md) for the methods and references,
[Choosing a Method](find-minima-choosing.md) for selection guidance
(including high-dimensional behavior), and notebooks
[19](https://github.com/jkitchin/pounce/blob/main/python/notebooks/19_find_minima_repulsion.ipynb),
[20](https://github.com/jkitchin/pounce/blob/main/python/notebooks/20_find_minima_restart.ipynb),
[21](https://github.com/jkitchin/pounce/blob/main/python/notebooks/21_find_minima_hopping.ipynb)
for the three families.

```python
from pounce import find_minima

r = find_minima(fun, x0, method="deflation", jac=jac, hess=hess,
                bounds=bounds, n_minima=6)
print(r.status, len(r), "minima; best f =", r.fun)
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

### Sparse Jacobian/Hessian compression (`sparse=`)

By default the constraint Jacobian and the Lagrangian Hessian are
computed *densely* — `jax.jacrev`/`jacfwd`/`hessian` build the full
matrix, which is then sliced to the detected sparsity pattern. The
reported structure is sparse, but the AD work and memory are `O(m·n)`
(Jacobian) and `O(n²)` (Hessian) **regardless of how sparse the true
matrices are**. On a 10,000-variable banded system that means computing
~10⁸ entries per iteration to keep ~50,000.

Passing `sparse=True` switches both derivatives to CPR-style **colored
AD** (pounce#83): structurally-orthogonal columns are colored, one
JVP (Jacobian) / HVP (Hessian) is taken per color — `k ≪ n` colors —
and the compressed result is scattered back to the known nonzeros. The
per-iteration cost drops from `O(n)` to `O(k)` AD passes. This is the
same compression strategy the Rust `.nl` tape path already uses for its
Hessian.

```python
prob = from_jax(f, g, n=4, m=1, lb=jnp.zeros(4), ub=jnp.full(4, 10.0),
                cl=jnp.zeros(1), cu=jnp.zeros(1),
                sparse=True)              # colored JVP/HVP instead of dense slice
```

The flag is also accepted by [`JaxProblem`](#build-once-solve-many-jaxproblem),
where it applies to both the single-solve and the batched
block-diagonal paths. The reported structure, the values, and the
solution are identical to the dense path either way — only the cost of
producing the derivative values changes. The differentiable backward
(`factor_reuse` / implicit diff) is unaffected.

**When to use it.** `sparse=True` wins on problems whose Jacobian/Hessian
are *genuinely* sparse with bounded per-row fill (banded, block, finite
differences/elements, PDE-constrained, separable). On a dense problem
the coloring finds no orthogonality (`k = n`) and the flag is a small,
bounded overhead, so it is **opt-in rather than the default**. Measured
on a banded family (`python/benchmarks/bench_sparse_ad_83.py`):

| n | colors (Jac / Hess) | per-eval Jacobian | per-eval Hessian | full solve |
|---|---|---|---|---|
| 800  | 2 / 3 | 6.2× faster | 2.0× faster | 1.3× faster |
| 2000 | 2 / 3 | 18.4× faster | 5.4× faster | 7.6× faster |
| 5000 | 2 / 3 | **560× faster** | **200× faster** | — |

The color count stays constant in `n` while the dense path grows
linearly, so the gap widens without bound as the problem scales.

**Pattern detection.** Sparsity is found by probing the dense derivative
at random points and recording where entries are nonzero. Under
`sparse=True` a mis-probe is costlier — it corrupts the compression
seed, not just a reported nonzero — so detection unions **3 probes** by
default (vs 1 for the dense path). Override with `n_probes=`. Truly
value-dependent structure (branchy `where`/`abs`) should still be
hand-rolled via the `Problem` API.

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
    # sparse=True,                                  # colored AD on sparse problems (see above)
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
