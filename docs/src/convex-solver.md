# Convex Solver: LP, QP, and SOCP

POUNCE ships a specialized **convex conic interior-point solver**
(`pounce-convex`) alongside the general NLP filter-IPM. It solves the
standard-form convex program

```text
minimize    ½ xᵀP x + cᵀx
subject to  A x = b
            G x ⪯_K h
            lb ≤ x ≤ ub
```

where `P ⪰ 0` and the inequality block lies in a product cone `K` of
nonnegative orthants and second-order cones. `P = 0` is an LP; an
all-orthant `K` is an LP/QP; second-order blocks make it an **SOCP**.

The method is a **Mehrotra predictor–corrector** primal–dual interior-point
algorithm with Nesterov–Todd scaling for the cones, sharing the pure-Rust
[`feral`](algorithm.md) sparse LDLᵀ backend with the NLP path. It reaches
optimality in materially fewer iterations than routing the same problem
through the general NLP solver (≈30–50% fewer on bound/inequality QPs).

> **Inspiration.** The conic interior-point design follows
> [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs) (Goulart &
> Chen) — handling a quadratic objective directly and a product of
> symmetric cones — and the presolve follows
> [PaPILO](https://github.com/scipopt/papilo) (the presolving library of
> [SCIP](https://www.scipopt.org/)). POUNCE does not wrap either (the
> pure-Rust guarantee) but ports their ideas; see
> [Acknowledgments](acknowledgments.md).

This chapter covers the **Python API** (`pounce.qp` and the differentiable
`pounce.jax` layers). For automatic CLI/Pyomo routing of `.nl` LPs/QPs, see
[LP / QP Solver Routing](lp-qp-routing.md). Runnable, progressive notebooks
live in [`python/notebooks/`](https://github.com/jkitchin/pounce/tree/main/python/notebooks):
`15_convex_qp.ipynb`, `16_socp.ipynb`, `17_differentiable_convex.ipynb`.

## Quadratic programs

```python
import numpy as np
from pounce.qp import solve_qp

# min ½·2‖x‖² − 3x₀ − 4x₁  s.t.  x₀ + x₁ ≤ 1,  0 ≤ x ≤ 1
r = solve_qp(
    P=np.diag([2.0, 2.0]),
    c=[-3.0, -4.0],
    G=[[1.0, 1.0]], h=[1.0],
    lb=[0, 0], ub=[1, 1],
)
r.status   # 'optimal'
r.x        # primal solution
r.y, r.z   # equality / inequality multipliers
r.z_lb, r.z_ub  # bound multipliers (≥ 0)
r.obj, r.iters
```

`P` (lower triangle used, assumed symmetric), `A`, and `G` accept dense
arrays or scipy-sparse matrices; any of them may be omitted. The result is
a `QpResult` dataclass with a `.success` property. The solver reports
**verified** infeasibility / unboundedness (`'primal_infeasible'` /
`'dual_infeasible'`) backed by a Farkas / recession certificate rather than
an iteration-limit guess.

## Second-order cone programs

A second-order (Lorentz) cone is `{ (t, x) : t ≥ ‖x‖₂ }`. Partition the
inequality rows of `Gx ⪯_K h` with `cones` — a list of `(kind, dim)` specs
(`"nonneg"` or `"soc"`; a bare int means a second-order cone). Each slack
block `s = h − Gx` must lie in its cone.

```python
from pounce.qp import solve_socp

# minimize ‖x − x*‖  ⇔  min t s.t. (t, x − x*) ∈ SOC
r = solve_socp(
    c=[1.0, 0.0, 0.0],                 # minimize t
    G=-np.eye(3), h=[0.0, -2.0, 1.0],  # s = (t, x₀−2, x₁+1) ∈ SOC(3)
    cones=[("soc", 3)],
)
r.x   # ≈ [0, 2, -1]:  t* = 0, x = x*
```

Mixed cones compose — e.g. `cones=[("nonneg", 1), ("soc", 2)]` puts the
first slack in `ℝ₊` and the next two in a 2-D second-order cone. Large
cones use a **sparse diagonal-plus-rank-1** KKT representation (one
auxiliary variable per cone, the ECOS/Clarabel "sparse SOC" trick) so the
factorization stays sparse.

## Warm starting

Feed a previous (or nearby) solution back to seed the interior-point
iteration — useful for parametric sweeps, receding-horizon MPC, and
branch-and-bound subproblems:

```python
base = solve_qp(P=P, c=c, G=G, h=h, lb=lb, ub=ub)
nxt  = solve_qp(P=P, c=c2, G=G, h=h, lb=lb, ub=ub, warm_start=base)
```

The warm start only affects the iteration count, never the solution (a
mismatch is ignored). The recentering is **adaptive** for the orthant
(sized to the warm point's KKT residual, so it exploits a nearby problem's
duals yet self-corrects when the active set moves) and re-centers the cone
duals for second-order blocks (a converged conic point sits on the cone
boundary, where the scaling is singular).

## Batching and factorization reuse

```python
from pounce.qp import solve_qp_batch, QpFactorization

# Solve many independent QPs in parallel (rayon, across instances).
results = solve_qp_batch([dict(P=P, c=c_k, G=G, h=h) for c_k in cs])

# Build the KKT symbolic factor once, solve many same-structure problems.
fac = QpFactorization(P=P, c=c0, G=G, h=h, lb=lb, ub=ub)
for c_k in cs:
    rk = fac.solve(P=P, c=c_k, G=G, h=h, lb=lb, ub=ub)  # reuses the factor
```

`solve_qp_batch` parallelizes across instances (outer-parallel /
inner-serial) and `QpFactorization` reuses the AMD ordering and symbolic
factorization across solves that share a structure — the two compose with
warm starting.

## Presolve (PaPILO-inspired)

Before the interior-point solve, POUNCE can apply a **transaction-stack
presolve** with full primal **and dual** postsolve, modeled on
[PaPILO](https://github.com/scipopt/papilo). The catalog:

- empty / **duplicate / parallel** (scalar-multiple) rows,
- fixed-variable elimination (singleton equalities),
- free columns and free-column singletons,
- activity-based redundancy and infeasibility detection,
- **forcing constraints** (a row at its activity extreme pins its variables),
- **dominated columns** (sign-definite columns optimal at a bound),
- **bound tightening** (domain propagation), with the active-bound
  multiplier re-attributed to its source row in postsolve,

iterated to a **fixpoint** so reductions cascade. Each reduction carries
the data to reverse itself, and the postsolve reconstructs a valid KKT
point of the *original* problem — the dual recovery is the contract, and is
verified by KKT-residual tests. A cone-aware variant (`presolve_conic`)
gates the `≤`-row reductions off second-order-cone blocks (which are
coupled) and recovers the reduced cone partition.

Presolve is applied automatically on the CLI LP/QP route; it lives in
`pounce-convex::presolve` for Rust callers. See
[LP / QP Solver Routing](lp-qp-routing.md).

## Differentiable convex layers (JAX)

`pounce.jax` exposes the solve as a differentiable JAX op via the
implicit-function theorem on the KKT system at the optimum (Amos & Kolter,
*OptNet*, 2017). The forward calls the solver; the backward is a single
linear solve through the same KKT matrix.

```python
import jax, jax.numpy as jnp
from pounce.jax import solve_qp, solve_socp, QpLayer

# x*(c) for a parametric QP, differentiable w.r.t. all of P, c, G, h, A, b.
def loss(c):
    x = solve_qp(P=P, c=c, G=G, h=h)
    return jnp.sum((x - target) ** 2)

grad_c = jax.grad(loss)(c0)        # exact gradient via implicit diff
J = jax.jacrev(lambda c: solve_qp(P=P, c=c, G=G, h=h))(c0)
```

- Gradients are provided w.r.t. **every** parameter that enters through the
  optimum: `c`, `b`, `h`, and the matrices `P`, `G`, `A` (the full OptNet
  matrix derivatives; `∇P` is the symmetric gradient).
- `solve_socp` differentiates SOCPs too — the complementarity row uses the
  cones' **arrow operators** in place of the orthant's diagonal.
- `QpLayer` captures a fixed `P`/`G`/`A` structure for use inside a larger
  JAX model, with `jax.grad` / `jacrev` / `vmap` and a parallel `.batch`.
- A warm start may be passed through (non-differentiated — it cannot change
  the solution or its gradients, only the iteration count).

All gradients are validated against finite differences in the test suite.
