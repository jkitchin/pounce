# Differentiable Solves & the `DiffHandoff` Contract

POUNCE solves are differentiable: a solve can sit inside a JAX or PyTorch
model and pass gradients with respect to the problem parameters. This
page documents the **handoff contract** — the well-defined bundle of
post-convergence data every solve produces — so that any consumer (the
built-in JAX/Torch layers, a downstream tool such as `discopt`, or your
own autodiff code) can differentiate a POUNCE solve from one stable
surface rather than from solver internals.

Design notes: `dev-notes/diff-handoff-contract.md`.

## What a differentiable backward needs

The gradient of an optimal solution `x*(p)` with respect to a parameter
`p` comes from the implicit-function theorem applied to the KKT
conditions at the solution. To assemble it, a backward pass needs:

1. the primal solution `x*` and the constraint / bound **multipliers**;
2. the **active set** — which variable bounds bind and which constraint
   rows are active — so inactive directions drop out correctly;
3. (for performance) the converged **KKT factorization**, reused as a
   back-solve rather than rebuilt.

POUNCE produces all three. The first two ride out in the solve `info`
dict; the third is reused automatically by `JaxProblem` (see
[Sessions](sessions.md)).

## The active-set masks (the `DiffHandoff` core)

Every NLP solve's `info` dict carries a precomputed active set, derived
once on the Rust side (`pounce_sensitivity::DiffHandoff`) so no consumer
re-derives it under its own tolerance:

| `info` key | Type | Meaning |
|---|---|---|
| `pinned_vars` | `bool[n]` | Variable `i` has an active bound — its sensitivity is zero (`dx_i/dp = 0`). True when `mult_x_L[i] > active_tol` or `mult_x_U[i] > active_tol`. |
| `active_constraints` | `bool[m]` | Constraint row `i` is active: an equality (`g_l[i] == g_u[i]`) or a binding inequality (`abs(mult_g[i]) > active_tol`). |
| `active_tol` | `float` | The activity threshold used to derive the two masks above (default `1e-6`). |

`pinned_vars` is the seam used for mixed-integer problems: a
branch-and-bound leaf fixes integer variables at their optimal values,
and those variables differentiate exactly like an active bound
(`dx/dp = 0`). A producer of a fixed-integer leaf adds them to the mask
(`DiffHandoff::pin` on the Rust side).

## Multiplier conventions (canonical mapping)

The same dual quantity is named differently across POUNCE's solver
surfaces — deliberately, because each surface preserves an external
contract. **The canonical field is `DiffHandoff.lambda`** (general
constraint multipliers); this table maps every surface onto it so a
consumer knows the correspondence:

| Surface | Problem form | General-constraint dual | Bound duals | Why this naming |
|---|---|---|---|---|
| **NLP** (`Problem`, C ABI) | `min f(x) s.t. g_l ≤ g(x) ≤ g_u, x_l ≤ x ≤ x_u` | `mult_g` | `mult_x_L`, `mult_x_U` | **cyipopt-compatible** — drop-in for cyipopt / JuMP / AMPL clients. |
| **Convex QP/SOCP** (`solve_qp`) | `min ½xᵀPx + cᵀx s.t. Gx ≤ h, Ax = b` | `z` (inequality `G`), `y` (equality `A`) | `z_lb`, `z_ub` | **OptNet / convex-solver** convention (Amos & Kolter 2017). |
| `DiffHandoff` (canonical) | general | `lambda` | `mult_x_lower`, `mult_x_upper` | one name for the contract. |

> **Caution.** The internal symbol `lam` is *not* a single quantity: in
> the NLP backward (`jax/_diff.py`) it is *all* constraint multipliers
> (`= mult_g`); in the QP backward (`jax/_qp.py`) it is the
> *inequality-only* duals (`= z`). Always map through the table above
> rather than assuming a shared name means a shared quantity.

These names are stable: the NLP keys are an external cyipopt contract and
will not be renamed.

## Consuming the contract

### JAX / PyTorch (built in)

`pounce.jax` and `pounce.torch` already differentiate solves; you do not
touch the masks directly. Use `pounce.jax.solve` / `JaxProblem` (or the
torch equivalents) and call `jax.grad` / `.backward()` as usual. For
batched and repeated solves, `JaxProblem` reuses the converged KKT factor
in the backward (`factor_reuse=True`, default) — see [Sessions](sessions.md).

### Across a language / tool boundary (e.g. discopt)

A downstream tool that drives POUNCE as its NLP backend and composes its
own autodiff reads the contract straight from the `info` dict returned by
`Problem.solve`:

```python
x, info = problem.solve(x0=...)
# primal + duals
lam   = info["mult_g"]      # general-constraint multipliers (the canonical λ)
z_L   = info["mult_x_L"]
z_U   = info["mult_x_U"]
# precomputed active set — do NOT re-derive |mult| > tol yourself
pinned = info["pinned_vars"]          # bool[n]: dx/dp = 0 on these
active = info["active_constraints"]   # bool[m]: rows in the KKT block
tol    = info["active_tol"]
```

Because the active set is computed once in the producer, every consumer
sees the *same* masks under the *same* tolerance — which is what makes a
gradient assembled on one side of the boundary agree with one assembled
on the other.

## Verification

The contract is exercised by the test suite:

- `python/tests/test_problem.py::test_diff_handoff_masks_in_info` asserts
  the masks against a problem with a known active set (HS071: one
  variable on its lower bound, a binding inequality, and an equality).
- `python/tests/test_jax.py` (85 finite-difference gradient checks) and
  `python/tests/test_parity_jax_torch.py` (JAX↔Torch gradient agreement)
  confirm the backward passes that rest on this data are correct and
  frontend-independent.
