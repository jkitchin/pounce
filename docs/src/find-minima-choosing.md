# Choosing a Multiple-Minima Method

All six `find_minima` methods drive the same local solver; they differ in
how they leave a minimum once found. Use this page to pick one.

## By goal

| Your goal | Prefer | Why |
|---|---|---|
| **Enumerate all minima** of a smooth, low-dimensional objective | `flooding`, `deflation` | repulsion clears each basin so the next solve finds a new one; analytic derivatives keep the inner solve fast |
| **Just the global minimum** | `basinhopping`, `tunneling` | both are biased downhill and do not try to cover the whole space |
| **A robust, parallel baseline** | `multistart` | independent starts, trivially parallel, no tuning |
| **Expensive solves on a funneling landscape** | `mlsl` | clustering avoids re-descending basins it has already mapped |
| **Rugged, high-dimensional landscape** (clusters, conformers) | `basinhopping` | a local random walk over minima; the standard tool at scale |

## By problem structure

* **Have an analytic Hessian?** Repulsion methods (`flooding`, `deflation`)
  exploit it directly and certify each result as a true minimum. Without a
  Hessian, saddle rejection is skipped and the restart/hopping methods are a
  safer default.
* **Constrained problem?** All methods pass `bounds`/`constraints` through.
  Repulsion only touches the objective, so it is the cleanest with general
  constraints; restart and hopping sample/perturb inside the bounds box.
* **No bounds?** `multistart`/`mlsl` fall back to jittering around `x0`
  (give a `bounds` box for genuine global coverage). `flooding`/`deflation`
  and `basinhopping` work without bounds.
* **Variables on very different scales?** Handled automatically. The
  repulsion bump widths (`sigma` / `length`) are per-dimension and `"auto"`
  by default — sized to each variable's bounds range — and the default dedup
  metric measures distance in that same scaled space, so a single `dedup`
  tolerance is scale-free. Give `bounds` so the scales can be inferred; pass
  an explicit scalar or length-`n` vector to override.
* **Symmetric or periodic coordinates** (e.g. a periodic box): pass a custom
  `distance=` metric so that images of the same minimum de-duplicate
  correctly.

## Tuning cheat-sheet

| method | key knobs | rule of thumb |
|---|---|---|
| `flooding` | `sigma`, `amplitude` | `sigma` is per-dimension and `"auto"` (a fraction of each variable's bounds range) by default — leave it; `amplitude` a few × basin depth |
| `deflation` | `eta`, `power`, `soft`, `length` | `length` is per-dimension `"auto"` by default; raise `eta` if the solver returns to a known minimum |
| `tunneling` | `eta`, `power` | increase `patience`; it descends in a chain |
| `multistart` | `sobol` | leave Sobol on for coverage |
| `mlsl` | `samples_per_round`, `gamma` | more samples/round on rugged landscapes |
| `basinhopping` | `step`, `temperature` | `step` ≈ basin spacing; raise `temperature` to cross higher barriers |

If a run stops at `converged` with fewer minima than you wanted, raise
`patience` (search longer before giving up) and/or `max_solves`. If it stops
at `budget_exhausted`, raise `max_solves`.

## Scaling to high dimensions

The honest headline: **enumerating *all* minima is intractable in high
dimensions for every method here** — and that is a property of the problem,
not of the solver. The number of local minima typically grows *exponentially*
with dimension (Rastrigin has on the order of `k^n`; molecular energy
landscapes grow exponentially with the number of atoms). No method can list
exponentially many minima. What changes with dimension is which *goal*
remains reachable and which methods stay efficient.

Two costs scale independently:

1. **Cost per local solve.** This is just pounce's interior-point solve and
   scales well with sparse, large `n` — *provided the objective stays
   sparse*. Here is the catch for repulsion methods: each Gaussian or pole
   term adds a **dense** `n×n` Hessian contribution, so with
   `K` found minima the augmented Hessian is the sparse base plus `K`
   dense updates. On large sparse problems this destroys sparsity and the
   inner solve slows sharply. Restart and hopping never modify the objective,
   so they keep the original sparsity and the per-solve cost scales as
   `minimize` itself does.
2. **Number of solves needed.** For coverage this grows exponentially for
   *all* methods. For the *global* minimum it grows much more slowly for the
   downhill-biased methods, which is why they remain usable at scale.

How each family behaves as `n` grows:

* **Repulsion (`flooding`, `deflation`).** Two problems compound. A Gaussian
  of width `σ` covers a vanishing volume fraction `~ σⁿ`,
  so filling space needs exponentially many bumps; and the bumps densify the
  Hessian (above). The standard high-dimensional fix — used by metadynamics
  in practice — is to flood in a **low-dimensional collective-variable
  subspace** rather than all `n` coordinates. In full coordinates these
  are best kept to roughly `n ≲ 10–20`.
* **Restart (`multistart`, `mlsl`).** Each start is cheap and parallel, but
  the number of starts to cover (or to hit the global basin) grows
  exponentially. MLSL's clustering relies on a reduced radius
  `∝ (ln N / N)^(1/n)`; as `n` grows that exponent `→ 0`,
  distances concentrate, and MLSL degenerates toward plain multistart. So
  MLSL's advantage is a low-to-moderate-dimension phenomenon; in high
  dimension prefer plain parallel `multistart` (for the global basin) and
  spend the budget on more starts.
* **Hopping (`basinhopping`).** This is the family that **scales best in
  practice**, and it is exactly what the chemistry/physics community uses for
  hundreds to thousands of degrees of freedom. It performs a *local* random
  walk in minimum-space — it never tries to cover the domain — keeps the
  objective (and its sparsity) untouched, and the Metropolis bias funnels
  toward low minima. Pair it with multiple independent chains for
  parallelism.

Practical guidance:

* **`n ≲ 10–20`, want all minima:** `flooding`,
  `deflation`, or `mlsl`.
* **High `n`, want the global (or a few good) minima:** `basinhopping`
  first; `multistart` with many parallel starts as a baseline; `tunneling`
  for a descending trail.
* **High `n` and you still want flooding-style biasing:** restrict the
  bumps to a handful of collective variables, as in metadynamics, rather
  than the full coordinate vector.
* **Always:** each individual solve inherits pounce's scalability; the
  bottleneck is the *number* of solves and, for repulsion, the loss of
  sparsity — not the local solver.
