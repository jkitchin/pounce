# pounce-simplex

A bounded-variable revised simplex LP solver, in pure Rust, for the POUNCE
spatial branch-and-bound global optimizer.

It exists for one job the interior-point method does badly: the spatial
solver's optimization-based bound tightening (OBBT) solves `2n` LPs per node
that share one polytope and differ only in the objective (`min`/`max` each
variable). A simplex method warm-starts from the previous optimal basis and
reaches the next optimum in a handful of pivots, where an IPM must re-walk the
central path from scratch every time. The same warm-start handles the small
bound changes between a parent box and its child.

```text
minimize    cᵀ x
subject to  A x = b
            l ≤ x ≤ u          (bounds may be ±∞; inequalities enter as slacks)
```

Status: Phase 6.1 — correctness-first cold solve (two-phase bounded-variable
revised primal simplex, explicit dense basis inverse with product-form updates
and periodic refactorization). Sparse LU (6.2), warm-start modes (6.3), and the
OBBT wiring (6.4) follow.
