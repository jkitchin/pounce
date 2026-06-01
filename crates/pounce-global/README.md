# pounce-global

Deterministic **global** optimization of factorable nonconvex NLPs by
spatial branch-and-bound. Where `pounce-convex` solves convex/conic problems
to the global optimum and the NLP filter-IPM finds a *local* KKT point,
this crate certifies a **global** optimum of a nonconvex problem, returning a
feasible point and a proven optimality gap.

```text
minimize    f(x)
subject to  cl_j ≤ g_j(x) ≤ cu_j        (j = 0 … m−1)
            x_lo ≤ x ≤ x_hi
```

`f` and the `g_j` are *factorable* expressions (built from `+ − × ÷`, integer
powers, `√`, `exp`, `ln`, `|·|`, `sin`, `cos`).

## Example

```rust
use pounce_global::{expr::var, solve_global, GlobalProblem, GlobalOptions, GlobalStatus};
use pounce_feral::FeralSolverInterface;

// Six-hump camel: six local minima, two global ones (value ≈ −1.0316).
let x = var(0);
let y = var(1);
let f = 4.0 * x.clone().powi(2) - 2.1 * x.clone().powi(4) + (1.0 / 3.0) * x.clone().powi(6)
    + x.clone() * y.clone() - 4.0 * y.clone().powi(2) + 4.0 * y.powi(4);

let prob = GlobalProblem::new(vec![-2.0, -1.5], vec![2.0, 1.5], &f);
let sol = solve_global(&prob, &GlobalOptions::default(), || Box::new(FeralSolverInterface::new()));

assert_eq!(sol.status, GlobalStatus::Optimal);
assert!((sol.objective - (-1.0316)).abs() < 1e-2); // certified global minimum
```

## Method

Best-first spatial branch-and-bound over variable boxes. At each node:

1. **Bound tightening** — feasibility-based (FBBT, via `pounce-presolve`) then
   **optimization-based** (OBBT): minimize/maximize each variable over the
   relaxation with an incumbent cutoff. Prune if the box collapses.
2. **Lower bound** — a McCormick polyhedral **relaxation** of the problem over
   the box, solved as a linear program through `pounce-convex`. It is exact in
   the zero-width-box limit, so the search converges.
3. **Upper bound** — probe feasible points (the relaxation solution, the box
   center) and polish with a local NLP solve (`pounce-algorithm`) for a sharp
   incumbent.
4. **Branch** — split the **most-violated** variable (whose nonconvexity drives
   the relaxation gap), falling back to the widest box side; the frontier is
   ordered by node lower bound.

Every relaxation cut is a *verified global* under/over-estimator, so adding
any of them only tightens the bound — the certified optimum never changes.

### The relaxation suite

The lower bound is where the work is. Reusing the `FbbtTape` factorable-
expression representation (`pounce-nlp`) and interval arithmetic
(`pounce-presolve`):

| Component | What it does |
|---|---|
| **Tight envelopes** | exact convex/concave hulls of every univariate atom; the tangent-from-endpoint construction for single-inflection arcs (odd powers across 0, `sin`/`cos` over a sub-π box) |
| **McCormick** | the exact convex hull of each bilinear product |
| **Sandwich cuts** | tangent cuts at the LP point, re-solved — tightens the bound without branching |
| **OBBT** | min/max each variable over the relaxation + incumbent cutoff — the strongest box reducer |
| **αBB** | a convex underestimator of the whole objective from a rigorous interval-Hessian spectral shift |
| **RLT** | level-1 reformulation-linearization (affine constraint × bound factors) |
| **Multilinear** | intersect all three groupings of a 3-way product, not just one nested McCormick |

Each is a tunable, individually-disablable knob on `GlobalOptions`; all are on
by default.

## When to use it

Reach for `pounce-global` when the problem is genuinely nonconvex and you need
a *certified* global optimum (with a gap) rather than a local one. For convex
or conic problems, `pounce-convex` is the right tool — local is already global
there. For polynomial problems specifically, the SOS/Lasserre optimizer in
`pounce-convex` (`sos_minimize`) is often a better fit (one SDP, no branching).

## Status & limits

A complete, correct continuous global solver. Current limits:

- **Continuous variables only** — no integer branching (MINLP) yet.
- **Branching** is most-violation (falling back to widest); pseudocost /
  reliability branching would cut node counts further still.
- The Lagrangian Hessian used by the local NLP upper bound is finite-
  differenced (a usable Newton direction, not exact second order).
- `sin`/`cos` over a box wider than π, division by an interval straddling
  zero, and unsupported atoms fall back to the (valid, weak) interval box
  bound, which branching then sharpens.

See the [Global Optimization](../../docs/src/global-optimization.md) chapter of
the book for the full treatment.
