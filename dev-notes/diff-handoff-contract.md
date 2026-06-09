# The `solve → DiffHandoff` contract — design exploration

**Status: design exploration.** This note proposes consolidating
POUNCE's several differentiable-solve backward passes onto a single,
solver-agnostic **handoff contract**: every solve emits one well-defined
bundle of post-convergence data, and every consumer (JAX, PyTorch, a
future C/Rust autodiff user, discopt) differentiates from *that*, not
from solver-specific internals.

This is the one piece of MIP/global groundwork that is unambiguously
POUNCE's to own (see
[`mip-global-routing.md`](./mip-global-routing.md), RESOLVED banner):
it is general-purpose differentiable-solver work, useful to every
consumer, and it references no downstream orchestrator.

## Why now: the backward pass is already forked four ways

POUNCE today differentiates solves in at least four places, and **each
re-derives the same implicit-function-theorem logic from a differently
named set of multipliers**:

| Surface | File | Duals it reads | Active-set logic |
|---|---|---|---|
| NLP / JAX | `python/pounce/jax/_diff.py` | `info["mult_g"]`, `info["mult_x_L/U"]` | re-derived in `bwd` |
| NLP / Torch | `python/pounce/torch/_diff.py` | same dict, repacked | re-derived again |
| QP / JAX | `python/pounce/jax/_qp.py` | `d["z"]` (`lam`), `nu` | re-derived (`_kkt_backward`) |
| QP / Torch | `python/pounce/torch/_qp.py` | same | re-derived again |

Each works. But they encode the *same* facts — "a bound is active when
`|mult| > tol`", "an equality row is always active", "pinned variables
get `dx/dp = 0`" — in four hand-written copies, under two naming
conventions (`mult_g`/`mult_x_L` vs `lam`/`nu`). Adding a conic surface,
a torch path for it, or a discopt-facing MINLP-leaf handoff multiplies
the copies again. That is exactly the "retrofit the backward onto each
solver" failure mode the contract is meant to prevent.

The Rust side is already *more* consolidated than the Python side:
`pounce_sensitivity::ConvergedState` + `Solver` (`crates/pounce-sensitivity/
src/solver.rs`) already expose `x`, `obj_val`, the converged KKT factor,
`kkt_solve` / `kkt_solve_many` / `parametric_step` /
`compute_reduced_hessian`. The contract is mostly about **naming and
surfacing** what the Rust core already computes, then having every
frontend consume it uniformly.

## The contract

A single struct, emitted by every solve that supports differentiation —
NLP, QP, LP, conic, and (for discopt) the fixed-integer leaf of a B&B:

```rust
/// Everything the implicit-function-theorem backward pass needs.
/// Solver-agnostic: produced identically by IPM-NLP, convex-QP/LP,
/// conic, and a B&B leaf (original problem with integers fixed).
pub struct DiffHandoff {
    // ---- primal/dual solution ----
    pub x: Vec<f64>,             // primal solution
    pub obj_val: f64,
    pub lambda: Vec<f64>,        // general-constraint multipliers (g / G / A)
    pub mult_x_lower: Vec<f64>,  // variable lower-bound multipliers
    pub mult_x_upper: Vec<f64>,  // variable upper-bound multipliers

    // ---- active set, computed ONCE, here ----
    /// Constraint rows in the differentiated KKT block: equalities
    /// (always) ∪ inequalities with |λ| > active_tol. Precomputed so no
    /// frontend re-derives it.
    pub active_constraints: BitVec,
    /// Variables pinned in the backward (dx/dp = 0): active bounds, and
    /// — for a B&B leaf — the integer variables fixed at the optimum.
    pub pinned_vars: BitVec,
    pub active_tol: f64,

    // ---- the reusable factor (the expensive object) ----
    /// Converged KKT factor, reused across back-solves. This is the
    /// pounce_sensitivity::Solver factor, surfaced — not recomputed.
    pub factor: Option<Factorization>,
}
```

Three design commitments make it compose:

1. **One multiplier convention.** `lambda` / `mult_x_lower` /
   `mult_x_upper` is *the* naming, everywhere. The QP path's `lam`/`nu`
   and the NLP path's `mult_g`/`mult_x_L` both map onto it at the
   boundary, and no backward pass sees the solver-specific names again.
2. **Active set is computed once, in the producer.** `active_constraints`
   and `pinned_vars` are *outputs of the handoff*, not something each
   frontend recomputes from `|mult| > tol`. This kills the four-way
   duplication: the rule lives in exactly one place. (It also makes the
   tolerance a single, documented knob instead of four ad-hoc
   `_ACTIVE_TOL`s.)
3. **The factor is surfaced, not rebuilt.** `factor` is the
   already-converged `pounce_sensitivity` KKT factor. The backward pass
   is then "assemble the RHS, call `factor.solve`", which is exactly
   what `kkt_solve` / `kkt_solve_many` already do — so batched/`vmap`
   backward is `kkt_solve_many` and needs no new linear algebra.

## What the backward pass becomes

With the handoff, every frontend's `bwd` collapses to the *same* three
steps, parameterized only by the cotangent and the problem's
parameter-Jacobians (which are the frontend's job — JAX `jacrev`, torch
autograd, or AD in Rust):

```
bwd(handoff, cotangent_x):
    1. mask:    use handoff.pinned_vars / active_constraints directly
                (no |mult| > tol recomputation)
    2. solve:   handoff.factor.solve(rhs)    # or kkt_solve_many for batches
    3. contract: combine with ∂_p (∇L, g) to get the parameter cotangent
```

Steps 1–2 are *identical* across NLP/QP/LP/conic and across JAX/torch.
Only step 3's parameter-Jacobian assembly is frontend-specific, and even
that is mechanical. The net effect: the implicit-diff logic lives once,
in terms of the contract, and a new surface (conic, a torch conic path,
a discopt leaf) is "produce a `DiffHandoff`" — the backward comes free.

## Layering — who produces, who consumes

```
producers (emit DiffHandoff)          consumers (differentiate from it)
  pounce-algorithm  (IPM-NLP)   ┐      ┌  pounce.jax   (custom_vjp)
  pounce-convex     (LP/QP/conic)├─▶ DiffHandoff ─▶┤  pounce.torch (autograd.Function)
  [discopt B&B leaf: fixed-int] ┘      └  C ABI / Rust autodiff (future)
                                          discopt (its own autodiff, across the seam)
```

- **Producers** already compute everything in the struct; the work is to
  *return it in this shape* rather than as a loosely-typed `info` dict
  with surface-specific keys.
- **Consumers** stop re-deriving active sets and KKT assembly; they read
  the contract. discopt is just another consumer — it composes its
  Python/JAX autodiff with whatever `nlp_pounce.py` returns, and the
  contract is what `nlp_pounce.py` should return.

## Why this is the right MIP/global groundwork

The RESOLVED decision put all B&B in discopt. The *only* thing POUNCE
must get right for discopt's differentiable MINLP to work is that **a
single continuous solve hands back a clean, complete, differentiable
bundle** — because a B&B leaf is just a continuous solve with integers
fixed, and discopt differentiates it exactly like any node solve. So:

- Solidifying this contract is necessary for discopt's MINLP
  differentiability **and** independently valuable for POUNCE's own
  JAX/torch QP/NLP/conic layers.
- It introduces **no** B&B, no `pounce-mip`, no discopt reference — it
  is pure general-purpose solver work (the test from the RESOLVED
  banner: "would a non-discopt user want it?" — yes, every differentiable
  layer wants it).
- It is the seam `nlp_pounce.py` should speak: extend `solve_nlp`'s
  return to carry the `DiffHandoff`, and discopt's gradient composes
  across the language boundary for free.

## Incremental path (each step shippable, no regressions)

1. **Name the struct, map existing surfaces onto it.** Introduce
   `DiffHandoff` in Rust (a thin re-shape of `ConvergedState` +
   multipliers); add a Python view. No behavior change — the existing
   `info` dict keeps working; the handoff is an additional, typed view.
2. **Move active-set computation into the producer.** Compute
   `active_constraints` / `pinned_vars` once and expose them; switch
   `_diff.py` and `_qp.py` to *read* them instead of recomputing
   `|mult| > tol`. Delete the duplicated masking. Lock with the existing
   finite-difference gradient tests (they must not move).
3. **Unify the multiplier naming at the boundary.** NLP and QP frontends
   consume `lambda`/`mult_x_*`; the `lam`/`nu`/`mult_g` names become
   internal mapping detail. One convention from here out.
4. **Surface the factor uniformly.** Route every backward through
   `factor.solve` / `kkt_solve_many` (batched), so `vmap`/batched
   backward is one code path. (NLP already does factor-reuse; bring QP
   onto the same handle.)
5. **Conic + torch parity for free.** A new surface implements *produce
   a `DiffHandoff`*; its backward is the shared steps 1–2. Validates the
   contract by construction.
6. **Expose across the seam.** `nlp_pounce.py` returns the handoff;
   document it as the stable contract discopt differentiates against.

## Verification

- **No-regression gate:** the existing finite-difference gradient checks
  in `python/tests/test_jax.py` (and the torch/QP equivalents) must pass
  unchanged at every step — the contract is a refactor of *how* the
  gradient is assembled, not *what* it is.
- **Cross-surface equivalence:** a problem expressible as both an NLP and
  a QP must yield the same gradient through both surfaces (same handoff →
  same backward). This is a new test the contract makes possible and is
  the strongest proof the duplication is truly gone.
- **Batched equivalence:** `vmap`/batched backward via `kkt_solve_many`
  matches the loop-over-singletons gradient.

## Open questions

- **Where does `DiffHandoff` live?** Candidate: `pounce-sensitivity`
  (it already owns `ConvergedState` and the factor). It would re-export
  from a common crate so `pounce-convex` can produce one without a
  circular dep. Needs a dependency-graph check.
- **Factor ownership / lifetime across the FFI + JAX callback boundary.**
  The factor is an `Rc`-backed Rust object; the existing JAX path stashes
  a handle in the `custom_vjp` residual (pounce#75–77 LRU). The contract
  should standardize that stashing rather than leave each frontend to
  reinvent it.
- **Degenerate / weakly-active sets.** A single `active_tol` in the
  producer is cleaner than four, but weak activity (multiplier ≈ tol) is
  where implicit-diff gradients are least stable. Worth a documented
  policy (and possibly a returned "near-active" mask) rather than a
  silent threshold.
