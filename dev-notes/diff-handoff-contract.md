# The `solve → DiffHandoff` contract — design exploration

**Status: design exploration.** This note proposes consolidating
POUNCE's several differentiable-solve backward passes onto a single,
solver-agnostic **handoff contract**: every solve emits one well-defined
bundle of post-convergence data, and every consumer (JAX, PyTorch, a
future C/Rust autodiff user, discopt) differentiates from *that*, not
from solver-specific internals.

This is general-purpose differentiable-solver work: useful to every
consumer, owned by POUNCE for all of them, and referencing no downstream
orchestrator (MIP / global / branch-and-bound all live in discopt, not
here). That ownership test — "would a non-discopt user want it?" — is why
the contract belongs in POUNCE while the combinatorial layers do not.

## Progress / current state (corrects the original framing)

The first implementation pass and a build-and-test sweep revised the
picture below. Recorded here so the rest of the note is read in light of
it:

- **`jax/_diff.py` is consolidated** — the three hand-copied NLP backward
  bodies (plain `solve`, `solve_with_warm`, batched `vmap_solve_parallel`)
  now route through one `_kkt_implicit_backward` helper. Verified:
  `test_jax.py` 85 passed.
- **The other three surfaces were *already* internally single-source** —
  `torch/_diff.py`, `jax/_qp.py`, `torch/_qp.py` each have exactly one
  backward helper, with batched paths `vmap`-ing over it. The "four
  hand-written copies" framing below was only literally true *within*
  `jax/_diff.py`; across files the duplication is **cross-framework**
  (jax↔torch namespace ports), not within-framework.
- **NLP and QP backward are different algorithms and must stay separate.**
  `_diff` is active-set implicit diff (the pounce#73 slack-row fix);
  `_qp` is OptNet (Amos & Kolter 2017) reading multipliers directly via
  `diag(λ)` / `diag(Gx−h)` cone scalings, with **no active set**
  (`torch/_qp.py` says so). A single shared backward across NLP+QP would
  be a correctness regression, not a cleanup. The contract unifies the
  *handoff data shape and naming*, not the backward algorithm.
- **Cross-frontend parity already holds.** `test_parity_jax_torch.py`
  directly compares `dL/dp` for the same Rust core under JAX and torch;
  it passes (35 passed across parity + torch suites; 20 across jax
  QP/SOCP). So the "one numerical backbone, any autodiff frontend" claim
  is *already true and tested* — the consolidation is about removing
  duplicate **derivation**, not about achieving parity.

Net: the remaining Python-side work is **naming unification** (small) and
**not** a four-way backward merge. The high-value remaining work is the
**Rust `DiffHandoff` struct + the `nlp_pounce.py` seam** (steps 5–6
below), which is where the discopt story and batched-`kkt_solve_many`
backward actually live.

## Why now: the backward derivation was forked (now partly consolidated)

POUNCE differentiates solves across NLP/QP × JAX/torch. The active-set +
KKT-assembly logic was hand-copied — three times inside `jax/_diff.py`
(now fixed), and still mirrored across the jax↔torch framework boundary:

| Surface | File | Duals it reads | Backward |
|---|---|---|---|
| NLP / JAX | `python/pounce/jax/_diff.py` | `info["mult_g"]`, `info["mult_x_L/U"]` | active-set; **now one helper** |
| NLP / Torch | `python/pounce/torch/_diff.py` | same dict, repacked | active-set; line-for-line jax port |
| QP / JAX | `python/pounce/jax/_qp.py` | `lam`, `nu` | OptNet (no active set) |
| QP / Torch | `python/pounce/torch/_qp.py` | same | OptNet; line-for-line jax port |

The NLP surfaces encode the *same* facts — "a bound is active when
`|mult| > tol`", "an equality row is always active", "pinned variables
get `dx/dp = 0`". Within `jax/_diff.py` that was three copies (fixed).
Across frameworks the jax and torch NLP backwards remain namespace-only
ports — unifying *those* needs an array-API shim over both frameworks and
is deferred (its only payoff is dropping a two-port maintenance burden;
parity is already tested). The QP surfaces are a *different* algorithm
and are intentionally not merged with the NLP ones.

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

Status tags reflect the build-and-test sweep recorded above.

0. **[DONE] Consolidate the in-file NLP backward.** `jax/_diff.py`'s
   three copies → one `_kkt_implicit_backward`. `test_jax.py` 85 passed.
1. **[DONE] Name the struct in Rust.** `pounce_sensitivity::DiffHandoff`
   (`from_solution` / `from_sens_result`, the precomputed
   `active_constraints` / `pinned_vars` masks, `DEFAULT_ACTIVE_TOL`, and
   `pin()` for B&B-leaf integer fixing). 3 unit tests; crate green.
2. **[DONE, producer side] Compute the active set once in the producer.**
   `build_info_dict` (PyO3) emits `pinned_vars` / `active_constraints` /
   `active_tol` on every NLP solve, computed via `DiffHandoff`. The masks
   now ride the info dict — so `nlp_pounce.py` / discopt already receive
   them (this is also most of step 6). `test_problem.py` asserts them
   against HS071's known active set; `test_jax.py` 85 passed (the
   producer addition is behavior-neutral on gradients).
   *Consumer side not done:* the JAX/torch backwards still recompute the
   mask. That rewiring is behavior-neutral by construction (same tol,
   same rule), so it has no test signal and was intentionally skipped —
   do it only when a *non-AD* consumer needs the mask it can't recompute.
3. **[REVISED — document, don't rename] Reconcile the multiplier naming.**
   Investigation showed the surface names are *deliberate public
   contracts*, not accidental drift, so renaming would break consumers:
   - `mult_g` / `mult_x_L` / `mult_x_U` (NLP) is the **cyipopt
     compatibility** contract — the `Problem` API is cyipopt-shaped and
     the C ABI exports exactly these symbols for cyipopt / JuMP / AMPL
     clients. Untouchable.
   - `x` / `y` / `z` / `z_lb` / `z_ub` (QP) is the **OptNet / convex-solver
     convention** for a structurally different problem
     (`min ½xᵀPx s.t. Gx≤h, Ax=b`; `y` = equality, `z` = inequality).
   - `lam` / `nu` are **internal** to `_qp.py` and already consistent.
     Note `lam` means *different things* across files (`_diff.py`: *all*
     constraint duals; `_qp.py`: *inequality-only*), so "one name" would
     be actively wrong.

   The legitimate intent — a single canonical convention consumers can
   rely on — is therefore satisfied by **documentation**, not renaming:
   `DiffHandoff.lambda` is the canonical field, and the seam doc (step 6)
   tabulates how each surface's duals map onto it. Folded into step 6.
4. **[ALREADY DONE — pounce#76/#77] Factor-reuse / batched backward.**
   `JaxProblem` already routes both the single (`_bwd_single_factor_reuse`)
   and batched (`_bwd_batched_factor_reuse`) backward through
   `Solver.kkt_solve_many` against the held LDLᵀ factor (default
   `factor_reuse=True`), avoiding the dense `O((n+m)³)` re-solve. Gradient-
   tested sparse-vs-dense in `test_jax.py`. The module-level *functional*
   API (`_diff.py` `solve` / `vmap_solve_parallel`) intentionally stays on
   the dense path: it holds no `Solver`, and `JaxProblem` is the
   documented path when backward performance matters. Bringing QP onto a
   uniform factor handle is the only remaining sub-item, low priority.
5. **[deferred] Cross-framework jax↔torch NLP merge.** Only behind an
   array-API shim; gated on the two-port maintenance burden actually
   biting. Parity is already tested, so this is pure de-duplication with
   no correctness upside — do it last, or not at all.
6. **[DONE] Expose + document across the seam.** The masks are in the
   info dict `nlp_pounce.py` reads, so the contract data already crosses
   the language boundary. The user-facing contract — info-dict mask keys,
   the canonical `lambda` field, and the per-surface multiplier mapping
   table (this absorbs step 3's documentation intent) — is written up in
   [`docs/src/differentiable-solves.md`](../docs/src/differentiable-solves.md)
   and linked from `SUMMARY.md`.

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
