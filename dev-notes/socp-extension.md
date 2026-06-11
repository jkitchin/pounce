# SOCP extension for the convex IPM — design note

**Status: Phases 1 + 2 landed — pounce solves SOCPs.** Captures the design
for adding a second-order cone (SOC) to `pounce-convex`'s interior-point
solver. Phase 1 (the `CompositeCone` refactor) and Phase 2 (the NT scaling,
the generalized dense-block KKT, and `solve_socp_ipm`) are implemented and
validated; the remaining items (cone-aware presolve gating, SOC warm
start, low-rank KKT for large cones, cone-aware differentiable layer) are
scoped below.

## Outcome (Phases 1–2)

`solve_socp_ipm(prob, &[ConeSpec], …)` solves `min ½xᵀPx+cᵀx s.t. Ax=b,
Gx ⪯_K h` over a product of nonnegative-orthant and second-order cones,
with closed-form-validated optima (norm minimization, linear-over-SOC,
Euclidean projection onto a cone) and a mixed orthant+SOC case — see
`tests/socp.rs`. Correctness is **intrinsic**: the IPM only reports
`Optimal` at a verified KKT point (residual below tolerance, `s,z` kept in
the cone), so no external reference solver is needed. The NT reduced
system (`block = W⁻² = η²Q_{w̄}`, `rhs = Arw(z)⁻¹ r_comp`, `recover_ds =
−rhs − W⁻²dz`) was derived to be self-consistent and reduces exactly to
the orthant in 1-D; the orthant LP/QP path is byte-identical (all prior
tests pass).

## Motivation

`pounce-convex` today solves LP/QP over the nonnegative orthant (plus a
box, expanded into orthant rows). Adding SOC moves pounce into the
*second-order cone program* class — the same problem class differentiable
GPU solvers (Moreau) and general conic solvers (Clarabel) target, and the
single highest-leverage gap versus them. Everything pounce already has —
presolve with dual postsolve, warm starting, rayon batching, symbolic
factor reuse, the JAX/OptNet differentiable layer — then applies to a much
larger problem class.

## What the driver already abstracts vs. bakes in

The [`cones::Cone`](../crates/pounce-convex/src/cones/mod.rs) trait already
owns `mu`, `scaling_diag`, `comp_residual`, `comp_residual_corrector`,
`recover_ds`, `max_step`, and `run_ipm` calls them generically. The
residuals (`r_d, r_p, r_g` via matvecs), `split_step`, factor reuse, and
the predictor–corrector structure are cone-agnostic.

Two orthant assumptions are **baked into the driver** and are the crux for
SOC:

1. **The `(z,z)` KKT block is diagonal.** `KktStructure` allocates exactly
   one entry per inequality row (`z_diag_pos[i]`); `update_scaling` writes
   `-scaling[i] - reg` there; `scaling_diag` returns a *vector*. SOC's
   Nesterov–Todd block `W²` is dense within each cone (diagonal + rank-1),
   so a per-row diagonal cannot represent it.
2. **`build_rhs` divides by `z` elementwise** (`-r_g[i] + r_c[i]/z[i]`) —
   the orthant's analytic elimination of the slack block. SOC replaces
   `1/z` with an NT-scaled apply.

## The math SOC adds

Jordan algebra of `K = { (s₀, s₁) : s₀ ≥ ‖s₁‖₂ }`, with
`J = diag(1,−1,…,−1)`, identity `e = (1,0,…,0)`, product
`(s∘z)₀ = sᵀz`, `(s∘z)₁ = s₀ z₁ + z₀ s₁`.

- **Rank / degree = 2** per SOC (independent of dimension):
  `μ = ⟨s,z⟩ / Σ rank`, orthant contributes `n`, each SOC contributes `2`.
- **NT scaling.** With `det(u) = u₀² − ‖u₁‖²`,
  `η = (det(s)/det(z))^{1/4}`, normalized `s̃ = s/√det(s)`,
  `z̃ = z/√det(z)`, `γ = √((1 + s̃ᵀz̃)/2)`, scaling point
  `w̄ = (s̃ + J z̃)/(2γ)`. The KKT block is
  ```
  W² = η²(2 w̄ w̄ᵀ − J) = η²·diag(−1, 1, …, 1) + 2η²·w̄ w̄ᵀ
  ```
  i.e. **diagonal + rank-1** — the structure that enables the sparse
  expansion.
- **Step to boundary** (`max_step`): largest `α` keeping `v + α dv` in
  `int(K)` — the smaller positive root of `det(v + α dv) = 0`, capped at 1.
- **Self-dual:** `K* = K`. Dual feasibility is `z ∈ K`; the verified
  Farkas/recession certificates change from `z ≥ 0` / `Gd ≤ 0` to
  `z ∈ K` / `Gd ∈ −K`.

## Architecture

### Composite cone (Phase 1)

The inequality block becomes a *product* of cones
`K = R₊^{n₀} × SOC(m₁) × SOC(m₂) × …`. A `CompositeCone` owns an ordered
list of `(offset, ConeKind)` blocks and dispatches every `Cone` method
block-wise (slicing `s`/`z`/`out` per block; `mu` sums `⟨s,z⟩` and ranks;
`max_step` takes the min). `ConeKind` is a closed enum (`Nonneg`, later
`SecondOrder`) — no `dyn` dispatch. The driver holds a `CompositeCone`
instead of a bare `NonnegCone`. With a single `Nonneg` block this is
bit-identical to today (Phase 1's correctness guarantee).

### Problem cone declaration (Phase 2)

```rust
pub enum ConeSpec { Nonneg(usize), SecondOrder(usize) }   // dims, row order
// QpProblem gains: pub cones: Vec<ConeSpec>   (empty ⇒ all-nonneg, back-compat)
```
Bounds keep expanding into `Nonneg` rows; SOC constraints append
`SecondOrder(mₖ)` blocks to `G`/`h`. Riding on `QpProblem` (rather than a
new type) keeps presolve / warm-start / batch / factor-reuse working
through the existing paths.

### Trait extension (Phase 2)

Promote the two baked-in operations to the trait:
```rust
fn kkt_block(&self, s, z, reg) -> ConeBlock;   // Diagonal | Dense | DiagPlusLowRank
fn rhs_comp_term(&self, s, z, r_c, out);       // generalizes r_c / z
```
`KktStructure`/`build_rhs` consume these instead of assuming diagonal.

### KKT `(z,z)` block: two tiers

- **Tier A (dense block, first):** reserve a dense lower-triangular
  `mₖ×mₖ` block per SOC; fill from `W²` each iteration. Correct and
  simple; fine for `mₖ ≲ 10–20`. Localized to `KktStructure::build`
  (layout) and `update_scaling` (write).
- **Tier B (sparse low-rank, later):** exploit `W² = D + ρ vvᵀ` — add 1–2
  auxiliary rows/cols per SOC so the augmented `(z,z)` stays
  diagonal-plus-sparse (ECOS/Clarabel trick), preserving fill on large
  cones.

## Presolve extension

Postsolve (transaction stack + global dual recovery) is unaffected — SOC
multipliers pass through the `kept_ineq` mapping. Reduction *detection*
must be **gated per cone**:

- *Keep, gated to nonneg/box rows & cols:* empty rows, fixed-var, free /
  free-singleton columns, duplicate / parallel rows — only when the
  rows/cols are not part of an SOC block (an SOC's rows are coupled).
- *Skip SOC rows:* activity-bound, forcing, dominated columns, bound
  tightening — these are `≤`-row reductions with no per-row meaning for a
  cone constraint. Add a "row ∈ SOC block ⇒ skip" guard in the detection
  passes.

## Warm-start extension

The adaptive recentering generalizes by replacing the positivity floor on
`s`/`z` with a floor on the **distance to the cone boundary**
`λ_min = s₀ − ‖s₁‖`, projecting the warm point back to `int(K)`. Same
structure, cone-aware primitive. Cold start seeds SOC blocks at the cone
identity `e = (1,0,…,0)`, not `1`.

## Differentiable-layer extension (last)

The OptNet backward currently linearizes complementarity as
`diag(λ)`/`diag(slack)` — pure orthant. SOC needs the Jordan-product / NT
differential (arrow blocks instead of `diag`). The forward already returns
`(x, z)` regardless of cone; only the backward KKT differential is
cone-specific. **Ship SOC forward/solve first; keep the differentiable
layer LP/QP-only**, then add cone-aware implicit diff as a distinct
follow-up (derive + finite-difference-validate per cone, as for the matrix
gradients).

## Phased plan

| Phase | Scope | Risk |
|---|---|---|
| **1** | `CompositeCone` + `ConeKind`; driver routed through it; `NonnegCone` behind it. **No behavior change.** | low — pure refactor, existing tests guard it |
| 2 | `ConeSpec` on `QpProblem`; trait gains `kkt_block`/`rhs_comp_term`; `SecondOrderCone` NT scaling; **Tier-A dense KKT block**; cold start at `e`; cone `max_step`/`mu`; solve standard-form SOCPs | **medium-high** — NT reduced-system algebra; validate vs known optima + a reference solver |
| 3 | Cone-aware infeasibility certificates; per-cone presolve gating | low–medium |
| 4 | Warm-start recentering on `λ_min`; SOCP input plumbing (CLI/`.nl`/Python wrapper) | low–medium |
| 5 | Tier-B sparse low-rank KKT expansion (large cones) | medium — fill/perf, not correctness |
| 6 | Cone-aware differentiable layer (JAX) | medium-high — new dual-diff derivation |

The single highest-risk artifact is the NT reduced-system algebra in
Phase 2 (`kkt_block` + `rhs_comp_term` + `recover_ds` must be mutually
consistent). Validate it the way everything else in this crate is:
known-optima tests plus a randomized KKT-residual check against a trusted
SOCP solver.

## Phase 1 — what lands now, and what is deliberately deferred

**Lands:** `CompositeCone`/`ConeKind` and the driver routed through a
single-`Nonneg` composite. This is a pure internal refactor: no public API
change, no behavior change, fully guarded by the existing convex test
suite. It creates the block-dispatch seam every later phase plugs into.

**Deferred to the start of Phase 2** (to avoid dead scaffolding that could
rot): the `QpProblem.cones` field and the `kkt_block`/`rhs_comp_term`
trait methods. They only earn their keep once a non-diagonal cone exists,
and adding them against an only-diagonal implementation now would be
unused surface. Phase 2 introduces them together with `SecondOrderCone`.
