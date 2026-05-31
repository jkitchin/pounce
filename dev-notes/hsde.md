# Homogeneous self-dual embedding for the convex IPM — design note

**Status: Phases H2–H4 landed — HSDE solves LP/QP/SOCP and is a
selectable driver (`QpOptions::use_hsde`). H5 (exponential cone) next.**
Chosen as the foundation for Clarabel cone parity (see
`clarabel-parity.md`): reformulate the interior-point driver into a
homogeneous self-dual embedding (HSDE), prove it reproduces every existing
LP/QP/SOCP result and infeasibility certificate, switch over, and *then*
add the non-symmetric (exp/power) and PSD cones onto the uniform HSDE
driver — the structure Clarabel, SCS, and ECOS use.

## Why HSDE

The current driver (`ipm.rs`) is an infeasible-start primal–dual method
with a **bolt-on** verified certificate check (`detect_infeasibility`). It
works, but:

- infeasibility/unboundedness is detected by watching the iterate diverge
  along a Farkas/recession ray — robust but heuristic in *when* it fires;
- there is no single self-starting iterate that handles primal- and
  dual-infeasible problems uniformly;
- non-symmetric cones (exp, power) are far better behaved inside HSDE — the
  embedding bounds the iterates and gives a clean central path.

HSDE folds primal, dual, and the infeasibility certificates into **one**
self-dual system. Its solution either has `τ > 0` (recover the optimal
primal–dual point by dividing by `τ`) or `κ > 0` (a certificate of
primal or dual infeasibility) — decided *at convergence*, not by a side
test.

## What is reused (the whole point)

The per-cone math — `kkt_block` (NT scaling `W²`), `rhs_comp_term`,
`recover_ds`, `comp_residual{,_corrector}`, `max_step`, `mu` — is **reused
verbatim**. So is `KktStructure`: the embedding borders the existing
symmetric `(x, y, z)` block

```text
      ⎡ P+δI   Aᵀ      Gᵀ      ⎤
  M = ⎢ A      −δI     0       ⎥        (exactly today's KKT matrix)
      ⎣ G      0     −W²−δI    ⎦
```

with one extra scalar `τ` (and its complement `κ`). The bordered system is
solved by **two** back-solves through the *same* factorization of `M` plus
a scalar Schur complement (the SCS/ECOS scheme), so the factorization, AMD
ordering, refactor-per-iteration, and the SOC aux-variable trick are
untouched. What changes is the outer iteration: residuals, the τ/κ row,
the step combination, the step length, and termination.

## The embedding — linear conic case (P = 0)

For `min cᵀx  s.t.  Ax = b, Gx + s = h, s ∈ K` with conic dual
`z ∈ K*` and free equality dual `y`, the self-dual embedding introduces
`τ ≥ 0, κ ≥ 0`:

```text
 (1)  Aᵀy + Gᵀz + c τ            = 0          (r_x, length n)
 (2)  A x            − b τ        = 0          (r_y, length m_eq)
 (3)  G x + s        − h τ        = 0          (r_z, length m_ineq)
 (4)  −cᵀx − bᵀy − hᵀz       − κ = 0          (r_τ, scalar)
      s ∈ K,  z ∈ K*,  τ ≥ 0, κ ≥ 0,  sᵀz = 0,  τκ = 0
```

This system is **self-dual** (the matrix is skew-symmetric apart from the
cone block). Goldman–Tucker: it has a solution with `τ + κ > 0`, and

- `τ > 0, κ = 0` ⇒ `(x, y, z, s)/τ` is an optimal primal–dual point;
- `τ = 0, κ > 0` ⇒ `cᵀx + bᵀy + hᵀz < 0` is impossible, so either
  `bᵀy + hᵀz < 0` with `Aᵀy+Gᵀz = 0, z ∈ K*` (primal-infeasible Farkas
  certificate) or `cᵀx < 0` with `Ax = 0, Gx + s = 0, s ∈ K`
  (dual-infeasible / unbounded recession ray).

### Central path and the Newton step

Relax the two complementarity conditions to `s ∘ z = σμ e` and
`τκ = σμ`, with `μ = (sᵀz + τκ)/(degree + 1)`. The Newton system for
`(Δx, Δy, Δz, Δs, Δτ, Δκ)` is the embedding matrix linearized. Eliminating
`Δs` via the cone (`Δs = −W²Δz − rhs_comp`, exactly `recover_ds`) and `Δκ`
via `τΔκ + κΔτ = σμ − τκ`, the reduced system is the bordered

```text
  ⎡ M   ⎤ ⎡Δx⎤   ⎡ ... ⎤        with border column   bcol = (c, −b, −h)
  ⎢   b ⎥ ⎢Δy⎥ = ⎢     ⎥        and  Δτ closing row    (−cᵀ,−bᵀ,−hᵀ)·(Δx,Δy,Δz)
  ⎣ col ⎦ ⎣Δz⎦   ⎣  .  ⎦                                 − (κ/τ) Δτ = r_τ + σμ/τ − κ
```

i.e. `M·Δw + Δτ·bcol = rhs_w` and `bcolᵀ·Δw − (κ/τ)Δτ = rhs_τ` (signs as in
(1)–(4)). **Two-solve scheme** (one factorization of `M`):

```text
  solve  M p = bcol        (the "constant" direction; depends only on data + scaling)
  solve  M q = rhs_w        (the "residual" direction)
  Δτ = (rhs_τ − bcolᵀ q) / (−κ/τ − bcolᵀ p)
  Δw = q − Δτ · p
```

`p` can be reused between the predictor and corrector (same `M`, same
`bcol`); only `q` and the scalars differ. So HSDE costs **one extra
back-solve per iteration** over the current method — the factorization is
shared exactly as today.

### Initial point, step, termination

- **Self-start:** `x = 0, y = 0, s = z = e` (cone identity), `τ = κ = 1`.
  Perfectly centered (`s∘z = e, τκ = 1`); no infeasible-start needed.
- **Step length:** fraction-to-boundary over the cone (`max_step` on
  `s, z`) **and** the rays `τ, κ > 0` — `α` is the min of the cone step and
  the `τ/κ` steps. One shared `α` (HSDE is symmetric in primal/dual).
- **Termination** (Clarabel/SCS style), all relative:
  - **optimal:** primal res `‖Ax−bτ‖/τ`, dual res `‖Aᵀy+Gᵀz+cτ‖/τ`, and gap
    `|cᵀx + bᵀy + hᵀz|/τ` all below `tol` (the `/τ` un-homogenizes);
  - **primal infeasible:** `τ` small, `bᵀy + hᵀz < 0`, `‖Aᵀy+Gᵀz‖` small;
  - **dual infeasible:** `τ` small, `cᵀx < 0`, `‖Ax‖, ‖Gx+s‖` small.
  These are the *same* certificate inequalities `detect_infeasibility`
  already checks; the embedding drives the iterate onto the Farkas/recession
  ray as `τ → 0`, and the HSDE driver **reuses** that verified relative check
  on the homogeneous `(x, y, z)` (rather than retiring it) — so both drivers
  share one certificate path.

## The quadratic objective (P ≠ 0)

With `P`, the embedding is no longer perfectly self-dual; we adopt
Clarabel's QP embedding. Stationarity (1) gains `Px`:

```text
 (1q)  P x + Aᵀy + Gᵀz + c τ = 0
 (4q)  κ = −(cᵀx + bᵀy + hᵀz) − xᵀP x / τ
```

(At `τ>0`, dividing recovers the QP duality-gap condition
`x̂ᵀPx̂ + cᵀx̂ + bᵀŷ + hᵀẑ = 0`.) **Landed (H3).** The Newton linearization
of (4q) shows the `P` coupling enters *only* the τ-row scalar:

- `ρ_τ = κ + cᵀx + bᵀy + hᵀz + xᵀPx/τ`,
- the τ-row gradient becomes `g̃ = (c + (2/τ)Px, b, h)` (used in `g̃ᵀp`,
  `g̃ᵀq`),
- the scalar Schur denominator gains a `−xᵀPx/τ²` term.

The border *column* is unchanged — `(1q)`'s τ-coefficient is still `c`, so
`p = M⁻¹(−c, b, h)` as in the linear case — and `P` already sits in `M`'s
`(x,x)` block and in `ρ_x`. Hence the two M-solves, the cone elimination,
and the step are **identical** to H2; only the τ-row scalar differs, and it
reduces to the linear case at `P = 0`. Validated against the direct driver
and closed-form optima (equality-constrained QP; box/inequality QP; QP with
a second-order cone) — all agree.

## Phased plan

| Phase | Scope | Risk |
|---|---|---|
| H1 | This note: exact embedding, two-solve scheme, termination. | low |
| **H2** | ✅ HSDE driver for **linear** conic (`P=0`): orthant + SOC, reusing `KktStructure`/`Cone`. `solve_conic_hsde` alongside the current solver. Validated optima + both certificates vs the existing solver. | med-high — embedding signs, two-solve combination |
| **H3** | ✅ Quadratic objective: the `(1q)/(4q)` τ-row with the `P` coupling. Validated on the QP suite (closed-form optima + QP-with-SOC) vs the direct driver. | high — τ-row P algebra |
| **H4** | ✅ *(revised)* HSDE promoted to a first-class **selectable** driver (`QpOptions::use_hsde`), routed through `solve_qp_core` and reachable from every public entry point (bound expansion + `z_lb`/`z_ub` split validated). **Not** forced as the universal default: doing so would regress warm starting — `warm_start_reduces_iterations_on_nearby_problem` asserts a *strict* iteration reduction that the direct method's adaptive recentering delivers and an IPM embedding inherently does not. End state is **automatic routing**: symmetric-only cones stay on the direct driver (warm start, factor reuse, differentiable layers); problems with non-symmetric cones (exp/power, H5+) use HSDE. Embedded warm start / factor reuse remain future work, gated on need. | med |
| H5 | **Exponential cone** on HSDE: barrier oracles, non-symmetric scaling, third-order corrector, neighborhood line search. Known-optima (GP, logistic, entropy) + KKT-residual validation. | high |
| H6 | **Power cone** (exp machinery + new barrier). | low after H5 |
| H7 | **PSD cone**: pure-Rust symmetric eig, svec/smat, dense `W⊗ₛW` block; small dense SDPs first, chordal decomposition later. | med-high |
| H8 | Cone-aware differentiable backward (JAX) for each new cone, FD-validated, as separate follow-ups. | med-high |

Validation discipline is unchanged and intrinsic: the IPM reports
`Optimal` only at a verified KKT point; each phase adds known-optima tests
plus randomized KKT-residual checks, and the orthant/SOC results stay
identical to the current solver (the cross-check that guards H2–H4). The
existing direct driver stays in place until H4 flips the default, so there
is no window where the crate regresses.

## Non-symmetric cones on HSDE (H5 — exponential cone)

The exponential and power cones are **not** self-scaled: there is no
Nesterov–Todd point `W` with `W²z = s`, no Jordan product `s∘z`. The
path-following method instead uses the primal barrier `F` directly
(Skajaa–Ye 2015; Dahl–Andersen 2021, the MOSEK exponential-cone
algorithm). `pounce-convex` already has the validated barrier oracles
(`BarrierCone`: `F`, `∇F`, `∇²F`, membership — see `cones/exp.rs`).

### Central path and the scaling block

The central path of the homogeneous model is, at parameter `μ`,
```text
  z = −μ ∇F(s),   τκ = μ,   μ = (sᵀz + τκ)/(ν + 1),
```
with `ν` the total barrier degree (exp cone: 3). `−∇F(s) ∈ int K*` for
`s ∈ int K`, so `z` stays dual-feasible. The Newton step toward the path at
a centered target `σμ` linearizes `z + σμ∇F(s) = 0`:
```text
  dz + σμ H(s) ds = −(z + σμ ∇F(s)),     H = ∇²F(s).
```
Eliminating `ds` (so the cone contributes a `(z,z)` block exactly as the
symmetric path does) gives
```text
  (z,z) block      :  −(1/σμ) H(s)⁻¹          [dense; exp cone is 3×3]
  r_c              :  z + σμ ∇F(s)
  rhs_comp_term    :  (1/σμ) H(s)⁻¹ r_c
  recover_ds       :  ds = −rhs_comp_term − (1/σμ)H(s)⁻¹ dz
```
**Orthant-reduction check (the correctness anchor).** For the orthant,
`F = −Σ log sᵢ`, `H⁻¹ = diag(sᵢ²)`, and on the path `zᵢ = σμ/sᵢ`, so the
block `(1/σμ)sᵢ² = sᵢ/zᵢ = W²` — it reduces *exactly* to the orthant
scaling. The whole derivation collapses to the symmetric one in 1-D, the
same anchor that de-risked the SOC reduced system.

### Why a separate loop (fixed-σ single step, not Mehrotra)

The block carries `1/σμ`, so the Mehrotra **predictor** (`σ = 0`) is
singular for a non-symmetric cone. Skajaa–Ye therefore use a
predictor (tangent to the path) **plus** a distinct centering corrector,
not a single combined `σ→σμ` step. The minimal robust version is a
**fixed-σ single-step path-follower**: each iteration pick `σ ∈ (0,1)`,
assemble the `(z,z)` block `−(1/σμ)H⁻¹`, solve the *same* bordered HSDE
system (two solves + the τ scalar, reused verbatim from H2/H3), then take a
**backtracking** step — there is no closed-form `max_step`, so shrink `α`
until `s+αds ∈ int K`, `z+αdz ∈ int K*` (via `BarrierCone` membership) and
the barrier decreases. More iterations than Mehrotra, but correctness
first; a Mehrotra/RK corrector is a later optimization.

### Implementation steps

1. **Dense `(z,z)` block in `KktStructure`.** Today's assembly handles
   `Diagonal` (orthant) and `DiagRank1` (SOC). Add a `DenseLower` path that
   reserves a `dim×dim` lower triangle at the cone's `(z,z)` position and
   fills it from `−(1/σμ)H⁻¹` each iteration. (This is the "Tier-A dense
   block" the SOC note deferred; the exp cone is only 3×3, so fill is
   trivial.)
2. **A non-symmetric HSDE loop** (`hsde::solve_conic_hsde_nonsym`, or a
   branch) sharing the residuals, the two-solve τ handling, and
   un-homogenizing — but with the fixed-σ step and barrier line search.
   Routed to when the cone product contains a non-symmetric block.
3. **`ExponentialCone` becomes a `Cone`/`ConeKind`** providing the
   `(z,z)`-block (dense `−(1/σμ)H⁻¹`), `r_c`, `recover_ds`, the central-ray
   identity start, `mu`, and a membership-based `max_step`.
4. **Validate** on known optima: an entropy maximization / `log-sum-exp`
   epigraph and a tiny geometric program (posynomial), plus a randomized
   KKT-residual check, all to intrinsic tolerance; the orthant/SOC paths
   stay byte-identical.

Sources for the non-symmetric algorithm:
[Skajaa & Ye, *A homogeneous interior-point algorithm for nonsymmetric
convex conic optimization*, Math. Prog. 2015](https://link.springer.com/article/10.1007/s10107-014-0773-1);
[Dahl & Andersen, *A primal-dual interior-point algorithm for nonsymmetric
exponential-cone optimization*, Math. Prog. 2021](https://link.springer.com/article/10.1007/s10107-021-01631-4).
