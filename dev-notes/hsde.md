# Homogeneous self-dual embedding for the convex IPM — design note

**Status: Phases H2–H4 landed — HSDE solves LP/QP/SOCP and is a
selectable driver (`QpOptions::use_hsde`). H5 (exponential cone) core
landed: the dual-aware scaling, the non-symmetric driver
(`hsde_nonsym::solve_conic_hsde_nonsym`), the third-order corrector, and
public-API routing (`ConeSpec::Exponential` → the driver) solve exp-cone
problems to known optima — see "H5 status" below. Remaining: broader
benchmarks (`pounce-nlp` cross-checks, CBLIB).**
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
| **H7** | ✅ **PSD cone**: pure-Rust symmetric eig, svec/smat, dense `W⊗ₛW` block; small dense SDPs (chordal decomposition later). Landed — see the H7 status note below. | med-high |
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
The scaling block uses the **current** `μ` (the `σ` enters only the target
`r_c`); linearizing `z + dz = −σμ(∇F(s) + H ds)` and eliminating `ds`
(so the cone contributes a `(z,z)` block exactly as the symmetric path
does) gives
```text
  (z,z) block      :  −(1/μ) H(s)⁻¹           [dense; exp cone is 3×3]
  r_c              :  z + σμ ∇F(s)
  rhs_comp_term    :  (1/μ) H(s)⁻¹ r_c
  recover_ds       :  ds = −rhs_comp_term − (1/μ)H(s)⁻¹ dz
```
**Orthant-reduction check (the correctness anchor).** For the orthant,
`F = −Σ log sᵢ`, `H⁻¹ = diag(sᵢ²)`, and on the path `zᵢ = μ/sᵢ`, so the
block `(1/μ)sᵢ² = sᵢ/zᵢ = W²` — it reduces *exactly* to the orthant
scaling, and `r_c = z − σμ/sᵢ` matches the symmetric `(s∘z − σμe)/s`. The
whole derivation collapses to the symmetric one in 1-D, the same anchor
that de-risked the SOC reduced system. (Putting `σμ` in the *block*
instead of `μ` — an early mistake — both mis-scales the step and
reintroduces a `σ=0` singularity; the `μ` form is the correct one.)

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
   stay byte-identical. **Cross-check against NLP solves:** each of these
   problems also has a smooth-NLP form — solve it through `pounce-nlp` and
   require the conic optimum to agree with the NLP optimum (objective and
   primal point) to tolerance. This is the strongest intrinsic check: two
   independent solvers (a conic IPM and a general NLP IPM) landing on the
   same KKT point.

### Prototype findings (what works, what's still needed)

A standalone prototype driver (assembling the dense bordered system and
reusing the two-solve τ handling) confirmed the **math is right**:

- the barrier oracles are exact (FD + the three log-homogeneity identities);
- the `(1/μ)H⁻¹` block and `r_c = z + σμ∇F(s)` give a correct first step —
  on `min z s.t. (0,1,z)∈K_exp` the opening iteration cuts primal and dual
  residuals by ~2× in the right direction.

But it **stalls** after a few iterations: with primal-only Hessian scaling
the **dual** iterate races to `∂K*` (proximity `ψ* → 0`) while `μ` is still
large, and the line search throttles `α → 0`. This persists across all `σ`
and across a central-path-neighborhood line search — it is the known
weakness of naive primal scaling, *not* a sign/algebra bug (the symmetric
reduction holds and the first step is correct).

**What's needed (resolved — item #1 in hand).** The stall is the known
weakness of primal-only Hessian scaling. The fix is a **dual-aware
primal–dual scaling** built from *both* the primal and dual cone iterates —
the Tunçel scaling, specialized to 3-D and computed by a BFGS update, exactly
as in MOSEK's exponential-cone solver. The construction is transcribed below
from **Dahl & Andersen (2021)** — the local copy is `~/Desktop/hsde-reference.pdf`
(this reference was *not* network-blocked after all; it was on disk).
Equation tags `(DA n)` below refer to that paper.

### The dual-aware scaling (item #1) — Tunçel/BFGS primal–dual scaling [Dahl & Andersen 2021]

This **replaces** the primal-only `−(1/μ)H(s)⁻¹` block of "Central path and
the scaling block" above, and supersedes the fixed-σ path-follower of "Why a
separate loop" (Dahl–Andersen fold predictor + corrector + centering into one
combined direction). Implements `[Dahl & Andersen 2021]`, which itself
specializes the primal–dual scalings of `[Tunçel 2001]` / `[Myklebust &
Tunçel 2014]` to the exponential cone.

**Notation / convention alignment (read this first).** Dahl–Andersen put the
*primal* cone variable in `x` and the *dual* in `s`; pounce's HSDE uses
`s ∈ K` (primal slack) and `z ∈ K*` (dual). Map **DA `x` → pounce `s`**,
**DA `s` → pounce `z`**. Their exp-cone ordering also differs:
`K_exp = cl{x₁ ≥ x₂·e^{x₃/x₂}}`, barrier `F = −log(x₂log(x₁/x₂) − x₃) − log x₁
− log x₂` (DA 2) — a coordinate **permutation** of pounce's `(x,y,z)` with
`ψ = y·log(z/y) − x` (`cones/exp.rs`): pounce `(x,y,z) = DA (x₃, x₂, x₁)`. Port
the appendix derivatives through that permutation, **or** (cheaper, less
error-prone) re-derive `F'''` directly in pounce's order and FD-check it
alongside the existing `F, ∇F, ∇²F` oracles.

In DA's convention (`x` = primal cone var, `s` = dual cone var), for an iterate
off the central path:

**Shadow iterates and scalars** (DA 7):
```
  x̃ := −F'_*(s)      (gradient of the conjugate barrier at the dual point)
  s̃ := −F'(x)        (gradient of the primal barrier at the primal point)
  μ  := ⟨x,s⟩/ϑ,     μ̃ := ⟨x̃,s̃⟩/ϑ          (μ·μ̃ ≥ 1, equality only on path)
```
`s̃ = −F'(x)` is free (reuse `∇F`). `x̃ = −F'_*(s)` has no closed form for the
exp cone: it is `x̃ = argminₓ{−⟨s,x⟩ − F(x)}`, i.e. solve `F'(x̃) = −s` by a
damped Newton iteration (DA p. 347); then `F''_*(s) = [F''(x̃)]⁻¹`.
`Y^T S ≻ 0` (with `S, Y` below) ⇔ the iterate is off the path.

**Secant equations — definition of a primal–dual scaling** (DA 8, DA 29). A
nonsingular `W` with the *double* secant property
```
  W x = W^{-T} s,     W x̃ = W^{-T} s̃        ⇔   (WᵀW)⁻¹ ∈ T₁(x,s),
```
where Tunçel's set is `T₁(x,s) = {T≻0 : T²s = x, T²F''(x) = F'_*(s)}` (DA 20).
On the central path this collapses to the self-scaled `WᵀW = μF''(x)` (DA 21);
**off** the path the dual data `s, s̃` genuinely enter — that is exactly the
"dual awareness" the primal-only block lacked.

**3-D closed form (this is what to implement).** In 3-D every such scaling is
(DA §5, end):
```
  WᵀW          = Y(YᵀS)⁻¹Yᵀ + t·z zᵀ
  W⁻¹W⁻ᵀ       = S(YᵀS)⁻¹Sᵀ + t⁻¹·r rᵀ          S := [x  x̃],  Y := [s  s̃]
```
with `Sᵀz = 0, Yᵀr = 0, ⟨r,z⟩ = 1, ‖z‖ = 1` — computed by **cross products**:
```
  z = (x × x̃) / ‖x × x̃‖ ,        r = (s × s̃) / ⟨s × s̃, z⟩ .
```
The entire non-symmetry is carried by the single scalar `t > 0`.

**Choosing `t` — the BFGS value** (DA 32):
```
  t = μ·‖ F''(x) − s̃s̃ᵀ/ϑ − (F''(x)x̃ − μ̃s̃)(F''(x)x̃ − μ̃s̃)ᵀ / (⟨x̃,F''(x)x̃⟩ − ϑμ̃²) ‖_F
```
— the Frobenius norm of the rank-3 BFGS update `H_BFGS − μF''(x)` (DA 30). DA
also give an "optimally bounded" `t` via bisection (DA 31; conjectured bound
`ξ* ≈ 1.253` for the exp cone), but report **no practical difference** vs the
BFGS `t` (largest observed `ξ ≤ 1.72`). **Use the BFGS `t` (DA 32)** — closed
form, no bisection.

**Factored scalings used in the loop** (DA §6) — the columns of `Wᵀ` / `W⁻¹`:
```
  Wᵀ   columns:  x/√⟨x,s⟩ ,   δ_s/√⟨δ_x,δ_s⟩ ,   √t · z
  W⁻¹  columns:  s/√⟨x,s⟩ ,   δ_x/√⟨δ_x,δ_s⟩ ,   r/√t
  δ_x := x − μ x̃ ,    δ_s := s − μ s̃ .
```
This dense 3×3 `WᵀW` is the `DenseLower` cone block of implementation step #1
— now `WᵀW` rather than `−(1/σμ)H⁻¹`. **Reconcile placement and signs with
pounce's elimination** (pounce keeps `Δz`, eliminates `Δs`; DA keep `Δx`,
eliminate `Δs` in *their* convention) using the **orthant-reduction anchor**:
on the path `WᵀW → μF''(s)`, and the block must collapse to the existing
`−W²` orthant/SOC block — pin the sign there, exactly as the `−(1/μ)H⁻¹`
derivation was pinned.

**The corrector (DA's headline contribution)** (DA 16) — a Mehrotra-like
*third-order* corrector for the non-symmetric case:
```
  η := −½ F'''(x)[ Δxᵃ , (F''(x))⁻¹ Δsᵃ ]
```
where `(Δxᵃ, Δsᵃ)` is the affine/predictor direction (DA 11). Evaluate via
(DA 34): `η = −½ F'''(x)[u, v]`, `u = Δxᵃ`, `v` solving `F''(x)v = Δsᵃ` (use
the factored `F'' = RRᵀ`, DA App. A.2, for stability). The exp-cone third
derivative `F'''(x)[u]` is DA App. A.3 (DA 33). DA Table 1 / Fig 2: this
corrector cuts iteration counts to roughly the symmetric-cone level — it is
the reason their method is competitive and the reason to prefer it over the
Skajaa–Ye Runge–Kutta corrector (which needs extra KKT factorizations).

**Centering and the combined step** (DA §6):
```
  α_a := step-to-boundary of the affine direction   (bisection on membership)
  γ   := (1 − α_a)·min{(1 − α_a)², 1/4}              (centering parameter)
  combined (DA 18):  G(Δz) = −(1 − γ)·G(z),
                     W Δx + W^{-T} Δs = −v + γμ ṽ − W^{-T} η,
                     v = Wx = W^{-T}s ,   ṽ = W x̃ = W^{-T} s̃ .
  update:  z ← z + α Δz,  largest α keeping the iterate in N(β),  β = 1e-6.
```
`N(β)` is the one-sided ∞-norm neighborhood `ϑ·⟨F'(xᵢ), F'_*(sᵢ)⟩⁻¹ ≥ βμ`
(DA §3). The reduced bordered linear system is DA §7.2: the cone block is
`WᵀW`, solved through an `LDLᵀ` of `[ −WᵀW  Aᵀ ; A  0 ]` — structurally the
**same** bordered two-solve already in `hsde.rs`, with the dense `WᵀW` in
place of the symmetric `W²`.

**Starting point** (DA §6): `x = s = −F'(x)` (solve `x + F'(x) = 0`, the min
of `½‖x‖² + F(x)`), `y = 0`, `τ = κ = 1`. For the exp cone DA give the constant
`x⁰ = s⁰ ≈ (1.290928, 0.805102, −0.827838)` (their ordering — permute to
pounce's). Then `z⁰ ∈ N(1)`, perfectly centered.

**Termination** (DA §7.3): relative primal/dual feasibility `ρ_p, ρ_d` and gap
`ρ_g`, plus infeasibility metrics `ρ_pi, ρ_di` and ill-posedness `ρ_ip` —
these mirror the relative optimal/infeasible checks already in "Initial point,
step, termination", so the existing certificate path is reused.

### H5 status — what landed

Implemented and validated (all to intrinsic tolerance, `cargo test -p
pounce-convex`):

- **Conjugate-barrier gradient** `x̃ = −F'_*(z)` (`cones/exp.rs`,
  `ExponentialCone::conjugate_grad`) — damped self-concordant Newton,
  validated by exact round-trip (`p → −∇F(p) → recover p`) and the residual
  equation `∇F(x̃) = −z`.
- **Dual-aware scaling** `M = WᵀW` (`ExponentialCone::scaling` →
  `ExpScaling`) — the closed form `Y(YᵀS)⁻¹Yᵀ + t·z_cp z_cpᵀ` with the BFGS
  `t` (DA 32). The driver needs only `M` (not `W`/`W⁻¹`): the secants
  pre-multiplied by `Wᵀ` are the exact, `W`-free identities `M·s = z`,
  `M·x̃ = s̃`, which the tests confirm; `M` is SPD and reduces to `μ∇²F` near
  the path.
- **Non-symmetric driver** (`hsde_nonsym::solve_conic_hsde_nonsym`) — the
  same homogeneous embedding + two-solve τ scheme as `hsde.rs`, with the
  cone `(z,z)` block `−M⁻¹` (dense 3×3, genuine off-diagonals reserved in a
  local `NsKkt`), `comp_term = −M⁻¹·rc`, `rc = −z + σμ·s̃`, and a
  backtracking step on cone membership. **For the orthant it reduces exactly
  to the symmetric Mehrotra step** (the correctness anchor). Validated on
  `min z : (1,1,z)∈K_exp` → `z = e`; `log-sum-exp` (2 exp + 1 orthant) →
  `log 2`; and a geometric program `min x + 1/x` → `2`.
- **Third-order corrector** (DA 16/34) — `ExponentialCone::third_dir_apply`
  computes `F'''(s)[u, v]` as a directional derivative of the Hessian
  (validated against the exact identity `F'''(s)[s,v] = −2∇²F·v`); the driver
  forms `η = −½ F'''(s)[ds_aff, ∇²F⁻¹ dz_aff]` and folds `−η` into `rc`. For
  the orthant `η_i = ds_aff_i dz_aff_i/s_i` — exactly the Mehrotra
  second-order term, so the orthant corrector *is* standard Mehrotra. Two
  safeguards keep it robust: a step-collapse fallback to pure centering, and
  gating the corrector off within `~1e3·tol` of convergence (its
  finite-difference perturbation otherwise stalls the endgame). The FD step is
  scaled `∝ 1/‖u‖` so the third derivative stays accurate for a tiny affine
  step.
- **Public-API routing** — `ConeSpec::Exponential`; `solve_socp_ipm` detects
  any exp spec and routes to `hsde_nonsym` (`solve_nonsym`), with bound
  expansion into a trailing orthant block and bound-dual splitting exactly as
  the symmetric path. SOC mixed with exp is not yet supported (returns
  `NumericalFailure`). End-to-end routing test
  (`routes_exponential_through_public_entry`) passes.
- **Python access** — `pounce.qp.solve_socp(..., cones=[("exp", 3), ...])`
  reaches the driver via `pounce-py`'s cone parser (`"exp"`/`"exponential"`,
  fixed dimension 3 validated; the SOC+exp mix raises a clear `ValueError`
  up front rather than returning an opaque status). Verified from Python on
  the GP (`→ 2`) and log-sum-exp (`→ log 2`) problems
  (`python/tests/test_socp.py`).
- **QP solve report** — the convex/QP CLI path (`run_convex_qp`) now emits the
  `pounce.solve-report/v1` JSON report (`--json-output`) like the NLP path,
  with real final KKT residuals via `QpSolution::kkt_residuals` →
  `QpResiduals` (in `pounce-convex`, tested with active bounds and a binding
  inequality), so the benchmark harness can compare QP/exp-cone solves to NLP
  solves uniformly. At `--json-detail full` the report also carries the
  **per-iteration convergence trace** (`iterations` array, same `IterRecord`
  schema as the NLP path): an opt-in `QpOptions::collect_iterates` makes the
  convex IPM record `obj / inf_pr / inf_du / μ / α` per iteration into
  `QpSolution::iterates` (off by default — no overhead), which `run_convex_qp`
  maps into the report.
- **Bug fixed:** `in_dual_cone` had `ψ* = v − u·log(−u/w)` instead of the
  correct `v − u + u·log(−u/w)` (it mislabeled dual-infeasible points as
  interior); cross-checked against DA p. 346 and regression-tested.

- **NLP cross-checks** (`crates/pounce-cli/tests/exp_cone_vs_nlp.rs`) — the
  geometric program (`= 2`), log-sum-exp (`= log 2`), and entropy
  maximization (`= −log n`) are each solved *twice*: as an exp-cone conic
  program (this driver) and as a smooth NLP (the independent IPOPT-style
  filter-IPM in `pounce-algorithm`). The two optima agree to ~1e-7 — strong
  evidence of correctness, since the conic and NLP paths share no code.
- **Endgame acceptance:** near the cone boundary `ψ → 0` makes `∇²F` blow up,
  so the scaling/factorization can break down a hair short of `tol`. When that
  happens with KKT residuals already within `~1e3·tol`, the driver accepts the
  current iterate (IPOPT's "solved to acceptable level") instead of reporting a
  spurious `NumericalFailure`.

**H6 (power cone) — landed.** The non-symmetric machinery was generalized
(`cones/nonsym.rs`): `conjugate_grad`, the dual-aware scaling
(`NonsymScaling`), and `third_dir_apply` are now generic over any 3-D
`BarrierCone` (which gained an `interior_reference` returning a point in
`K ∩ K*`). The exp and power cones supply only their barrier oracles. The
`PowerCone { alpha }` (`cones/power.rs`) implements `K_α = {|x| ≤ y^α z^{1−α}}`
with the degree-3 barrier `−log(y^{2α}z^{2−2α} − x²) − (1−α)log y − α log z`
(FD- and identity-validated). The driver dispatches over a `NonsymCone`
enum (Exp/Power) that implements `BarrierCone`, so the loop, corrector, and
step length are cone-agnostic; the generic machinery is validated on both
cones via the secants `M·s=z`, `M·x̃=s̃`. Wired through `ConeSpec::Power(α)` →
`solve_socp_ipm` → `solve_nonsym`, and Python `solve_socp(cones=[("pow", α)])`
(exponent validated to `(0,1)`). Known-optimum tests
(`max x s.t. (x, 2, 0.5) ∈ K_α` → `2^α 0.5^{1−α}`) pass for several α in Rust
and Python.

**SOC mixing — landed.** The non-symmetric driver now also accepts
second-order-cone blocks (`NsBlock::SecondOrder`): they are self-scaled, so
they reuse `SecondOrderCone`'s NT machinery — a dense `W² = diag(d)+uuᵀ`
block, the Jordan `comp_residual`/corrector, the arrow `rhs_comp_term`, and
the closed-form `max_step` — alongside the dual-aware exp/power blocks in one
KKT. A SOC may be freely mixed with an exp/power cone (`solve_socp_ipm` routes
any exp/power/SOC mix to `solve_nonsym`; Python `solve_socp` likewise). Tested:
SOC-only and `min t + z s.t. (t,3,4)∈SOC ∧ (1,1,z)∈K_exp` → `t=5, z=e` in Rust
and Python.

**Warm-start — landed (primal hook).** `solve_conic_hsde_nonsym_warm` seeds
the primal `x` from a previous (nearby) solution while keeping the cones
centered, lowering the initial primal residual. Honest scope: the HSDE
embedding's iteration count is start-dependent and not guaranteed to drop, so
this is a primal hook, **not** a promised speedup — the property tested is
*start-independence* (warm from the optimum, a bad point, or an ignored
mismatched vector all reach the same optimum). Higher-level routing
(`solve_socp_ipm_warm` for the non-symmetric path, Python) and factor reuse
remain optional follow-ups, gated on a demonstrated need.

### H7 status — PSD cone landed (small dense SDPs)

The semidefinite cone is **self-scaled**, so unlike exp/power it lives on the
*symmetric* driver (`hsde.rs` / `solve_socp_ipm`), not the non-symmetric one.

- **Oracles** (`cones/psd.rs`) — `svec`/`smat` (the `√2`-off-diagonal isometry
  so `⟨X,Y⟩_F = svec·svec`), the `−log det` barrier + gradient `−X⁻¹` +
  Hessian action, membership / fraction-to-boundary via eigenvalues, and the
  Nesterov–Todd scaling `W = S^{1/2}(S^{1/2}ZS^{1/2})^{-1/2}S^{1/2}`, validated
  against `W Z W = S`. Eigendecompositions reuse
  `pounce_linalg::symmetric_eigen`.
- **`Cone` impl** — the matrix-Jordan machinery: `kkt_block` → the dense
  symmetric Kronecker `H = W ⊗ₛ W` (`ConeBlock::DenseLower`), validated to
  satisfy `H·svec(z) = svec(s)`; `comp_residual` uses the Jordan product
  `(SZ+ZS)/2`; `rhs_comp_term` = `Arw(z)⁻¹ r` via a Lyapunov solve
  `ZD+DZ = 2·smat(r)`; `recover_ds = −Arw(z)⁻¹ r − H·dz`, all cross-checked.
- **Driver integration** — `ConeSpec::Psd(n)` / `ConeKind::Psd`; `KktStructure`
  gained a fully-dense `(z,z)` block path (a third `block_shapes` class
  alongside the orthant's diagonal and the SOC's diag+rank-1 aux-var trick).
  Validated end to end on `max λ s.t. M − λI ⪰ 0 ⇒ λ_min(M)` for a diagonal
  and a non-diagonal `M` (the latter exercising the off-diagonal scaling).

- **Python** — exposed via `pounce.qp.solve_socp(cones=[("psd", n)])` (the
  value is the matrix size `n`; the slack block is `svec(X)`). The
  PSD-with-exp/power mix raises a clear `ValueError`.
- **Sparsity (block-diagonal)** — `decompose_psd` splits a block-diagonal
  `Psd(n)` cone into independent PSD cones over the connected components of
  its sparsity graph (one dense `O(m²)` KKT block → several small ones,
  exploited by the sparse factorization). Solution-equivalent: the primal /
  objective are unchanged and the dropped (structurally-zero) cross rows have
  empty `G` rows, so their dual is `0`.
- **Sparsity (chordal range-space)** — `chordal_decompose` (built on
  `cones/chordal.rs`: chordal extension + maximal cliques) handles the
  *general* connected-sparse case via Agler's theorem: `s ⪰ 0` ⟺
  `s = Σ_k Tᵀ S_k T`, introducing clique blocks `S_k ⪰ 0` and one consistency
  equality per clique-covered entry. Runs after the block-diagonal split;
  the dual is reconstructed through both layers (PSD entry duals from the
  consistency-equality multipliers). Equivalence-tested against the dense
  solve on a path-pattern SDP (`x`, objective).
- **CBF SDP input** — the CBF reader parses affine PSD constraints
  (`PSDCON` + `HCOORD`/`DCOORD`): `D_c + Σ_k x_k H_{c,k} ⪰ 0` maps directly
  onto `s = svec(D) − Σ x_k svec(H_k) ∈ Psd` (√2-scaled). Validated on a
  synthetic SDP (`max λ s.t. M − λI ⪰ 0`).

Remaining for PSD: primal `PSDVAR` matrix variables in the CBF reader (the
`OBJFCOORD`/`FCOORD` form) — affine `PSDCON` is done; and PSD cannot be mixed
with exp/power cones in one problem (different drivers; the mix fails
cleanly). The chordal elimination uses the natural variable order — a
fill-reducing ordering (AMD) would shrink the cliques further on large
instances.

Remaining (overall): only — if a need emerges — embedded factor-reuse for the
non-symmetric path. The CBLIB exp- and power-cone tiers, the cross-check,
and the benchmarks-harness integration all landed (see below).

### CBLIB benchmark tier — landed (exp + power cones)

**Status: landed.** The reader, the CBF→pounce mapping, the independent NLP
cross-check, and the benchmarks-harness integration are implemented and
green for both the exponential-cone GPs and the 3-D power cone.

- **CBF reader** (`pounce_cli::cbf`) — parses the Conic Benchmark Format
  (`VER`/`OBJSENSE`/`POWCONES`/`VAR`/`CON`/`OBJACOORD`/`OBJBCOORD`/`ACOORD`/`BCOORD`)
  with the cone kinds `F`/`L=`/`L+`/`L-`/`EXP`/`Q` and the 3-D power cone
  (`@k:POW` resolving its exponent `α = α₀/(α₀+α₁)` against the `POWCONES`
  table). Unsupported kinds (PSD `DCOORD`, rotated SOC `QR`, dual power
  cones) are rejected with a clear error rather than mis-parsed. Unit-tested
  on the section grammar, the exp-dim and cone-sum checks, the `POWCONES`
  α-resolution + permutation, and unsupported-cone / bad-`@k` rejection.
- **`CbfModel::to_conic`** — maps an instance to a pounce conic program
  (`QpProblem` + `Vec<ConeSpec>`): VAR cones → slack `s = −Gx ∈ K`, CON
  cones → `s = Ax+b ∈ K`, `L=` → equality `Ax = −b`. The non-symmetric
  triples are permuted into pounce cone order: exp **reversed** (CBF
  bound-first `(a,b,c)` → pounce bound-third `(c,b,a)`), power **rotated**
  (CBF `x₀^β₀ x₁^β₁ ≥ |x₂|` → pounce `(x,y,z) = (x₂,x₀,x₁)`, `α = β₀`).
- **Conic solve on real instances** (`tests/cblib_cbf.rs`) — three vendored
  CBLIB GPs (`demb761`, `beck751`, `fang88`) plus a hand-authored synthetic
  power-cone instance (`pow3_synthetic.cbf` — the real `2013_fir*` are
  ~120 MB), each under `crates/pounce-cli/tests/data/cblib/`, parse, map,
  and reach a verified `Optimal`. The power instance hits its closed-form
  optimum `x₂ = 2^½·½^½ = 1`.
- **Independent NLP cross-check** (`tests/cblib_vs_nlp.rs`) — exactly the
  `exp_cone_vs_nlp` strategy: each instance is also built as a smooth NLP
  (exp triple → `u₀ − u₁·exp(u₂/u₁) ≥ 0`; power cone → the epigraph
  `u₀^α u₁^{1−α} ∓ x_bnd ≥ 0`; both with exact gradient + Hessian, `L=`/`L-`
  rows linear) and solved by the filter-IPM, **cold-started independently**
  of the conic solution. The two solvers — sharing no code — agree to ~1e-8
  relative: `demb761 → 22.31086`, `beck751 → 7.50095`, `fang88 → −10.38004`,
  `pow3 → 1.0`. (CBLIB ships no reference solution files, so the cross-check
  *is* the reference.)
- **Benchmarks-harness integration** — the `pounce_cblib` binary solves a
  `.cbf` and emits a `pounce.solve-report/v1` JSON (status / iters / time /
  objective, per-iteration trace at `--json-detail full`; input descriptor
  kind `cbf-file`). `benchmarks/cblib/run_cblib.py` runs it over the
  vendored instances (offline) — or a `--dir` of a local CBLIB checkout —
  and projects each report into the composite suite schema at
  `cblib/pounce.json`.

Extensions left for when needed: the large power-cone instances
(`2013_fir*`, ~120 MB — fetch into a `--dir` rather than vendoring),
constraint-side exp/SOC cones in the NLP cross-check form (the conic
mapping already handles them), and the rotated SOC (`QR`) cone kind.

#### Original plan (kept as the implementation record)

The literal benchmark instances from the source papers live in CBLIB
(`https://cblib.zib.de/download/all/<name>.cbf.gz`, reachable) and are the
gold-standard broad validation:

- **Geometric programs** (small, exp cones, pure-continuous): `demb761/762/763`,
  `beck751/752/753`, `fang88`, `jha88`, `car`, `rijc786/787`, `mra01/02`.
- **Logistic regression** (pure-continuous exp): `LogExpCR-n{20,100,500}-m{400…2000}`.
- **Power cone**: `2013_fir*`.
- (`batch*`/`rsyn*` are MINLPs — solve the *continuous relaxation* if used.)

**CBF → pounce conversion** (verified against a full dump of `demb761`):
the `.cbf` has `VAR` (cones over variables) and `CON` (cones over `Ax+b`),
plus sparse `OBJACOORD` (obj `c`), `OBJBCOORD` (obj constant `c₀`), `ACOORD`
(`A`), `BCOORD` (`b`).
- VAR `EXP 3` → variable triple in `K_exp`; **CBF order `(a,b,c)` permutes to
  pounce `(c,b,a)`** (CBF `x1 ≥ x2 e^{x3/x2}` vs pounce `z ≥ y e^{x/y}`).
  Realize as `s = x_triple ∈ K` via `G = −I`, `h = 0`.
- VAR `POW` → `K_α` (read the exponent); VAR `Q`/`QR` → SOC; `F` → free.
- CON `L=` → equality `Ax = −b`; `L-` → `Ax ≤ −b`; `L+` → `Ax ≥ −b`
  (nonneg slack `s = −(Ax+b)`); CON cone blocks (EXP/POW/Q) → cone rows.

**Validation strategy (no published reference objectives — they 404):** use
the same cross-check as `exp_cone_vs_nlp` — parse each `.cbf` into *both* a
conic program (this driver) and a smooth NLP (`pounce-nlp`, with the exp/pow
epigraph constraints and their analytic Jacobians) and assert the two
independent solvers agree on the objective. Report status / iters / time /
KKT residuals per instance (feeding the JSON solve report into the existing
`benchmarks/` harness). Build the CBF reader as its own carefully-tested unit
first (round-trip on `demb761`) before wiring the harness.

## Sources (local copies — read and transcribed)

- **Skajaa, A. & Ye, Y. (2015).** *A homogeneous interior-point algorithm for
  nonsymmetric convex conic optimization.* Mathematical Programming Ser. A
  **150**(2), 391–422. DOI [10.1007/s10107-014-0773-1](https://doi.org/10.1007/s10107-014-0773-1).
  Local copy: `~/Desktop/hsde-2.pdf`. Provides the homogeneous model and the
  primal-only Hessian scaling with a separate centering corrector — the `μH`
  scaling the prototype used (and the Runge–Kutta corrector DA improve on).
- **Dahl, J. & Andersen, E. D. (2021).** *A primal-dual interior-point
  algorithm for nonsymmetric exponential-cone optimization.* Mathematical
  Programming Ser. A **194**(1–2), 341–370. DOI
  [10.1007/s10107-021-01631-4](https://doi.org/10.1007/s10107-021-01631-4).
  Local copy: `~/Desktop/hsde-reference.pdf`. **Source of item #1**: the
  Tunçel/BFGS dual-aware primal–dual scaling (this is MOSEK's exp-cone
  algorithm), the third-order corrector, and the exp-cone barrier derivatives
  (Appendix A) — the `(DA n)` equations cited above.
- Underlying scaling theory: **Tunçel, L. (2001)**, *Generalization of
  primal–dual interior-point methods to convex optimization problems in conic
  form*, Found. Comput. Math. **1**(3), 229–254; **Myklebust, T. & Tunçel, L.
  (2014)**, *Interior-point algorithms for convex optimization based on
  primal–dual metrics*, arXiv:1411.2129 — the secant / multiple-secant BFGS
  scalings DA build on.
