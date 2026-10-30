# Native second-order cones in the NLP interior-point method

Workstream C of [`calipso-contact-plan.md`](calipso-contact-plan.md).

## Why

Measured in `benchmarks/calipso/` (workstream A): written with `t − ‖x‖ ≥ 0`
cones, the drifting car is physical on no route for either POUNCE or Ipopt;
written with squared cones `t² − ‖x‖² ≥ 0`, the best route is physical on
5/5 at ~1759 iterations; CALIPSO, which treats the cones natively, solves it
in 130. The quadruped: 1718 against 168. Both smooth lowerings are
degenerate at the cone tip — `t − ‖x‖` is not differentiable there and the
gradient of `t² − ‖x‖²` vanishes there — and a contact with no friction
force sits exactly at the tip. No reformulation removes that: the cone's
boundary is not smooth at its tip. A primal-dual cone barrier never
evaluates the boundary, so it does not see the kink.

## The formulation

A cone is declared over constraint rows: `(g_{r0}(x), g_{r1}(x), …) ∈ K`,
`K = { u : u₀ ≥ ‖u_{1..}‖ }`. The rows are free (`cl = −∞, cu = +∞`); the
cone is their only restriction. In the IPM's own split they are `d` rows,
`d(x) − s = 0`, and each cone's block of slacks carries the cone:
`s_K ∈ K`.

**The cone dual is the row multiplier.** For a lower-bounded slack Ipopt
keeps `v_L` beside `y_d` and the `s`-stationarity row `−y_d − v_L = 0` ties
them. A cone slack has no bounds, so `s`-stationarity reads `−y_d − z = 0`
for its cone dual `z`, and `z := −y_d` exactly. Nothing is added to the
iterate vector: the cone dual lives in `y_d`'s cone rows, the
`s`-stationarity row of those entries is satisfied by construction, and the
complementarity `s ∘ z = μ e` (Jordan product, `e = (1, 0, …, 0)`) takes
its place in the Newton system.

**Why not a bound slot.** The alternative — give each cone row an internal
lower bound `0` so the cone dual lives in `v_L` beside every other slack
dual — reuses more machinery, and was started and backed out. About 370
sites read `d_L`, `v_L` and the slack-bound quantities, and each one a cone
change misses would read a cone entry as `s_i ≥ 0`: a tail entry is negative
as often as not, so a missed site is a wrong answer. The constraint-violation
measures (`unscaled_nlp_constraint_violation_max`,
`relative_d_infeasibility_max`) were the first two found. With the rows free,
a missed site sees a free row, which is safe: the slack stays strictly inside
the cone and `d − s` is measured everywhere. So the cone is added where it
must act, and is invisible everywhere else.

**The Newton row.** Eliminating `v` from Ipopt's system leaves the `s` row
`Σ ds − dy_d = y_d + μ/s` (`Σ = V/S`). The cone's row is the same with
`z = −y_d`: `H ds − dy_d = y_d + μ s⁻¹`, `H` the NT block — an ordinary
symmetric row with a dense `(s, s)` block, so iterative refinement and any
other right-hand side have their slot.

**Conventions** (pinned by tests in `pounce_algorithm::soc_slack`):

| quantity | value |
|---|---|
| central path | `s ∘ z = μ e`, so `⟨s, z⟩ = μ` per cone |
| barrier term | `B(s) = −(μ/2)·log det s`, `det s = s₀² − ‖s_{1..}‖²` |
| barrier gradient | `∇B = −μ s⁻¹`, `s⁻¹ = J s / det s`, `J = diag(1, −1, …)` |
| scaling block (the `Σ` analogue) | Nesterov–Todd `H` with `H s = z`, symmetric positive definite |
| complementarity count | 1 per cone in the average-complementarity `μ` measure |

**Newton system.** In the augmented system the `(s, s)` block is diagonal
`Σ_s = V/S` for bounded slacks. For a cone block it is the dense NT block
`H` (dimension of the cone), and the `s` right-hand side for those rows is
`−(μ s⁻¹ − z)` in place of `−(μ/s − v)`. Every other block is unchanged.
After the solve, `dz = −dy_d` on cone rows; no recovery step is needed.

**Step to the boundary.** The primal step is capped so `s + α ds` stays in
`int K` (fraction `τ`). The cone rows of `y_d` — the cone duals — step with
the **dual** step size, capped so `z + α dz` stays in `int K`, as a bound
multiplier does; the other rows of `y_d` keep `alpha_y`. Tying the cone
duals to the primal step was the first version and failed: on Rosenbrock in
an inactive cone the dual direction, driven by the objective gradient
through `x`-stationarity, left the cone within two iterations and the dual
cap then held `alpha_pr` at `1e-3 … 1e-16` until `Error_In_Step_Computation`.
Both caps are the cone-boundary root (`pounce-convex`'s
`SecondOrderCone::max_step` computes the same quantity).

## What must change, and what must not

With no cones declared every new path is skipped, so a model without cones
runs the code it runs today. The fixture sweep must diff empty.

| place | change |
|---|---|
| application / `OrigIpoptNlp` | accept cone row blocks (user `g` indices); map them to `d` indices; refuse a cone row that is not free, a row in two cones, a cone of dimension < 2 (✅ C1/C2a) |
| NLP scaling | every row of one cone gets one factor (a per-row factor changes the cone) (✅ C2a) |
| presolve, any row-renumbering wrapper | declined when cones are declared, out loud |
| `IpoptCq` | barrier objective and its `s`-gradient; complementarity, average complementarity, centrality, barrier error; the NT blocks |
| search direction | the `s` right-hand side on cone rows gains `μ s⁻¹` (so it reads `y_d + μ s⁻¹`) |
| multipliers | `y_d` on cone rows must stay in `−int K`: initialised there, capped by the cone fraction-to-boundary, re-projected after any least-squares recompute (or that recompute refused) |
| augmented system | dense `(s, s)` blocks at cone positions (structure fixed at first assembly) |
| line search | cone fraction-to-boundary for `s` and for `y_d`'s cone rows |
| initialization | `s` and `z` pushed into `int K` (`s₀ ≥ ‖s_{1..}‖ + margin`) |
| μ update / convergence | the complementarity error includes `‖s ∘ z − μ e‖` per cone |
| restoration | first version: a problem with cones that enters restoration reports it and fails rather than running a restoration that ignores the cones; cone-aware restoration is a later increment |
| sensitivity, SQP arm, convex routing, `--minima` | refuse cones (first version) |

## Increments

1. ✅ **C1 — math and declaration.** `soc_slack` module: `det`, Jordan product,
   inverse, NT block, barrier value/gradient, max step, interior push,
   with tests that pin every convention above. The declaration surface on
   the application and its validation. No change to any solve.
2. ✅ **C2 — the solve.** CQ, augmented system, line search, initialization,
   convergence, behind "cones declared". Unit tests on small SOCPs with
   known optima; restoration refuses.
   *Done* (`crates/pounce-algorithm/tests/soc_native.rs`): a disk whose
   optimum has both tail entries negative (5 iterations), a cone active at
   its tip (5), Rosenbrock in an inactive cone (23) and an active one (13),
   all `Solve_Succeeded` at the closed-form answer. Presolve declined; the
   barrier strategy forced monotone; `mu_strategy=adaptive`, Mehrotra,
   `alpha_for_y≠primal`, `recalc_y`, warm start and the SQP arm refused
   with the reason; restoration not installed (a solve that needs it fails
   honestly). With no cones every path is the old one: 645 crate unit
   tests unchanged.
3. **C3 — measure correctness.** CBLIB SOCPs through the NLP arm against
   `pounce-convex` (global optima): objectives match to 1e-6 on ≥ 95%.
   Fixture sweep empty.
   *Measured* (`pounce_cblib_nlp`, one build, default options). 611 CBLIB
   instances downloaded (portfolio families sampled 30 each, every instance
   of the smaller SOC families, < 1 MB compressed); 192 skipped for rotated
   cones (`QR`, which the `.cbf` reader does not take), 234 with no
   second-order cone, 24 with other cones, 2 timeouts (`sched_*_scaled`).
   Of the 153 the convex driver solves: **121 match to 1e-6 (79%)** — below
   the 95% bar. By family: `classical`, `robust`, `shortfall` 30/30 each,
   `flay` 10/10, `estein` 8/9, `tls` 5/6, `nb` 1/3, `nql` 2/3, `qssp` 1/2,
   `turbine` 2/7, `slay` 2/11, `clay` 0/12. The 32 misses:
   - **14 `Restoration_Failed`** (`slay`, `clay` `h` variants): restoration
     is not installed with cones. The largest single gap.
   - **8 succeed *below* the global optimum** (`clay` `m` variants, ~1e-5):
     `bound_relax_factor` widening big-M orthant rows — not the cones. With
     it at 0, `clay0203m` matches (4.5e-8 vs 2.5e-10).
   - **6 `Error_In_Step_Computation`**, **3 above the optimum**, 1 iteration
     limit.
   Fixed on the way: the cones' fraction-to-boundary is capped at 0.99
   (Ipopt's `τ = 1 − μ` stepped `flay02m`'s active dual outside the cone by
   rounding; 67 of the first 98 compared instances ended
   `Error_In_Step_Computation` before the cap). Pinned by
   `crates/pounce-cli/tests/soc_nlp_vs_convex.rs`. Fixture sweep, both legs,
   252 lines: diff empty.
   *Tried and not kept:* starting the cone dual on the central path
   (`z = μ₀ s⁻¹`) instead of `e` — `clay0203m` 703 → 375 iterations, but
   `slay05m` went from a slow match (1791 iterations) to the iteration limit
   far from the optimum. And `bound_relax_factor = 0` as a cone default —
   it fixes the `clay` `m` misses and breaks `robust_20_2` and
   `classical_20_18` (orthant duals diverge to 3e10, then
   `Error_In_Step_Computation`).
   *Iterations:* the NLP arm takes 3–15× the convex driver's (median
   `classical` 168 vs 17, `robust` 134 vs 16).
   *Next:* cone-aware restoration (14 of 32 misses); then the remaining
   step failures.

   **Against main (the before).** On main a cone in an NLP has to be
   written as a smooth row; `pounce_cblib_nlp` runs both ways on the same
   152 instances (one build, default options, NLP arm throughout):

   | | `t² − ‖x‖² ≥ 0` (main) | `t − ‖x‖ ≥ 0` (main) | native (branch) |
   |---|---|---|---|
   | reaches the convex optimum (1e-6) | **125** | 121 | 121 |
   | reports success at a wrong answer | 11 | 13 | **10** |
   | median iterations | **18** | **18** | 128 |
   | median time | **45 ms** | 47 ms | 322 ms |

   On the 112 both solve, native takes a median 151 iterations against 17.
   Native solves 2 the smooth forms do not (`nb_L1`, `turbine07GF`); the
   smooth forms solve 14 native does not (`slay*`, `qssp60`, `estein4_C`).
   **On this corpus native cones are not an improvement over main.** The
   tip — where the smooth forms degenerate and the case native cones exist
   for — is rare in convex SOCP optima and common in contact (a sliding-free
   contact sits at it); the drifting car is the test that decides C. First
   suspect for the iteration gap: the barrier strategy is forced monotone for
   cones, while the smooth runs use the default adaptive one.
4. **C4 — entry points.** Python `Problem(..., cones=[[rows], …])`; the
   CALIPSO ports' cones declared natively.
5. **C5 — measure the target.** Drifting car: 5/5 physical, cone-tip
   violation ≤ 1e-8 under default options, iterations against 1759 and
   CALIPSO's 130; quadruped; rocket thrust cones. Then cone complementarity
   (`β ∘ η = 0`) through workstream B's driver.

## Verdict on the first implementation (C4/C5 measured)

**1. The iteration gap is structural, not a setting.** Measured on CBLIB:
the barrier strategy is the same (monotone) in both arms; gentler μ
schedules move individual instances both ways and close nothing; a cone
analogue of Ipopt's `kappa_sigma` reset (re-centre `z = μ s⁻¹` when
`⟨s, z⟩ > κμ` or `√(det s det z) < μ/κ`) was worse than none at every κ in
{3, 10, 100, 1000} and was removed. The logs show the mechanism: after
each μ decrease the native cone pairs leave the central path, the cone
fraction-to-boundary then clips every step to α ≈ 0.03–0.3 for 10–30
iterations, while bound slacks absorb the same decrease in 2–5 full steps.
Conic interior-point codes avoid this with a predictor–corrector
(Mehrotra) step with a cone-aware second-order term; the Ipopt-style loop
has none for cones.

**2. The target model fails.** The drifting car with its friction cones
declared natively (`benchmarks/calipso`, `drifting_native`; the β·η
complementarity on each route's scalar lowering): **0/25 physical** (5
routes × 5 seeds), every run `Error_In_Step_Computation` — at 8–11
iterations on the single-solve routes (the dual step collapses to 1e-16
with a cone dual pinned on its boundary), at ~245 on the continuations.
On main the same model with squared cones is 5/5 at 0.2026
(`scholtes_then_ineq`, 1759 iterations).

**Conclusion.** Native cones in this form are worse than main on both
corpora and fail the model they were built for. Making them work needs a
cone predictor–corrector in the IPM — a new piece of the algorithm, not a
fix to this one. Until then workstream C should not merge.

## Continuing C: corrector, dual lift, and the coupling they run into

Measured after the verdict above, on the same corpora.

- **Mehrotra corrector on the cone rows** (`soc_slack::mehrotra_term`,
  predictor with the cone rows' μ at 0, one factorization). Verified: the
  corrected row closes the scaled second-order residual to 1e-10. Effect on
  CBLIB: mixed (`estein4_C` 118 → 42 and solved; `robust_20_2` 32 → 92).
  Allowing one μ decrease per iteration
  (`mu_allow_fast_monotone_decrease=no`) helps about as much and the two do
  not stack. Neither touches the hard failures. On the drifting car it
  changes nothing: those runs fail before μ first decreases.
- **Cone dual lift** (the lower half of `kappa_sigma`: raise `z₀` minimally
  when `√(det s det z) < μ/κ`). This is what the drifting car needed first.
  A dual reached `det z = 4e-14` against `det s = 0.07` at `μ = 0.1`, and the
  0.99 fraction rule let it close 99% of the gap per iteration. With the
  lift, `scholtes_then_ineq` gets through `τ = 0.01` and `τ = 0.001`
  (objective 0.2019, against 0.2026 for squared cones) and stalls at
  `τ = 1e-4`: primal feasible to 1e-8 with full primal steps, dual step
  ~1e-4 for 1500 iterations. κ ∈ {1e2, 1e4, 1e6} all stall there or earlier;
  `ncp_eq` does not converge at any κ. Not yet physical.

**The coupling.** In this design the cone dual *is* the row multiplier
(`z = −y_d`). Every `x`-stationarity correction to `y_d` moves the cone dual
directly, and the dual has to stay in the cone at every iterate. Ipopt's
bound slacks are not built that way: their multiplier `v` is a separate
variable, tied to `y_d` by `−y_d − v = 0` only at convergence. `y_d` is free
to move while `v` stays positive under its own step rule. The drifting car
fails in the shape that coupling predicts: the dual direction keeps pushing
a cone dual out while its slack is interior. Each safeguard (the
fraction-to-boundary cap, the lift) moves the failure to a later stage
without removing it.

**What would remove it:** a separate cone dual `z` in the iterate, with the
`s`-row residual `−y_d − z` relaxed like Ipopt's `grad_lag_s` and `z` stepped
by its own cone rule — the bound-slack structure, for cones. That is the
storage design that was backed out in C2a for safety (a missed bound-reading
site would impose `s_i ≥ 0` on a cone entry). It can be done safely with a
dedicated cone-dual component rather than borrowing `v_L`. Doing so means a
ninth iterate component that every iterate-vector operation, the warm start,
restoration and the sensitivity layer would have to carry.

## The decoupled cone dual (option 1) — measured, parked

Implemented (`80c880e`): each cone's dual `z` is its own variable
(`IpoptData::soc_z`), tied to `y_d` only through the `s`-stationarity
residual `−y_d − z`. The reduced Newton row is unchanged (z cancels), `dz`
is recovered after each solve, `z` steps with `alpha_dual` under its own
cone cap, `y_d` steps freely. Agreed stopping rule: the drifting car, seed
0, continuation route, must be physical (violation ≤ 1e-4) and clearly
better than main's 1759 iterations, or C is parked.

| `scholtes_then_ineq`, seed 0 | furthest τ | objective | violation | iterations |
|---|---|---|---|---|
| decoupled, lift κ=1e4, corrector | 1e-6 | **0.2026** | 1.8e-3 | 11 659 |
| decoupled, lift, no corrector | 1e-5 | 0.2036 | 5.7e-3 | 7 890 |
| decoupled, no lift, no corrector | 1e-4 | 0.2031 | 1.8e-2 | 2 484 |
| decoupled, no lift, corrector | fails at 0.1 | — | 1.8 | 1 332 |
| coupled `z = −y_d` (before), lift | 1e-3 | 0.2019 | 5.8e-2 | 6 550 |
| **main, squared cones** | finished | 0.2026 | ≤ 1e-4 | **1 759** |

`ncp_eq` and `direct` end `Restoration_Failed` (restoration is not
installed with cones). The decoupling is a real improvement — the
continuation reaches the correct trajectory, which the coupled design
never did — but it fails the stopping rule on both counts. Where the
iterations go: the tightest stages (τ ≤ 1e-5) run to the 3000-iteration
cap with the dual step small, and with the corrector on τ = 0.1 alone
takes 1894. **C is parked here.** Everything is on
`feature/contact-mpcc-cones` behind declared cones (with none declared,
every path is the old one: fixture sweep empty, crate tests unchanged), so
it can be resumed or reverted as a unit. Resuming would mean the endgame
(the τ ≤ 1e-5 stall) and cone-aware restoration, against a target that
main already meets with squared cones.

## Resuming C: where the iterations go, and what moves them

Measured on `feature/contact-mpcc-cones` after the parking verdict above,
seed 0, `scholtes_then_ineq`, one build per table. Every number below is a
single stage re-run from the saved primal point the previous stage ended
at (`solve_once` with `print_level=5`, a per-iteration cone diagnostic
behind an env switch that was not kept), or the whole continuation
stage by stage.

**The reference is not what it looked like.** Main's 1759 is 383
iterations for the nine Scholtes stages and 1376 for the `prod_ineq`
finish — and the stages are that cheap only because each one warm-starts
the duals at the previous stage's μ (`WarmStart`, `mu_init ≈ 1e-9`). The
same squared-cone model with every stage started primal-only at
`mu_init = 0.1`, which is what the native-cone runs did, costs 1469 for
the stages and 4065 in all, and ends at a different point (0.2071). So the
native-cone stage counts were being compared against a continuation
protocol they were not allowed to use; the honest per-stage gap, cold
against cold, was 735 vs 175 at τ = 1e-5 and 2426 vs 206 at τ = 1e-6.

**Where a cold stage spends its iterations.** The τ = 1e-6 stage, 2426
iterations: 19 at μ = 0.1, 69 at 1.5e-4, ~1200 at 1.8e-6 and ~1100 at
2.5e-9. Each μ drop (superlinear, ×80–700) is followed by one full step
that re-centres `⟨s, z⟩` and then **hundreds of iterations with
`alpha_pr`, `alpha_du` in 1e-5 … 1e-1, clipped by the cones' own
fraction-to-boundary, while `inf_du` sits at 1–60**. The pairs that clip
are the *active* friction cones (β on its boundary: sliding) and the η
cones they pair with — on this model most contacts slide, so the pair has
`s` and `z` both against their boundaries, `det s ≈ 1e-7 … 1e-12` with
`z₀ ≈ 50 … 1000` at μ = 1.5e-4 (the Scholtes multipliers grow as τ
shrinks). The mechanism, read off the logs and then checked by hand:

- A pair on the central path with `det s` that small has `det z =
  μ²/det s`: the dual is itself within rounding of its boundary even when
  perfectly centred.
- The dual step `dz = μ s⁻¹ − z − H ds` carries `H ds`, and the NT block's
  large eigenvalue is `≈ μ/λ_min(s)²` with `λ_min(s) = det s/(2 s₀)`.
  A *tangential* primal move `t` (the friction direction turning) gives a
  tangential `dz ≈ μ t/λ_min²`, which leaves the cone as soon as
  `|t| > λ_min(s) ≈ 1e-7`, while the primal itself tolerates
  `|t| ≈ √(det s) ≈ 3e-4`. **A boundary dual cannot be rotated by a linear
  step.** The squared form never has this problem: its dual is the scalar
  `v` and its direction `v·∇(det s) = 2 v J s` rotates with `s` for free.
- After the μ drop the first step has `alpha_du = 0.95` with `|dz| ≈ |z|
  ≈ 1e3`, and the `(1 − α_du) dz` left in `−y_d − z` is the `inf_du = 68`
  the log shows; the mismatched primal and dual step lengths then leave
  `s ∘ z` nowhere near `μ e`, and the clipped crawl begins.

**What moved the τ = 1e-6 stage (2426 at the parked default).**

| change | iterations | note |
|---|---|---|
| dual lift floor κ = 1e2 / 10 / 3 | 1990 / 1499 / 1282 | monotone in κ |
| **dual lift floor κ = 1** (`√(det s det z) ≥ μ`) | **570** | kept, new default |
| κ = 0.5 / 0.3 | 3000, wrong point (0.2106) | over-inflated dual changes the answer |
| κ = 1, `mu_superlinear_decrease_power = 1.01` | 1012 | gentler drops are worse |
| κ = 1, + `mu_linear_decrease_factor = 0.5` | 1439 | |
| `mu_allow_fast_monotone_decrease = no` | 2309 | |
| corrector off | 3000 (cap) | |
| per-cone dual step lengths | 2032 | κ = 10 + per-cone: 1624 |
| cone dual steps with the primal α (common step, as conic codes) | 1229 | both caps on the primal step: 1142; with κ = 1: 718 |
| full dual step, head lifted back into the cone (no dual cap) | 926 (κ = 1), fails at κ = 1e4 | |
| re-align `z` onto the central ray when it collapses, keeping `⟨s, z⟩` | 2327 … 3000 | worse than the head-only lift |
| cone dual ≡ `μ s⁻¹` (primal barrier, no dual step) | 3000, wrong point (0.2136) | |
| upper half of `kappa_sigma` (scale `z` down when `⟨s, z⟩ > κ_up μ`), κ = 1, κ_up = 10 / 100 | 1479 / 733 | |

Only the lift floor at κ = 1 is a large effect, and it is not specific to
this model: on 60 CBLIB SOCPs (`pounce_cblib_nlp`, the first 60 of the C3
list) κ = 1 against the parked 1e4 keeps every one of the 49 matches,
turns the five `clay*m` "succeeded at a wrong objective" into honest
`Restoration_Failed`, and takes the matched instances from 8378 to 2557
iterations (`classical_40_40` 404 → 54, `estein4_A` 2207 → 153). It is
now the default of `POUNCE_SOC_KAPPA`.

**Warm starts for cones** are implemented (`init_soc_from_warm_start`:
slacks pushed into the interior, `z = −y_d` lifted to the central path at
`mu_init`; `warm_start_init_point` is no longer refused). On this
continuation they do not pay: at the harness's `mu_init = 1e-9` the next
stage's 0.09 infeasibility meets a barrier that is already gone, the cone
caps hold every step at 1e-3 … 1e-5, and the line search fails into the
restoration that is not installed — τ = 0.1 drops 1894 → 86 but τ = 0.01,
1e-3, 1e-6 and 1e-7 all need the restart rung, 4610 in all at κ = 1. A
larger `mu_init` (1e-2) needs the rung as often (3836 and a τ = 1e-6
cap). The squared form survives the same start because the orthant's
`τ = 1 − μ` lets a bound slack cross 99.9999999% of its distance in one
step; a cone crosses 99%. The harness keeps the primal-only stage start
for cones (`POUNCE_SOC_WARM=1` enables the dual start).

**The result against the stopping rule** (κ = 1, the harness unchanged
otherwise, seed 0): the stages through τ = 1e-6 take 1397 iterations
(were 5634) and reach the same point as the squared form (0.2026 at
1.8e-3), and **the τ = 1e-7 stage runs to the 3000-iteration cap on a cold
start and on the restart rung alike**. `run_route`, seed 0: 7457
iterations (both τ = 1e-7 attempts at the cap, the finish
`Restoration_Failed`), the continuation's τ = 1e-6 answer kept, 0.2026 at
violation 1.8e-3. Seed 1: τ = 1e-7 ends `Restoration_Failed` at 107 on
both attempts, 1549 iterations, 1.8e-3. With the dual warm start enabled
(`POUNCE_SOC_WARM=1`), seed 0 gets through every stage with the restart
rung and fails the finish: 4610 iterations, 0.2027 at 1.8e-4 — the closest
run to physical, and still neither physical nor under 1759. The τ = 1e-7
stage is a different failure from the one above:
2440 of its 3000 iterations sit at μ = 1.8e-6 with `inf_pr ≤ 1e-7`, full
or half primal steps that the filter backtracks (`ls` 3–5, watchdog
entries), `inf_du` between 1e-3 and 1e2 never settling, and the objective
creeping down at ~1e-7 per iteration — a crawl along the degenerate
valley the Scholtes relaxation leaves at τ = 1e-7, where the η cones of
the sticking contacts are within 2e-7 of their tips. The squared form's
cold τ = 1e-7 stage is 156 iterations, with `inf_du` at 1e9 and
regularisation at 1e10 on the way — Ipopt's inertia machinery handling the
indefinite `2 v J` the squared row adds, which the cone barrier does not
have.

**Conclusion.** The decoupled cone dual with the central-path lift is a
4× better algorithm than what was parked, on the target and on CBLIB,
and it still does not pass: it cannot close the last two decades of the
Scholtes continuation in a useful number of iterations. Every remaining
idea that keeps Ipopt's step structure — separate primal and dual step
lengths, a pre-set monotone μ, a filter on the primal — was tried above
and is worse or marginal. What the measurements point at is the thing
CALIPSO does instead: no Scholtes stages at all, the cone complementarity
`β ∘ η = κ e` relaxed by the solver's own barrier parameter inside one
solve, so the degenerate endgame is never posed as a separate NLP. That is
workstream B's driver with cone pairs, not a fix to this arm. C stays
parked, with κ = 1 and the warm-start initializer merged into the branch.
