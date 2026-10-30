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
