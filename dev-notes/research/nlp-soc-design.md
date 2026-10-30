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
`int K` (fraction `τ`); the step for `y_d` on cone rows is capped so
`z + α dz` does. Both are the cone-boundary root (`pounce-convex`'s
`SecondOrderCone::max_step` computes the same quantity).

## What must change, and what must not

With no cones declared every new path is skipped, so a model without cones
runs the code it runs today. The fixture sweep must diff empty.

| place | change |
|---|---|
| application / `OrigIpoptNlp` | accept cone row blocks (user `g` indices); map them to `d` indices; refuse a cone row that is not free, a row in two cones, a cone of dimension < 2 |
| NLP scaling | every row of one cone gets one factor (a per-row factor changes the cone) |
| presolve, any row-renumbering wrapper | declined when cones are declared, out loud |
| `IpoptCq` | barrier objective and its `s`-gradient; `grad_lag_s` zero on cone rows; complementarity, average complementarity, centrality, barrier error; Σ blocks |
| augmented system | dense `(s, s)` blocks at cone positions (structure fixed at first assembly) |
| line search | cone fraction-to-boundary for `s` and for `y_d`'s cone rows |
| initialization | `s` and `z` pushed into `int K` (`s₀ ≥ ‖s_{1..}‖ + margin`) |
| μ update / convergence | the complementarity error includes `‖s ∘ z − μ e‖` per cone |
| restoration | first version: a problem with cones that enters restoration reports it and fails rather than running a restoration that ignores the cones; cone-aware restoration is a later increment |
| sensitivity, SQP arm, convex routing, `--minima` | refuse cones (first version) |

## Increments

1. **C1 — math and declaration.** `soc_slack` module: `det`, Jordan product,
   inverse, NT block, barrier value/gradient, max step, interior push,
   with tests that pin every convention above. The declaration surface on
   the application and its validation. No change to any solve.
2. **C2 — the solve.** CQ, augmented system, line search, initialization,
   convergence, behind "cones declared". Unit tests on small SOCPs with
   known optima; restoration refuses.
3. **C3 — measure correctness.** CBLIB SOCPs through the NLP arm against
   `pounce-convex` (global optima): objectives match to 1e-6 on ≥ 95%.
   Fixture sweep empty.
4. **C4 — entry points.** Python `Problem(..., cones=[[rows], …])`; the
   CALIPSO ports' cones declared natively.
5. **C5 — measure the target.** Drifting car: 5/5 physical, cone-tip
   violation ≤ 1e-8 under default options, iterations against 1759 and
   CALIPSO's 130; quadruped; rocket thrust cones. Then cone complementarity
   (`β ∘ η = 0`) through workstream B's driver.
