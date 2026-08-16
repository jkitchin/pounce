VERDICT: no-improvement

Fixes #608

## Problem

#90 delivered a predictor–corrector continuation engine around the held KKT
factor, but only for the differentiable frontend: `pounce.jax.PathFollower` is
written against `JaxProblem` and reaches for `jax.grad` / `jax.jacobian`. The
same capability was not available as a reusable repeated-NLP initialization
path for the generic `Problem`, Pyomo, CLI or GAMS frontends, so anyone tracing
a parametric NLP sequence through those frontends rebuilt the orchestration —
transfer, warm start, predictor, trace — in user code.

This is an integration issue, not a request to reimplement #90.

The measurable question underneath it is whether the tangent predictor is worth
anything on an interior-point method once a plain previous-solution warm start
is already in play. Measured on pounce's own warm-start corpus, **it is not**,
and this PR records that as the headline rather than burying it.

| arm | iterations | vs. `warm-ipm` |
|---|---|---|
| `cold-ipm` | 2492 | +102.8% |
| `warm-ipm` (previous solution) | 1229 | — |
| `pred-ipm` (linear predictor) | 1258 | **+2.4%** |
| `predcorr-ipm` (predictor–corrector) | 1242 | **+1.1%** |

Baseline `cfc1121`, `warm_start_recentering=residual`,
`mpc_horizon_{10,20,40,80}` × `{tiny,small,large}` — 12 cells, 240 solves per
arm.

## Cause

The mechanism is a floor, not a tuning failure. In the continuation regime a
warm-started IPM solve already converges in **one iteration per step**, so a
better seed has nothing left to remove. At larger steps there is headroom, but
there the active set moves and a first-order predictor is extrapolating through
a kink — error `O(Δθ²)` at best and unbounded across a critical-region
boundary. That is why `pred-ipm` is *reliably worse* at the `large` scale
(+8.8% overall, +17.5% on `mpc_horizon_80`).

This is the interior-point analogue of the well-known result that IPMs
warm-start weakly: the barrier restart, not the seed's distance from the
solution, is what the iterations are spent on.

## Fix

A frontend-neutral driver, `pounce.Continuation`, running #90's loop through
the public `Problem` / `Solver` API, plus adapters for Pyomo, the CLI and GAMS.
`run(thetas)` traces a prescribed repeated-NLP sequence; `follow(theta_of_s,
s_span)` traces an adaptive path and may accept a point on the predictor alone.
`StepController` is shared verbatim with `PathFollower`, so "how far to step
next" has one implementation in the tree.

Three things this branch previously stopped short of, now implemented:

**`run` subdivides.** A corrector rejection between two prescribed points used
to end the trace with a runaway iterate recorded as an answer. The caller
choosing the points they *want* does not forbid the driver visiting more, so
`run` now halves the gap and re-predicts from the last point known good.
Inserted points carry `prescribed=False` and are counted by
`trace.n_inserted`; every prescribed point still appears, in order. It is a
**no-op on a healthy path** — asserted, because otherwise every existing
caller's step count moves silently. `subdivide_tol` adds an opt-in
monitor-driven trigger, deliberately a separate knob from `monitor_tol`, whose
`1e-6` default would subdivide on essentially every step.

**Pseudo-arclength past folds.** The previous stopping point was that
`Solver.parametric_step` back-solves against the factor of a *converged* solve
and a fold has none. That is true and it is the wrong place to stop: a dense
Jacobian is not required. The tangent is the null vector of the `(d, d+1)`
augmented matrix `[∂R/∂z | ∂R/∂θ]`, obtained by **bordering** it with the
previous tangent and solving

```
[ ∂R/∂z   ∂R/∂θ ] [ t ]   [ 0 ]
[     t_prevᵀ   ]       = [ 1 ]
```

which is nonsingular *at* a simple fold — that is the point of the
pseudo-arclength formulation — and needs no SVD, unlike #90's dense route. `R`
and its Jacobian are assembled sparsely from the problem's own cyipopt-shaped
callbacks; the Hessian of the Lagrangian is exactly `∂/∂x` of the stationarity
block, so nothing is approximated and no third derivative appears.

*Rejected alternatives, so the experiment is not repeated.* Reusing the held
factor cannot work: past a fold there is no solution at that `θ` for any factor
to belong to. Posing the arclength step as an NLP (`min f` subject to the
constraints plus the arclength row, with `θ` a variable) is **wrong**, not
merely inconvenient — stationarity in `θ` then forces `λ = ν·t_θ`, which is not
the original curve. It has to be a root-find on the KKT system, which is what
this does.

*The cost, measured and stated:* one sparse LU per Newton iteration, where
parameter continuation gets a back-solve against a factor the solver already
built. That is the price of going round the corner at all.

**CLI and GAMS.** #608's scope note names the mechanism — "CLI/GAMS can follow
using a path manifest or repeated-solve protocol" — and both are now built.
`pounce-continue path.json` traces a whole path: one command, one report, one
process per point.

There is **no tangent predictor on the CLI path and there cannot be**: the
predictor is a back-solve against a factor that does not survive `exec`. The
trace reports the predictor as absent rather than leaving it to be inferred.
What *does* cross the boundary is more than the primal point — an AMPL `.nl`
carries an initial primal point (`x` segment) *and* initial duals (`d`
segment), both honoured by pounce's reader — so the transfer is primal-dual,
missing only the bound multipliers and the barrier parameter, which `.nl` has
nowhere to put.

`pounce.gams.continuation.trace` routes a GAMS path through the same driver.
The pip link builds an ordinary `Problem` from a GMO view, so driven from one
Python process GAMS gets the **whole** driver, tangent predictor included.

## Blast radius

**No Rust changed.** `git diff cfc11218 --name-only` matches nothing under
`crates/`, so the CLI binary is unchanged and the fixture sweep is empty by
construction. Run anyway, both binaries built from this tree:

```
scripts/sweep-fixtures.sh <cfc1121 binary> /tmp/sweep-base.txt
scripts/sweep-fixtures.sh target/release/pounce /tmp/sweep-new.txt
diff → empty, 0 lines moved across 57 fixtures
```

The sweep was also run in the regime that reaches the changed code — the
benchmark corpus with the continuation arms on, 240 solves per arm at two
`warm_start_recentering` settings — and the pre-existing arms were checked
against the parent directly. Running `cfc1121`'s own `benchmarks/` tree (which
drives `Problem.solve`) against this branch's (which drives `pounce.Solver`, so
the predictor can reach the factor the solve leaves behind):

```
cold-ipm + warm-ipm steps compared : 480
identical in BOTH iters and objective: 480
differing                            : 0
```

So rerouting the adapter through `pounce.Solver` is **bit-identical** on the
two pre-existing arms. Separately, and for a different reason, those two arms
are also unchanged between the two `warm_start_recentering` settings (0 of 240
steps each) — see below, that one is a finding about #620, not about this
branch.

Python-side blast radius: `link.solve_view` was split, with its problem
construction extracted as `problem_from_view`. Behaviour is unchanged and the
existing `test_gams_link.py` suite (80 tests) covers it.

## Verification

### The four-way benchmark, at both recentering settings

Baseline **`cfc1121`**. `warm_start_recentering` is swept because the warm
baseline is what every margin here is quoted against, and #606/#620 is the
merge most likely to have moved it.

**`recentering=residual`**

| family | scale | cold | warm | pred | predcorr | pred vs warm | predcorr vs warm |
|---|---|---|---|---|---|---|---|
| mpc_horizon_10 | tiny | 155 | 28 | 28 | 28 | +0.0% | +0.0% |
| mpc_horizon_10 | small | 172 | 46 | 40 | 41 | −13.0% | −10.9% |
| mpc_horizon_10 | large | 170 | 104 | 100 | 100 | −3.8% | −3.8% |
| mpc_horizon_20 | tiny | 178 | 39 | 36 | 36 | −7.7% | −7.7% |
| mpc_horizon_20 | small | 192 | 98 | 101 | 112 | +3.1% | +14.3% |
| mpc_horizon_20 | large | 189 | 152 | 165 | 153 | +8.6% | +0.7% |
| mpc_horizon_40 | tiny | 214 | 39 | 37 | 38 | −5.1% | −2.6% |
| mpc_horizon_40 | small | 224 | 111 | 92 | 117 | −17.1% | +5.4% |
| mpc_horizon_40 | large | 233 | 215 | 228 | 214 | +6.0% | −0.5% |
| mpc_horizon_80 | tiny | 255 | 49 | 44 | 45 | −10.2% | −8.2% |
| mpc_horizon_80 | small | 264 | 125 | 125 | 135 | +0.0% | +8.0% |
| mpc_horizon_80 | large | 246 | 223 | 262 | 223 | +17.5% | +0.0% |
| **total** | | **2492** | **1229** | **1258** | **1242** | **+2.4%** | **+1.1%** |

**`recentering=none`**

| family | scale | cold | warm | pred | predcorr | pred vs warm | predcorr vs warm |
|---|---|---|---|---|---|---|---|
| mpc_horizon_10 | tiny | 155 | 28 | 28 | 28 | +0.0% | +0.0% |
| mpc_horizon_10 | small | 172 | 46 | 40 | 41 | −13.0% | −10.9% |
| mpc_horizon_10 | large | 170 | 104 | 100 | 100 | −3.8% | −3.8% |
| mpc_horizon_20 | tiny | 178 | 39 | 36 | 36 | −7.7% | −7.7% |
| mpc_horizon_20 | small | 192 | 98 | 101 | 96 | +3.1% | −2.0% |
| mpc_horizon_20 | large | 189 | 152 | 165 | 152 | +8.6% | +0.0% |
| mpc_horizon_40 | tiny | 214 | 39 | 37 | 36 | −5.1% | −7.7% |
| mpc_horizon_40 | small | 224 | 111 | 92 | 120 | −17.1% | +8.1% |
| mpc_horizon_40 | large | 233 | 215 | 227 | 213 | +5.6% | −0.9% |
| mpc_horizon_80 | tiny | 255 | 49 | 44 | 44 | −10.2% | −10.2% |
| mpc_horizon_80 | small | 264 | 125 | 125 | 128 | +0.0% | +2.4% |
| mpc_horizon_80 | large | 246 | 223 | 262 | 221 | +17.5% | −0.9% |
| **total** | | **2492** | **1229** | **1257** | **1215** | **+2.3%** | **−1.1%** |

By step scale, `recentering=residual`:

| scale | cold | warm | pred | predcorr |
|---|---|---|---|---|
| tiny | 802 | 155 | 145 (−6.5%) | 147 (−5.2%) |
| small | 852 | 380 | 358 (−5.8%) | 405 (+6.6%) |
| large | 838 | 694 | 755 (**+8.8%**) | 690 (−0.6%) |

### The named failure mode did not occur, and here is the evidence

The previous report warned: *"If it raises early-iteration counts on some
paths, the predictor could look better without being better — which is the
failure mode to watch for."*

It did not happen, and the reason is mechanical rather than lucky. Between the
two recentering settings, per-step iteration counts differ on:

| arm | steps differing | steps identical |
|---|---|---|
| `cold-ipm` | **0** | 240 |
| `warm-ipm` | **0** | 240 |
| `pred-ipm` | 1 | 239 |
| `predcorr-ipm` | **14** | 226 |

`warm_start_recentering` measures the supplied iterate and adapts to how far
off-centre it is. `predcorr-ipm` is the only arm that supplies a *perturbed*
multiplier seed — the previous solution stepped along the tangent — so it is
the only arm with anything for that measurement to act on. The others hand over
either a cold point or an exactly-converged one, and on these single-block,
zero-inequality-row models (22–162 equality rows, 0 inequality rows) there is
nothing to change.

Consequently the **warm baseline is 1229 at both settings**. The `−1.1%` at
`recentering=none` is not a better predictor; it is the same predictor with its
off-centre dual seed left alone, landing better on 14 steps, against an
identical baseline. The predictor cannot "look better because the baseline
rose", because the baseline did not rise.

The baseline *did* move against the previous measurement on `70bf53de` (warm
1097 → 1229, cold 2360 → 2492, +12% / +5.6%). Since `recentering=none` is by
definition pre-#606 behaviour and reproduces 1229 **exactly**, #606/#620
contributes **none** of that move; it comes from the other merges in between
(#605/#619, #602/#614, #607/#623).

### The fold fixture

`min x₀³/3 + x₁²/2` s.t. `x₀ + x₁ = θ`. Stationarity gives `λ = −x₀²` and
`x₁ = x₀²`, so the solution curve is `x₀ + x₀² = θ`, folding at
**`x₀ = −1/2, θ = −1/4`**. Built so LICQ holds at the fold (`∇g = (1,1)`) and
`λ = −1/4` stays finite there — a fold resting on a vanishing constraint
gradient sends `λ` to infinity and no arclength scheme in `(x, λ, θ)` passes
it, which would be a fixture built to flatter the method rather than test it.

| | result |
|---|---|
| `run`, θ: 2.0 → 1.0 → 0.0 → −0.4, `subdivide=False` | **fails** at θ = −0.4, `Diverging_Iterates`, x₀ ≈ −9.9×10¹⁰ |
| `run`, same path, `subdivide=True` | walks in via θ = −0.200, −0.250 (x₀ = −0.499945), fails at θ = −0.251172; `subdivision_exhausted`, 3 inserted points, 10 rejections absorbed — **locates the fold to 1.2×10⁻³** and returns 3 extra valid solutions where the un-subdivided path returned one garbage one |
| `trace_arclength`, same start, `ds=0.25` | **traverses**: θ decreases to −0.239, turns, and increases again while x₀ continues past −1/2 to −5.97 (41 points); every accepted point satisfies `x₀ + x₀² = θ` and `‖R‖∞ ≤ 7.0×10⁻¹⁰` |
| `trace_arclength`, `ds=0.02` | localises the turning point to θ_min = −0.249994, i.e. `|θ_min − (−1/4)| = 6×10⁻⁶` — the discretisation is the only thing keeping it off the exact fold (the test asserts < 10⁻³) |

### What the CLI / GAMS path is measured to be worth

20-point van der Pol NMPC path, horizon 40 (n = 122, m = 82), against repeated
cold `pounce model.nl` invocations:

| | iterations | evaluations | wall (mean of 3) |
|---|---|---|---|
| repeated cold | 226 | 1230 | 233 ms |
| `pounce-continue` | 193 | 1166 | 221 ms |
| | **−14.6%** | **−5.2%** | **−5%** |

Iteration counts are exactly reproducible run to run (226/193 on all three
repeats); the wall-clock spread overlaps. The gap between −14.6% of iterations
and −5% of wall clock is process startup, `.nl` parse and presolve, which at
these sizes dominate the solve. Repeating at horizon 300 (n = 902) moves the
iteration saving not at all (225 → 193) and the wall time not at all
(705 → 713 ms). **If you are paying per process, that overhead is what you are
paying, and the warm transfer does not address it.**

One trap worth recording: a linear-quadratic MPC is a convex QP, and the CLI
routes convex QPs to `pounce-convex`, which never reaches the NLP warm-start
path. Measured on an LQ path the answer is exactly 0% — for that reason, not
the interesting one. The fixture uses van der Pol dynamics because of it.

**GAMS.** The native C link's `sqp_state_file` does **not** carry more than the
pip link for this purpose, and the source says so precisely: it holds the
discrete working set only — one byte of `bound_status` per variable and one of
`cons_status` per constraint, behind a magic string and a checksum over
`(n, m, bounds)`. No primal point, no multipliers, no barrier parameter; it
feeds `IpoptSetWarmStartWorkingSet` on the active-set SQP path. For
interior-point continuation that is strictly *less* than the pip link already
holds in memory, and the checksum over the bounds is invalidated by exactly the
horizon shifts and remeshes the transfer map exists to serve.

### Fixture sweep

Empty diff, 0 lines moved across 57 fixtures, baseline `cfc1121`. No Rust
source changed, so this is expected by construction; it was run rather than
argued because "it cannot produce a wrong answer" is not the relevant safety
property.

## Tests

`python/tests/test_continuation.py` grows from 21 to 32 tests. The 11 new ones
**fail on the parent commit** (`ce04c00b`) for the right reason:
`AttributeError: 'Continuation' object has no attribute 'trace_arclength'` for
the six arclength tests, and `TypeError: Continuation.run() got an unexpected
keyword argument 'subdivide'` for the five subdivision tests. The 21
pre-existing tests pass on the parent unchanged.

New files: `python/tests/test_continuation_cli.py` (13 tests — manifest
parsing, the `.nl` initial-point rewrite, `.sol` parsing, and three end-to-end
traces gated on a built binary plus the fixture corpus).
`python/tests/test_gams_link.py` gains 4 continuation tests driving the same
license-free in-memory `GmoView` fake the rest of that file uses, so the GAMS
path is covered without a GAMS install.

Deliberately not pinned:

- **The benchmark numbers themselves.** They move with the solver and are
  recorded in `docs/src/continuation.md` with their baseline commit, not
  asserted in a test.
- **The live `gamsapi` adapter** (`solve_from_control_file`), unchanged here and
  still the only CI-untestable surface in the GAMS link.
- **Wall-clock figures.** Reported as a mean of three with the spread stated;
  not asserted.

---

- [x] Tests fail on the parent commit for the stated reason
- [x] `CHANGELOG.md` `[Unreleased]` entry, in the user's terms
- [x] Book page under `docs/src/` updated, and linked from `SUMMARY.md`
- [x] `cargo fmt --all -- --check`, `cargo clippy`, `cargo test` clean
      (`cargo test --workspace --exclude pounce-hsl`: 2950 passed, 0 failed;
      `python/tests`: 1055 passed, 19 skipped; `pyomo-pounce/tests`: 294
      passed, 6 skipped; `check-release-consistency.sh` and
      `check-docs-consistency.sh` both OK)
- [x] Every claim in this body is true of the code as it stands now

---

## Acceptance criteria

### The four criteria from #608

1. **"A parametric NLP sequence can be traced through the generic `Problem` API
   without rebuilding orchestration in user code."** — **MET.**
   `pounce.Continuation(update, pins=...).run(thetas)` / `.follow(theta_of_s,
   s_span)`. Covered by `python/tests/test_continuation.py`.

2. **"A Pyomo MPC/horizon-shift example supplies a transfer map and reuses
   primal/dual/barrier state."** — **MET.** `pyomo_pounce.continuation` with
   `shift_map(shift=k)`; the corrector reuses Var values plus the `dual` /
   `ipopt_zL_in` / `ipopt_zU_in` suffixes and the previous barrier `μ`. Covered
   by `pyomo-pounce/tests/test_continuation.py`, which checks every traced
   objective against an independent cold solve at the same parameter.

3. **"The driver reports predictor residual, corrections, step rejections,
   active-set events, and total evaluations."** — **MET.**
   `ContinuationTrace.{n_corrections, n_rejections, n_active_set_events,
   total_evals}` plus per-step `predictor_residual`; `n_inserted` was added for
   subdivision. `trace.report()` prints all of them.

4. **"Benchmarks compare cold solve, previous-solution warm start, linear
   predictor, and predictor-corrector."** — **MET.** `cold-ipm`, `warm-ipm`,
   `pred-ipm`, `predcorr-ipm` in `benchmarks/warmstart`, tabulated above at two
   `warm_start_recentering` settings.

### The scope bullets

| bullet | status |
|---|---|
| accepts a parameter path `theta(s)` and a callback/model update | **MET** — `update(theta) -> Problem`; `follow` takes `theta_of_s` |
| transfers the complete previous iterate | **MET** — `WarmStart.from_info` carries `x`, `λ`, `z_L`, `z_U`, `μ` |
| uses sensitivity/KKT solves for a tangent predictor when available | **MET** — `Solver.parametric_step` / `parametric_step_full` under `pins=` |
| applies an explicit user transfer/prolongation map for horizon or mesh changes | **MET** — `transfer=` on #607's `WarmStart.transfer` mapper protocol; `shift_map` on the Pyomo side |
| performs a warm-started corrector | **MET** — every corrected step |
| adapts step size from residuals and corrector work | **MET** — `StepController` in `follow`; `run` subdivides on rejection and (opt-in) on the monitor |
| detects active-set events and reanchors | **MET** — bound-multiplier fingerprint; `StepController.corrected(..., active_set_event=True)` drops back to `ds0` |
| optionally supports pseudo-arclength near folds | **MET** — `trace_arclength`, traversal demonstrated above. Scope is #90's v1: scalar `θ`, equality/unconstrained, fixed active set |
| falls back to zero-order warm transfer when sensitivities are unavailable | **MET** — omit `pins`; `has_tangent` reports which you got |
| thin adapters for generic `Problem` and Pyomo first | **MET** |
| CLI/GAMS follow using a path manifest or repeated-solve protocol | **MET** — `pounce-continue` + manifest v1; `pounce.gams.continuation.trace` |

### Not met

None of the four acceptance criteria or scope bullets is unmet.

Three things are **out of scope by design**, stated so the boundary is a known
one rather than a silent gap — all three are also out of scope in #90, and none
is a deferral of work this issue asked for:

- **Bifurcation detection and branch switching.** `trace_arclength` follows the
  branch it starts on. At a bifurcation (as opposed to a simple fold) the
  bordered matrix is singular and the trace stops rather than picking a branch.
- **Folds with a moving active set.** The arclength residual `R` treats every
  general row as an active equality, so two-sided inequality rows are rejected
  with an explicit error rather than mis-traced.
- **Vector-parameter arclength.** "Past the fold" is not defined for a solution
  manifold of dimension > 1; the driver raises and says to reparametrise onto a
  scalar path.

### On the verdict

`VERDICT: no-improvement` refers to the measured question the benchmark asks —
**does the tangent predictor beat a previous-solution warm start on an
interior-point method?** It does not: +2.4% / +1.1% at
`recentering=residual`, +2.3% / −1.1% at `=none`, all inside ±3%, and reliably
worse (+17.5%) at the largest step scale. That verdict now covers a *complete*
implementation rather than a partial one, which makes it stronger, not weaker:
the arclength mode, the subdivision and the CLI/GAMS adapters all landed and
none of them changes the answer to that question.

What the branch *is* worth is separable from that and is positive: warm
starting itself is 2.03×; `follow` skips solves outright; `run` subdivision
turns a diverging trace into three extra valid solutions and a fold located to
1.2×10⁻³; `trace_arclength` reaches a branch parameter continuation cannot see
at all; and the CLI path is −14.6% iterations against repeated cold
invocations. None of those is the predictor beating warm start, and this report
does not present them as if they were.
