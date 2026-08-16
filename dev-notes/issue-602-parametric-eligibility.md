# issue #602 — `solve_parametric` eligibility, measured

[#602](https://github.com/jkitchin/pounce/issues/602) observes that
`solve_parametric` admits a warm parametric solve on a guard that checks only
shape and `H`, while the homotopy it then runs interpolates only `g` and the
row bounds — and proposes extending the guard to also require an unchanged
constraint matrix, unchanged variable bounds, and an unchanged Hessian-inertia
declaration.

The observation is correct on all three counts. This note records what each of
the three actually costs today, because the fix that follows from the data is
not the one the issue proposes.

Short version:

* This is a **cost** question, not a correctness one. Every route measured
  lands on the same answer to `1e-14`; the path is a predictor and the
  corrector re-solves against the true problem.
* Bullets 1 and 2 (`A`, variable bounds) are real and measurable. Bullet 3
  (inertia) has **no effect on this path at all** — `hessian_inertia` is never
  read by the homotopy or by the KKT factorization.
* **Rejecting an ineligible pair usually costs more than admitting it**,
  because the fallback in the current code is a *cold* solve — and on this
  family the ineligible cases beat cold in 10 of 13 rows. The useful change is
  not a stricter guard, it is a **better fallback**: `solve_with_working_set`,
  which is already what the SQP driver uses and which is the fastest of the
  three routes in 12 of 17 rows measured.

**Outcome.** The better fallback shipped. The stricter guard was implemented,
measured, and **declined** — it is better or equal in 14 of 14 rows at `n = 30`
and worse in 9 of 14 at `n = 20`, on the same family. Sections at the bottom
record both, and they are the parts to read first: the measurement changed the
recommendation this note opens with, and the reasoning that produced the wrong
recommendation is worth more than the conclusion.

---

## What the guard checks, and what the path models

`solve_parametric_scoped` (`crates/pounce-qp/src/solver.rs`) admits the warm
path when `(n, m)` match, `H` is bit-identical (nonzeros, values, `irows`,
`jcols`), `sol_prev.status == Optimal`, and `sol_prev.x.len() == qp_new.n`.

`solve_homotopy`'s warm arm (`crates/pounce-qp/src/homotopy.rs`) then starts at
`x = sol_prev.x` with `working = sol_prev.working`, and traces to `t = 1`
carrying exactly two moving quantities:

* `dg = qp.g − prev.g`, entering the direction solve as the primal right-hand
  side `−dg`;
* row bounds, interpolating `prev.bl → qp.bl` and `prev.bu → qp.bu` through
  `bound_rate`.

Everything else the tracer touches is taken from the **new** problem and
treated as constant along the path: `qp.a` in both `a_times_x` calls and in
`assemble_active_set_kkt`, and `qp.xl` / `qp.xu` in the pinned box. So three
quantities can differ between the two problems with the path modelling the
difference not at all:

| quantity | checked by the guard? | modelled by the path? |
|---|---|---|
| `n`, `m`, `H` | yes | `H` held fixed (correctly — it is not interpolated) |
| `g`, `bl`, `bu` | no (correctly) | **yes**, interpolated |
| `A` (structure and values) | **no** | **no** |
| `xl`, `xu` | **no** | **no** |
| `hessian_inertia` | **no** | not read at all — see below |

The mechanism behind the issue's concern is worth stating precisely, because it
bounds how bad the damage can get. The tracer is a **pure predictor with no
corrector inside the loop**: each step solves for `d/dt (x, λ)` and takes a
linear step. It never evaluates a residual and never projects back onto the
manifold. So an initial off-manifold error does not grow *through a feedback
mechanism* — it is simply never removed, and the linear extrapolation is taken
about the wrong point for the whole path. That is why the failure shows up as a
degraded active-set *prediction* rather than as a wrong answer.

## Bullet 3 first: the inertia declaration is a no-op here

`hessian_inertia` is read in exactly one place in the crate — `elastic.rs`,
where `ElasticProblem::as_qp` maps it onto the augmented problem. The homotopy
never reads it, and neither does `factorize_with_inertia_control`, which takes
its `expected_neg` as an explicit argument and is documented as deciding from
the factor rather than the hint:

> `expected_neg` is required (no bypass) so the inertia signal is always
> checked. The `HessianInertia::Indefinite` hint merely tells the caller
> "shifts may be needed"; the algorithm decides what to do based on the
> factor's report.
> — `solver.rs`

Measured, flipping the declaration with `H` bit-identical changes nothing:

| change | homotopy |
|---|---|
| `Psd` → `Psd` (baseline) | `Optimal chg=4` |
| `Psd` → `Indefinite` | `Optimal chg=4` |
| `Psd` → `Unknown` | `Optimal chg=4` |

Since `H` must already be bit-identical to reach this path, a differing
declaration means the *caller* contradicted itself about one matrix. That reads
as worth rejecting on hygiene grounds, and this note originally recommended it.

It was measured and it is not: because the tracer never reads the declaration,
declining on it cannot improve any path — it can only swap a working path for
the fallback. Measured at `n = 20`, that swap costs 2 working-set changes
becoming 5, for nothing. See "Step 2, declined" below. A hygiene check with a
2.5× price is not hygiene, and the general lesson is that "the caller was
inconsistent" is a reason to *say so*, not a reason to do less work well.

## Bullet 2 is real, and the cause is worse than "not interpolated"

The issue asks for unchanged variable bounds "unless box-bound interpolation is
implemented". Interpolating the box would not be sufficient, because the
homotopy has **no bound-adding event at all**. The `Event` enum is

```rust
enum Event { AddRowLower(usize), AddRowUpper(usize), DropRow(usize), DropBound(usize) }
```

— rows can be added and dropped, and bounds can be *dropped*, but no inactive
variable bound can ever *become* active on the path. The primal ratio test
loops `for i in 0..m` over general rows only, and `worst_path_violation`
documents the matching blind spot ("variable bounds do not move along this
path, so they are not checked here").

The consequence is not confined to the parametric entry point: on the cold arm
too, nothing stops `x(t)` walking straight out of its box. For #602 it means
that if the new problem tightens `xu` below the previous solution, the path
starts outside the new box, never notices, and hands the corrector a working
set built along an infeasible trajectory. Measured (`xu` capped, `g`/`b` moved
slightly):

| change | homotopy | cold | ws-only |
|---|---|---|---|
| `xu = 2` | `chg=4` 8.1 ms | `chg=97` 304.7 ms | `chg=3` **5.2 ms** |
| `xu = 0.5` | `chg=77` 241.4 ms | `chg=94` 299.3 ms | `chg=7` **11.9 ms** |
| `xu = 0.1` | `chg=77` 217.2 ms | `chg=103` 338.5 ms | `chg=75` **210.0 ms** |

The `xu = 0.5` row is the sharpest single result in this note: the homotopy
spends **20×** what simply reusing the previous working set costs, and a
stricter guard as proposed would have replaced it with the cold solve, which is
**25×**. The waste is real; the proposed remedy is the wrong end of it.

## Bullet 1 is real, and its damage is not monotone

Perturbing every entry of `A` by a relative `da` (structure fixed), with `g`
and `b` moved slightly:

| `da` | homotopy | cold | ws-only |
|---|---|---|---|
| 0.02 | `chg=3` 8.6 ms | `chg=18` 56.2 ms | `chg=3` 5.6 ms |
| 0.05 | `chg=65` 189.7 ms | `chg=82` 257.2 ms | `chg=64` 182.0 ms |
| 0.10 | `chg=5` 17.5 ms | `chg=19` 61.5 ms | `chg=4` 6.9 ms |
| 0.20 | `chg=6` 19.1 ms | `chg=19` 71.6 ms | `chg=5` 9.3 ms |
| 0.30 | `chg=8` 23.7 ms | `chg=19` 64.8 ms | `chg=7` 12.7 ms |
| 0.40 | `chg=53` 180.8 ms | `chg=16` **59.7 ms** | `chg=52` 157.0 ms |
| 0.50 | `chg=54` 167.8 ms | `chg=16` 52.7 ms | `chg=9` **14.0 ms** |
| 0.60 | `chg=53` 152.2 ms | `chg=17` **63.1 ms** | `chg=52` 156.8 ms |
| 0.80 | `chg=55` 157.4 ms | `chg=71` 215.7 ms | `chg=54` 156.8 ms |
| 1.00 | `chg=45` 138.7 ms | `chg=61` 183.4 ms | `chg=45` 140.9 ms |

Two things to read off it. The homotopy loses to cold at `da ∈ {0.40, 0.60}` —
so the issue's concern reproduces. And the damage is **not monotone in `‖ΔA‖`**:
`da = 0.05` is expensive on all three routes (a harder target, not a warm-start
failure) while `da = 0.30` is cheap and `da = 1.00` is fine again. A guard
thresholded on how much `A` moved would be fitting noise — which is precisely
the trap [#434](https://github.com/jkitchin/pounce/issues/434) documented when
its own candidate guard turned out to be separated from destroying a genuine
gain by a 3% margin on one instance (`dev-notes/issue-434-homotopy-cost.md`,
"Result 2 — the guard, declined").

What *is* monotone is a quantity the path already computes. The
`POUNCE_HOMOTOPY_DEBUG` handoff line reports the target violation of the point
the path hands the corrector, and it tracks `‖ΔA‖` cleanly:

| `da` | handoff violation |
|---|---|
| 0.00 | 5.3e-2 |
| 0.02 | 1.5e-1 |
| 0.10 | 2.1e-1 |
| 0.30 | 3.5e-1 |
| 0.60 | 5.7e-1 |

That makes it a candidate *runtime* signal (measure the path, do not predict
it — #434's preferred shape), but it is measured here on one synthetic family
and separates the good rows from the bad ones only loosely: `da = 0.30` at
3.5e-1 is cheap and `da = 0.60` at 5.7e-1 is not, with nothing in between to
place a threshold on. It needs the real suite before it is worth anything.

## Correctness: not at risk

Across all 17 rows measured, `|x_warm − x_cold| ≤ 8.4e-15` and
`|obj_warm − obj_cold| ≤ 1.4e-14`, every route `Optimal`. That is by
construction rather than luck: `trace_path` never reports its own point as a
solution, it hands the discovered working set to `solve_with_working_set`,
which pins a primal, routes through `solve`'s infeasible-warm-start pre-check
into l1-elastic phase-1 when the pin is unusable, and finishes under
`audit_and_repair`. The M5 audit re-checks every row and bound against
`feas_tol` before an `Optimal` is allowed out.

So #602 should be scoped as a performance issue. Nothing here justifies a
correctness-flavoured urgency, and framing it that way would invite exactly the
"it cannot produce a wrong answer, therefore ship it" reasoning that `CLAUDE.md`
warns about — in the opposite direction.

## Two incidental findings

**`solve_parametric` ignores `use_homotopy`.** `solve_scoped` gates the cold
homotopy on `ws.is_none() && opts.use_homotopy`; `solve_parametric_scoped`
calls `solve_homotopy` unconditionally. Verified: the parametric route produces
identical statistics (`chg=4 refac=8`) with the flag `false` and `true`. Since
`use_homotopy` defaults to `false` in `pounce-qp` — deliberately, per the #434
work — the public parametric entry point is the one place the homotopy runs
without an opt-in and without a kill switch. Whether that is intended is a
separate question from #602, but any option-level mitigation for #602 would be
built on a flag that this path does not currently honour.

**`solve_parametric` has no production caller.** The only callers in-tree are
`crates/pounce-qp/tests/homotopy.rs` and `crates/pounce-rs/tests/qp_surface.rs`.
The SQP driver (`pounce-algorithm/src/sqp/sqp_alg.rs`) warm-starts through
`solve_with_working_set`, precisely because each SQP linearization moves `A` and
translates the row bounds by `−c(x_k)`, so the previous *primal* does not carry
over. It is worth being explicit about what that implies for #602's framing:
the natural consumer of a parametric API is the SQP outer loop, and in that
loop `A` changes at **every** iteration. A guard that requires `A` unchanged
would make `solve_parametric` permanently ineligible for the workload it exists
to serve. The eligible population is the fixed-`A` parametric family — MPC with
fixed dynamics, an RHS/objective sweep, a continuation step — which is real, but
is a narrower claim than the crate currently makes.

(Relatedly, `pounce-qp`'s package description advertises "the parametric
corrector inside `pounce-sensitivity`". `pounce-sensitivity` does not depend on
`pounce-qp`. Cosmetic, ships to crates.io, not part of this issue.)

## What to do

Ranked by measured value, not by how closely it matches the issue text.

1. **Change the fallback, not the guard.** — **done, see below.**
   `solve_parametric_scoped`'s ineligible branch was
   `self.solve(qp_new, None, opts)` — a cold solve that throws away a working
   set the caller just handed us. Routing it to
   `solve_with_working_set(qp_new, &sol_prev.working, opts)` costs nothing when
   the guard already passes and turns every rejection from "start over" into
   "keep the active-set guess". `ws-only` is the fastest of the three routes in
   **12 of the 17** rows above, and never loses to the homotopy by more than
   noise. The three rows it loses (`da ∈ {0.40, 0.60, 1.00}`) are ones where
   *both* warm routes lose to cold, i.e. where the previous active set is
   genuinely a bad guess — a case for a runtime bail-out, not for preferring a
   cold restart by default. This is worth doing on its own merits, before any
   eligibility change, and it is what makes a stricter guard affordable.

2. **Then tighten the guard, in this order.** With (1) in place, rejecting an
   ineligible pair is cheap, so the guard can be honest about what the path
   models: require identical `A` (structure *and* values — compare
   `nonzeros/values/irows/jcols`, the same test already applied to `H`), and
   identical `xl`/`xu`. Reject a mismatched `hessian_inertia` too if desired,
   but document it as a caller-consistency check, not a path-cost one.

3. **Or fix the box blind spot instead**, which is the larger prize and is not
   specific to #602: add `AddBoundLower` / `AddBoundUpper` events and the
   matching primal ratio test over `j in 0..n`. That removes the reason
   variable-bound changes hurt, benefits the **cold** arm as well (where `x(t)`
   can currently leave the box on any problem), and is the prerequisite for the
   box-bound interpolation the issue asks about. Interpolating `xl`/`xu`
   without it would move a bound the ratio test still cannot see.

4. **Interpolating `A` is a different algorithm — do not scope it here.** With
   `A(t) = (1−t)A₀ + tA₁` the KKT matrix itself becomes `t`-dependent, so
   `(x(t), λ(t))` is no longer affine along a segment and the two ratio tests
   in `t` stop being exact. It needs a genuine predictor-corrector continuation
   (a Newton correction back onto the manifold at each step), not an extension
   of the current tracer. Given (1), the payoff over `solve_with_working_set`
   would have to be demonstrated before the complexity is worth it.

## What must be measured before any of it merges

All four options above reroute which correction the solver reaches for, which
makes them **trajectory changes** under `CLAUDE.md` — `scripts/sweep-fixtures.sh`
against a baseline binary, diffed before merge, with every moving line
explained. In addition:

* The synthetic family in this note is `n = 30`, `m = 20`, diagonal PD `H`. It
  is an instrument for *mechanism*, not evidence about the shipped workload.
  The real comparison is the 138-problem Maros-Mészáros sweep through
  `crates/pounce-convex/examples/homotopy_sweep.rs` and the warm-start suite in
  `benchmarks/warmstart` (whose `-hom` arms differ by exactly one option).
* Option 1 needs an A/B on the warm-start suite specifically, since that is
  where a changed fallback shows up.
* Any threshold-shaped rule needs the #434 treatment: replay candidate rules
  against recorded per-step trajectories, not against endpoint summaries, and
  decline the rule if nothing separates the losses from the gains with margin.

## Reproducing

```text
cargo run -p pounce-qp --example parametric_eligibility_sweep
POUNCE_HOMOTOPY_DEBUG=1 cargo run -p pounce-qp --example parametric_eligibility_sweep
```

The tables above are from the debug profile, so read the ratios between columns
rather than the absolute milliseconds. With the trace enabled each row is
preceded by three `[hom] summary` lines — previous solve, warm path, cold solve
— and the middle one carries the handoff violation quoted above.

---

## Step 1, implemented

`solve_parametric_scoped`'s declined branch now routes to
`solve_with_working_set` when the previous solve reached `Optimal` and its
working set is dimensionally valid for the new problem, falling back to a cold
solve otherwise.

The measurement above could not show this, because every row in it is a pair
the guard *admits* — the fallback is never reached. The sweep therefore gained
a section that perturbs `H`, which is the one quantity the guard rejects on, so
`solve_parametric` never reaches the path and the `homotopy` column is the
fallback and nothing else:

| change | before | after | `ws-only` |
|---|---|---|---|
| `H` 1% | `chg=18` 14.8 ms | `chg=3` 1.3 ms | `chg=3` 1.2 ms |
| `H` 1%, `A` 10% | `chg=19` 16.7 ms | `chg=4` 1.5 ms | `chg=4` 1.8 ms |
| `H` 5% | `chg=18` 15.5 ms | `chg=3` 1.1 ms | `chg=3` 1.1 ms |
| `H` 10% | `chg=18` 14.6 ms | `chg=3` 1.1 ms | `chg=3` 1.3 ms |
| `H` 10%, `A` 10% | `chg=84` 58.1 ms | `chg=67` 42.5 ms | `chg=67` 35.8 ms |
| `H` 25% | `chg=18` 15.3 ms | `chg=3` 1.4 ms | `chg=3` 1.7 ms |
| `H` 50% | `chg=17` 15.1 ms | `chg=2` 0.8 ms | `chg=2` 0.8 ms |
| `H` 50%, `A` 10% | `chg=82` 51.6 ms | `chg=65` 37.1 ms | `chg=65` 34.8 ms |

Two things worth reading off it. The "before" column is *identical* to the
`cold` column, which is the direct confirmation that the declined branch was a
cold solve; the "after" column is identical to `ws-only`, which is the
confirmation that it now isn't. And the hint survives a **50%** Hessian
perturbation (2 changes against a cold solve's 17) — because what a working set
encodes is which constraints bind, and that is far more stable under a change
of `H` than the iterate is. This is why the change is worth making even though
the guard rejects these pairs for a good reason.

Answers are unchanged: agreement with a cold solve of the same target within
`1.8e-15` on every row.

### Conditions on the hint

* `sol_prev.status == Optimal`. A `TimeLimit` result carries
  `WorkingSet::cold` — all-inactive, i.e. no information — and a `MaxIter` one
  carries a set that was still moving. Neither is covered by the measurement,
  so neither gets the hint.
* `sol_prev.working.validate_dims(qp_new.n, qp_new.m).is_ok()`. Without it, a
  shape change reaches `solve_with_working_set` and comes back
  `WarmStartDimensionMismatch` — converting "no warm start available" into
  "solve failed" for a call that has a perfectly good cold answer available.

### Test

`ineligible_parametric_reuses_the_working_set` in
`crates/pounce-qp/tests/homotopy.rs` asserts the answer matches a cold solve
**and** that the change count is strictly below it. The second half is the
point: the old behaviour returned the right answer and would pass any
answer-only assertion, which is precisely the gh #544 shape. Verified to fail
against the previous code (7 changes against cold's 7) and pass after.

The two existing closed-form cases are too small to carry that assertion — a
cold solve reaches their optima in 0–2 working-set changes — so the test builds
an `n = 12`, `m = 8` case with several rows binding.

### Fixture sweep

`scripts/sweep-fixtures.sh` over all 57 CLI fixtures, baseline against new, on
both the default arm and `algorithm=active-set-sqp`: **empty diff on both**.

That is the expected result and it is *not* evidence the change is good.
`solve_parametric` has no production caller, so no CLI fixture can reach the
branch that moved. The sweep's value here is confirming the change did not leak
into anything else in `solver.rs` — it bounds collateral damage, it does not
measure the intended effect. The intended effect is the table above, on a
synthetic family, and it stays synthetic until something in-tree calls
`solve_parametric`.

### Review finding: reusing a working set needs more than matching dimensions

[@GermanHeim caught a bug](https://github.com/jkitchin/pounce/issues/602#issuecomment-5303884538)
in the change above, and it was a **wrong answer**, not a slow one. Reproduced
before fixing:

> prev: `min ½x² s.t. x == 1` → `x = 1`
> new:  `min x²  s.t. x ≤ 2` (`H` changed, so the guard declines)
>
> | route | status | `x` | objective |
> |---|---|---|---|
> | parametric | `Optimal` | **−1e19** | **1e38** |
> | cold | `Optimal` | 0 | 0 |

The dimensional check the change shipped with is necessary but not sufficient.
`ConsStatus::Equality` and `BoundStatus::Fixed` are not "active" markers — they
are assertions about the *problem's* bound topology (`bl == bu`, `xl == xu`),
and they carry the semantics that follow: always in the working set, multiplier
unrestricted in sign, and a drop score hard-wired to `0.0` — never dropped.
Carried onto a problem where the row is a range, that is a false statement
nothing downstream can retract. `pin_working_set` pins an `Equality` row to
`qp.bl[i]`, which for the new problem is the `-1e20` infinity sentinel; the
resulting iterate is feasible for `x ≤ 2`, so the M5 audit passes it; no drop
test can reconsider it; and the solve reports `Optimal`.

The same shape reproduces through `Fixed` when a previously-pinned variable is
freed (`x = [−0, −1e19]`), which the report predicted and which the fix covers.

**Why the audit does not catch it.** `audit_and_repair` checks feasibility, and
`x = −1e19` *is* feasible. The point is optimal for the problem the working set
describes; it is the working set that describes the wrong problem. No
feasibility-based check can see that.

**The distinction that matters.** `AtLower` pinned to a lower bound the new
problem does not have lands on the same sentinel — but it is *droppable*, so
the dual ratio test sees a multiplier of the wrong sign and removes it. The
damage is iterations. Only `Equality` and `Fixed`, which no drop test can
reconsider, convert a bad pin into a wrong answer. That is the whole reason the
first two rules of the fix are load-bearing and the rest are hygiene.

**The fix.** `WorkingSet::reconciled_with(qp, opts)` re-derives both
topology-dependent statuses from the new problem and drops any remaining status
that names a bound the new problem does not have, matching the predicates the
solver uses when it builds a working set itself (exact `bl == bu` for rows,
both-finite-within-`feas_tol` for bounds). What survives is the part of a
working set that is only ever a guess about which constraints bind — the part
that costs at most iterations when it is wrong.

Four regression tests in `crates/pounce-qp/tests/homotopy.rs`. Two of them
(`Equality` → range, `Fixed` → free) fail against the unfixed change; the other
two pin rules that currently cost iterations rather than correctness, and their
doc comments say so rather than implying more coverage than they have. The
step-1 measurements are unchanged — reconciliation is a no-op when the topology
matches, which is every row of the sweep.

### The same hazard exists in `solve_with_working_set` itself

Worth recording separately, because the fix above does **not** address it.
`solve_with_working_set` is public API and applies no such reconciliation: a
caller who passes a working set from a problem whose bound topology has changed
gets the `Optimal`-at-`-1e19` behaviour directly. What the change above did was
make `solve_parametric` reach that hazard *on its own*, from two well-formed
problems and no caller mistake — so fixing it there was the regression fix, not
the whole problem.

The in-tree caller that matters is the active-set SQP driver, which is safe by
construction: an SQP linearization moves `A` and translates the row bounds, but
a constraint that is an equality stays an equality across iterations, and a
fixed variable stays fixed. That is why this has never been observed there, and
also why it is not a reason to relax about it — it is an unstated precondition
holding by accident of the caller, not by contract.

Options, none of them taken here: document the precondition on
`solve_with_working_set`; make it reconcile internally (a trajectory change for
the SQP driver, so it needs its own A/B on `benchmarks/warmstart`); or promote
`reconciled_with` to public API so callers can opt in. The middle one is
probably right, and it is the kind of change that should be measured rather than
assumed.

---

## Step 2, declined

Step 1 made rejection cheap, which was supposed to make the guard #602 asks for
affordable. It was implemented — `same_a` (structure and values), `same_box`,
`same_inertia`, alongside the existing `same_h` — measured, and **backed out**.

The measurement is the reason. Because the verdict looked size-dependent, the
instrument gained a size dimension, and that is what killed the guard. Below,
working-set changes with the guard off (what shipped before) against on, over
every row it changes, at two sizes of the same family:

| size | row | before | after | cold | |
|---|---|--:|--:|--:|---|
| n=20 | A 0.02 | 2 | 5 | 12 | worse |
| n=20 | A 0.05 | 2 | 3 | 12 | worse |
| n=20 | A 0.10 | **2** | **34** | 9 | worse |
| n=20 | A 0.20 | 4 | 31 | 9 | worse |
| n=20 | A 0.30 | 5 | 30 | 9 | worse |
| n=20 | A 0.40 | 5 | 32 | 9 | worse |
| n=20 | A 0.50 | 8 | 32 | 9 | worse |
| n=20 | A 0.60 | 7 | 30 | 9 | worse |
| n=20 | A 0.80 | 32 | 29 | 11 | better |
| n=20 | A 1.00 | 30 | 29 | 11 | better |
| n=20 | xu = 2 | 2 | 5 | 51 | worse |
| n=20 | xu = 0.5 | 47 | 45 | 54 | better |
| n=20 | xu = 0.1 | 46 | 44 | 64 | better |
| n=20 | inertia | **2** | **5** | 12 | worse |
| n=30 | A 0.02 | 3 | 3 | 18 | — |
| n=30 | A 0.05 | 65 | 64 | 82 | better |
| n=30 | A 0.10 | 5 | 4 | 19 | better |
| n=30 | A 0.20 | 6 | 5 | 19 | better |
| n=30 | A 0.30 | 8 | 7 | 19 | better |
| n=30 | A 0.40 | 53 | 52 | 16 | better |
| n=30 | A 0.50 | **54** | **9** | 16 | better |
| n=30 | A 0.60 | 53 | 52 | 17 | better |
| n=30 | A 0.80 | 55 | 54 | 71 | better |
| n=30 | A 1.00 | 45 | 45 | 61 | — |
| n=30 | xu = 2 | 4 | 3 | 97 | better |
| n=30 | xu = 0.5 | **77** | **7** | 94 | better |
| n=30 | xu = 0.1 | 77 | 75 | 103 | better |
| n=30 | inertia | 4 | 3 | 18 | better |

**16 better, 10 worse, 2 unchanged** — and the split is not noise, it is the
size: at `n = 30` the guard is better or equal in 14 of 14, at `n = 20` it is
worse in 9 of 14. Same family, same generator, same perturbations. `A 0.10` at
`n = 20` goes from 2 working-set changes to 34.

That is [#434](https://github.com/jkitchin/pounce/issues/434)'s situation
restated: a rule that fires on the losses and the gains alike, whose apparent
success depends on which instances you happened to measure. #434's standard —
"if none does, this issue should be closed rather than shipping a guess; the
failure mode of a bad threshold is giving back more than it recovers, silently"
— applies unchanged, and this guard does not meet it.

The **inertia** condition is the clearest single argument, because it is the one
with no upside available even in principle. `hessian_inertia` is not read by the
tracer or by `factorize_with_inertia_control`, so declining on it cannot improve
a path — it can only replace one with a fallback. Measured: 2 working-set
changes become 5, in exchange for nothing. A guard justified as "hygiene" that
costs 2.5× is not hygiene.

### What the reasoning got wrong

The argument for step 2 was that the path does not *model* `A` or the box, so
tracing it is unjustified extrapolation. That part is true and the measurement
does not touch it. What does not follow is that declining is therefore better:
rejection is not a return to correctness, it is a switch to a *different
heuristic* — the working-set hint — which has its own failure mode. The guard
only pays when the hint is the better of the two, and nothing here predicts
when that is.

Put another way: the choice is not "trace a path that models the change" versus
"trace one that doesn't". It is between two guesses at the new active set, and
#602's premise — that the modelled one must be better — is what the data
declines.

### What would settle it

The discriminator #434 also looked for and did not find: something observable at
runtime that says whether the previous active set is a good guess *for this
problem*. The handoff violation is monotone in `‖ΔA‖` and so is a candidate, but
it separates the good rows from the bad only loosely (§ above). Anything else
needs the Maros-Mészáros sweep and `benchmarks/warmstart`, not this family.

Recorded in `solver.rs` at the guard itself, so the next reader who has the same
idea finds the measurement before writing the code.

## Step 1's own losses, which the size sweep also exposed

The same size dimension put a number on where the step-1 fallback loses, which
the single-size measurement could not. Over the `H`-perturbed block — the one
where the fallback is the whole behaviour — working-set changes, fallback
against the cold solve it replaced:

| size | row | ws fallback | cold |
|---|---|--:|--:|
| n=20 | H 0.01 | **5** | 12 |
| n=20 | H 0.01 + A 0.1 | 34 | **9** |
| n=20 | H 0.10 | **5** | 12 |
| n=20 | H 0.10 + A 0.1 | 34 | **9** |
| n=20 | H 0.50 | **5** | 12 |
| n=20 | H 0.50 + A 0.1 | 35 | **12** |
| n=30 | H 0.01 | **3** | 18 |
| n=30 | H 0.01 + A 0.1 | **4** | 19 |
| n=30 | H 0.10 | **3** | 18 |
| n=30 | H 0.10 + A 0.1 | **67** | 84 |
| n=30 | H 0.50 | **2** | 17 |
| n=30 | H 0.50 + A 0.1 | **65** | 82 |

**9 wins, 3 losses**, and the pattern is legible: the hint is good when the
previous problem is genuinely close, and bad at `n = 20` when `A` also moved —
i.e. exactly when the previous active set is not a good guess. Net positive with
large wins (18 → 3) and real losses (9 → 34).

Step 1 stays: its losses revert to the cold solve it replaced, which is not
uniformly better either, and the wins are the larger effect. But this is a
measured regression on a subset, and per `CLAUDE.md` that needs an issue and an
owner rather than a line in a commit message — it is the same open question as
the declined guard above, and as #434, seen from a third angle: **there is no
runtime signal for "is the previous active set worth keeping".** All three want
the same missing thing.

---

## Step 3, implemented and measured

The homotopy had no way to say "an inactive variable bound became active". The
`Event` enum carried `AddRowLower/AddRowUpper/DropRow/DropBound`, and the primal
ratio test looped `for i in 0..m` over general rows only, so `x(t)` crossed
inactive bounds with nothing capping the step — on the **cold** arm as much as
the warm one — and `worst_path_violation` skipped the box, so nothing reported
it either. This adds `AddBoundLower` / `AddBoundUpper`, the matching ratio test
over `j in 0..n`, a `tabu_bounds` mirroring `tabu_cons` (the rank repair can now
fight the new test the way it did for rows), and the box to the feasibility
report.

The bound test is simpler than the row test because variable bounds do not move
along this path: the bound's own rate is zero, so `dx` alone governs.

### Instrument

`benchmarks/warmstart`, whose `-hom` arms differ from their twins by exactly one
option (`sqp_qp_use_homotopy`) — the same instrument #434 used. 42 family×scale
combinations, 8 arms, run against the Python bindings built from this tree. The
metric is `n_qp_ws_changes`, inner-QP working-set changes summed over every
step, which is what #434 reported.

The Maros-Mészáros corpus is **not** available in this container
(`BENCH_DATA_ROOT` unset), so `pounce-convex`'s driver is not covered. See the
gap at the end.

### Result

| arm | ws changes, base → new | solved |
|---|---|---|
| `cold-ipm`, `cold-qp-ipm`, `warm-ipm`, `warm-qp-ipm` | 0 → 0 | unchanged |
| `cold-sqp` | 13038 → **13038** | unchanged |
| `warm-sqp` | 1009 → **1009** | unchanged |
| `cold-sqp-hom` | 36726 → **28487** (−22%) | unchanged |
| `warm-sqp-hom` | 2240 → **2005** (−10%) | unchanged |

The non-homotopy arms are **bit-identical**, which is the control this needed:
the change is confined to the tracer. Solved counts are unchanged everywhere
(855/855 SQP, 549/549 QP), so nothing traded correctness for speed.

In #434's framing — the homotopy's excess inner work over the conventional path
— this recovers about a quarter of it:

| | base | new |
|---|---|---|
| cold, homotopy ÷ conventional | 2.82× | **2.18×** |
| warm, homotopy ÷ conventional | 2.22× | **1.99×** |

### The distribution, which the aggregate hides

| arm | rows better | rows worse | unchanged |
|---|--:|--:|--:|
| `cold-sqp-hom` | 4 | 16 | 22 |
| `warm-sqp-hom` | 9 | 9 | 24 |

And the net is one family:

| arm | net | `mpc_horizon_80` | every other family |
|---|--:|--:|--:|
| `cold-sqp-hom` | −8239 | **−10580** | **+2341** |
| `warm-sqp-hom` | −235 | −546 | +311 |

`mpc_horizon_80` drops roughly 4× (`4749 → 1229`, `5115 → 1605`, `4708 → 1158`
across scales). Everything else gets modestly worse.

Two readings, and both are partly right. Some of the "worse" is real. But some
of it is an artefact of the metric: the path now *takes* pivots it previously
skipped by walking through the bound, and every one of those counts as a
working-set change. A path that walks through a bound is cheap per step and
wrong about the active set; the corrector then pays for it, off this metric.

Wall-clock is the honest check on that, and it is **flat**: 41.8 s → 42.7 s
(`cold-sqp-hom`), 3.9 s → 4.2 s (`warm-sqp-hom`). So outside `mpc_horizon_80`
this buys correctness of the path, not speed.

That `mpc_horizon_80` is where the win lands is not a coincidence worth
shrugging at: MPC problems are mostly box constraints, which is exactly the
structure the missing events were blind to, and MPC is the workload §4.2 cites
as the reason for a parametric active-set method at all.

### Fixture sweep

* default arm: **empty diff**
* `algorithm=active-set-sqp`: **empty diff** (the option defaults to homotopy
  off, so this arm does not reach the tracer)
* `algorithm=active-set-sqp sqp_qp_use_homotopy=yes`: **one line moves**

```
- jit1   SearchDirectionBecomesTooSmall  it=8  obj=170563.5973
+ jit1   MaximumIterationsExceeded       it=2  obj=165557.7682
```

Explained: `jit1` does not solve on the active-set-SQP path in *any*
configuration. Homotopy off it is `MaximumIterationsExceeded it=0`, before and
after this change alike (that arm's diff is empty). It solves only on the
default IPM path (`SolveSucceeded it=24`). So the moving line is a fixture that
fails on this path either way changing *which* failure it reports — not a
capability regression. Worth stating plainly rather than leaving as an
unexplained diff, since an unexplained moving line is how #544 shipped.

### The gap, which matters before release

`pounce-convex`'s active-set QP driver sets `use_homotopy: true` — it is the one
shipped path where the tracer runs **by default**, and it is precisely the path
this measurement does not cover, because the Maros-Mészáros corpus is not
available here. #434's own numbers came from that corpus and are the reason the
homotopy's cost profile is known at all.

So: the SQP evidence above is real and positive, and the convex-QP evidence does
not exist yet. Running
`cargo run -p pounce-convex --release --example homotopy_sweep` over the 138
problems, homotopy-on before and after, is the missing step, and it should
happen before this reaches a release rather than after.

---

## The runtime signal: built, measured, declined

Three decisions in this crate wanted the same missing thing, and the note says
so three times: #434's abandon-the-path guard, step 1's fallback choice, and
step 2's declined guard all reduce to **"is the previous active set worth
keeping?"** So: build it and find out.

#434 refuted the obvious *predictor* — `n_eq / n`, computable from problem data
with no solve — and concluded that a discriminator, if one exists, has to be
**measured rather than predicted**. The candidate measurement here is the
cheapest honest one available:

> Pin the hinted active rows, then count the rows and bounds *outside* the hint
> that the pinned point violates.

That is a property of the hint applied to the target problem, so it needs no
model of what changed between the two problems. It costs one pinned-KKT
factorization — which `solve_with_working_set` already pays, so a caller that
goes on to keep the hint pays nothing extra for having asked. Implemented as
`ParametricActiveSetSolver::hint_pin_quality`; the sweep is
`crates/pounce-qp/src/tests/hint_signal.rs`:

```text
cargo test -p pounce-qp --release --lib hint_signal -- --ignored --nocapture
```

360 pairs — 5 sizes from `n = 12` to `n = 50`, crossed over `H`, `A`, `g`, row
bound and box perturbations — scored against ground truth (the working-set
changes each route actually took). Both arms are swept, because the fallback's
opponent depends on the caller's options: `pounce-qp` defaults `use_homotopy`
off, `pounce-convex`'s driver turns it on, and they are very different
opponents.

### Arm 1 — conventional cold (`pounce-qp`'s default): nothing to decide

| policy | total working-set changes |
|---|--:|
| always keep the hint | **17497** |
| always cold | 22338 |
| oracle (per-case best) | 17488 |

"Always keep the hint" is wrong on **2 of 360** cases, and the oracle beats it
by **9 changes out of 17497 — 0.05%**. There is no prize here. Every threshold
tested, on either normalization, lands at or above 17497.

That is a real result rather than a null one: it says step 1's unconditional
"keep the working set" is not a heuristic awaiting a guard, it is very close to
optimal on this arm, and a signal would be machinery in front of a decision that
does not need making.

### Arm 2 — homotopy cold (`pounce-convex`'s default): a real prize the signal cannot reach

| policy | total working-set changes |
|---|--:|
| always keep the hint | 17497 |
| always cold | 19062 |
| **oracle (per-case best)** | **12693** |

Here "always keep" is wrong on **158 of 360**, and the oracle is 27% below it.
The prize is real. No rule on this signal reaches any of it:

| rule | cost | | rule | cost |
|---|--:|---|---|--:|
| `violated/active ≤ 0.2` | 17613 | | `violated ≤ 5` | 17787 |
| `violated/active ≤ 0.3` | 17833 | | `violated ≤ 8` | 18088 |
| `violated/active ≤ 0.5` | 19680 | | `violated ≤ 12` | 18396 |
| *always keep* | **17497** | | *always keep* | **17497** |

**Every threshold is worse than not having a rule at all.** Best error count is
156 of 360 against a 158 baseline — two cases, i.e. noise.

### Why, and it is not a tuning problem

The refutation is direct: samples carrying an *identical* signal disagree about
the answer, so no rule reading only that signal can be right about both.

```text
collision: active=20 violated=1 — same signal, opposite answers
   n30 dh0   da0     dg0.15 xu-     ws  3  cold 21   keep
   n30 dh0   da0.05  dg0.15 xu-     ws 64  cold 22   go cold
```

Same hint size, same number of violated rows, and the right answers differ by
20× in one direction and 3× in the other.

The structural reason is worth stating, because it also says what *would* work.
The signal measures **"is the hint good?"**. The decision is **"is the hint
better than the alternative?"** — and on arm 2 the alternative is the homotopy,
whose cost is a property of the *path*: how many events it hits between `t = 0`
and `t = 1`. The hint says nothing about that. A hint-quality signal is
answering a different question from the one being asked, and the collision above
is what that looks like in data.

Which is exactly why #434 wanted a guard that *measures the path*, and it
remains the shape with a live prize: 27%, on the one arm that ships the tracer
by default. #434 replayed candidate path rules against recorded trajectories and
declined them for want of margin; this note adds that the hint side is not the
missing half either.

### What ships

The instrument and the negative result — no rule. `hint_pin_quality` is
`#[cfg(test)]`, so nothing dead ships, and the next person who has this idea
finds the sweep and the collision table rather than only the idea.

That is now three declined guards in this area — #434's, #602 step 2's, and this
one — and they are consistent rather than merely discouraging. The discriminator
is not in the problem data (#434), not in the change between the two problems
(step 2), and not in the hint (here). What is left is the path itself, measured
while it runs, which is where the remaining 27% lives.

---

## PR review: the topology hazard had a second entrance, and a third

[@GermanHeim's review of #614](https://github.com/jkitchin/pounce/pull/614#issuecomment-5304670936)
found that the `reconciled_with` fix above was incomplete, and he was right on
both counts he raised.

**1. The traced path never reached the reconciliation.** `reconciled_with` was
applied only on the *declined* branch. When `H` is identical the guard
**admits**, so `solve_homotopy`'s warm arm clones `sol_prev.working` as-is and
hands it to the corrector at `t = 1`, still claiming `Equality`. Measured, with
`H` bit-identical:

| case | `use_homotopy` | parametric | cold |
|---|---|---|---|
| equality → range | false | **`Optimal`, x = −1e19** | x = 0 |
| equality → range | true | x = 0 | x = 0 |
| fixed → free | false | **`Optimal`, x = −1e19** | x = 0 |
| fixed → free | true | **`Optimal`, x = −1e19** | x = 0 |

Three of four wrong — worse than the original report, and invisible to the
first round of tests because every one of them perturbed `H` and so only ever
exercised the fallback.

The reason the path cannot be patched into handling this: **the row type does
not interpolate.** A row that is an equality at `t = 0` is a *range* at every
`t > 0` (the lower bound's rate is zero when the new lower bound is infinite,
so the interpolated row is `[1, 1+t]`). So `Equality` is correct only at the
single point `t = 0`, the tracer has no mechanism to re-type it, and no drop
test can either. That makes it a change the path genuinely cannot model —
exactly the same category as `H`.

Fixed by a `same_topology` condition on the guard: equality-ness of every row
and fixed-ness of every variable must agree between the two problems, using the
predicates the solver itself uses. Declining routes the pair to the fallback,
which reconciles.

**This is a correctness guard, and that distinction is the whole reason it
stands where the `A` / box guards fell.** Step 2 was declined because it traded
one heuristic for another with no reliable winner — a cost question. This one
prevents a wrong answer. They are not the same decision and should not be
weighed the same way. On the CLI fixture sweep the topology guard is inert on
all three arms, since `solve_parametric` has no production caller.

**2. `solve_with_working_set` accepts impossible statuses from any caller.**
Recorded above as known-and-not-fixed, on the reasoning that fixing it is a
trajectory change for the SQP driver and wants its own A/B. He asked for it
anyway, so it got the A/B:

| | base → new |
|---|---|
| `cold-sqp`, `warm-sqp`, all IPM arms | **bit-identical** |
| `cold-sqp-hom` | 28487 → **27828** |
| `warm-sqp-hom` | 2005 → **1755** (−12%) |

Solved counts unchanged; every per-family move is an improvement. On the CLI
fixture sweep five lines move, all on fixtures already failing — and one stops
failing: `jit1` on the homotopy arm goes from `MaximumIterationsExceeded` to a
converged solve in 11 iterations (dual infeasibility `9.3e-9`, constraint
violation `6.5e-19`, objective 173346.0967 against the IPM path's 173345.3768 —
two independently converged KKT points 4e-6 apart).

So the prediction that it would move the SQP trajectory was right, and the
inference that this made it too risky to include was wrong: it moves it in the
right direction, and the arms where the driver actually lives are bit-identical.
The residual difference comes from hints where a variable's bounds sit within
`feas_tol` of each other without being exactly equal, which reconciliation
promotes to `Fixed`.

Three regression tests, all verified failing on the pre-fix code:
`traced_path_survives_an_equality_becoming_a_range`,
`traced_path_survives_a_fixed_variable_being_freed` (both across
`use_homotopy ∈ {false, true}`), and
`solve_with_working_set_reconciles_a_stale_hint`, which passes the stale hint
straight into the public entry point as an external caller would.

### What this says about the first fix

The original `reconciled_with` change treated the symptom at the point it had
been observed rather than at the point the hazard enters. The hint reaches the
solver by three routes — the declined fallback, the traced path, and the public
entry point directly — and only the first was covered. Worth remembering next
time a fix is derived from a single reproduction: the reproduction picks the
route, and the route is rarely the whole surface.
