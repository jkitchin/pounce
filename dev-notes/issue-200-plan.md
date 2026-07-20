# Issue #200 — plan: stop certifying optimality behind an extreme objective scale

Status: **implemented**, with two corrections to the plan below that were
forced by measurement. See §8 — read it before §3, which is superseded in part.

## 1. Problem recap (verified against current code)

On `quartc` / `dqrtic` the initial objective gradient max is ~4e12, so
gradient-based scaling (`scale_gradient_based`,
`crates/pounce-nlp/src/orig_ipopt_nlp.rs:833-875`) computes
`df = nlp_scaling_max_gradient / max_grad_f ≈ 2.5e-11` and floors it at
`nlp_scaling_min_value = 1e-8`. The strict convergence test
(`OptErrorConvCheck::check_convergence_with_state`,
`crates/pounce-algorithm/src/conv_check/opt_error.rs:208-237`) gates on the
**scaled** aggregate `nlp_err ≤ tol` plus the pounce#173 per-component
**unscaled** gates. Those per-component gates do not save us because the
default `dual_inf_tol = 1.0` and quartc's unscaled dual infeasibility at the
false stop is **0.84** — under the gate, 8 orders of magnitude above `tol`.
Result: `Solve_Succeeded` at obj 248.88 while the true minimum (≈0) is
reachable — `nlp_scaling_method=none` follows the *same trajectory*
(constant obj scale cancels from the Newton step and the Armijo/filter tests
are scale-invariant) and reaches obj 6.8e-9 at iter 39.

## 2. What has already been tried and must NOT be repeated (issue comment, empirical)

1. **Judge the aggregate on `curr_unscaled_nlp_error`** → effectively
   tightens `tol` by `1/obj_scale`; `hs1`/`hs38` (obj_scale ≈ 0.04) regress
   to `Search_Direction_Becomes_Too_Small`. Rejected.
2. **Require unscaled aggregate ≤ `acceptable_tol` for the strict
   certificate** → fixes quartc/dqrtic/penalty1 but downgrades ~19
   correctly-converged problems (`avion2`, `dallasl`, `dallasm`, `denschnd`,
   `lakes`, `meyer3`, `sawpath`, `steenbrc`, …). Rejected.

The discriminating signal is the **scale factor itself**: the false
certificates all have `obj_scale` at/near the 1e-8 floor; the collateral
problems sit around 1e-2.

## 3. Decision

Owner preference (2026-07-19): *"the right outcome, or the one with least
surprise — we should be able to actually solve it."* A pure
downgrade-at-termination (the issue comment's minimal direction) still stops
at obj 248.88, just relabeled. Since the iterates provably continue to the
true minimum when the stop is refused, the chosen design is:

**Masked-certificate veto with continuation, plus a relabel backstop.**

> **Correction (see §10.1).** The "never worse" guarantee below is stated
> conjunctively ("status no worse AND objective no worse"). That reading is not
> satisfiable and never was: on the cross-status upgrade paths the mechanism
> returns a *better status at a higher raw objective*, and the two objectives sit
> at different constraint violations, where raw comparison is not meaningful. The
> guarantee holds under a **lexicographic, status-dominant** reading, with the
> objective clause claimed only within an equal-status pair. Read §10 before
> relying on anything in this section.

Define, at the convergence check:

```
masked := obj_scale < obj_scale_certificate_threshold   // new option, default 1e-4
          && curr_unscaled_nlp_error() > acceptable_tol // 1e-6 default
```

- When the strict test (`passes_component_tols`) passes but `masked` is
  true: **do not return `Converged` — continue iterating** (log a one-shot
  diagnostic). On quartc/dqrtic the run then proceeds exactly like the
  `nlp_scaling_method=none` run and terminates strict-`Converged` at the
  true minimum (~15 extra iters), because near the minimum the unscaled
  error falls below `acceptable_tol` and the veto lifts.
- Apply the same veto to the **acceptable-level termination counter** in
  `check_convergence_with_state` (both branches), so the run doesn't just
  swap the false Optimal for a false-ish Acceptable at a nearby wrong point.
  Do **not** veto acceptable-point *storage*
  (`current_is_acceptable_with_state` / `store_acceptable_point`,
  `ipopt_alg.rs:623-631`) — the stashed point is the rollback target.
- **Backstop (guarantees never-worse):** if a veto fired during the run and
  the run later terminates without a strict certificate (tiny step,
  max_iter, restoration failure), restore the stored acceptable point
  (`restore_acceptable_point`) and return
  `StopAtAcceptablePoint`/`Solved_To_Acceptable_Level` — i.e. today's point
  with an honest label. Additionally extend `apply_kkt_fidelity_gate`
  (`crates/pounce-algorithm/src/application.rs:912-932`) with a default-on
  masked-condition relabel (same `masked` predicate at the final point) so
  the SQP path and any missed IPM exit path can never report
  `Solve_Succeeded` at a masked point. Keep the existing opt-in
  `kkt_fidelity_tol` untouched.
- **Opt-out:** `obj_scale_certificate_threshold = 0` disables the whole
  mechanism → bit-for-bit Ipopt-faithful behavior (Ipopt itself reports
  quartc at 248.88 as optimal). Register the option in
  `upstream_options.rs` next to `kkt_fidelity_tol` and document the
  deliberate deviation the same way pounce#173 is documented.

Why this respects the earlier negative results: the veto only fires when
`obj_scale < 1e-4` (quartc/dqrtic/penalty1 sit at ~1e-8; hs1/hs38 at ~4e-2;
the 19 collateral problems at ~1e-2), and even then the strict tolerance in
scaled space is unchanged — we only refuse to *stop early*, we never demand
more precision than `acceptable_tol` in unscaled space to lift the veto.

## 4. Implementation steps

1. **Plumb the inputs.** `IpoptCq` already exposes
   `curr_unscaled_nlp_error()` (`ipopt_cq.rs:834`) and the nlp handle has
   `obj_scaling_factor()` (used at `ipopt_cq.rs:784`). Add a small cq
   accessor for the obj scale if one doesn't exist. Extend
   `OptErrorConvCheck` with the threshold field + a `veto_fired: bool`.
2. **Veto in `check_convergence_with_state`** (strict branch and
   acceptable-counter branch). Pure-helper form
   (`fn certificate_masked(obj_scale, unscaled_err) -> bool`) so it unit
   tests without a full cq, matching the existing style
   (`passes_component_tols`).
3. **Failure-path fallback.** In `ipopt_alg.rs`, on
   `MaxIterExceeded` / tiny-step / restoration-failure exits, if
   `veto_fired` and an acceptable point is stored, restore it and return
   `StopAtAcceptablePoint`. Verify which of these paths already call
   `restore_acceptable_point` (restoration failure does, `ipopt_alg.rs:1427-1439`;
   check the tiny-step and max-iter paths) and wire only what's missing,
   gated on `veto_fired` so non-veto behavior is unchanged.
4. **Backstop relabel** in `apply_kkt_fidelity_gate` (rename or add a
   sibling fn): default-on masked-condition downgrade
   `Solve_Succeeded → Solved_To_Acceptable_Level`, one-shot
   `tracing::info!` diagnostic naming `obj_scale`, scaled and unscaled KKT
   error, and the option that disables it.
5. **Option registration + docs**: `upstream_options.rs`, options doc page,
   CHANGELOG entry (this is a behavior change vs Ipopt — say so explicitly,
   citing #200), and a paragraph in the scaling section of the docs.

## 5. Tests (all in-repo, no benchmark data needed)

- Pure-helper unit tests in `opt_error.rs`: veto fires/does-not-fire around
  both thresholds; `0` disables; veto blocks strict AND acceptable
  termination; storage still happens.
- Synthetic quartic integration test (new, `crates/pounce-algorithm/tests/`):
  `f = Σ_{i=1..n} (x_i − i)^4`, `x0 = 2`, `n = 1000` (max_grad_f ≈ 4e9 →
  df ≈ 2.5e-8 < 1e-4). Assert: default options now reach final unscaled
  obj < 1e-6 with `Solve_Succeeded`; with
  `obj_scale_certificate_threshold=0` the old false stop returns (guards the
  opt-out).
- Stall fallback test: same quartic but `max_iter` capped below the true
  convergence iteration → expect `Solved_To_Acceptable_Level` at the stored
  point, not `Maximum_Iterations_Exceeded` at a worse one (and never
  `Solve_Succeeded`).
- Existing HS suite (`hs1`, `hs38`, `optimize_hs71.rs`, …) must stay green
  with identical statuses — they are the recorded regression victims of the
  rejected designs.

## 6. Benchmark validation (needs local .nl data — likely jkitchin's machine)

The 733-problem Vanderbei + Mittelmann sweeps need `POUNCE_BENCH_DATA`
(~2 GB, gitignored; `benchmarks/Makefile`). If the implementing session
lacks the data, deliver the code + tests and hand this checklist back:

```
make -C benchmarks vanderbei-rerun mittelmann-rerun benchmark-report
```

Acceptance criteria:
- `quartc`, `dqrtic` now finish at obj ≤ 1e-6 with `Solve_Succeeded`;
  `penalty1` reaches its true 0.0097 (it did under rejected design #2, so it
  should here too — verify).
- **Zero** problems previously `Solve_Succeeded` at the *correct* objective
  lose the strict status (the 19-problem collateral list from design #2 is
  the canary: `avion2`, `dallasl`, `dallasm`, `denschnd`, `lakes`, `meyer3`,
  `sawpath`, `steenbrc` must be untouched). If any of them has
  `obj_scale < 1e-4`, lower the threshold (1e-6) rather than accept
  collateral, and re-run.
- Total sweep wall time within ~5% (veto-driven continuation is bounded by
  `max_iter` but should only trigger on the handful of masked problems —
  log and count veto firings across the sweep).
- Byte-compare the disagreement list before/after; the only diffs should be
  the intended flips.

## 7. Close-out

- Update #200 with the benchmark table and close it when the acceptance
  criteria hold.
- If the benchmark pass forces a threshold change, record the measured
  `obj_scale` distribution in this file for posterity.


## 8. Corrections found during implementation (2026-07-19)

Two things in §3 did not survive contact with the data. Both are recorded here
so they are not re-derived.

### 8.1 The `obj_scale` gate does **not** separate the collateral — §3's premise is wrong

§3 asserts the false certificates sit at the 1e-8 scale floor while the
collateral sits around 1e-2, and prescribes "lower the threshold to 1e-6" if a
canary trips. Measured objective scales on the Vanderbei suite:

| problem | obj_scale | baseline status |
|---|---|---|
| quartc, dqrtic, penalty1 | **1e-8** | falsely `Optimal` |
| **meyer3** | **1e-8** | correctly `Optimal` |
| lakes | 1.8e-4 | correctly `Optimal` |
| avion2, dallasm | ~1e-3 | correctly `Optimal` |

`meyer3` sits at *exactly* the same floor as the problems being fixed, so **no
`obj_scale` threshold can separate them** and §3's prescribed remedy cannot
work. A scale-only veto downgrades `meyer3` and `hs084` from `Solve_Succeeded`,
violating §6's zero-strict-loss criterion.

### 8.2 A second threshold on the unscaled error was tried and rejected

The obvious repair is a second bar on the unscaled KKT error, since the two
groups do separate on it (a 36x gap: `dqrtic` 3.5e-1 … `vardim` 9.9e-3), and a
cutoff at the gap's geometric centre (5e-2) does give zero status losses.

**Rejected, for two reasons:**

1. `unscaled_err` is a **dimensional** quantity — it carries the units of the
   objective gradient. The same problem with its objective multiplied by 100
   moves across any absolute cutoff. A scale-sensitive threshold is the wrong
   shape of fix for a bug that is about scaling.
2. The cutoff would be fitted to 16 problems from one suite, 5 on one side and
   11 on the other. Nothing predicts where a new problem lands.

It also failed on its own terms: one bar cannot serve both roles. Engaging at
5e-2 is what spares `meyer3`, but *releasing* at 5e-2 stops the rescued run far
too early — `quartc` halted at objective 1.92 instead of 8.8e-7, because a
quartic's unscaled error drops under 5e-2 long before its objective reaches the
minimum. A hysteresis band (engage at 5e-2, release at `acceptable_tol`) fixes
that but keeps both objections above.

### 8.3 What was implemented instead: test the hypothesis, don't predict it

Whether a stop is genuinely false **cannot be read off the residuals**. So the
veto does not try. It fires on the scale condition alone (§3's original
predicate), refuses to stop, and *continues*:

- If the iterates go somewhere, the stop was false — the run reaches the true
  minimum and issues an honest certificate.
- If they do not, the refused point is restored and returned **with the status
  it would originally have had** (`Solve_Succeeded`, via `terminate_vetoed_or`),
  because that point had already passed the strict test.

The second branch is what makes this safe without any fitted constant: trying
and failing costs iterations, never correctness, so the mechanism is
**never worse than not having it, by construction**. `meyer3` and `hs084` come
back byte-identical in status without any threshold tuned to them.

Measured outcome (baseline = `obj_scale_certificate_threshold=0`):

| problem | baseline | after | status |
|---|---|---|---|
| quartc | 248.88 | **8.78e-07** | unchanged (`Optimal`) |
| dqrtic | 39.36 | **7.03e-07** | unchanged |
| penalty1 | 6.44 | **0.0097** (true) | unchanged |
| denschnd | 2.22e-04 | 3.18e-10 | unchanged |
| vardim | 2.46e-09 | 8.06e-30 | unchanged |
| porous1 | 1.43e-08 | 7.04e-17 | unchanged |
| meyer3, hs084, avion2, dallasl, dallasm, lakes, sawpath, steenbrc, … | — | — | unchanged |

Zero status changes across all 16 gate-eligible problems and the full §6 canary
list. The only surviving threshold is `obj_scale_certificate_threshold` on the
scale factor itself, which is dimensionless and whose 1e-8 floor is a
documented clamp (`nlp_scaling_min_value`), not a fitted value.

### 8.4 Also fixed in passing: the console hid the discrepancy

`print_solve_summary` passed the *scaled* residual to both the "(scaled)" and
"(unscaled)" columns, so `quartc` reported dual infeasibility `8.38e-09` twice
when the unscaled value is `8.38e-01`. A user auditing the suspicious
certificate was shown a report that agreed with it. The unscaled statistics
were already computed and already surfaced through the Python bindings; only
the console dropped them. Upstream Ipopt prints the unscaled value correctly,
so this was a porting defect, not a deviation.


## 9. Findings from adversarial review (2026-07-19)

Two independent reviews were commissioned specifically to attack the fix. Both
found defects that the fix's own tests had missed; all are fixed, and each is
now pinned by a test that fails when the fix is reverted.

### 9.1 Blocked acceptable-level termination had no safety net (REGRESSION)

The veto gates the acceptable-level branch on `!masked`, but the fallback that
undoes a refusal was armed only by `refusing_strict`. A run whose best available
outcome was `Solved_To_Acceptable_Level` therefore had its exit blocked with
nothing to catch it, and surfaced a bare failure:

```
f = A(x-a)^4 - K*sqrt(1+y^2)      (quartic pins obj_scale at 1e-8;
                                   the second term holds the unscaled dual
                                   infeasibility on a plateau above dual_inf_tol)
  baseline  Solved_To_Acceptable_Level  f=-6.83e14   40 iters
  veto      Maximum_Iterations_Exceeded             300 iters
```

Fixed: acceptable-level refusals are tracked separately (`acceptable_veto_fired`,
`vetoed_acceptable_iterate`) and restored as `StopAtAcceptablePoint` — claiming
`Success` for one would over-report. Since that termination is count-based, a
shadow counter follows the suppressed streak so the refusal is recorded at the
iterate where the baseline would really have stopped.

### 9.2 The unscaled accessors divided by a SIGNED scale factor (PRE-EXISTING)

`obj_scaling_factor` is signed (`-1` is the documented way to maximize) while
`curr_dual_infeasibility_max` / `curr_complementarity_max` are max-norms.
Dividing one by the other returned a **negative** "max-norm", which passes every
`<= tol` comparison trivially. Consequences:

- The veto was silently disabled on every maximization.
- More broadly `passes_component_tols` compares those same values against
  `dual_inf_tol` / `compl_inf_tol`, so the unscaled residual gate added for
  **pounce#173 was defeated on maximization** — a false certificate could pass
  the very check added to catch false certificates. This predates #200.

Verified on a concave quartic (`max g = -sum (x-a)^4`, optimum 0) against the
identical minimization: `2.27 -> 4.05e-8` (min) vs `-2.27 -> -2.27` (max).
Fixed by taking the magnitude; now symmetric.

### 9.3 The SQP path does NOT share the bug — the §3 backstop is unnecessary

§3 called for a default-on relabel in `apply_kkt_fidelity_gate` so no exit path
could report `Solve_Succeeded` at a masked point, and §8 dropped it. Measured
rather than assumed: the SQP path has no `ConvCheck` (so it cannot use the veto)
but does evaluate through `OrigIpoptNlp`, so exposure was an open question. On
the masked quartic whose minimum is 0:

```
SQP:  Solve_Succeeded  obj = 2.66e-5      (IPM unfixed stops at ~2.27)
```

It converges to essentially the right answer, so there is no false certificate
to relabel. Adding the backstop would have been actively harmful: the same
predicate that fires on `quartc` also fires on `meyer3`, so a relabel would
reintroduce precisely the collateral §8.1 removed. Pinned by
`sqp_path_behaviour_on_a_masked_objective_is_pinned`.

### 9.5 Post-optimal sensitivity after a fallback restore — NOT a defect

Reported here first as an open defect. **That was wrong**, and the correction is
recorded because the wrong reasoning is easy to repeat.

`PdSensBacksolver` reuses the solver's **held factor**, so restoring an earlier
iterate after the run has travelled on raises a fair question about whether the
factor still describes the returned point. The first probe appeared to answer
yes-there-is-a-problem: both arms returned a byte-identical `x` and objective,
and the sensitivity differed by nine orders of magnitude (`4.5e10` vs `9.7e19`).

**The probe was the problem.** Its objective carried a `−k·√(1+y²)` term, which
is unbounded below; the run drove `y` to `6.8e13` and stopped at a point that is
not a minimum. With no minimum the KKT system is singular along the unbounded
direction, so "sensitivity" there is an artifact of whichever inertia-correction
perturbation happened to be in force — the quantity is not defined, and the two
arms are under no obligation to agree.

On a **well-posed** masked problem (`Σ(xᵢ−a)⁴`, genuine minimum at `x = a`),
with the vetoed run cut off at the baseline's own iteration count so it must
fall back to exactly that point:

```
baseline (never vetoes)  Solve_Succeeded  sens = [13617951.376845466, 0]
fallback                 Solve_Succeeded  sens = [13617951.376845466, 0]
                                          max |diff| = 0
```

Bit-identical. Pinned by
`crates/pounce-sensitivity/tests/masked_certificate_sensitivity.rs`.

Two hypotheses were also eliminated along the way, by direct instrumentation
rather than argument:

- **The factorization is not stale.** Forcing a re-factorization at the restored
  point (via `compute_search_direction`, after clearing `delta` — it
  short-circuits when a regularization delta is pending) produces a genuine
  cache miss and rebuilds the factor, confirmed by instrumenting the 13-tag
  dependency cache in `pd_full_space_solver.rs`. It changed nothing on either
  problem, so it was **removed** rather than kept: it costs a KKT factorization
  on every fallback and buys nothing.
- **`curr_mu` is not the carrier** either, though restoring it is correct on its
  own merits and is kept (9.6).

### 9.6 Barrier parameter restored with the iterate

`curr_mu` lives on `IpoptData`, not in the `IteratesVector`, so a restore
rewound `x` but not `mu` while `stats.final_mu` is read afterwards — reporting
the continued run's barrier parameter beside the refused run's point. Fixed for
both the strict and acceptable-level snapshots. Latent rather than observable:
in every reachable fallback case `mu` has bottomed out at its floor, making the
two coincide (2.506e-9 measured in both arms).

### 9.4 Still open

All three items listed here previously have since been closed, and one new
defect (9.5) was opened by the same pass:

- **Wall time — CLOSED.** Bounded by `VETO_MAX_EXTRA_ITERS`; the 9.1 case went
  300 → 115 iterations. Same-code A/B spot check: controls (3000-iteration and
  large-dimension problems that never veto) show −1.0% … +1.3%, mean +0.26% —
  no detectable per-iteration cost. The affected set totals 11.7s of a 4413s
  suite, so whole-suite impact is well under 1%.
- **User scaling — CLOSED (was a real defect).** The gate read the *product*
  `df * user_obj_factor`, so a user who deliberately scaled a well-conditioned
  objective down was second-guessed: 13 → 18 iterations for no benefit. Now
  gates on the solver-computed `df` alone.
- **Fallback retry suppression — CLOSED, was never real.** `mu_strategy_fallback`
  and `l1_fallback_on_restoration_failure` give identical status and objective in
  both arms. A refused strict certificate is exactly what the baseline returns as
  `Success`, so neither arm retries; a refused acceptable-level one restores as
  `StopAtAcceptablePoint`, which still triggers the retry.
- **The §6 benchmark sweep — DONE.** 780 problems across both suites: zero
  strict-status losses, zero objective changes among still-strict runs, all
  targets met.
- **9.5 turned out not to be a defect** — see the correction there.

### 9.7 Configuration sweep (2026-07-20)

Every configuration that changes *when* or *how* a run terminates is a candidate
for an interaction the default-options fuzz cannot see. The ones that had been
named but never checked are now checked, each against the same never-worse
invariant (status never degrades, objective never worse) with a guard requiring
that the veto actually engaged:

| configuration | why it was suspect | cases / engaged | result |
|---|---|---|---|
| `hessian_approximation=limited-memory` | the scale-invariance argument is weakest here — curvature comes from accumulated gradient differences built on the pre-veto trajectory | 80 / 31 | holds |
| `mu_strategy=adaptive` | changes where the unscaled error crosses `acceptable_tol`, i.e. when the veto lifts | 80 / 49 | holds |
| `acceptable_iter=0` | disables acceptable-level termination, making the strict fallback the *only* rollback | 80 / 28 | holds |
| `kkt_fidelity_tol=1e-4` | acts on the same signal as the veto; the restored point has a large unscaled error by construction | 80 / 40 | holds |
| warm-start chain | `final_mu` feeds the next solve's `mu_init`; a latent inconsistency in what is carried across shows up in a chain, not a single solve | 60 / 6 | holds |

L-BFGS is worth singling out: it is the case the scale-invariance justification
covers least well, and it holds anyway — which is the point. Correctness does
not rest on that argument. The fallback restores the refused certificate however
the continuation behaves, so a weaker trajectory argument costs iterations, not
correctness.


## 10. Formal analysis (2026-07-20)

A state-machine analysis of the mechanism — as opposed to more testing — was
commissioned to either establish the never-worse invariant or exhibit a state
that breaks it. It exhibited one, and the defect is now fixed.

### 10.1 The invariant, stated properly

Because the veto's check is side-effect-free with respect to step computation,
the vetoed run's trajectory is bit-identical to the unvetoed run's on every
iteration both execute. So there is a well-defined **first deviation** k\*: the
iterate where the baseline terminates and the mechanism continues. The baseline
returns `(status_B, x_B)` there, and `status_B ∈ {Success, StopAtAcceptablePoint}`
— those are the only two decisions the veto suppresses.

The invariant therefore needs only a fragment of an order, not a total one:
`Success ⪰ everything`, `StopAtAcceptablePoint ⪰ everything but Success`, plus
reflexivity. It must be read **lexicographically, status-dominant**, with the
objective clause claimed only *within* an equal-status pair. The conjunctive
phrasing used earlier in this file overstates it: when both a strict and an
acceptable snapshot exist, the mechanism prefers the strict one and can return a
*better status at a higher objective* than the baseline's acceptable-level
answer. That is an improvement, not a violation — and the raw objectives are not
comparable there anyway, being measured at different constraint violations.

### 10.2 The defect: `masked` is not constant over a run

`obj_scale` is fixed per run, but `masked` also requires
`unscaled_err > acceptable_tol`, and that quantity crosses the bar during the
endgame — the crossing **is** the veto lifting. So an acceptable-level streak
can straddle the boundary, and the implementation kept two disjoint counters
(`acceptable_count`, `shadow_acceptable_count`), each zeroed by the other's
phase.

Concretely, with the default `acceptable_iter = 15`: fourteen unmasked
qualifying iterates leave the real count at 14; one masked qualifying iterate
zeroes it and starts the shadow at 1. The baseline would have reached 15 and
returned `Solved_To_Acceptable_Level`. The mechanism instead falls through to
`max_iter` (or a tiny step, or a time budget) and returns a **bare failure**,
with no snapshot armed — the shadow having only just started — so the fallback
is inert. Never-worse, violated.

The window is mechanically reachable on constrained problems: with `df` at the
1e-8 floor and constraint violation inside `(tol, acceptable_tol]`, the strict
test fails in both arms every iterate, the acceptable test passes in both, and
`masked` flips purely on the user-space dual residual crossing `acceptable_tol`
— exactly what wobbles during a masked run's endgame.

Empirical fuzzing was unlikely to find this: it needs the unscaled dual to hover
at `acceptable_tol` while the violation sits in a two-decade band for fifteen
consecutive iterates. It is a state-machine seam, not a numerical regime.

### 10.3 The fix

One counter, advanced on `acceptable_now` regardless of `masked`; `masked`
decides only what happens when it crosses the threshold — terminate, or record
that a termination was refused *here*, which is exactly the iterate the baseline
would have returned. `shadow_acceptable_count` is gone. Pinned by
`acceptable_streak_survives_a_masked_boundary_mid_streak`, which covers both
straddle directions and the reset semantics, and which fails when the old
two-counter behaviour is restored.

### 10.4 Findings accepted without code change

- **First-vs-best snapshot.** Keeping the *first* refused certificate is
  never-worse *by identity* — it is precisely what the baseline returned.
  A later refusal at a lower objective is discarded, which leaves value on the
  table but cannot violate the invariant. "Best-of-refused" would weakly
  dominate; not adopted, as it trades a by-identity argument for a numerical one.
- **`UserRequestedStop → Success`.** The stop request necessarily arrives
  *after* k\*, so in the baseline world the callback was never fired — the
  mechanism returns what the baseline returned to a user who was never asked.
  Sound, but a caller doing callback-driven early stopping receives a different
  `x` than the iterate it observed; a documented deviation from upstream
  semantics rather than a defect.
- **Objective comparison on the Success path** is valid for the invariant: both
  points passed `passes_component_tols`, so both are equally-valid certificates
  by the solver's own definition, and either branch is ⪰ baseline. It can prefer
  a lower-objective point at a larger (still sub-tolerance) violation; a merit
  comparison would be more defensible, but cannot affect never-worse.
- **Termination and state consistency** verified: the veto suppresses only the
  two convergence verdicts, never `max_iter` or the time budgets, so the
  driver-loop termination argument is intact; statistics are drained after the
  restore, so they describe the returned point. The restoration inner IPM is
  immune — its adapter delegates to the *scalar* `check_convergence`, which has
  no veto logic.


### 10.5 Re-analysis of the fixed machine

The formal analysis was re-run against the single-counter version, since the
analysis that found the defect had been performed on code that no longer existed.

**Verdict: the fix is correct**, and correct *by the structure of the machine*
rather than by tuning. The three properties that carry it:

1. The mechanism's termination decisions are a strict **subset** of the
   baseline's — `Converged` and `ConvergedToAcceptable` each require the
   baseline's own condition plus `!masked`, and no other exit is suppressed.
   So the first difference is always a suppression, never a different or earlier
   exit, and the first-deviation iterate `k*` stays well defined.
2. **Every suppression is flagged in the call it happens**, and the first flag of
   each kind is snapshotted at that same iterate with `curr` unmoved. There is no
   third suppression, hence no unflagged deviation — the hole that produced the
   original counterexamples is closed by construction.
3. The counter is now bit-identical to the baseline's on every shared iterate,
   because it breaks on exactly one condition (`!acceptable_now`) and `masked`
   plays no part in advancement. The old masked-phase reset was an *extra* break
   the baseline never had.

Two new post-deviation behaviours were checked and are harmless: the count now
stays at or above threshold through a suppressed phase, so `ConvergedToAcceptable`
can fire immediately once the veto lifts (that result is then overwritten by the
snapshot, i.e. baseline-exact); and the acceptable counter advances on iterates
after a strict refusal where the baseline never ran — unreachable, since every
fallback from that state prefers the strict snapshot.

**One violation was found and is fixed here:** a strict certificate can pass at an
iterate whose objective is non-finite (`passes_component_tols` never inspects
`f`), and the baseline returns exactly that. Refusing it armed a snapshot the
restore then declined, surfacing a failure where the baseline reported success.
The veto now declines to engage at a non-finite objective, which keeps that case
bit-identical to the baseline. The acceptable-level side already had the property,
finite `f` being a precondition of qualifying there.

**Preconditions the guarantee rests on**, all verified: the convergence check is
trajectory-inert; the computed objective scale is constant per solve; the snapshot
code stays adjacent to the check with `curr` unmoved and keeps its first-only
guards; `honour_refused_certificate` remains the single post-loop chokepoint with
strict-over-acceptable preference; and "no worse" is read lexicographically.
