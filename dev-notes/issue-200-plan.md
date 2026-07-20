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

### 9.5 Post-optimal sensitivity after a fallback restore — OPEN DEFECT

Verified present, cause not found, **not fixed**. Filed rather than hidden.

On the gh #200 masked problem both arms return the *identical* point and status,
and the sensitivity object differs by nine orders of magnitude:

```
baseline (threshold=0)  Solved_To_Acceptable_Level  f=-6.833859e14
                        x=[99999.99095622574, 68338588703875.97]
                        solve_scaled_space(e0) = [4.5283e10, 0]
veto (falls back)       Solved_To_Acceptable_Level  f=-6.833859e14
                        x=[99999.99095622574, 68338588703875.97]   <- byte-identical
                        solve_scaled_space(e0) = [9.6844e19, 0]     <- 2.1e9 relative
```

Baseline does not have this problem: it stops at the point it factorized, so its
held factor and its returned iterate agree. The fallback restores an earlier
iterate after the run travelled on, so `PdSensBacksolver` — which explicitly
reuses the **held factor** — solves against state belonging to a different
point. A consumer receives a `dx/dp` that does not correspond to the solution it
was handed, silently.

Three hypotheses were tried and all three were wrong, which is why this is
reported rather than patched:

1. **Refresh the factorization** at the restored point via
   `PdSearchDirCalc::compute_search_direction` (clearing `delta` first, since it
   short-circuits when a regularization delta is pending). It ran (`ok=true`)
   and changed nothing.
2. **Restore `curr_mu`** with the iterate. Correct on its own merits and kept
   (see 9.6), but it did not change the sensitivity either.
3. **A malformed probe** — `solve_scaled_space` rejects a mismatched length and
   it returned `true`, so the compound dimension really is 2 here and the probe
   is valid.

Whatever carries the discrepancy is therefore neither the factor as refreshed by
that entry point, nor `mu`, nor the returned iterate. Likely candidates not yet
eliminated: cached `IpoptCq` quantities that `set_trial`/`accept_trial_point`
outside the main loop does not invalidate, or a `pd` solver instance distinct
from the one `compute_search_direction` refreshes.

Scope: runs that fall back **and** have an `on_converged` sensitivity consumer
attached. Correctness of the returned point, objective and status is unaffected.

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
- **9.5 is open** and is the one item that should not be forgotten.
