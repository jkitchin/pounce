# gh#690 — the adaptive τ tail in the HSDE driver

**Status: the blocker is gone, and it was never real.** Re-measured a third
time, on `8b5d5217` (main, post-#712/PR#719), 60 CLI fixtures, both sweep legs.
Against the *honest* baseline the "20× trajectory regression" that this issue
was held open for is a **31% improvement**, and there are no status changes and
no objective regressions anywhere in the corpus on either leg.

The study has been run three times because the first two were measured through
defects that were later removed. The history is kept below, because the reason
this took three passes is itself the lesson.

## Result (post-#712)

Control first: with `POUNCE_HSDE_TAU_STUDY` unset the scaffold sweeps
bit-identical to `main` on both legs — verified by `diff`, empty — so the
tables below are the τ rule and nothing else.

### Corpus, default `max_iter = 200`

| variant | exact leg | lbfgs leg | status changes | objective regressions |
|---|---|---|---|---|
| off | 2163 | 5272 | — | — |
| orthant | 2164 (+0.0%) | 5273 (+0.0%) | 0 | 0 |
| ray | 2145 (−0.8%) | 5254 (−0.3%) | 0 | 0 |
| **both** | **2073 (−4.2%)** | **5182 (−1.7%)** | **0** | **0** |

### Corpus, `max_iter = 5000`

`scaled_feasible_a` is the only fixture in the corpus that reaches the default
cap, so it is the only line that moves between the two tables. Lifting the cap
is what puts it on the same footing as everything else.

| variant | exact leg | lbfgs leg |
|---|---|---|
| off | 5560 | 8669 |
| ray | 5542 (−0.3%) | 8651 (−0.2%) |
| **both** | **4348 (−21.8%)** | **7457 (−14.0%)** |

### `scaled_feasible_a` alone, `max_iter = 5000`

| variant | status | iters | objective | final_kkt_error |
|---|---|---|---|---|
| off | `SolveSucceeded` | 3596 | 0 | 1.88e-10 |
| orthant | `SolveSucceeded` | 3181 | 0 | 3.91e-03 |
| ray | `SolveSucceeded` | 3596 | 0 | 1.88e-10 |
| **both** | `SolveSucceeded` | **2474** | 0 | 1.22e-10 |

## 1. The blocker was an artefact of the baseline, not of the τ rule

#690's second measurement recorded `scaled_feasible_a` at 123 → 2474 iterations
and called it a 20× regression with the gh#544 signature. #712 then found that
the 123-iteration point was never converged: the gh#293 Ruiz retry was returning
an uncertified `Optimal`, and the genuineness guard's relative arm normalized
complementarity by the objective displaced by its degree-0 term, so an absolute
KKT error of `2.28e+03` read as `4.57e-09` and passed.

With that removed, the baseline for this fixture is **3596** iterations. The τ
tail's 2474 is unchanged from the previous measurement — it was always a real
convergence — so the comparison flips sign entirely:

    123  → 2474    "20× regression"   (against a point that did not exist)
    3596 → 2474    "31% improvement"  (against the point the solver certifies)

At the default cap both the baseline and every variant report
`MaximumIterationsExceeded` at 199, so on today's `main` this fixture is not a
status change under any variant. The single status change that blocked this
issue is gone.

This is the second time a number in this study was invalidated by a defect in
what it was measured against, and the pattern is worth naming: an iteration
count is only as trustworthy as the stopping test that produced it. A
"regression" measured against an early-stopped baseline is not a weak result,
it is not a result at all.

## 2. `orthant` and `ray` are each worth nothing; the gain is the interaction

The previous note inferred, from `orthant` at +0.0% and `both` at −4.1%, that
"the τ/κ ray isn't merely where the gain is, it's where *all* of it is", and
recommended studying the ray variant alone. **That inference was wrong, and the
`ray` variant was added to this study to test it.** Measured directly:

| variant | exact | lbfgs |
|---|---|---|
| orthant only | +0.0% | +0.0% |
| ray only | −0.8% | −0.3% |
| both | −4.2% | −1.7% |

Neither half delivers alone. The mechanism is the `min`:

    α = min( ray_step(τ, dτ), ray_step(κ, dκ), cone.max_step(s, ds), cone.max_step(z, dz) )

Relaxing the fraction-to-boundary parameter on one group leaves the other group
binding, and α does not move. `scaled_feasible_a` shows this in its cleanest
form: under `ray` it is **bit-identical to the baseline** — same 3596
iterations, same `final_kkt_error` to every digit — because the cone block is
what limits the step there, and the ray is merely slack. Under `orthant` the
cone block relaxes and the ray becomes binding, which buys 3596 → 3181. Only
`both` releases the actual constraint: 2474.

So the answer to the question `QpOptions::tau_max`'s doc comment asked —

> **direct driver only** — the HSDE loop's step is also limited by the τ/κ ray,
> so the same idea needs its own study there.

— is that the doc comment was right to refuse the port, and right about why.
gh#417's rule applied verbatim to HSDE's orthant blocks buys **exactly zero**,
because HSDE's step, unlike the direct driver's, is additionally capped by the
homogenizing pair. The rule has to be extended to the ray to do anything at
all. That is a substantive difference between the two drivers and not a
parameter-tuning detail.

## 3. The accuracy moves are all in the right direction

17 fixture-legs move under `both` on each leg, and every objective that changes
moves *toward* the exact optimum:

| fixture | off | both |
|---|---|---|
| `bound_active_qp` | 5.000000002 | **5** |
| `lp_row_constant` / `_expr` | −5.999999998 | **−6** |
| `presolve_overflow_feasible` | 1.000000001 | **1** |
| `lp_afiro` | −464.7531428 | −464.7531429 |
| `tame` | 2.366582716e-30 | 3.056836008e-30 (both 0, same `x`) |

The rest hold their objective exactly and cut iterations (`8 → 5` on the small
LP/QP family, `feasible_x0_wide_scale` `80 → 37`, `feasible_x0_extreme_row`
`33 → 29`, `scaled_feasible_b` `47 → 46`). Both legs move the identical set, so
L-BFGS again surfaced no exposure the exact leg missed — which is a result, not
a reason to have run one leg.

## 4. The suite under `both`: two calibrated literals and one real gap

Full workspace suite (`--no-fail-fast`, `pounce-hsl` excluded — it fails to
link locally for want of the proprietary HSL sources, unrelated to this study)
under `POUNCE_HSDE_TAU_STUDY=both`: **6 tests fail across 2 targets, everything
else passes.** They are not one kind of failure, and the difference matters.

### 4a. A real defect, exposed rather than caused (gh#535 reroute gate)

Five of the six are `crates/pounce-cli/tests/issue_535_lp_falls_back_to_nlp.rs`.
The cause is worth stating carefully, because the τ tail is not the defect.

`lp_declines_to_nlp` (`crates/pounce-cli/src/main.rs:2208`) gates the gh#535
LP → NLP reroute on

    QpStatus::OptimalInaccurate | QpStatus::IterationLimit

`QpStatus::NumericalFailure` is **not** in that list, and never has been. So an
LP whose convex solve ends in a numerical failure is reported as the last word
— which is precisely the shape gh#535 was filed to prevent ("a specialized fast
path displaced a general one, and when the fast path failed there was no
fallback left").

That hole is on `main` today; nothing in this study touches `main.rs`. What the
τ tail does is make a fixture *reach* it. On `lp_afiro` at the test's
deliberately unreachable `tol=1e-20`:

| | convex verdict | reroute fires? |
|---|---|---|
| off | `Maximum iterations exceeded`, 199 iters | yes → NLP path |
| both | `Numerical failure (no verified KKT point)`, 85 iters | **no** |

The mechanism is unsurprising: at an unreachable tolerance the τ tail drives
iterates harder against the boundary, so the KKT system goes singular at 85
iterations instead of the budget running out at 199. Same non-certification,
different status, and only one of the two statuses reroutes.

**Verified fix:** adding `QpStatus::NumericalFailure` to the gate turns 5
failures into 1, and — checked explicitly — does not change the suite with the
scaffold off (8/8 pass on `main` behaviour, unchanged). It is arguably the
correct gate independently of this study, and it should land as its own change
with its own issue rather than riding along inside a τ-rule PR.

### 4b. Two literals calibrated to today's step rule

The remaining two failures are threshold literals, not contracts. In both cases
every assertion carrying the test's actual claim still passes under `both`;
only a number tuned to the point the current step rule happens to land on does
not.

1. `ipm::false_optimum_metric_tests::the_false_optimum_is_invisible_unscaled_and_obvious_equilibrated`
   — fails on its **premise** assertion:

       premise: the certified point's own KKT error is huge
       (got 2.301e2, the issue reports 8.3e3)

   With that one threshold instrumented out, the substantive assertions all
   still hold under `both`: the point is still certified by the un-repaired
   embedding, still invisible to the unscaled relative test, still `O(1)` or
   worse in the equilibrated metric, and the true optimum still sits below
   `1e-6`. The τ tail simply lands the un-repaired solve on a *less bad* false
   optimum — `2.30e2` where the control gives `8.28e3`. The separation the test
   exists to assert is `2.3e2` against `<1e-6`, nine orders, so the claim
   survives comfortably; only the `> 1e3` literal does not.

   This test keeps #712's extra machinery explained, so re-tuning its threshold
   is a decision to make out loud in the PR, not a number to quietly adjust.

2. `an_explicitly_selected_convex_solve_is_not_rerouted` — the sixth failure,
   and the one the 4a fix does not resolve. It asserts
   `stdout.contains("Maximum iterations exceeded")`, the same status literal
   4a is about. Its two load-bearing assertions — exactly one convex verdict
   line, and no reroute for a named engine — both pass under `both`. Only the
   hard-coded failure mode does not.

### 4c. The scaffold is not the implementation

It reads an environment variable per solve. Shipping means the rule applies
unconditionally to the HSDE corrector (never the predictor), driven by the
existing `tau` / `tau_max` pair, plus the doc comment on `QpOptions::tau_max`
rewritten to record the answer instead of the question — including §2's
finding, which a future reader will otherwise re-derive.

## Verdict

**Ship `both`, behind one companion change.** −4.2% exact / −1.7% lbfgs at the
default cap, −21.8% / −14.0% with the cap lifted, zero status changes and zero
objective regressions across 60 fixtures on both legs, and accuracy
improvements on six. The fixture that held this issue open for two rounds is
31% faster than the baseline that can actually certify it.

Order of work:

1. **First, independently:** widen the gh#535 reroute gate to include
   `QpStatus::NumericalFailure` (§4a). This is a hole in `main` today, it needs
   its own issue and its own reasoning, and it should not be discovered inside
   a τ-rule PR.
2. **Then:** implement the τ tail unconditionally on the HSDE corrector, adjust
   the two calibrated literals in §4b with the reasoning stated in the PR body,
   and rewrite the `QpOptions::tau_max` doc comment to carry §2's answer.

## History — why this took three passes

| pass | measured on | headline | invalidated by |
|---|---|---|---|
| #588 Q9a (2026-08-18) | `feat/588-q9-correctors` | −3.0%, blocked on ~2× objective moves | #696: HSDE normalized the duality gap by an objective displaced by its degree-0 term (`Σaᵢ² ≈ 5e11`), buying `tol·|Σaᵢ²| ≈ 5e3` of gap slack |
| re-run (2026-08-19) | `ac18ba6` | −4.1% excl. a 20× regression on `scaled_feasible_a` | #712: the same fixture's 123-iteration baseline was an uncertified `Optimal` from the gh#293 Ruiz retry, absolute KKT error `2.28e+03` |
| this note (2026-08-20) | `8b5d5217` | −4.2% / −1.7%, no blocker | — |

Each pass's objection dissolved when the thing it was measured against was
fixed, and in both cases the τ rule was never the defect. Both passes were also
correct to refuse to ship: neither had the evidence to, and the second's
objection was strictly better-founded than the first's.

## The scaffold

Applies to `8b5d5217`. Off unless `POUNCE_HSDE_TAU_STUDY` is set to `orthant`,
`ray`, or `both`; with it unset the sweep is bit-identical to `main`, which is
the control that makes the tables above mean anything. **Not for merge** — it
reads an environment variable per solve and exists only to be measured. The
`ray` arm is new in this pass and is what disproved §2's earlier inference.

Reproduce with:

    scripts/sweep-fixtures.sh <bin> /tmp/off.txt                     # env unset
    POUNCE_HSDE_TAU_STUDY=both scripts/sweep-fixtures.sh <bin> /tmp/both.txt
    diff /tmp/off.txt /tmp/both.txt

```diff
diff --git a/crates/pounce-convex/src/hsde.rs b/crates/pounce-convex/src/hsde.rs
index e917f215..8de56626 100644
--- a/crates/pounce-convex/src/hsde.rs
+++ b/crates/pounce-convex/src/hsde.rs
@@ -38,7 +38,8 @@ use crate::cones::{CompositeCone, Cone};
 use crate::correctors;
 use crate::debug::{ConvexDebugState, fire};
 use crate::ipm::{
-    QpOptions, build_factorization, build_rhs, detect_infeasibility_cone, dot, inf_norm, split_step,
+    QpOptions, adaptive_tau, build_factorization, build_rhs, detect_infeasibility_cone, dot,
+    inf_norm, split_step,
 };
 use crate::qp::{QpIterate, QpProblem, QpSolution, QpStatus};
 use pounce_common::debug::{Checkpoint, DebugAction, DebugHook};
@@ -227,6 +228,45 @@ fn true_kkt_error(
     )
 }
 
+/// gh#690 STUDY SCAFFOLD -- NOT FOR MERGE. Selects an adaptive-tau tail on the
+/// HSDE *corrector* step (never the predictor), off unless the environment
+/// variable `POUNCE_HSDE_TAU_STUDY` is set to `orthant`, `ray`, or `both`.
+#[derive(Clone, Copy, PartialEq, Eq)]
+enum TauStudy {
+    /// Shipped behaviour: static `opts.tau` everywhere.
+    Off,
+    /// Adaptive tau on the orthant cone blocks; the tau/kappa ray stays static.
+    Orthant,
+    /// Adaptive tau on the tau/kappa ray only; the cone blocks stay static.
+    Ray,
+    /// Adaptive tau on the orthant blocks *and* the tau/kappa ray.
+    Both,
+}
+
+impl TauStudy {
+    fn from_env() -> Self {
+        match std::env::var("POUNCE_HSDE_TAU_STUDY").as_deref() {
+            Ok("orthant") => Self::Orthant,
+            Ok("ray") => Self::Ray,
+            Ok("both") => Self::Both,
+            _ => Self::Off,
+        }
+    }
+
+    /// `(tau_orthant, tau_ray)` for the corrector step at this mu.
+    fn taus(self, mu: f64, opts: &QpOptions) -> (f64, f64) {
+        match self {
+            Self::Off => (opts.tau, opts.tau),
+            Self::Orthant => (adaptive_tau(mu, opts), opts.tau),
+            Self::Ray => (opts.tau, adaptive_tau(mu, opts)),
+            Self::Both => {
+                let t = adaptive_tau(mu, opts);
+                (t, t)
+            }
+        }
+    }
+}
+
 /// Fraction-to-boundary step for a positive scalar ray `v + α dv > 0`,
 /// scaled by `tau` and capped at 1 (the scalar analogue of `Cone::max_step`
 /// for the homogenizing variables `τ`, `κ`).
@@ -416,6 +456,7 @@ where
     // record at the converged iterate (α = 0).
     let mut trace: Vec<QpIterate> = Vec::new();
 
+    let tau_study = TauStudy::from_env();
     for it in 0..opts.max_iter {
         iters = it;
         if crate::deadline::expired() {
@@ -850,11 +891,12 @@ where
             // degenerate NETLIB GEN family (α_p ≫ α_d) that blows ρ_x up from
             // ~1e-8 to ~5e-2. The symmetric step keeps the embedding's clean
             // (1−α) residual decrease.
-            alpha = ray_step(tau, dtau, opts.tau).min(ray_step(kappa, dkappa, opts.tau));
+            let (tau_orthant, tau_ray) = tau_study.taus(mu, opts);
+            alpha = ray_step(tau, dtau, tau_ray).min(ray_step(kappa, dkappa, tau_ray));
             if m_ineq > 0 {
                 alpha = alpha
-                    .min(cone.max_step(&s, &ds, opts.tau))
-                    .min(cone.max_step(&z, &dz, opts.tau));
+                    .min(cone.max_step_split(&s, &ds, tau_orthant, opts.tau))
+                    .min(cone.max_step_split(&z, &dz, tau_orthant, opts.tau));
             }
 
             if alpha >= CENTERING_MIN_STEP
@@ -939,10 +981,11 @@ where
                     step_s[i] = ds[i] + cds[i];
                     step_z[i] = dz[i] + cdz[i];
                 }
-                let a_new = ray_step(tau, dtau + dtau_c, opts.tau)
-                    .min(ray_step(kappa, dkappa + dkappa_c, opts.tau))
-                    .min(cone.max_step(&s, &step_s, opts.tau))
-                    .min(cone.max_step(&z, &step_z, opts.tau));
+                let (tau_orthant, tau_ray) = tau_study.taus(mu, opts);
+                let a_new = ray_step(tau, dtau + dtau_c, tau_ray)
+                    .min(ray_step(kappa, dkappa + dkappa_c, tau_ray))
+                    .min(cone.max_step_split(&s, &step_s, tau_orthant, opts.tau))
+                    .min(cone.max_step_split(&z, &step_z, tau_orthant, opts.tau));
                 let keep = correctors::accepts(a_new, alpha);
                 tally.record(keep, a_new - alpha);
                 if keep {
diff --git a/crates/pounce-convex/src/ipm.rs b/crates/pounce-convex/src/ipm.rs
index 97a3ee63..dc0209f1 100644
--- a/crates/pounce-convex/src/ipm.rs
+++ b/crates/pounce-convex/src/ipm.rs
@@ -112,7 +112,7 @@ const TAU_CEIL: f64 = 1.0 - 1e-12;
 /// rather than in a logarithm of the perturbation (gh #417). Far from the
 /// solution (μ ≥ 1 − `tau`, and on badly-scaled data where μ is large) it
 /// reduces to the static `opts.tau`, so early iterations are unchanged.
-fn adaptive_tau(mu: f64, opts: &QpOptions) -> f64 {
+pub(crate) fn adaptive_tau(mu: f64, opts: &QpOptions) -> f64 {
     // `tau` wins if a caller sets an inverted pair (`tau_max < tau`), which is
     // how the static behaviour is requested (`tau_max == tau`).
     let hi = opts.tau_max.min(TAU_CEIL).max(opts.tau);
```
