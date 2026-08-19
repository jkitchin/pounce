# gh#690 — the adaptive τ tail in the HSDE driver, re-measured after #696

**Status: still not shipped, for a different reason than the one #690 was
filed with.** The original objection dissolved; a stronger one appeared in its
place.

Measured on `ac18ba6` (main, 2026-08-19), 58 CLI fixtures, **both sweep legs**.
The scaffold that produced it is reproduced verbatim at the end of this note so
the next reader does not have to reconstruct it — the first run of this study
(#588 Q9a) was a throwaway that was reverted, and it is not recoverable from
any branch.

## Why it had to be re-measured

#690 reported `−3.0%` corpus-wide against a baseline of 2082 iterations, and
declined to ship on the grounds that three fixtures moved their reported
objective by ~2× while still returning `SolveSucceeded`. It named the fork
itself: either those fixtures have flat optima, or "the τ tail is letting the
HSDE step reach iterates the success test should not be accepting", and said
the second "outranks the τ question entirely".

It was the second. #689 / PR #696 found HSDE normalizing the duality gap by an
objective displaced by the model's degree-0 term (`Σaᵢ² ≈ 5e11` on this pair),
which bought a blanket `tol·|Σaᵢ²| ≈ 5e3` of absolute gap slack. All three
fixtures in #690's table now report their true optimum of 0. The corpus
baseline moved with the fix: 2082 → 2280 on the same 57 fixtures, and the +198
is exactly the four models #696 made do the work they had been skipping
(`scaled_feasible_a` 16→123, `scaled_feasible_b` 21→47,
`feasible_x0_wide_scale` 16→80, `feasible_x0_extreme_row` 32→33). The two
totals reconcile to the iteration, which is the evidence that nothing else in
the corpus moved — #690 published only a total, so this is an inference from
it rather than a per-fixture check.

So every number in #690 was measured through the defect #696 removed, and the
study is worth nothing until re-run.

## Result

Control first: with `POUNCE_HSDE_TAU_STUDY` unset the scaffold sweeps
bit-identical to `main` on both legs, so the numbers below are the τ rule and
nothing else.

| variant | exact leg | lbfgs leg | status changes | objectives moved |
|---|---|---|---|---|
| off (today) | 2292 | 5438 | — | — |
| orthant | 2369 (+3.4%) | 5515 (+1.4%) | 1 | 2 |
| both | 2278 (−0.6%) | 5424 (−0.3%) | 1 | 7 |

`scaled_feasible_a` is the single status change in both variants and it
dominates the totals. Excluding it:

| variant | exact leg | lbfgs leg |
|---|---|---|
| orthant | +0.0% | +0.0% |
| both | **−4.1%** | **−1.7%** |

### 1. The objection #690 was filed on is gone

Every objective that moves under `both` moves *toward* the exact optimum:

| fixture | off | both |
|---|---|---|
| `bound_active_qp` | 5.000000002 | **5** |
| `lp_row_constant` | −5.999999998 | **−6** |
| `lp_row_constant_expr` | −5.999999998 | **−6** |
| `presolve_overflow_feasible` | 1.000000001 | **1** |
| `lp_afiro` | −464.7531428 | −464.7531429 |
| `tame` | 2.366582716e-30 | 3.056836008e-30 (both are 0, same `x`) |

No fixture returns a materially different answer, and the three that used to
move ~2× (`scaled_feasible_a`, `scaled_feasible_b`, `feasible_x0_extreme_row`)
now hold at 0. The reason not to ship, as written in #690, no longer exists.

### 2. A harder objection replaced it

`scaled_feasible_a` is a **20× trajectory regression**:

| max_iter | status | iters | objective | final_kkt_error |
|---|---|---|---|---|
| 200 (default) | `MaximumIterationsExceeded` | 199 | 6.10e-05 | 3.18e-07 |
| 400 | `MaximumIterationsExceeded` | 399 | 0 | 4.10e-03 |
| 1000 | `MaximumIterationsExceeded` | 999 | 6.10e-05 | 4.84e-08 |
| 3000 | `SolveSucceeded` | **2474** | 0 | 1.22e-10 |

against 123 iterations on `main`. It is a stall, not a divergence, and the
shape of the stall is the interesting part: by iteration 199 the τ tail has
already reached `final_kkt_error` 3.2e-07 — four orders of magnitude *better*
than the 4.6e-03 point the baseline certifies at 123 — and then wanders
non-monotonically (4.1e-03 at 399, 4.8e-08 at 999) for another two thousand
iterations before the stopping test accepts it.

This is the gh#544 signature exactly: the right answer, slowly, with status and
objective both intact, and it would have been invisible to a suite that asserts
those two. It is a hard blocker under CLAUDE.md, and it is a *better* reason
than the one #690 was holding out for.

The near-certification at 199 also says the blocker is not purely the step
rule. `scaled_feasible_a` is the model whose objective constant is `Σaᵢ² ≈
5e11`, i.e. the one that stresses the post-#696 stopping test hardest, and the
τ tail is producing points that test will not accept while accepting worse
ones. Whether that is the τ rule's fault or a remaining gap in the stopping
test is the question a follow-up has to answer, and it should be answered
before the −4.1% is taken.

### 3. `orthant` alone is now worthless

+0.0% on both legs excluding the regression, i.e. it buys nothing and still
breaks `scaled_feasible_a`. #690 read the split as "the τ/κ ray is where the
gain is"; post-#696 it is where *all* of it is. Any follow-up should study the
ray variant only, and the doc comment on `QpOptions::tau_max` — which scopes
the rule away from HSDE because "the HSDE loop's step is also limited by the
τ/κ ray, so the same idea needs its own study there" — is pointing at exactly
the right thing.

### 4. On the second leg

Both legs move the identical 17-model set, so the L-BFGS leg surfaced no
exposure the exact leg missed. That is a result and not a reason to have run
one leg: #690 reported a single total, and there was no way to know which of
the two it was until both were run. The gain is also materially smaller on the
L-BFGS leg (−1.7% vs −4.1%), because that leg's iteration count is dominated
by fixtures the convex driver never sees.

## Verdict

Keep #690 open. The τ/κ ray is worth −4.1% on the exact leg and the accuracy
moves are all in the right direction, but `scaled_feasible_a` at 20× is not
shippable, and diagnosing it is a phase of its own — plausibly a stopping-test
phase rather than a τ phase.

## The scaffold

Applies to `ac18ba6`. Off unless `POUNCE_HSDE_TAU_STUDY` is set to `orthant`
or `both`; with it unset the sweep is bit-identical to `main`, which is the
control that makes the table above mean anything. **Not for merge** — it reads
an environment variable per solve and exists only to be measured.

```diff
diff --git a/crates/pounce-convex/src/hsde.rs b/crates/pounce-convex/src/hsde.rs
index e917f21..0e8144b 100644
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
@@ -228,6 +229,41 @@ fn true_kkt_error(
 }
 
 /// Fraction-to-boundary step for a positive scalar ray `v + α dv > 0`,
+/// gh#690 STUDY SCAFFOLD -- NOT FOR MERGE. Selects an adaptive-tau tail on the
+/// HSDE *corrector* step (never the predictor), off unless the environment
+/// variable `POUNCE_HSDE_TAU_STUDY` is set to `orthant` or `both`.
+#[derive(Clone, Copy, PartialEq, Eq)]
+enum TauStudy {
+    /// Shipped behaviour: static `opts.tau` everywhere.
+    Off,
+    /// Adaptive tau on the orthant cone blocks; the tau/kappa ray stays static.
+    Orthant,
+    /// Adaptive tau on the orthant blocks *and* the tau/kappa ray.
+    Both,
+}
+
+impl TauStudy {
+    fn from_env() -> Self {
+        match std::env::var("POUNCE_HSDE_TAU_STUDY").as_deref() {
+            Ok("orthant") => Self::Orthant,
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
+            Self::Both => {
+                let t = adaptive_tau(mu, opts);
+                (t, t)
+            }
+        }
+    }
+}
+
 /// scaled by `tau` and capped at 1 (the scalar analogue of `Cone::max_step`
 /// for the homogenizing variables `τ`, `κ`).
 fn ray_step(v: f64, dv: f64, tau: f64) -> f64 {
@@ -416,6 +452,7 @@ where
     // record at the converged iterate (α = 0).
     let mut trace: Vec<QpIterate> = Vec::new();
 
+    let tau_study = TauStudy::from_env();
     for it in 0..opts.max_iter {
         iters = it;
         if crate::deadline::expired() {
@@ -850,11 +887,12 @@ where
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
@@ -939,10 +977,11 @@ where
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
index 635885b..0ba8583 100644
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
