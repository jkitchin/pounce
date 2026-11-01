//! Regression: rapid infeasibility detection must not declare a *feasible*
//! problem infeasible because the constraint scaling flattered its stationarity
//! measure.
//!
//! The detector fires when two conditions hold together: the constraint
//! violation is bounded away from zero, and a stationarity surrogate
//! `‖Jᵀc‖ / max(1, ‖c‖)` falls under an absolute tolerance — i.e. supposedly no
//! local move reduces the violation. That surrogate is not scale-invariant:
//! under a row scaling `dc` the numerator carries `dc²` while the denominator
//! clamps at 1, so an aggressive scaling drives it toward zero regardless of
//! where the iterate is.
//!
//! Hock–Schittkowski 13 makes it concrete:
//!
//! ```text
//!   min  (x₁ - 2)² + x₂²   s.t.  (1 - x₁)³ - x₂ ≥ 0,  x ≥ 0        f* = 1
//! ```
//!
//! From `x₀ = (1e4, 1e4)` the starting Jacobian is ~3e8, so gradient-based
//! scaling picks `dc ≈ 3.3e-7` and the surrogate reads `5e-14` — far under the
//! `1e-8` tolerance — at a point whose violation is **0.51**, whose `‖∇θ‖` is
//! 1.40, and where neither bound is active to block descent. One step downhill
//! reaches a feasible point. POUNCE reported `Infeasible_Problem_Detected` (AMPL
//! band 200) on a problem Ipopt solves.
//!
//! That verdict is the dangerous kind for a branch-and-bound driver: a false
//! *unbounded* is loud and retryable, but a false *infeasible* silently prunes a
//! node that may contain the optimum.
//!
//! NO THRESHOLD FIXES THIS, which is why the fix is not a retune. Measured over
//! 800 corpus models: the scaled surrogate is not separable on the targeted
//! cases; an unscaled surrogate needs a tolerance ≥ 1e-2 to fire at all, which
//! introduces new false infeasibility on 3+ corpus models while still losing 2
//! correct detections; and a scale-invariant `‖Jᵀc‖ / ‖c‖²` is not separable
//! either. A single absolute threshold on a surrogate cannot separate these.
//!
//! So the surrogate is kept as a cheap pre-filter and the claim the status
//! actually makes is confirmed directly: probe for a materially less-violating
//! point nearby, and withhold the verdict if one exists. Comparing `θ` to `θ` is
//! scale-free and needs no calibration. See
//! `IpoptCalculatedQuantities::infeasibility_descent_available`.
//!
//! Note this is HS13's *hard* start, not its published one: it is deliberately
//! remote so that gradient-based scaling picks an extreme `dc`. The ordinary
//! start is covered by `hock_schittkowski_subset` and is unaffected.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::ApplicationReturnStatus;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

fn tmp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_falseinfeas_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

fn solve(fixture_name: &str, extra: &[&str]) -> SolveReport {
    let json_path = tmp_path(&format!("{fixture_name}.json"));
    let sol_path = tmp_path(&format!("{fixture_name}.sol"));
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(fixture_name))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    for o in extra {
        cmd.arg(o);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// Ipopt on the identical `.nl` from the identical start: 0.98492871533797743
/// in 29 iterations. (HS13's published optimum is `f* = 1`; both solvers stop
/// slightly short because LICQ and MFCQ both fail at `x*`, which is what makes
/// this problem a classic.)
const HS13_IPOPT: f64 = 0.984_928_715_337_977_43;

/// The headline: a feasible problem must not be reported infeasible.
///
/// Pre-fix this returned AMPL band 200 with objective 0.28583751 at a point
/// whose constraint violation was 0.51.
#[test]
fn hs13_from_remote_start_is_not_reported_infeasible() {
    let report = solve("hs13_bigstart.nl", &[]);
    let code = report.solution.solve_result_num;
    assert!(
        !(200..300).contains(&code),
        "HS13 reported INFEASIBLE (solve_result_num={code}, status={:?}) — it is \
         feasible, with f* = 1 and Ipopt reaching {HS13_IPOPT} from this same \
         start. The returned point had constraint violation 0.51 and was not a \
         stationary point of the infeasibility (‖∇θ‖ = 1.4, no active bound)",
        report.solution.status,
    );
}

/// And it must actually converge — to Ipopt's answer under Ipopt's inertia
/// correction, and to the neighbourhood of `x* = (1, 0)` under any.
///
/// Where a solve on HS13 stops is set by the trajectory, not by the
/// optimality conditions: `x*` admits no KKT multiplier (LICQ and MFCQ both
/// fail there), so every solver stops on a tolerance short of it, with the
/// multiplier growing without bound and the objective a little under
/// `f* = 1` because the stopping point sits just outside the cusp, within
/// `constr_viol_tol`. Ipopt's `0.98492871533797743` is one such point. It is
/// pinned at `perturb_slack_fact=1`, which is Ipopt's `δ_s = δ_x`, because
/// matching it to 1e-6 is a trajectory-parity claim and holds only on
/// Ipopt's trajectory (gh#1098). At `perturb_slack_fact=0.03` the same start
/// stops at `0.98168`, `x = (1.00920, -1e-8)`, in 31 iterations — the same
/// kind of point (`Solve_Succeeded`, unscaled constraint violation 7.6e-7
/// against 4.1e-7 at factor 1, unscaled dual infeasibility 0.70 against 0.69
/// behind the same `large_dual_scale` warning), a little further along the
/// cusp. That is not a regression and is not pinned as a value; what is
/// pinned at the default is that the solve converges near `x*`, which is
/// what the pre-fix answer — objective 0.2858 at a point violating the
/// constraint by 0.51, reported infeasible — did not do.
#[test]
fn hs13_from_remote_start_reaches_the_optimum() {
    let report = solve("hs13_bigstart.nl", &["perturb_slack_fact=1"]);
    let obj = report.solution.objective;
    let rel = (obj - HS13_IPOPT).abs() / HS13_IPOPT.abs();
    assert!(
        rel < 1e-6,
        "HS13 from the remote start at Ipopt's δ_s = δ_x: got {obj}, expected \
         Ipopt's {HS13_IPOPT} (rel err {rel:.3e})",
    );

    let default = solve("hs13_bigstart.nl", &[]);
    assert_eq!(
        default.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "HS13 from the remote start at the default inertia correction",
    );
    let obj = default.solution.objective;
    assert!(
        (0.97..=1.0 + 1e-6).contains(&obj),
        "HS13 at the default: objective {obj} is not near f* = 1 (Ipopt \
         stops at {HS13_IPOPT}; the pre-fix answer was 0.2858)",
    );
    let x = &default.solution.x;
    assert!(
        (x[0] - 1.0).abs() < 1e-2 && x[1].abs() < 1e-2,
        "HS13 at the default: x = {x:?} is not near x* = (1, 0)",
    );
}

/// The returned point must be feasible **in the user's own units**, checked by
/// evaluating HS13's constraint on the returned `x` directly.
///
/// This deliberately does not derive the unscaled residual from the report's
/// scaled `final_constr_viol`: an earlier version divided it by the *objective*
/// scale factor, which is a different factor from the *constraint* row scale
/// `dc`, and the test passed pre-fix for that reason — reporting 3.4e-5 where the
/// true violation was 0.51. Evaluating the constraint closed-form avoids the
/// whole class of mistake, and is possible because HS13 is two variables:
///
/// ```text
///   c(x) = (1 - x₁)³ - x₂ ≥ 0
/// ```
#[test]
fn hs13_returned_point_is_feasible_in_user_units() {
    let report = solve("hs13_bigstart.nl", &[]);
    let x = &report.solution.x;
    assert_eq!(x.len(), 2, "HS13 has two variables");
    let (x1, x2) = (x[0], x[1]);
    let c = (1.0 - x1).powi(3) - x2;
    // Violation of `c >= 0`, in the user's units.
    let violation = (-c).max(0.0);
    assert!(
        violation < 1e-4,
        "returned point x = ({x1}, {x2}) violates (1-x1)^3 - x2 >= 0 by \
         {violation} — pre-fix POUNCE stopped at (1.5698, 0.31744), a violation \
         of 0.51, and called it infeasible",
    );
    // Bounds, for completeness.
    assert!(x1 >= -1e-8 && x2 >= -1e-8, "bounds violated: ({x1}, {x2})");
}

/// Disabling the scaling was the diagnostic that isolated the bug: it removes
/// the space mismatch by removing the scaling, and the problem solved cleanly
/// that way even before the fix. Post-fix, neither path may report infeasible.
///
/// The two paths are NOT required to agree on the point. HS13 fails both LICQ
/// and MFCQ at `x*`, so convergence is slow and where a solver stops depends on
/// its trajectory: POUNCE reaches 0.98493 scaled and 0.99458 unscaled, and Ipopt
/// reaches 0.98493 on the scaled path too. Both bracket the published `f* = 1`
/// from below. An earlier version of this test asserted the two objectives
/// matched to 1e-4 and failed for that reason — the fix unifies the *space the
/// stationarity test is measured in*, not the trajectory.
#[test]
fn hs13_neither_scaling_path_reports_infeasible() {
    // Published optimum. Both paths must land near it, from below.
    const HS13_STAR: f64 = 1.0;
    for opts in [vec![], vec!["nlp_scaling_method=none"]] {
        let r = solve("hs13_bigstart.nl", &opts);
        let code = r.solution.solve_result_num;
        assert!(
            !(200..300).contains(&code),
            "HS13 reported INFEASIBLE with opts {opts:?} \
             (solve_result_num={code}, status={:?})",
            r.solution.status,
        );
        let obj = r.solution.objective;
        assert!(
            (obj - HS13_STAR).abs() < 0.05,
            "HS13 with opts {opts:?}: objective {obj} is not near the published \
             f* = {HS13_STAR}",
        );
    }
}

/// gh #379: the two shapes where the *numerical* path called a feasible model
/// infeasible, found by the feasible-by-construction property sweep in
/// `pyomo-pounce/tests/test_infeasibility_no_false_positives.py`.
///
/// Both are linear-constrained sum-of-squares models — convex, so the auto-route
/// sends them to the QP IPM and a user on defaults never sees this. They reach
/// the numerical path only under `solver_selection=nlp`, which is also what every
/// genuinely nonlinear model uses, where there is no second solver to disagree
/// and no way to notice.
///
/// `feasible_x0_extreme_row` (property seed 294) is the sharper of the two: the
/// model's own starting point `(5e5, 5e5)` satisfies both rows *exactly*, and
/// `pounce verify` reports `0.000e0` constraint and bound violation on the convex
/// route's solution. Its first row carries `±1e30` coefficients; in the scaled
/// space that row's slack initializes far from anything `x` can follow, the outer
/// line search walks the violation up rather than down, restoration burns its
/// 3000-iteration budget, and the cycle gate reported
/// `Infeasible_Problem_Detected` — AMPL 200 — about a model the solver was
/// *handed a feasible point for*.
///
/// The fix is a refutation rather than a retune: no numerical path may claim
/// infeasibility while the starting point satisfies every constraint. See
/// `pounce_algorithm::infeasibility_refutation`. The solve still fails here — the
/// `±1e30` row is genuinely hostile — but it fails as AMPL 500, which a caller
/// can see, instead of AMPL 200, which is a wrong answer they cannot.
///
/// `feasible_x0_wide_scale` (property seed 102) is the same family at milder
/// scale (`1e8` row coefficients, right-hand sides near `5e13`). It was a false
/// positive before the scale-relative tolerance work (#382–#386) and solves
/// cleanly now; it is pinned here so that stays true.
#[test]
fn feasible_models_with_a_feasible_start_are_not_reported_infeasible() {
    for model in ["feasible_x0_extreme_row.nl", "feasible_x0_wide_scale.nl"] {
        let report = solve(
            model,
            &["solver_selection=nlp", "presolve=no", "print_level=0"],
        );
        let code = report.solution.solve_result_num;
        assert!(
            !(200..300).contains(&code),
            "{model} is feasible — its own starting point satisfies every \
             constraint, and POUNCE's convex QP route solves it to optimality — \
             but the NLP path reported the AMPL infeasible band \
             (solve_result_num={code}, status={:?})",
            report.solution.status,
        );
    }
}

/// The contradiction that made gh #379 unambiguous: POUNCE's own convex solver
/// returns `Optimal Solution Found` on the same models. Pinned so the two routes
/// cannot drift back into disagreeing about whether the feasible set is empty.
#[test]
fn the_convex_route_still_solves_both_shapes() {
    for model in ["feasible_x0_extreme_row.nl", "feasible_x0_wide_scale.nl"] {
        let report = solve(model, &["print_level=0"]);
        let code = report.solution.solve_result_num;
        assert_eq!(
            code, 0,
            "{model} is a convex QP the auto-route solves; got \
             solve_result_num={code}, status={:?}",
            report.solution.status,
        );
    }
}

/// gh #524: the local-infeasibility second-opinion ladder must be
/// *algorithmically* diverse, not just numerically diverse.
///
/// `cresc4` is a CUTE problem (Ph. Toint, 1993): fit the smallest-area crescent
/// containing four given points. Six variables, eight constraints, nonconvex,
/// feasible, with a known optimum — LOQO, SNOPT and Ipopt-MA57 all reach
/// `0.8718976`, Ipopt in 71 iterations. POUNCE on defaults converges to a point
/// with constraint violation 0.51, objective ~2e-09 (a degenerate zero-area
/// crescent), and reports `Infeasible_Problem_Detected`.
///
/// What makes it a ladder bug rather than one more false-infeasibility case is
/// what happened next: the MC64 re-solve fired, reproduced the original
/// trajectory character-for-character through iteration 15, diverged at
/// iteration 16 in the eighth significant digit, landed in the same basin
/// anyway, and the CLI announced the verdict "corroborated by a second
/// scaling". Rung 1 varies only the linear algebra, so whether its ULP-scale
/// perturbation escapes the basin is luck — on `discs.nl`, the case the guard
/// was written for, it was lucky. Here it was not, and the unlucky draw was
/// reported as corroboration.
///
/// `mu_strategy=adaptive` changes the iterate sequence itself and walks
/// straight to `0.8718975`. So does `nlp_scaling_method=none` — a third
/// trajectory disagreeing with the first two is what tells you the verdict was
/// never about scaling at all.
///
/// Fixture provenance: transcribed from CUTEst `mastsif/CRESC4.SIF` (the model,
/// bounds and start point are the SIF's, and the objective it reaches matches
/// the published `0.8718976`), then written to `.nl` by Pyomo. It is *not*
/// byte-identical to the benchmark corpus's AMPL-generated `cresc4.nl` from the
/// issue, which was not reachable from this environment. It reproduces the
/// reported failure signature exactly — `Infeasible_Problem_Detected` at a
/// near-zero objective with a constraint violation near a half, recovered under
/// both `mu_strategy=adaptive` and `nlp_scaling_method=none` — but the
/// iteration counts differ (the corpus file exits at 74, this one at 61).
const CRESC4_STAR: f64 = 0.871_897_6;

#[test]
fn cresc4_is_not_reported_infeasible() {
    let report = solve("cresc4.nl", &[]);
    let code = report.solution.solve_result_num;
    assert!(
        !(200..300).contains(&code),
        "cresc4 reported INFEASIBLE (solve_result_num={code}, status={:?}) — it \
         is feasible, with f* = {CRESC4_STAR} reached by LOQO, SNOPT and Ipopt \
         (71 iterations). Pre-fix POUNCE stopped at a point with constraint \
         violation 0.51 and objective ~2e-09, and the MC64 rung of the retry \
         ladder re-traced the same trajectory and 'corroborated' it",
        report.solution.status,
    );
    let obj = report.solution.objective;
    let rel = (obj - CRESC4_STAR).abs() / CRESC4_STAR;
    assert!(
        rel < 1e-6,
        "cresc4: got objective {obj}, expected the known optimum {CRESC4_STAR} \
         (rel err {rel:.3e})",
    );
}

/// gh #693: no rung carries `cresc4` any more — the base trajectory reaches the
/// optimum on its own.
///
/// This test used to assert the opposite. It pinned the *mechanism* behind the
/// test above: that rung 1 alone still failed, so the barrier rung was what
/// recovered `cresc4`, and a future change collapsing the ladder back to a
/// single MC64 re-solve would be caught here rather than surfacing as an
/// unexplained failure upstairs. Its message asked whoever made it pass to
/// re-derive whether the barrier rung was still carrying the case. gh #693 made
/// it pass, and the re-derivation is this:
///
/// ```text
///                                    main (0.10.0)                this change
///   both rungs off      Infeasible_Problem_Detected, 75    SolveSucceeded, 69
///   rung 2 off          Infeasible_Problem_Detected, 75    SolveSucceeded, 69
///   defaults            SolveSucceeded, 75 (promoted)      SolveSucceeded, 69
/// ```
///
/// With the *whole* ladder disabled the monotone-mu trajectory now walks to
/// `0.8718975393` — the Ipopt-MA57 reference figure to all ten published
/// digits, and closer to it than the ladder-promoted `0.871897548` main
/// returned. The false-infeasibility verdict is gone at its cause rather than
/// second-guessed after the fact, so the assertion is inverted: what is pinned
/// now is that neither rung is needed.
///
/// The cause was the damped equality-multiplier initializer. `y0` came out of a
/// Tikhonov-perturbed least-squares solve (`delta = 1e-8`), which biases `y`
/// by `O(delta / sigma_min(J)^2)`; on `cresc4` that bias was enough to send
/// iteration 0 down the basin containing the degenerate zero-area crescent.
/// See `LeastSquareMults::calculate_y_eq`.
///
/// **What this costs in coverage, stated plainly.** The ladder's rung-2
/// diversity property — that varying the linear algebra is not a second opinion
/// and varying the trajectory is — no longer has a witness in this repo.
/// `discs.nl`, rung 1's own canonical case, is not a fixture here either. Both
/// rungs remain on by default and are still exercised end-to-end by the unit
/// tests in `main.rs`'s `scaling_retry_tests`, but nothing in-tree now
/// demonstrates that the ladder changes an outcome on a real model. That is a
/// real gap and it is tracked, not shrugged off.
#[test]
fn cresc4_no_longer_needs_either_rung_of_the_ladder() {
    let unaided = solve(
        "cresc4.nl",
        &[
            "infeasibility_mu_strategy_retry=no",
            "feral_infeasibility_scaling_retry=no",
        ],
    );
    assert!(
        !(200..300).contains(&unaided.solution.solve_result_num),
        "cresc4 with the whole second-opinion ladder disabled reported \
         INFEASIBLE (status={:?}) — gh #693 removed the damped `y0` that put \
         the base trajectory in the wrong basin, so the monotone-mu path is \
         expected to reach the optimum unaided. If this fails, the initializer \
         change has been undone or diluted and gh #524 is being carried by the \
         ladder again",
        unaided.solution.status,
    );
    let obj = unaided.solution.objective;
    let rel = (obj - CRESC4_STAR).abs() / CRESC4_STAR;
    assert!(
        rel < 1e-6,
        "cresc4 unaided: got objective {obj}, expected the known optimum \
         {CRESC4_STAR} (rel err {rel:.3e})",
    );

    // …and the two knobs the rungs apply still work by hand, which is what
    // keeps them worth having for the models this repo does not carry.
    for opts in [
        vec!["mu_strategy=adaptive"],
        vec!["nlp_scaling_method=none"],
    ] {
        let r = solve("cresc4.nl", &opts);
        assert!(
            !(200..300).contains(&r.solution.solve_result_num),
            "cresc4 with {opts:?} should solve; got status={:?}",
            r.solution.status,
        );
    }
}

/// A rung that would not change anything must not burn a solve: with the
/// baseline already on `mu_strategy=adaptive` there is no second opinion left
/// to ask, and the ladder is empty.
#[test]
fn ladder_rungs_are_skipped_when_the_baseline_already_has_them() {
    let report = solve(
        "infeasible_equalities.nl",
        &["mu_strategy=adaptive", "feral_scaling=mc64"],
    );
    assert!(
        (200..300).contains(&report.solution.solve_result_num),
        "genuinely infeasible problem must still report infeasible with both \
         ladder knobs already set at baseline; got status={:?}",
        report.solution.status,
    );
}

/// The other direction, and the test whose absence caused two wrong fixes.
///
/// The probe withholds the verdict when a materially less-violating point sits
/// nearby, so it can only make the detector fire LESS. Genuine infeasibility
/// must therefore still be detected — otherwise the cure is worse than the
/// disease, since a solve that would have reported infeasibility in ~25
/// iterations instead grinds to `max_iter`.
///
/// `x³ + y³ == 1` and `x³ + y³ == 2` cannot both hold. From `x₀ = (1e4, 1e4)`
/// the Jacobian is steep, which is exactly the regime where the scaling games
/// above bite. Near the least-squares point the residuals are `±0.5` and only
/// ~0.07 % further descent exists — well under the probe's material-descent
/// margin — so the verdict is correctly issued.
///
/// Two earlier attempts at this fix passed every other test here and failed
/// this one: measuring the surrogate unscaled at the shipped `1e-8` tolerance
/// turned this into `Maximum_Iterations_Exceeded` at 3000 iterations.
#[test]
fn genuinely_infeasible_problem_is_still_detected() {
    let report = solve("infeasible_equalities.nl", &[]);
    let code = report.solution.solve_result_num;
    assert!(
        (200..300).contains(&code),
        "genuinely infeasible problem was NOT detected (solve_result_num={code}, \
         status={:?}). The descent probe must not suppress correct verdicts — \
         before this fix POUNCE reported local infeasibility here in ~25 \
         iterations",
        report.solution.status,
    );
}
