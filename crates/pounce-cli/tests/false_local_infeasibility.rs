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

/// And it must actually converge, to Ipopt's answer.
#[test]
fn hs13_from_remote_start_reaches_the_optimum() {
    let report = solve("hs13_bigstart.nl", &[]);
    let obj = report.solution.objective;
    let rel = (obj - HS13_IPOPT).abs() / HS13_IPOPT.abs();
    assert!(
        rel < 1e-6,
        "HS13 from the remote start: got {obj}, expected Ipopt's {HS13_IPOPT} \
         (rel err {rel:.3e})",
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
