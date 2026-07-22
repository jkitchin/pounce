//! Regression: the dual-divergence guard's diversion must never leave the
//! solve worse off than a point it already had in hand (pounce#250 follow-up).
//!
//! The guard (`dual_diverging_streak`, added for the emfl050 warm-start stall)
//! routes a solve into restoration once the dual infeasibility has grown for 15
//! consecutive iterations in an elevated regime. That is a *bet*: usually a good
//! one — on the MINLPLib corpus it rescues twice as many models as it harms —
//! but nothing made losing it safe.
//!
//! On `autocorr_bern55-06` the bet loses. The guard fires at iteration 23; the
//! diverted run then reaches the true optimum (-2304.0000278, which Ipopt also
//! finds) and holds it from iteration 57 to 86, but the dual residual sawtooths
//! between 1e-8 and 2e-1 there, so it never strings together the
//! `acceptable_iter` consecutive qualifying iterates that would stop the solve.
//! It enters restoration a second time, wanders into a worse basin, and stops at
//! -2263.46 — 1.8 % worse, with an overall NLP error of **1.0**, reported as
//! "solved to acceptable level".
//!
//! The better point had already passed the acceptable test; it was simply
//! overwritten, because `store_acceptable_point` keeps the latest rather than
//! the best. The fix records the best acceptable iterate once the guard has
//! fired and hands it back if the diverted run ends worse.
//!
//! Tuning the guard's trigger instead was tried and rejected: every streak
//! setting that spares this model (>= 25) also loses the `deb7`/`deb9`/`deb8`
//! rescues, which need exactly the default 15. Hence fixing the consequence,
//! not the trigger — and hence `dual_guard_rescue_is_preserved` below, which
//! pins one of those rescues so a future retune cannot quietly trade it away.

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
        "pounce_issue250_{}_{}_{suffix}",
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

/// Ipopt's answer on the identical `.nl`: -2304.0000278027342, reached in 72
/// iterations with an overall NLP error of 3.7e-9.
const AUTOCORR_OPTIMUM: f64 = -2304.000_027_802_734_2;

/// With the guard at its default the diversion must not cost the objective.
/// Pre-fix this returned -2263.4612448099933.
#[test]
fn dual_guard_diversion_does_not_return_a_worse_point() {
    let report = solve("autocorr_bern55-06.nl", &["max_wall_time=10"]);
    let obj = report.solution.objective;
    let rel = (obj - AUTOCORR_OPTIMUM).abs() / AUTOCORR_OPTIMUM.abs();
    assert!(
        rel < 1e-6,
        "dual-divergence guard diverted the solve into a worse basin: got {obj}, \
         expected the optimum {AUTOCORR_OPTIMUM} that the same solve already \
         visited (rel err {rel:.3e}; pounce#250 follow-up)",
    );
}

/// The returned point must also be a stationary one. The pre-fix answer was
/// feasible (constraint violation 3.3e-12) but carried an overall NLP error of
/// 1.0, so asserting on the objective alone would not have caught a future
/// regression that returned a different feasible-but-not-KKT point.
#[test]
fn dual_guard_diversion_returns_a_stationary_point() {
    let report = solve("autocorr_bern55-06.nl", &["max_wall_time=10"]);
    let err = report.statistics.final_kkt_error;
    assert!(
        err < 1e-4,
        "returned point is not stationary: overall KKT error {err} (pre-fix this \
         was 1.0 — feasible but nowhere near a KKT point; pounce#250 follow-up)",
    );
}

/// The guard is load-bearing at its default streak of 15: this model is only
/// solved *because* it fires. That is the reason the fix targets the guard's
/// consequence rather than its trigger, so pin it — a future retune that raises
/// the streak past ~25 to "fix" autocorr would silently trade this away.
///
/// `deb9` and `deb8` in the same corpus family exercise the identical mechanism
/// (`deb9` 104.53 -> 97.56, `deb8` 4005.89 -> 1350.42); only `deb7` is vendored,
/// since three ~150 kB fixtures would buy no coverage the first does not.
#[test]
fn dual_guard_rescue_is_preserved() {
    let expected = 97.559_934_894_163_59;
    let report = solve("deb7.nl", &["max_wall_time=10"]);
    let obj = report.solution.objective;
    let rel = (obj - expected).abs() / expected.abs();
    assert!(
        rel < 1e-6,
        "deb7: lost the dual-divergence guard's rescue — got {obj}, expected \
         {expected}. Disabling the guard (or raising its streak past ~25) \
         returns ~104.95 here (pounce#250 follow-up)",
    );
}
