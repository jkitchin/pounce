//! The dual-divergence guard (`dual_diverging_streak`, pounce#246) is **opt-in**,
//! and when opted into its diversion must never leave the solve worse off than a
//! point it already had in hand (pounce#250 follow-up).
//!
//! WHY IT IS OPT-IN. The guard routes a solve into restoration once the dual
//! infeasibility has grown for N consecutive iterations in an elevated regime. It
//! shipped default-on at N=15 to bound a reported emfl050 bad-warm-start grind,
//! but that justification did not survive being reproduced: the reported
//! measurement was caller-side JAX compilation, and the build predating the guard
//! solves both emfl050 instances to the same optimum in the same time. What
//! remained was an effect on four of 1284 MINLPLib models that is knife-edge and
//! non-monotone in the threshold:
//!
//!   * `deb7`/`deb9` reach a better local optimum (104.95 -> 97.56) at *exactly*
//!     15, and at no other value tried (0, 5, 25, 32, 40, 60) — on macOS/FERAL.
//!     On the Linux CI runner the same setting makes `deb7` *worse*: 97.56 with
//!     the guard off, 127.87 with it on. The effect differs by host in **sign**.
//!   * `pooling_rt2stp` turns from `Solve_Succeeded` into
//!     `Maximum_Iterations_Exceeded` at 10 and 15 only — solving cleanly at 0, 5,
//!     25 and 40.
//!
//! That is basin luck on nonconvex problems, not a property, and the two sides
//! are not commensurate: the upside is a better local optimum on an
//! already-solved problem, the downside is a clean solve becoming a failure. So
//! the guard stays available but is no longer imposed. `pooling_rt2stp_solves_at_default_settings`
//! pins that decision.
//!
//! THE NEVER-WORSE-OFF PROPERTY, for anyone who does opt in. On
//! `autocorr_bern55-06` at N=15 the guard fires at iteration 23; the diverted run
//! reaches the true optimum (-2304.0000278, which Ipopt also finds) and holds it
//! from iteration 57 to 86, but the dual residual sawtooths between 1e-8 and 2e-1
//! there, so it never strings together the `acceptable_iter` consecutive
//! qualifying iterates that would stop the solve. It then entered restoration a
//! second time, wandered into a worse basin, and stopped at -2263.46 — 1.8 %
//! worse, with an overall NLP error of **1.0**, reported as "solved to acceptable
//! level". The better point had already passed the acceptable test; it was simply
//! overwritten, because `store_acceptable_point` keeps the latest rather than the
//! best. POUNCE now records the best acceptable iterate and hands it back if the
//! diverted run ends worse.
//!
//! That fallback bounds the *objective* downside but cannot manufacture a point
//! the solve never reached — which is exactly why `pooling_rt2stp` still needed
//! the default change rather than more machinery.

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
const AUTOCORR_OPTIMUM: f64 = -2_304.000_027_802_734_2;

/// The guard is opt-in as of the pounce#250 follow-up (`dual_diverging_streak`
/// defaults to 0). Every test below must therefore enable it explicitly — at the
/// former default of 15 — or it would exercise nothing and pass vacuously.
const GUARD_ON: [&str; 2] = ["max_wall_time=10", "dual_diverging_streak=15"];

/// With the guard at its default the diversion must not cost the objective.
/// Pre-fix this returned -2263.4612448099933.
#[test]
fn dual_guard_diversion_does_not_return_a_worse_point() {
    let report = solve("autocorr_bern55-06.nl", &GUARD_ON);
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
    let report = solve("autocorr_bern55-06.nl", &GUARD_ON);
    let err = report.statistics.final_kkt_error;
    assert!(
        err < 1e-4,
        "returned point is not stationary: overall KKT error {err} (pre-fix this \
         was 1.0 — feasible but nowhere near a KKT point; pounce#250 follow-up)",
    );
}

// `deb7` carries NO absolute assertion, deliberately, and the reason is the
// single most decisive measurement in this investigation.
//
// The guard's effect on this model differs by host **in sign**:
//
// | host              | guard off | guard on (streak 15) |
// |-------------------|-----------|----------------------|
// | macOS / FERAL     |    104.95 |            **97.56** |
// | Linux CI runner   |     97.56 |           **127.87** |
//
// It helps on one platform and hurts on the other, from the same source, on the
// same fixture. Two earlier versions of this test tried to pin a value here —
// first "the guard rescues deb7 to 97.56", then the weaker "enabling the guard
// still reaches ~97.56" — and CI falsified both. There is no cross-platform
// claim to make.
//
// That is the finding, not an obstacle to testing it. A heuristic whose sign
// depends on the host is not a property of the algorithm, and it is why
// `dual_diverging_streak` is off by default. It also bounds what the
// best-acceptable fallback can promise: on Linux the guard costs deb7 30 % of
// its objective and the fallback cannot recover it, because 127.87 is the best
// acceptable point the diverted run ever reached — see
// `honour_best_acceptable_after_dual_guard`.
//
// deb7 is still exercised below, by the relative hair-trigger comparison, which
// is host-independent by construction.

/// The guard must stay off unless asked for. This is the model that decided it:
/// `pooling_rt2stp` solves cleanly with the guard disabled and at streaks 5, 25
/// and 40, but at 10 and 15 the diversion turns `Solve_Succeeded` into
/// `Maximum_Iterations_Exceeded` with an unscaled NLP error of ~18 and a 0.26 %
/// worse objective.
///
/// The best-acceptable fallback cannot rescue this one — the diverted solve never
/// reaches an acceptable point at all, so there is nothing to hand back. That is
/// what makes it a defaults question rather than more machinery.
///
/// Fails if the default is flipped back on.
#[test]
fn pooling_rt2stp_solves_at_default_settings() {
    // Ipopt on the identical .nl: -3273.9549927585640, NLP error ~1e-8.
    const EXPECTED: f64 = -3_273.954_991_374_754_6;
    let report = solve("pooling_rt2stp.nl", &["max_wall_time=10"]);
    let code = report.solution.solve_result_num;
    assert!(
        (0..100).contains(&code),
        "pooling_rt2stp did not converge at default settings \
         (solve_result_num={code}, status={:?}) — has dual_diverging_streak been \
         re-enabled by default? At 15 this model returns \
         Maximum_Iterations_Exceeded (pounce#250 follow-up)",
        report.solution.status,
    );
    let obj = report.solution.objective;
    let rel = (obj - EXPECTED).abs() / EXPECTED.abs();
    assert!(
        rel < 1e-6,
        "pooling_rt2stp: got {obj}, expected {EXPECTED} (rel err {rel:.3e})",
    );
}

/// The "never worse off" property under **maximum firing pressure**.
///
/// `dual_diverging_streak=1` is a hair trigger: the guard diverts on the first
/// growing dual-infeasibility step in the elevated regime, so it fires early and
/// often on models that would otherwise converge untouched. If the fallback were
/// incomplete, this is where it would show — a diversion landing somewhere the
/// solve cannot recover from.
///
/// This is also the test that covers the gap the first version of the fix had.
/// Recording was originally gated on `dual_guard_fired`, but the guard returns
/// to the driver *before* the recording site on the iteration it fires, so
/// nothing at or before the diversion was captured. `autocorr_bern55-06` did not
/// expose it — its better point arrives at iteration 86, long after the guard
/// fires at 23 — so a hair trigger, which moves the diversion as early as it can
/// go, is the sharper probe.
///
/// DELIBERATELY RELATIVE, no absolute reference. An earlier version asserted that
/// the guard-off arm reproduced a hard-coded objective, to stop fixture drift
/// making the test vacuous. That assertion was itself wrong: which local optimum
/// these nonconvex models land in is **platform-dependent**. `deb7` with the
/// guard off returns 104.95 on macOS/FERAL and 97.56 on the Linux CI runner, so
/// the hard-coded value failed in CI. Both are legitimate local optima; POUNCE
/// claims neither globally.
///
/// That platform dependence is not noise around the property being tested — it
/// *is* the property. A heuristic whose benefit varies by host is basin luck, and
/// it is the reason `dual_diverging_streak` is off by default.
///
/// Vacuity is covered by the two tests above, which pin that the guard fires on
/// `autocorr_bern55-06` at streak 15 and that the fallback recovers the optimum.
/// Here the comparison is between two live solves on the same machine and build,
/// so it cannot go stale.
#[test]
fn hair_trigger_guard_never_degrades_the_answer() {
    for name in ["autocorr_bern55-06.nl", "deb7.nl"] {
        let off = solve(name, &["max_wall_time=20", "dual_diverging_streak=0"]);
        let hair = solve(name, &["max_wall_time=20", "dual_diverging_streak=1"]);
        // Minimization: the hair-trigger arm must not return a higher objective
        // than the same build reaches with the guard disabled. A small relative
        // slack absorbs last-digit differences between two legitimately
        // different trajectories.
        let off_obj = off.solution.objective;
        let hair_obj = hair.solution.objective;
        let slack = 1e-6 * off_obj.abs().max(1.0);
        assert!(
            hair_obj <= off_obj + slack,
            "{name}: hair-trigger guard (dual_diverging_streak=1) returned \
             {hair_obj} — worse than the {off_obj} the same build reaches with \
             the guard disabled. The diversion left the solve worse off, which \
             the best-acceptable fallback exists to prevent (pounce#250 follow-up)",
        );
    }
}
