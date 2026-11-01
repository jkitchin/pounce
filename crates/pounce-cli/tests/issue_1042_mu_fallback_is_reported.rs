//! gh#1042 C1 — a `mu_strategy_fallback` retry must leave a trace in the
//! solve report.
//!
//! `run_with_mu_strategy_fallback` re-solves under the flipped `mu_strategy`
//! after a first attempt that ended `Solved_To_Acceptable_Level` or
//! `Maximum_Iterations_Exceeded`, and promotes the re-solve on
//! `Solve_Succeeded`. On a promotion `statistics.iteration_count` and
//! `statistics.restoration_calls` are the re-solve's alone, so before
//! `statistics.mu_strategy_fallback` existed nothing in the report said the
//! first attempt had happened. That is how gh#1042's R2b was mis-scored:
//! `pooling_rt2stp` exact read `492 -> 109` in the fixture sweep while its
//! first attempt had gone `Solve_Succeeded`/492 → `Solved_To_Acceptable_Level`
//! /494 (the 109 was the fallback's) — the gh#850 blind spot one mechanism
//! over.
//!
//! The fixtures reach the three shapes the record must tell apart, measured on
//! x86_64:
//!
//! | fixture (exact leg)        | first attempt        | re-solve                    | reported   |
//! |----------------------------|----------------------|-----------------------------|------------|
//! | `csfi2_stall_start`        | Acceptable / 12      | adaptive, Solve_Succeeded / 9 | promoted, `it=9` |
//! | `mu_fallback_point_floor`  | Acceptable / 55      | adaptive, Infeasible / 31   | kept, `it=55` |
//! | any, `mu_strategy_fallback=no` | —                | none                        | no record  |
//!
//! The promoted row was `csfi2` (Acceptable/35 → Solve_Succeeded/21) until
//! gh#1098: that first-attempt stall depends on the inertia correction's
//! `δ_s = δ_x`, and at `perturb_slack_fact=0.03` `csfi2.nl` certifies
//! directly (`Optimal`/37) with no record to read. `csfi2_stall_start.nl` is
//! the same model from a start on which the stall and the promotion above are
//! measured identically at `perturb_slack_fact` 1, 0.3, 0.1, 0.03 and 0.01,
//! and 17/17 under the round-off screen at each.
//!
//! Mutation-checked: with the record never written (`mu_strategy_fallback`
//! left `None`) the first two tests fail; with `promoted` hard-wired `false`
//! the first fails; with `retry_iteration_count` read before the re-solve the
//! first fails on the iteration identity.

use pounce_solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

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

fn solve(name: &str, tag: &str, opts: &[&str]) -> SolveReport {
    let json =
        std::env::temp_dir().join(format!("pounce_1042_c1_{tag}_{}.json", std::process::id()));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(pounce_exe())
        .arg(fixture(name))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .args(opts)
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|_| {
        panic!(
            "no JSON report for {name}:\n{}",
            String::from_utf8_lossy(&out.stdout)
        )
    });
    let _ = std::fs::remove_file(&json);
    serde_json::from_str(&text).expect("deserialize report")
}

/// A promoted re-solve: the record names the first attempt, and the reported
/// count is the re-solve's alone.
#[test]
fn a_promoted_fallback_records_the_first_attempt() {
    let r = solve("csfi2_stall_start.nl", "promoted", &[]);
    let mf = r.statistics.mu_strategy_fallback.as_ref().expect(
        "csfi2_stall_start's first attempt stops at the acceptable level and the fallback runs",
    );
    assert!(
        mf.promoted,
        "the re-solve should have been promoted: {mf:?}"
    );
    assert_eq!(mf.retry_status, "Solve_Succeeded", "{mf:?}");
    assert_eq!(
        mf.base_status, "Solved_To_Acceptable_Level",
        "the first attempt's verdict must be recorded: {mf:?}"
    );
    assert!(mf.base_iteration_count > 0, "{mf:?}");
    assert_eq!(
        mf.retry_iteration_count, r.statistics.iteration_count,
        "on a promotion the reported count is the re-solve's: {mf:?}"
    );
    assert_eq!(
        mf.total_iteration_count(),
        mf.base_iteration_count + r.statistics.iteration_count
    );
    assert!(
        mf.retry_mu_strategy == "adaptive" || mf.retry_mu_strategy == "monotone",
        "{mf:?}"
    );
}

/// A losing re-solve: the record says it ran and what it cost, and every other
/// number is the first attempt's (pounce#870's floor).
#[test]
fn a_losing_fallback_records_the_retry_and_keeps_the_first_attempt() {
    let r = solve("mu_fallback_point_floor.nl", "kept", &[]);
    let mf = r
        .statistics
        .mu_strategy_fallback
        .as_ref()
        .expect("the fallback runs on this fixture and loses (pounce#870)");
    assert!(!mf.promoted, "{mf:?}");
    assert_ne!(mf.retry_status, "Solve_Succeeded", "{mf:?}");
    assert_eq!(mf.base_status, "Solved_To_Acceptable_Level", "{mf:?}");
    assert_eq!(
        mf.base_iteration_count, r.statistics.iteration_count,
        "a floored run reports the first attempt's count: {mf:?}"
    );
    assert!(mf.retry_iteration_count > 0, "{mf:?}");
}

/// No retry, no record — including on a run that would otherwise have one.
#[test]
fn no_record_without_a_retry() {
    let r = solve("csfi2_stall_start.nl", "off", &["mu_strategy_fallback=no"]);
    // The run must be one that *would* retry: an acceptable-level exit on
    // stock termination options, the trigger the promoted test above reaches.
    assert_eq!(
        r.solution.status,
        pounce_nlp::return_codes::ApplicationReturnStatus::SolvedToAcceptableLevel,
        "this fixture no longer stalls, so the absence of a record proves nothing",
    );
    assert!(
        r.statistics.mu_strategy_fallback.is_none(),
        "{:?}",
        r.statistics.mu_strategy_fallback
    );
}
