//! gh#1042 — under `solver_selection=qp-active-set`, the reported
//! `iteration_count` is the work the convex active-set driver did, every
//! attempt included.
//!
//! The driver runs up to four attempts (an LP's simplex-seeded one, the
//! unscaled one, the Ruiz retry, the seeded last resort) and returned the
//! winner's count alone, so a model that needed three attempts reported one.
//! Inside an attempt, `pounce-qp`'s l1-elastic recovery did the same with its
//! passes: when phase-1 ended infeasible, the feasibility passes and the
//! Farkas probe were never counted, and a recovery that landed reported only
//! its own changes. `convex_qp_share1b` reported 2004 for 4125 changes done,
//! and `convex_qp_qscfxm1` 2195 for 17 848. Both were reporting only — no
//! budget, cap or cycle guard reads these counts.
//!
//! Oracle: the `POUNCE_AS_DEBUG` trace prints one `[as] engine=…` line per
//! attempt with that attempt's own engine count (`engine_iters=`), written in
//! `solve_translated` before the driver sums anything. The report must equal
//! their sum. `pounce-qp`'s `issue1042_elastic_accounting_tests` holds the
//! elastic passes inside an attempt to the same rule, against a ledger
//! written at the passes themselves.
//!
//! Mutation-checked: returning the winning attempt's own count from the
//! driver (`charged` a no-op) fails `every_driver_attempt_is_counted`.

use std::path::PathBuf;
use std::process::Command;

use pounce_solve_report::SolveReport;

fn fixture(name: &str) -> PathBuf {
    let mut fx = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fx.push("tests");
    fx.push("fixtures");
    fx.push(format!("{name}.nl"));
    fx
}

fn tmp_path(tag: &str, ext: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_gh1042_accounting_{tag}_{}.{ext}",
        std::process::id()
    ));
    p
}

/// `(report, per-attempt engine counts from the trace)`.
fn solve_traced(name: &str) -> (SolveReport, Vec<u64>) {
    let json = tmp_path(name, "json");
    let sol = tmp_path(name, "sol");
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fixture(name))
        .arg("--sol-output")
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .arg("solver_selection=qp-active-set")
        .env("POUNCE_AS_DEBUG", "1")
        .output()
        .expect("run pounce");
    let text = std::fs::read_to_string(&json).expect("JSON report");
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&sol);
    let report: SolveReport = serde_json::from_str(&text).expect("parse SolveReport JSON");
    let trace = String::from_utf8_lossy(&out.stderr);
    let attempts = trace
        .lines()
        .filter(|l| l.starts_with("[as] engine="))
        .map(|l| {
            let tail = l
                .rsplit_once("engine_iters=")
                .unwrap_or_else(|| panic!("{name}: no engine_iters in {l:?}"))
                .1;
            tail.trim().parse::<u64>().expect("engine_iters is a count")
        })
        .collect();
    (report, attempts)
}

/// `mu_fallback_point_floor` (n = 19, two equalities) ends every attempt
/// `OptimalInaccurate`, so the driver runs all three — unscaled, Ruiz, seeded
/// — and keeps the first. Measured 34 reported for 3 attempts before the fix.
#[test]
fn every_driver_attempt_is_counted() {
    let name = "mu_fallback_point_floor";
    let (r, attempts) = solve_traced(name);
    assert_eq!(r.solution.engine, "qp-active-set", "{name}: engine");
    assert!(
        attempts.len() >= 2,
        "{name}: {attempts:?} — the fixture no longer runs more than one \
         attempt, so it says nothing about the driver's accounting"
    );
    let total: u64 = attempts.iter().sum();
    assert!(
        attempts.iter().all(|&a| a < total),
        "{name}: {attempts:?} — one attempt did all the work, so the test \
         cannot tell its count from the sum"
    );
    assert_eq!(
        r.statistics.iteration_count as u64, total,
        "{name}: reported {} for attempts {attempts:?}",
        r.statistics.iteration_count
    );
}
