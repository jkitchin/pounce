//! gh#1176: the second-opinion ladder's presolve rung (rung 6).
//!
//! The Waechter-Biegler textbook example is the case: from its standard
//! start the interior point settles at the locally infeasible `x1 = -1`
//! (violation 1.5), rungs 1-3 all agree, and the model is feasible with
//! optimum `x1 = 1`. Presolve's bound tightening reaches that optimum, so
//! the rung turns the default run's wrong verdict into the right answer.
//!
//! Two branches, two tests: the default run promotes the rung, and the same
//! run with the rung switched off keeps the infeasible verdict. The second
//! is what shows the rung is the thing doing the work. The third checks the
//! `.sol` the CLI writes, which takes its point and duals from a different
//! capture when this rung promotes (`presolve_rung_promoted` in `main.rs`),
//! on a variant where presolve drops a row: the duals must be the original
//! model's, at full length, and pass `pounce verify --require-optimal`.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_algorithm::second_opinion::PRESOLVE_RUNG_LABEL;
use pounce_cli::solve_report::SolveReport;
use pounce_nlp::return_codes::ApplicationReturnStatus;

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
        "pounce_issue1176_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

const WB: &str = "issue981_wachter_biegler.nl";
/// Waechter-Biegler plus `x2 + x3 >= -1`, which the bounds `x2, x3 >= 0`
/// imply, so presolve drops it: the rung's reduced problem has 2 rows where
/// the model has 3.
const WB_REDUNDANT_ROW: &str = "issue1176_wb_redundant_row.nl";

/// Solve a fixture; return the JSON report and the `.sol` path.
fn solve(name: &str, extra: &[&str]) -> (SolveReport, PathBuf) {
    let json_path = tmp_path("report.json");
    let sol_path = tmp_path("out.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(name))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path)
        .arg("max_wall_time=300");
    for o in extra {
        cmd.arg(o);
    }
    let _ = cmd.output().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    (
        serde_json::from_str(&text).expect("deserialize SolveReport"),
        sol_path,
    )
}

#[test]
fn the_presolve_rung_recovers_the_wachter_biegler_example() {
    let (r, sol) = solve(WB, &[]);
    let _ = std::fs::remove_file(&sol);
    let so = r
        .second_opinion
        .as_ref()
        .expect("the ladder did not run: the base solve no longer fails");
    assert_eq!(so.base_status, "Infeasible_Problem_Detected", "{so:?}");
    assert_eq!(
        so.promoted_by.as_deref(),
        Some(PRESOLVE_RUNG_LABEL),
        "{so:?}"
    );
    assert_eq!(
        so.tried.last().map(String::as_str),
        Some(PRESOLVE_RUNG_LABEL)
    );
    assert_eq!(r.solution.status, ApplicationReturnStatus::SolveSucceeded);
    let x = &r.solution.x;
    assert!((x[0] - 1.0).abs() < 1e-6, "x1 = {}; the optimum is 1", x[0]);
    assert!(
        (r.solution.objective - 1.0).abs() < 1e-6,
        "objective {}",
        r.solution.objective
    );
}

#[test]
fn without_the_presolve_rung_the_verdict_stays_infeasible() {
    let (r, sol) = solve(WB, &["infeasibility_presolve_retry=no"]);
    let _ = std::fs::remove_file(&sol);
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::InfeasibleProblemDetected,
        "something other than the presolve rung now recovers this model, so \
         the test above no longer shows the rung doing the work"
    );
    if let Some(so) = r.second_opinion.as_ref() {
        assert!(!so.tried.iter().any(|t| t == PRESOLVE_RUNG_LABEL), "{so:?}");
    }
}

/// The `.sol` for a promoted rung must carry the original model's duals. The
/// rung's reduced problem has one row fewer than the model here; before the
/// CLI took this rung's answer from `finalize_solution`, the `.sol` carried
/// the reduced problem's 2 duals against 3 constraints, and `pounce verify`
/// refused it ("solution carries 2 dual values but the problem has 3
/// constraints") -- checked by disabling that branch, not assumed.
#[test]
fn the_promoted_presolve_rung_writes_the_original_models_duals() {
    let (r, sol) = solve(WB_REDUNDANT_ROW, &[]);
    let so = r.second_opinion.as_ref().expect("the ladder did not run");
    assert_eq!(
        so.promoted_by.as_deref(),
        Some(PRESOLVE_RUNG_LABEL),
        "{so:?}"
    );
    assert_eq!(r.solution.status, ApplicationReturnStatus::SolveSucceeded);
    let lambda = &r.solution.lambda;
    assert_eq!(lambda.len(), 3, "lambda {lambda:?}");
    let out = Command::new(pounce_exe())
        .arg("verify")
        .arg(fixture(WB_REDUNDANT_ROW))
        .arg(&sol)
        .arg("--require-optimal")
        .output()
        .expect("spawn pounce verify");
    let _ = std::fs::remove_file(&sol);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        text.contains("VERDICT: VERIFIED"),
        "the .sol written for the promoted rung is not a KKT point of the \
         original model:\n{text}"
    );
}
