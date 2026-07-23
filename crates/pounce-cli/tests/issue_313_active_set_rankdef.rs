//! gh #313 — `qp-active-set` on a rank-deficient-but-consistent equality QP.
//!
//! Fixture `rankdef_eq_qp.nl` is the strictly convex QP
//!   min 0.5 (x0²+x1²+x2²) − x0 − 2 x1 − 3 x2
//!   s.t.  x0 + x1 == 2
//!         2 x0 + 2 x1 == 4        (exactly 2× the first row — redundant)
//!         −10 ≤ xi ≤ 10
//! whose equality block is exactly rank-deficient but consistent. Unique
//! optimum x* = (0.5, 1.5, 3), objective −6.75.
//!
//! Redundant/scaled-duplicate equality rows are routine in modelling, so this
//! must be handled without adversarial intent. Two things are pinned:
//!
//! 1. The active-set path SOLVES it — status optimal, objective −6.75 — the
//!    same answer `qp-ipm` and `nlp` find. (The issue was originally filed
//!    against a stale binary that aborted with "INTERNAL ERROR: Unknown
//!    SolverReturn value" / exit 1; a current binary solves it.)
//! 2. The console summary reports a FINITE scaled objective, not `nan`. The
//!    SQP result path left `final_scaled_objective` at its NaN default, so
//!    even a clean optimal active-set solve printed
//!    "Objective ...: nan  <unscaled>".

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("rankdef_eq_qp.nl");
    p
}

/// The rank-deficient equality QP must SOLVE on the active-set path to the
/// correct optimum — not crash, not fail, not return a wrong objective.
#[test]
fn active_set_solves_rank_deficient_equality_qp() {
    let json = std::env::temp_dir().join("pounce_issue313.json");
    let _ = std::fs::remove_file(&json);

    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg("solver_selection=qp-active-set")
        .output()
        .expect("spawn pounce");

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    assert_eq!(
        out.status.code(),
        Some(0),
        "should solve, exit 0:\n{combined}"
    );
    assert!(
        !combined.contains("INTERNAL ERROR") && !combined.contains("Unknown SolverReturn"),
        "must not hit the internal-error path:\n{combined}"
    );

    let report = std::fs::read_to_string(&json).expect("json report written");
    // Correct optimum objective is −6.75.
    let obj = extract_objective(&report);
    assert!(
        (obj + 6.75).abs() < 1e-6,
        "active-set objective {obj} != −6.75 (correct optimum):\n{report}"
    );
    assert!(
        report.contains("\"status\": \"SolveSucceeded\"") || report.contains("SolveSucceeded"),
        "expected SolveSucceeded status:\n{report}"
    );
}

/// The console must not print a `nan` scaled objective (gh #313): the SQP
/// result path now mirrors the unscaled objective into
/// `final_scaled_objective` instead of leaving it at the NaN default.
#[test]
fn active_set_solve_reports_finite_scaled_objective() {
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("solver_selection=qp-active-set")
        .output()
        .expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);

    // Find the "Objective ...:" summary line and confirm neither column is nan.
    let obj_line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("Objective"))
        .unwrap_or_else(|| panic!("no Objective summary line:\n{stdout}"));
    assert!(
        !obj_line.to_ascii_lowercase().contains("nan"),
        "scaled/unscaled objective must be finite, got:\n{obj_line}"
    );
    assert!(
        obj_line.contains("-6.75"),
        "objective summary should show the −6.75 optimum:\n{obj_line}"
    );
}

/// Minimal JSON scrape of the `"objective"` number from the solve report.
fn extract_objective(report: &str) -> f64 {
    let key = "\"objective\":";
    let i = report.find(key).expect("objective key in report");
    let tail = &report[i + key.len()..];
    let tail = tail.trim_start();
    let end = tail
        .find(|c: char| c == ',' || c == '}' || c == '\n')
        .unwrap_or(tail.len());
    tail[..end].trim().parse().expect("parse objective number")
}
