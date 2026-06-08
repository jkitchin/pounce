//! Integration tests for the LP/QP dispatch routing (Phase 1).
//!
//! See `dev-notes/lp-qp-routing.md`. Phase 1 wires the `solver_selection`
//! option and the classifier but routes everything to the existing NLP
//! solver, so the only externally observable behavior is:
//!
//!   * `auto` / `nlp` solve exactly as before (no regression);
//!   * an unknown `solver_selection` value is rejected;
//!   * a forced specialized solver that does not match the detected
//!     problem class errors with a clear message (the plan's integration
//!     test: `--solver=lp` on an NLP should error).
//!
//! These use the `rosenbrock` builtin so they are hermetic — no `.nl`
//! fixture or fetched benchmark cache required.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

#[test]
fn auto_solves_builtin_unchanged() {
    let output = Command::new(pounce_exe())
        .arg("--problem")
        .arg("rosenbrock")
        .arg("solver_selection=auto")
        .output()
        .expect("spawn pounce");
    assert_eq!(
        output.status.code(),
        Some(0),
        "auto should solve rosenbrock; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn default_has_no_solver_selection_regression() {
    // Omitting solver_selection entirely must behave exactly as before.
    let output = Command::new(pounce_exe())
        .arg("--problem")
        .arg("rosenbrock")
        .output()
        .expect("spawn pounce");
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn forced_lp_on_nlp_errors() {
    // The plan's named integration test: forcing an LP solver on a
    // general NLP must error, naming both the detected class and the
    // forced solver.
    let output = Command::new(pounce_exe())
        .arg("--problem")
        .arg("rosenbrock")
        .arg("solver_selection=lp-ipm")
        .output()
        .expect("spawn pounce");
    assert_eq!(
        output.status.code(),
        Some(2),
        "forced mismatch should exit 2"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("NLP") && stderr.contains("lp-ipm"),
        "error should name detected class and forced solver: {stderr}"
    );
}

#[test]
fn forced_qp_solvers_on_nlp_error() {
    // The qp-family entry points (qp-ipm, qp-active-set) forced onto a
    // general NLP must error just like lp-ipm does — never fall through to
    // a wrong solve. The error names the detected class and forced solver.
    for sel in ["qp-ipm", "qp-active-set"] {
        let output = Command::new(pounce_exe())
            .arg("--problem")
            .arg("rosenbrock")
            .arg(format!("solver_selection={sel}"))
            .output()
            .expect("spawn pounce");
        assert_eq!(
            output.status.code(),
            Some(2),
            "{sel} on an NLP should exit 2"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("NLP") && stderr.contains(sel),
            "{sel}: error should name detected class and forced solver: {stderr}"
        );
    }
}

#[test]
fn unknown_solver_selection_rejected() {
    // `lp-simplex` was removed from scope; it must be rejected, not
    // silently accepted.
    let output = Command::new(pounce_exe())
        .arg("--problem")
        .arg("rosenbrock")
        .arg("solver_selection=lp-simplex")
        .output()
        .expect("spawn pounce");
    assert_eq!(output.status.code(), Some(2));
}
