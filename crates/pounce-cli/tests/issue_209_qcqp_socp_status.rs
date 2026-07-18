//! Regression for pounce#209: a feasible convex QCQP solved correctly by the
//! `.nl` → SOCP conic path was reported as a failure.
//!
//! Fixture `qcqp_ball.nl` is `min 3x₀ + 4x₁ s.t. x₀² + x₁² ≤ 4`, box
//! `[-10, 10]` — the linear objective over a ball, whose closed-form optimum
//! needs no oracle: `x* = −r·c/‖c‖ = (−1.2, −1.6)`, `f* = −r‖c‖ = −10`.
//!
//! Two independent defects made that solve look like an internal error:
//!
//!   1. The end-of-run KKT summary measured the SOC rows with the orthant's
//!      per-row `Gx ≤ h` test. A converged second-order cone block routinely
//!      has individual rows with `Gx > h` (only `s₀ ≥ ‖s₁‖` must hold), so a
//!      feasible point reported a large `Constraint violation` — in fact
//!      `≈ √2·max(0, −min xᵢ)`, a quantity with nothing to do with the
//!      quadratic constraint at all.
//!   2. The conic driver's convergence test reads the *homogeneous* residuals,
//!      which carry the internal slack's consistency `Gx + s − hτ` alongside
//!      the real KKT quantities. That term is bookkeeping (`s` is never
//!      returned) and it floors out once μ ~1e-16 makes the NT scaling
//!      ill-conditioned: here it bottomed at 1e-8, wandered back up to 1e-4,
//!      and the solve ground on to a factorization breakdown — all while the
//!      iterate itself was accurate to 1e-14. The verdict is now taken from the
//!      true KKT error of the point actually returned, so this reports
//!      `Optimal`.
//!
//! The checks below pin the user-visible contract: right answer, success-family
//! status, exit 0, and a summary whose residuals actually describe the point.

use pounce_solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("qcqp_ball.nl");
    p
}

fn run(selection: &str) -> (String, Option<i32>) {
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg(format!("solver_selection={selection}"))
        .output()
        .expect("spawn pounce");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        out.status.code(),
    )
}

/// Pull a labelled value out of the Ipopt-style end-of-run block, e.g.
/// `Constraint violation....:   0.0000e+00    0.0000e+00` → the first number.
fn summary_value(stdout: &str, label: &str) -> f64 {
    let line = stdout
        .lines()
        .find(|l| l.starts_with(label))
        .unwrap_or_else(|| panic!("no `{label}` line in summary; stdout=\n{stdout}"));
    let rhs = line.split(':').nth(1).expect("label line has a colon");
    rhs.split_whitespace()
        .next()
        .expect("value after colon")
        .parse()
        .expect("summary value parses as f64")
}

/// The conic path must solve the ball QCQP and *say* it solved it: a
/// success-family status and exit 0. Previously: "Numerical failure in KKT
/// factorization", exit 1.
#[test]
fn feasible_qcqp_on_conic_path_reports_success_and_exits_zero() {
    for selection in ["socp", "auto"] {
        let (stdout, code) = run(selection);
        assert_eq!(
            code,
            Some(0),
            "solver_selection={selection} must exit 0; stdout=\n{stdout}"
        );
        let lower = stdout.to_lowercase();
        assert!(
            !lower.contains("numerical failure"),
            "solver_selection={selection} must not report a numerical failure; stdout=\n{stdout}"
        );
        assert!(
            lower.contains("optimal solution found"),
            "solver_selection={selection} should report a clean optimum, not reduced \
             accuracy; stdout=\n{stdout}"
        );
        // Sanity: it really did take the conic path (not a silent NLP fallback).
        assert!(
            lower.contains("pounce-convex"),
            "solver_selection={selection} should route to the conic IPM; stdout=\n{stdout}"
        );
    }
}

/// The summary block must describe the point that was returned. A feasible
/// iterate has ~zero cone violation; the orthant-only metric reported ≈3.2 here
/// (`√2·max(0, −min xᵢ)` at `x = (−1.2, −1.6)`), which then dominated the
/// "Overall NLP error" and made a solved problem look badly infeasible.
#[test]
fn conic_summary_reports_cone_violation_not_per_row_orthant_violation() {
    let (stdout, _) = run("socp");
    let viol = summary_value(&stdout, "Constraint violation");
    let nlp_err = summary_value(&stdout, "Overall NLP error");
    assert!(
        viol < 1e-6,
        "feasible QCQP must report ~zero constraint violation, got {viol}; stdout=\n{stdout}"
    );
    assert!(
        nlp_err < 1e-6,
        "overall NLP error must be small for a solved QCQP, got {nlp_err}; stdout=\n{stdout}"
    );
}

/// The JSON report's status and objective must agree with the console, and the
/// primal must be the closed-form optimum. `InternalError` here is what a
/// Pyomo/AMPL wrapper or a CI gate would trip on.
#[test]
fn conic_json_report_status_is_success_family_with_correct_optimum() {
    let json = std::env::temp_dir().join("pounce_issue_209_qcqp_ball.json");
    let _ = std::fs::remove_file(&json);
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg("solver_selection=socp")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "must exit 0");

    let text = std::fs::read_to_string(&json).expect("read JSON report");
    let report: SolveReport = serde_json::from_str(&text).expect("parse JSON report");
    let status = format!("{:?}", report.solution.status);
    // Not merely a success-family status: this point satisfies the KKT
    // conditions to ~1e-14, six orders inside `tol`, so anything short of a
    // clean `SolveSucceeded` would be understating a converged solve.
    assert_eq!(
        status, "SolveSucceeded",
        "a solve accurate to ~1e-14 must report a clean success, got {status}"
    );
    assert!(
        (report.solution.objective - -10.0).abs() < 1e-6,
        "objective should be −10, got {}",
        report.solution.objective
    );
    let x = &report.solution.x;
    assert!(
        (x[0] - -1.2).abs() < 1e-5 && (x[1] - -1.6).abs() < 1e-5,
        "primal should be (−1.2, −1.6), got {x:?}"
    );

    let _ = std::fs::remove_file(&json);
}
