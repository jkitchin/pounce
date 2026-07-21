//! Issue #248 regression: POUNCE must not report a spurious UNBOUNDED
//! status (AMPL `solve_result_num` in the 300 range) on the MINLPLib
//! `jit1` continuous relaxation, which is bounded below with a finite
//! optimum (`obj ≈ 173345`, as ipopt and SCIP find).
//!
//! `jit1`'s objective is `Σ cᵢ/xᵢ` plus a badly-scaled linear tail
//! (coefficients up to 1e7). The magnitude-only divergence guard could
//! trip on the resulting large excursion and tag the finite optimum as
//! `DivergingIterates` — Ipopt's unboundedness verdict. Two fixtures pin
//! the fix:
//!
//! * `jit1.nl` — the model verbatim. At the default
//!   `diverging_iterates_tol` it already converges; this guards against a
//!   future regression that starts tagging it unbounded.
//! * `jit1_boxed.nl` — the reporter's experiment, every variable clamped
//!   to a finite box (±100). A bounded box cannot be unbounded, yet with a
//!   low `diverging_iterates_tol` (the kind a branch-and-bound driver sets
//!   to abort runaway nodes) the pre-fix guard reported UNBOUNDED anyway.
//!   Post-fix the structural guard recognises the finite box and returns
//!   the finite optimum instead.

use std::path::PathBuf;
use std::process::Command;

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
    let mut p = std::env::temp_dir();
    p.push(format!("pounce_issue248_{}_{suffix}", std::process::id()));
    p
}

/// Solve `fixture_name` with the given extra `KEY=VALUE` options and return
/// the parsed report.
fn solve(fixture_name: &str, extra_opts: &[&str]) -> SolveReport {
    let json_path = tmp_path(&format!("{fixture_name}.json"));
    // Direct the `.sol` output to tmp too; without an explicit path pounce
    // writes `<stem>.sol` next to the input, polluting the fixtures dir.
    let sol_path = tmp_path(&format!("{fixture_name}.sol"));
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(fixture_name))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    for opt in extra_opts {
        cmd.arg(opt);
    }
    let status = cmd.status().expect("spawn pounce");
    // A non-unbounded, non-success terminal status (e.g. "search direction
    // too small") still exits non-zero; we assert on the report contents,
    // not the process code.
    let _ = status;
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

const JIT1_OPTIMUM: f64 = 173_345.0;

fn assert_not_unbounded_at_optimum(report: &SolveReport, ctx: &str) {
    let code = report.solution.solve_result_num;
    assert!(
        !(300..400).contains(&code),
        "{ctx}: reported UNBOUNDED (solve_result_num={code}, status={:?}); \
         jit1 is bounded below with a finite optimum (issue #248)",
        report.solution.status,
    );
    let obj = report.solution.objective;
    assert!(
        (obj - JIT1_OPTIMUM).abs() / JIT1_OPTIMUM < 1e-3,
        "{ctx}: objective {obj} is not the known optimum ~{JIT1_OPTIMUM}",
    );
}

/// The model as published: at the default divergence threshold it converges
/// to the finite optimum and must not be flagged unbounded.
#[test]
fn jit1_default_is_not_unbounded() {
    let report = solve("jit1.nl", &[]);
    assert_not_unbounded_at_optimum(&report, "jit1 (default)");
}

/// The published model has *free* / unbounded-above variables, so the
/// structural guard alone cannot rule out unboundedness. With a low
/// `diverging_iterates_tol` (the kind a branch-and-bound driver sets to
/// abort runaway nodes) `jit1`'s iterate briefly climbs past the threshold —
/// `|x|` peaks around 16 — before receding to the finite optimum near 2.9.
/// The growth-persistence check recognises the excursion is transient and
/// lets the solve converge instead of reporting a spurious UNBOUNDED.
#[test]
fn jit1_free_vars_low_diverging_tol_is_not_unbounded() {
    let report = solve("jit1.nl", &["diverging_iterates_tol=2"]);
    assert_not_unbounded_at_optimum(&report, "jit1 (free vars, diverging_iterates_tol=2)");
}

/// The reporter's strengthened case: every variable boxed to ±100. A bounded
/// feasible region cannot be unbounded, so even with a divergence threshold
/// low enough to trip on the (finite) iterates, the solve must return the
/// finite optimum rather than a spurious UNBOUNDED.
#[test]
fn jit1_boxed_is_not_unbounded_even_with_low_diverging_tol() {
    let report = solve("jit1_boxed.nl", &["diverging_iterates_tol=2"]);
    assert_not_unbounded_at_optimum(&report, "jit1 boxed (diverging_iterates_tol=2)");
}
