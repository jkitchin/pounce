//! Regression test for the presolve dual-block length bug (code-review
//! 2026-06 item M13).
//!
//! The NLP-path presolve wrapper (`pounce_presolve::PresolveTnlp`) drops
//! redundant constraint rows, so the solver works in a reduced
//! (kept-row) space. The CLI captures the converged duals from *outside*
//! that wrapper, so before the fix the `.sol` / JSON dual block carried
//! only the reduced `m_out` rows — shorter than the originating `.nl`'s
//! `m`. AMPL / Pyomo read the dual block positionally against the `.nl`,
//! so a short block mis-aligns or is rejected.
//!
//! `PresolveTnlp::finalize_solution` already lifts the duals back to the
//! original row order (with dropped-row multiplier recovery); the CLI now
//! prefers that full-length vector. This test pins:
//!   - presolve genuinely drops rows on the fixture (else it proves
//!     nothing), and
//!   - the lifted dual block has the *original* `.nl` constraint count,
//!     matching the no-presolve baseline and the report's own
//!     `n_constraints` (which `SolutionInfo::lambda` is documented to
//!     equal).

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

fn tmp_json(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_m13_{}_{seq}_{tag}.json",
        std::process::id()
    ));
    p
}

/// Run the CLI through the general NLP path and return
/// `(report, stderr)`. `solver_selection=nlp` is required: under the
/// default `auto` route `lp_afiro` would dispatch to the convex IPM,
/// which never wraps `PresolveTnlp`.
fn run(extra: &[&str]) -> (SolveReport, String) {
    let json_path = tmp_json("e2e");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture("lp_afiro.nl"))
        .arg("solver_selection=nlp")
        .arg("--json-output")
        .arg(&json_path)
        .arg("--json-detail")
        .arg("full");
    for a in extra {
        cmd.arg(a);
    }
    let output = cmd.output().expect("spawn pounce");
    // The presolve summary ("... dropped N redundant rows ...") prints to
    // stdout; the "wrote .json/.sol" lines to stderr. Combine both so the
    // dropped-rows guard below can see the summary.
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    assert!(
        output.status.success(),
        "pounce failed (exit {:?}); output:\n{combined}",
        output.status.code(),
    );
    let text = std::fs::read_to_string(&json_path).unwrap();
    let _ = std::fs::remove_file(&json_path);
    let report = serde_json::from_str(&text).expect("parse SolveReport JSON");
    (report, combined)
}

#[test]
fn presolve_dual_block_keeps_original_nl_length() {
    let (baseline, _) = run(&["presolve=no"]);
    let m_full = baseline.solution.lambda.len();
    assert!(
        m_full > 0,
        "baseline produced no duals; fixture/route changed"
    );
    // The report's own invariant: lambda length == n_constraints.
    assert_eq!(
        m_full, baseline.problem.n_constraints as usize,
        "baseline lambda length disagrees with n_constraints"
    );

    let (presolved, stderr) = run(&["presolve=yes"]);

    // Guard: the test is only meaningful if presolve actually dropped
    // rows on this fixture. The stderr line reads
    // "... dropped N redundant rows ...".
    let dropped_some = stderr
        .split("dropped ")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|n| n.parse::<u32>().ok())
        .map(|n| n > 0)
        .unwrap_or(false);
    assert!(
        dropped_some,
        "presolve dropped no rows — test no longer exercises M13; output:\n{stderr}"
    );

    // The lifted dual block must regain the original .nl row count (it
    // was the reduced kept-row count before the fix), and stay
    // consistent with the report's n_constraints.
    assert_eq!(
        presolved.solution.lambda.len(),
        m_full,
        "presolve dual block length {} != original .nl m {m_full}",
        presolved.solution.lambda.len(),
    );
    assert_eq!(
        presolved.solution.lambda.len(),
        presolved.problem.n_constraints as usize,
        "presolve lambda length disagrees with reported n_constraints"
    );
}
