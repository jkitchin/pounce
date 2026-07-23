//! gh #326 — `qp-active-set` on 3+ exact-duplicate / integer-combination
//! rank-deficient equality rows (a residual of #313/#321/#323).
//!
//! The model is the projection of `(1, 2)` onto the line `x + y = 2`:
//!   min (x − 1)² + (y − 2)²   s.t.  <redundant encodings of x + y = 2>
//! whose unique optimum is `x* = (0.5, 1.5)`, `f* = 0.5`.
//!
//! The #313 fix taught the equality+bounds active-set path to prune a
//! rank-deficient equality block, but the pure-equality / no-bounds fast path
//! `solve_equality_only` was left without that guard. Under
//! `solver_selection=qp-active-set` (unbounded variables → that fast path),
//! three identical rows or an integer-combination triple made the pinned KKT
//! singular; no H-block inertia shift can rescue a rank-deficient *constraint*
//! block, so the solve aborted with `LinearSolverFailure` → `InternalError` /
//! exit 1. `solve_equality_only` now delegates such a block to the
//! rank-deficiency-aware `solve_general`.
//!
//! Pinned here:
//!   * the 3-exact-duplicate and integer-combination encodings SOLVE to f = 0.5;
//!   * an *inconsistent* rank-deficient block (`x+y=2` and `x+y=3`) is still
//!     reported infeasible, not silently "solved" — the redundancy prune must
//!     not mask genuine infeasibility.

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

fn run_active_set(fixture_name: &str) -> (Option<i32>, String, Option<f64>) {
    let json = std::env::temp_dir().join(format!("pounce_issue326_{fixture_name}.json"));
    let _ = std::fs::remove_file(&json);

    let out = Command::new(pounce_exe())
        .arg(fixture(fixture_name))
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
    let obj = std::fs::read_to_string(&json)
        .ok()
        .and_then(|r| extract_objective(&r));
    (out.status.code(), combined, obj)
}

/// Three exactly-identical equality rows `x+y=2` must SOLVE (f = 0.5), not
/// crash with `InternalError` / exit 1.
#[test]
fn active_set_solves_three_exact_duplicate_equality_rows() {
    let (code, combined, obj) = run_active_set("rankdef_triple_eq_qp.nl");
    assert_eq!(code, Some(0), "should solve, exit 0:\n{combined}");
    assert!(
        !combined.contains("INTERNAL ERROR")
            && !combined.contains("Unknown SolverReturn")
            && !combined.contains("SQP solve failed"),
        "must not hit the internal-error path:\n{combined}"
    );
    let obj = obj.expect("json report with objective");
    assert!(
        (obj - 0.5).abs() < 1e-6,
        "objective {obj} != 0.5:\n{combined}"
    );
}

/// Integer-combination redundant rows `[x+y=2 ; 3x+3y=6 ; 4x+4y=8]`
/// (row3 = row1 + row2) must SOLVE (f = 0.5).
#[test]
fn active_set_solves_integer_combination_equality_rows() {
    let (code, combined, obj) = run_active_set("rankdef_intcomb_eq_qp.nl");
    assert_eq!(code, Some(0), "should solve, exit 0:\n{combined}");
    assert!(
        !combined.contains("INTERNAL ERROR")
            && !combined.contains("Unknown SolverReturn")
            && !combined.contains("SQP solve failed"),
        "must not hit the internal-error path:\n{combined}"
    );
    let obj = obj.expect("json report with objective");
    assert!(
        (obj - 0.5).abs() < 1e-6,
        "objective {obj} != 0.5:\n{combined}"
    );
}

/// An *inconsistent* rank-deficient equality block (`x+y=2` and `x+y=3`) must
/// still be reported infeasible — the redundancy prune must not mask genuine
/// infeasibility by dropping a contradictory row.
#[test]
fn active_set_rejects_inconsistent_rank_deficient_rows() {
    let (_code, combined, _) = run_active_set("inconsistent_eq_qp.nl");
    // Must NOT crash with an internal error, and must NOT declare an optimal
    // solution — the redundancy prune must not mask genuine infeasibility.
    // (An infeasible verdict legitimately exits nonzero.)
    assert!(
        !combined.contains("INTERNAL ERROR") && !combined.contains("Unknown SolverReturn"),
        "must not hit the internal-error path:\n{combined}"
    );
    assert!(
        combined.to_ascii_lowercase().contains("infeasib"),
        "inconsistent block must be reported infeasible:\n{combined}"
    );
    assert!(
        !combined.contains("Optimal Solution Found"),
        "inconsistent block must NOT be reported optimal:\n{combined}"
    );
}

/// Minimal JSON scrape of the `"objective"` number from the solve report.
fn extract_objective(report: &str) -> Option<f64> {
    let key = "\"objective\":";
    let i = report.find(key)?;
    let tail = report[i + key.len()..].trim_start();
    let end = tail
        .find(|c: char| c == ',' || c == '}' || c == '\n')
        .unwrap_or(tail.len());
    tail[..end].trim().parse().ok()
}
