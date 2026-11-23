//! gh#1251: the CLI reads QPS / MPS files directly, with no `.nl`
//! conversion. `pounce model.qps` / `pounce model.mps` go through
//! `problem_file::read_problem_file`, which builds the same in-memory
//! `NlProblem` the `.nl` reader does, so classification, the convex
//! dispatch, the NLP arm and the `.sol` writer all apply unchanged.
//!
//! Each fixture is solved on both arms — `auto` (which routes an LP / convex
//! QP to the convex engine) and `nlp` — because the conversion feeds both and
//! a mistake in it (a missing ½ on the diagonal, a dropped OBJSENSE) would
//! show on each independently.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;

fn tmp_path(name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "pounce_issue1251_{}_{n}_{name}",
        std::process::id()
    ))
}

/// Write `text` to a file named `name`, solve it, return the JSON report.
fn solve(name: &str, text: &str, selection: &str) -> SolveReport {
    let input = tmp_path(name);
    std::fs::write(&input, text).unwrap();
    let json = tmp_path("report.json");
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(&input)
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg(format!("solver_selection={selection}"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "pounce {name} failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let report: SolveReport =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&json);
    report
}

/// min ½(x₁² + x₂²) + ½x₁x₂ − x₁ − 2x₂ + 3  s.t.  x₁ + x₂ ≤ 1,  0 ≤ x ≤ 1.
///
/// The QUADOBJ off-diagonal entry is listed once (lower triangle), so the
/// objective carries `H₂₁·x₁x₂ = ½x₁x₂`, and the RHS on the objective row
/// contributes `−(−3) = +3`. On the edge x₂ = 1 − t, x₁ = t the objective is
/// `½t² + ½t + 1.5`, minimized at t = 0, and the free gradient there,
/// (−½, −1), points into the row: the optimum is 1.5 at (0, 1). Dropping the
/// ½ on the diagonal would give 2.0. The off-diagonal term vanishes at
/// (0, 1), so its coefficient is pinned instead by the unit test in
/// `problem_file.rs`, which evaluates the objective at (2, 3).
const QP: &str = "\
NAME          QP1251
ROWS
 N  COST
 L  C1
COLUMNS
    X1        COST       -1.0   C1         1.0
    X2        COST       -2.0   C1         1.0
RHS
    RHS       C1         1.0    COST      -3.0
BOUNDS
 UP BND       X1         1.0
 UP BND       X2         1.0
QUADOBJ
    X1        X1         1.0
    X2        X1         0.5
    X2        X2         1.0
ENDATA
";

/// max 3x + 2y  s.t.  x + y ≤ 4,  x + 3y ≤ 6,  x, y ≥ 0  →  12 at (4, 0).
const LP_MAX: &str = "\
NAME LP1251
OBJSENSE
    MAX
ROWS
 N obj
 L c1
 L c2
COLUMNS
    x obj 3 c1 1
    x c2 1
    y obj 2 c1 1
    y c2 3
RHS
    rhs c1 4 c2 6
ENDATA
";

#[test]
fn qps_quadratic_objective_solves_on_both_arms() {
    let want = 1.5;
    for sel in ["auto", "nlp"] {
        let r = solve("qp1251.qps", QP, sel);
        assert!(
            (r.solution.objective - want).abs() < 1e-6,
            "{sel}: objective {} != {want}",
            r.solution.objective
        );
    }
    let r = solve("qp1251.QPS", QP, "auto");
    assert_eq!(
        r.solution.engine, "cvx-qp",
        "convex QP should reach the convex arm"
    );
}

#[test]
fn mps_lp_with_objsense_max_solves_on_both_arms() {
    for sel in ["auto", "nlp"] {
        let r = solve("lp1251.mps", LP_MAX, sel);
        assert!(
            (r.solution.objective - 12.0).abs() < 1e-6,
            "{sel}: objective {} != 12 (OBJSENSE MAX dropped?)",
            r.solution.objective
        );
    }
}
