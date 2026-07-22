//! Issue #257 regression: POUNCE must issue a termination certificate on the
//! branch-and-bound *node* subproblems that a spatial B&B driver actually
//! generates for MINLPLib `jit1` — not just on the published root relaxation
//! that #248 / #252 pinned.
//!
//! `jit1_node.nl` is the first node at which POUNCE reported failure during a
//! real discopt B&B run: `jit1`'s continuous relaxation on a tightened box,
//! captured verbatim (bounds and starting point both). Its distinguishing
//! feature is a scale mix the earlier fixtures do not have — 12 tightly boxed
//! reciprocal-objective variables around 1e-3 alongside 9 variables with
//! `ub = +inf` started far out at ~50 — which drives the computed objective
//! scaling factor down to `1e-5`.
//!
//! That scale factor was the whole bug, and it is *not* a divergence problem:
//! the μ floor is `min(tol, compl_inf_tol) / (barrier_tol_factor + 1)`, but
//! `compl_inf_tol` is enforced on the **unscaled** complementarity while μ
//! lives in scaled space. At `df = 1e-5` the floor sat 1e5× too high, so μ
//! bottomed out at `9.09e-9` — an unscaled complementarity of `9.09e-4`,
//! permanently over the `1e-4` component tolerance. POUNCE therefore sat *on*
//! the optimum, with a scaled NLP error 10× under `tol`, unable to certify it;
//! μ-at-floor plus the vanishing step exited
//! `Search_Direction_Becomes_Too_Small`, which drivers read as UNBOUNDED.
//!
//! The tell is the inversion these tests pin: the failure appeared at *looser*
//! tolerances (1e-7, 1e-6) and not at the default 1e-8, because a smaller `tol`
//! pushed the floor below what the unscaled tolerance needed by accident.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

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
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_issue257_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

fn solve(fixture_name: &str, extra_opts: &[&str]) -> SolveReport {
    let json_path = tmp_path(&format!("{fixture_name}.json"));
    let sol_path = tmp_path(&format!("{fixture_name}.sol"));
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(fixture_name))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    for opt in extra_opts {
        cmd.arg(opt);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// The node's optimum, as reported by Ipopt (via cyipopt) on the identical
/// evaluator, starting point, and box: `173345.37683089852`.
const NODE_OPTIMUM: f64 = 173_345.376_830_898_52;

/// `solve_result_num` 0..100 is AMPL's "solved" band. A driver that maps
/// anything else onto its own status enum is what turned this into a spurious
/// UNBOUNDED downstream, so assert on the band, not on one code.
fn assert_solved_at_optimum(report: &SolveReport, ctx: &str) {
    let code = report.solution.solve_result_num;
    assert!(
        (0..100).contains(&code),
        "{ctx}: did not converge (solve_result_num={code}, status={:?}); \
         this node has a finite optimum ~{NODE_OPTIMUM} (issue #257)",
        report.solution.status,
    );
    let obj = report.solution.objective;
    assert!(
        (obj - NODE_OPTIMUM).abs() / NODE_OPTIMUM < 1e-6,
        "{ctx}: objective {obj} is not the known node optimum {NODE_OPTIMUM}",
    );
}

/// The tolerance a B&B driver actually requests for a node solve, and the one
/// that failed: `tol = 1e-7`.
#[test]
fn jit1_node_certifies_at_driver_tolerance() {
    let report = solve("jit1_node.nl", &["tol=1e-7"]);
    assert_solved_at_optimum(&report, "jit1 node (tol=1e-7)");
}

/// The whole loose-tolerance band must certify. Before the fix every one of
/// these failed while the default `1e-8` passed — a tolerance a user *loosens*
/// must never cost them the certificate.
#[test]
fn jit1_node_certifies_across_loosened_tolerances() {
    for tol in ["1e-8", "1e-7", "1e-6", "1e-5"] {
        let report = solve("jit1_node.nl", &[&format!("tol={tol}")]);
        assert_solved_at_optimum(&report, &format!("jit1 node (tol={tol})"));
    }
}

/// The unscaled complementarity — the quantity `compl_inf_tol` is actually
/// enforced on — must land under `compl_inf_tol` (1e-4). This pins the
/// mechanism rather than the symptom: pre-fix it was `9.09e-4`, held there by
/// a μ floor that could not go any lower, so no amount of further iteration
/// could have produced a certificate.
///
/// The report carries the *scaled* residual, so recover the objective scaling
/// factor from the two objective columns and undo it, exactly as
/// `curr_unscaled_complementarity_max` does.
#[test]
fn jit1_node_unscaled_complementarity_clears_compl_inf_tol() {
    let report = solve("jit1_node.nl", &["tol=1e-7"]);
    let stats = &report.statistics;
    let obj_scale = stats.final_scaled_objective / stats.final_objective;
    assert!(
        (obj_scale - 1e-5).abs() / 1e-5 < 1e-6,
        "expected this node to be scaled by df≈1e-5 (the condition that \
         triggers #257); got {obj_scale} — the fixture no longer exercises \
         the bug",
    );
    let unscaled_compl = stats.final_compl / obj_scale;
    assert!(
        unscaled_compl <= 1e-4,
        "unscaled complementarity {unscaled_compl} exceeds compl_inf_tol=1e-4, \
         so no strict certificate is reachable no matter how long the solve \
         runs (issue #257)",
    );
}
