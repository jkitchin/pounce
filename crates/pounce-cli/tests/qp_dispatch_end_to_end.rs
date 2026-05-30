//! End-to-end: a convex-QP `.nl` file routed through the CLI dispatch to
//! the `pounce-convex` interior-point solver (Phase 2 wiring).
//!
//! Fixture `convex_qp.nl` is `min x0² + x1²  s.t.  x0 + x1 = 2`, whose
//! optimum is (1, 1) with objective 2. The tests check that:
//!   - `solver_selection=auto` classifies it as a convex QP and routes
//!     it to the convex IPM (banner names pounce-convex),
//!   - `solver_selection=qp-ipm` (forced) also solves it,
//!   - the `.sol` primal matches the known optimum,
//!   - `solver_selection=nlp` still solves the same file (no regression /
//!     same answer via the general path).

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("convex_qp.nl");
    p
}

#[test]
fn auto_routes_convex_qp_to_pounce_convex() {
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("solver_selection=auto")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0), "should solve");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("pounce-convex"),
        "auto should route the convex QP to pounce-convex; stdout=\n{stdout}"
    );
    assert!(
        stdout.contains("Optimal Solution Found"),
        "should report optimal; stdout=\n{stdout}"
    );
}

#[test]
fn forced_qp_ipm_solves() {
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("solver_selection=qp-ipm")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("pounce-convex"), "stdout=\n{stdout}");
}

#[test]
fn nlp_path_still_solves_same_file() {
    // No regression: the general NLP path must still handle the file.
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("solver_selection=nlp")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Optimal Solution Found"),
        "NLP path stdout=\n{stdout}"
    );
}

#[test]
fn sol_primal_matches_known_optimum() {
    let dir = std::env::temp_dir();
    let sol = dir.join("pounce_convex_qp_test.sol");
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--sol-output")
        .arg(&sol)
        .arg("solver_selection=auto")
        .output()
        .expect("spawn pounce");
    assert_eq!(out.status.code(), Some(0));
    let text = std::fs::read_to_string(&sol).expect("read .sol");
    // The primal block lists x0 then x1, each ≈ 1.0. Parse the trailing
    // floats and check the two that are closest to 1.0 are present.
    let near_one = text
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .filter(|v| (v - 1.0).abs() < 1e-5)
        .count();
    assert!(
        near_one >= 2,
        "expected two primal values ≈ 1.0 in .sol:\n{text}"
    );
}

/// The convex QP path's recovered constraint dual must match the NLP
/// path's dual on the same `.nl` file (the reference convention). For
/// `min x0²+x1² s.t. x0+x1=2` the equality multiplier is −2.
#[test]
fn qp_and_nlp_duals_agree() {
    let dir = std::env::temp_dir();

    let run = |sel: &str, out: &std::path::Path| {
        let _ = std::fs::remove_file(out);
        let status = Command::new(pounce_exe())
            .arg(fixture())
            .arg("--sol-output")
            .arg(out)
            .arg(format!("solver_selection={sel}"))
            .output()
            .expect("spawn pounce");
        assert_eq!(status.status.code(), Some(0), "{sel} failed");
        std::fs::read_to_string(out).expect("read .sol")
    };

    // The single constraint dual is the value closest to −2 in each
    // `.sol`'s float block.
    let dual_near = |text: &str| -> f64 {
        text.lines()
            .filter_map(|l| l.trim().parse::<f64>().ok())
            .min_by(|a, b| (a - (-2.0)).abs().partial_cmp(&(b - (-2.0)).abs()).unwrap())
            .expect("a float in .sol")
    };

    let qp_sol = run("qp-ipm", &dir.join("pounce_dual_qp.sol"));
    let nlp_sol = run("nlp", &dir.join("pounce_dual_nlp.sol"));

    let qp_dual = dual_near(&qp_sol);
    let nlp_dual = dual_near(&nlp_sol);
    assert!((qp_dual - (-2.0)).abs() < 1e-5, "QP dual {qp_dual} != −2");
    assert!(
        (qp_dual - nlp_dual).abs() < 1e-5,
        "QP dual {qp_dual} disagrees with NLP dual {nlp_dual}"
    );
}
