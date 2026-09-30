//! gh#974 follow-through — NETLIB `SHARE1B` and `DEGEN2` under
//! `solver_selection=qp-active-set`.
//!
//! All three reached the gh#974 hard error on the cold homotopy route, and with
//! it removed they exposed the engine defects behind it rather than an answer:
//! `Internal_Error` on each (after 7 s, 87 s and 286 s on `main`). Each step
//! from a feasible point should keep it feasible, and here several did not:
//!
//! * a Schur-complement block with a smallest pivot of `6.3e-16` was accepted
//!   and returned a step of `‖p‖∞ = 6e19`;
//! * the EXPAND ratio test's step-length window assumed O(1) rates, and with
//!   rates of 4e15 it overshot rows by 4.7e4;
//! * a δ-shifted step lengthened to `α = 6e8` carried the active rows' own
//!   round-off (`‖A_W p‖∞ = 4.2e-15`) 2.5e-6 off their bounds;
//! * the feasibility audit sent a phase-2 optimum carrying 1.8e-6 of EXPAND
//!   drift straight to l1-elastic, which ended in a false `Unbounded`;
//! * and, for the LPs, the homotopy was tried before the simplex seed, which
//!   for an LP is the simplex method itself.
//!
//! The oracle is the convex IPM on the same file (`solver_selection=auto`),
//! whose objectives are recorded in the fixture sweep: `-76589.31858`,
//! `720078.3182` and `-1435.178`.

use std::path::PathBuf;
use std::process::Command;

fn run(name: &str) -> String {
    let mut fx = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fx.push("tests");
    fx.push("fixtures");
    fx.push(name);
    let sol = std::env::temp_dir().join(format!("pounce_974_{name}.sol"));
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fx)
        .arg(&sol)
        .arg("solver_selection=qp-active-set")
        .output()
        .expect("run pounce");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn objective(stdout: &str) -> f64 {
    stdout
        .lines()
        .find(|l| l.trim_start().starts_with("Objective."))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no objective line:\n{stdout}"))
}

fn solved(stdout: &str) -> bool {
    stdout.contains("EXIT: Optimal Solution Found.")
        || stdout.contains("EXIT: Solved To Acceptable Level.")
}

#[test]
fn lp_share1b_is_solved_by_the_active_set_engine() {
    let out = run("lp_share1b.nl");
    assert!(solved(&out), "{out}");
    let f = objective(&out);
    assert!((f - -76589.31858).abs() < 1e-6 * 76589.3, "f = {f}");
}

#[test]
fn convex_qp_share1b_is_solved_by_the_active_set_engine() {
    let out = run("convex_qp_share1b.nl");
    assert!(solved(&out), "{out}");
    let f = objective(&out);
    assert!((f - 720078.3182).abs() < 1e-6 * 720078.3, "f = {f}");
}

/// NETLIB `DEGEN2`: the homotopy spent 60 s reaching a handoff 10.6 infeasible
/// and every recovery after it ran its full budget (past 1100 s for the first
/// attempt alone). As an LP it now takes the simplex seed first. `main`
/// reported `Internal_Error` after 286 s. Oracle: the IPM's `-1435.178`.
#[test]
fn lp_degen2_is_solved_by_the_active_set_engine() {
    let out = run("lp_degen2.nl");
    assert!(solved(&out), "{out}");
    let f = objective(&out);
    assert!((f - -1435.178).abs() < 1e-6 * 1435.178, "f = {f}");
}
