//! gh#1191 — the active-set SQP (limited-memory) reached the optimum of
//! `palmer8e` and `hs99exp` and stopped there with
//! `Search_Direction_Becomes_Too_Small` instead of certifying it.
//!
//! Both are endgames that gh#1173's filter rules (PR #1190) judged on
//! quantities that had stopped carrying information:
//!
//! * **`hs99exp`**: `r`, `q` reach `1e7`, so the last steps of a superlinear
//!   endgame are below round-off in `x`. Those are "tiny steps", taken
//!   unjudged, and two in a row were treated as a stall. Yet each one still
//!   divided the stationarity residual by four (`5.8e-7 → 1.4e-7 → 3.0e-8`),
//!   and the limit fired one step short of `tol = 1e-8`. A run of tiny
//!   steps that brings the residual 10% below its best is now progress, not
//!   a stall.
//! * **`palmer8e`**: at stationarity 1.7e-8 the step's predicted change
//!   `∇fᵀp = −1.2e-18` is below `φ`'s resolution (`10·ε·|φ| = 1.4e-17`), and
//!   `φ` moved by `±1e-15` of evaluation noise at every trial length. Armijo
//!   compared the noise and backtracked to failure. Such a "φ-tiny" step is
//!   now taken at full length, provided it does not raise `θ`.
//!
//! The fixtures are pyomo transcriptions of Vanderbei's CUTE `.mod` files
//! (`hs99exp` with the fixed `r₁ = q₁ = s₁ = 0` substituted, as AMPL's
//! presolve does). Both legs failed on `5463adb` exactly as the issue
//! reports on the corpus `.nl` (`palmer8e` 134 iterations,
//! `hs99exp` 57).
//!
//! **Not covered: `gilbert`**, the issue's third model. It is not the same
//! failure. Its limited-memory endgame (n = 1000) alternates stationarity
//! between ~1e-6 and ~1e-8 for tens of iterations on both binaries. The base
//! happened to land a low phase below `tol` at iteration 145, and this branch
//! certifies at 204, past the 200 cap. See `dev-notes/known-limitations.md`.

use std::path::PathBuf;
use std::process::Command;

fn solve(name: &str) -> serde_json::Value {
    let mut fx = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fx.push("tests");
    fx.push("fixtures");
    fx.push(name);
    let json = std::env::temp_dir().join(format!("pounce_1191_{name}.json"));
    let sol = std::env::temp_dir().join(format!("pounce_1191_{name}.sol"));
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&fx)
        .arg(&sol)
        .args([
            "algorithm=active-set-sqp",
            "hessian_approximation=limited-memory",
            "--no-options-file",
            "--json-output",
        ])
        .arg(&json)
        .output()
        .expect("run pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "no JSON report ({e}); stdout:\n{}",
            String::from_utf8_lossy(&out.stdout)
        )
    });
    serde_json::from_str(&text).expect("parse JSON report")
}

fn assert_certified(name: &str, f_star: f64, rel: f64) {
    let r = solve(name);
    let status = r["solution"]["status"].as_str().unwrap_or("<none>");
    let f = r["solution"]["objective"].as_f64().unwrap_or(f64::NAN);
    let iters = &r["statistics"]["iteration_count"];
    let dinf = &r["statistics"]["final_dual_inf"];
    assert_eq!(
        status, "SolveSucceeded",
        "{name}: {status} after {iters} iterations at f = {f} (dual inf {dinf})"
    );
    assert_eq!(r["solution"]["engine"], "sqp-active-set", "{name}: engine");
    assert!(
        (f - f_star).abs() <= rel * f_star.abs(),
        "{name}: f = {f}, expected {f_star}"
    );
}

#[test]
fn palmer8e_takes_the_step_phi_cannot_resolve() {
    assert_certified("cute_palmer8e.nl", 6.339_307_43e-3, 1e-8);
}

#[test]
fn hs99exp_tiny_steps_that_make_progress_are_not_a_stall() {
    assert_certified("cute_hs99exp.nl", -1.008_062_5e9, 1e-10);
}
