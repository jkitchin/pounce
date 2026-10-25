//! The `.sol`'s `sens_sol_state_1` block after a solve whose answer was
//! replayed from a floor is the reported attempt's step, not the losing
//! retry's.
//!
//! The CLI's `on_converged` hook computes the sIPOPT step and the reduced
//! Hessian per converged attempt. It used to overwrite one set of captures in
//! place, so when a retry converged and was declined the `.sol` carried the
//! base attempt's point (the finalize-payload backfill put that back) next to
//! the *retry's* step. Measured on `parametric.nl` before the fix:
//! `x_perturbed[0]` read `0.576480…` against `0.576310…` for the same solve
//! without the retry.
//!
//! Trigger, as in `pounce-sensitivity/tests/floored_solve_sensitivity.rs`:
//! `tol` below reach makes every attempt exit at `Solved_To_Acceptable_Level`,
//! an explicit `mu_strategy_fallback=yes` makes that retry-worthy, and the
//! acceptable retry does not promote. The reference is the same run with
//! `mu_strategy_fallback=no`.

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

/// `(sens_sol_state_1 values, JSON report)` for one run.
fn run(fallback: &str) -> (Vec<f64>, serde_json::Value) {
    let dir = std::env::temp_dir();
    let tag = format!("{}_{fallback}", std::process::id());
    let sol = dir.join(format!("pounce_floored_sens_{tag}.sol"));
    let json = dir.join(format!("pounce_floored_sens_{tag}.json"));
    let out = Command::new(pounce_exe())
        .arg(fixture("parametric.nl"))
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .arg("tol=1e-30")
        .arg("acceptable_tol=1e-2")
        .arg("acceptable_iter=2")
        .arg(format!("mu_strategy_fallback={fallback}"))
        .arg("print_level=0")
        .output()
        .expect("spawn pounce");
    assert!(
        out.status.success(),
        "pounce exited with {:?}; stderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    let sol_text = std::fs::read_to_string(&sol).expect("read .sol");
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).expect("read json")).expect("json");
    let _ = std::fs::remove_file(&sol);
    let _ = std::fs::remove_file(&json);

    // `suffix … sens_sol_state_1` header, then `index value` lines.
    let mut lines = sol_text.lines();
    lines
        .by_ref()
        .find(|l| l.trim_end().ends_with("sens_sol_state_1"))
        .expect("sens_sol_state_1 suffix block present");
    let sens = lines
        .map_while(|l| {
            let mut it = l.split_whitespace();
            let (_i, v) = (it.next()?, it.next()?);
            if it.next().is_some() {
                return None;
            }
            v.parse::<f64>().ok()
        })
        .collect::<Vec<_>>();
    assert_eq!(sens.len(), 5, "parametric.nl has five variables");
    (sens, report)
}

#[test]
fn the_sens_step_after_a_declined_retry_is_the_reported_attempts() {
    let (floored, report) = run("yes");
    let (reference, ref_report) = run("no");

    // Premise: the retry ran and lost, and both runs report the same point.
    let fb = &report["statistics"]["mu_strategy_fallback"];
    assert_eq!(fb["promoted"], serde_json::json!(false), "{fb}");
    assert_eq!(
        fb["retry_status"],
        serde_json::json!("Solved_To_Acceptable_Level"),
        "the retry must converge, or its hook never fires: {fb}"
    );
    assert_eq!(report["solution"]["x"], ref_report["solution"]["x"]);

    for (k, (got, want)) in floored.iter().zip(&reference).enumerate() {
        assert!(
            (got - want).abs() <= 1e-12,
            "x_perturbed[{k}] = {got} after the declined retry, {want} without \
             it: the .sol carries the losing retry's sensitivity step"
        );
    }
}
