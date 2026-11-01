//! A feasible LP whose feasible points lie far out must not be certified
//! primal infeasible by an approximate Farkas ray — and genuinely infeasible
//! LPs with no variable box (so the exact `farkas_box_proof` cannot help)
//! must still be.
//!
//! `farkas_far_feasible_no_box.nl` (18 × 18 `Gx ≤ h`, no bounds, data at
//! `~1e6`) is feasible: `G` is nonsingular and `Gx = h` has an exact rational
//! solution with `‖x‖∞ ≈ 1.07e9`; HiGHS reports the LP unbounded. The
//! convex QP HSDE used to report it `Infeasible_Problem_Detected` at
//! iteration 9 off a ray with `hᵀz/‖z‖ = −1.9e-5` and `‖Gᵀz‖ = 5.5e-12` —
//! both inside the absolute Farkas tolerances, but a ray that only excludes
//! feasible points within `|hᵀz|/‖Gᵀz‖ ≈ 3.4e6`, i.e. within ~1× the
//! model's own scale. The `farkas_margin` test in `pounce-convex` now asks
//! for `1e4×` that scale, and the solve ends `Diverging_Iterates`.
//!
//! The two infeasible fixtures have an *exact* left null vector (`y = 1`:
//! the last row is the negated sum of the others) and `Σh < 0`, with data at
//! `1e6` and `1e3`; both are certified through the ratio path, the second
//! off a ray with a non-zero residual (radius `~1e5`, the closest true
//! certificate in the measured families above the threshold). These are the
//! fixtures for the branch that must stay certified — the reproducer alone
//! only exercises the rejecting one.

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// `(status, engine)` from the JSON report of a default solve.
fn solve(model: &str) -> (String, String) {
    let dir = std::env::temp_dir();
    let tag = format!("pounce_farkas_{}_{}", std::process::id(), model);
    let sol = dir.join(format!("{tag}.sol"));
    let json = dir.join(format!("{tag}.json"));
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fixture(model))
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .arg("print_level=0")
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "{model}: no JSON report ({e})\n{}",
            String::from_utf8_lossy(&out.stdout)
        )
    });
    let _ = std::fs::remove_file(&sol);
    let _ = std::fs::remove_file(&json);
    let v: serde_json::Value = serde_json::from_str(&text).expect("JSON parses");
    let s = &v["solution"];
    (
        s["status"].as_str().unwrap_or("").to_string(),
        s["engine"].as_str().unwrap_or("").to_string(),
    )
}

#[test]
fn far_feasible_lp_is_not_certified_infeasible() {
    let (status, engine) = solve("farkas_far_feasible_no_box.nl");
    assert_eq!(engine, "cvx-qp", "the defect is on the convex QP arm");
    assert_ne!(
        status, "InfeasibleProblemDetected",
        "feasible LP (exact rational point at |x| ~ 1e9) certified infeasible"
    );
    // The LP is unbounded (HiGHS agrees); with the false certificate gone
    // the recession ray is what the HSDE finds.
    assert_eq!(status, "DivergingIterates");
}

#[test]
fn far_feasible_lp_with_zero_objective_is_optimal() {
    let (status, _) = solve("farkas_far_feasible_no_box_c0.nl");
    assert_eq!(status, "SolveSucceeded");
}

#[test]
fn no_box_infeasible_lps_stay_certified() {
    for model in [
        "farkas_infeasible_no_box_exact.nl",
        "farkas_infeasible_no_box_near.nl",
    ] {
        let (status, engine) = solve(model);
        assert_eq!(engine, "cvx-qp", "{model}");
        assert_eq!(status, "InfeasibleProblemDetected", "{model}");
    }
}
