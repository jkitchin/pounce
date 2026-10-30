//! Second-order cones carried natively by the NLP interior-point method
//! (workstream C, `dev-notes/research/nlp-soc-design.md`) against the convex
//! conic driver, on CBLIB instances: same continuous relaxation, the convex
//! driver's optimum is global, so the objectives must agree.
//!
//! `flay02m` pins the cones' fraction-to-boundary ceiling: with Ipopt's
//! `τ = max(0.99, 1 − μ)` the dual of its active cone (magnitude 51) stepped
//! outside the cone by rounding one iteration from convergence, and the solve
//! ended `Error_In_Step_Computation`.

use std::path::PathBuf;
use std::process::Command;

fn run(name: &str) -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/cblib")
        .join(format!("{name}.cbf"));
    let out = Command::new(env!("CARGO_BIN_EXE_pounce_cblib_nlp"))
        .arg(&path)
        .output()
        .expect("spawn pounce_cblib_nlp");
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text
        .lines()
        .rev()
        .find(|l| l.starts_with('{'))
        .unwrap_or("");
    serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {text}"))
}

fn check(name: &str) {
    let r = run(name);
    assert_eq!(r["convex"]["ok"], true, "{r}");
    assert_eq!(r["nlp"]["status"], "Solve_Succeeded", "{r}");
    let (c, n) = (
        r["convex"]["obj"].as_f64().unwrap(),
        r["nlp"]["obj"].as_f64().unwrap(),
    );
    assert!(
        (c - n).abs() <= 1e-6 * c.abs().max(1.0),
        "{name}: convex {c} vs nlp {n}"
    );
}

#[test]
fn flay02m_matches_the_convex_optimum() {
    check("flay02m");
}

#[test]
fn robust_20_2_matches_the_convex_optimum() {
    check("robust_20_2");
}
