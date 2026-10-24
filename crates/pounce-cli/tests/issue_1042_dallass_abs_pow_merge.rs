//! gh#1042 R3: `dallass` solves without a single restoration entry once the
//! `.nl` reader merges `|x|^2·|x|^0.85` into `|x|^2.85`.
//!
//! Vanderbei's (CUTE) `dallass` writes each head-loss term as
//! `(k·|x|^2)·|x|^0.85`. Differentiated as written, the product rule meets
//! `0·∞` whenever a flow lands on exactly `x = 0`, the Hessian is NaN, the
//! KKT factorization refuses it and the step hands off to restoration. On
//! 0.13.0 (FERAL 0.21) the trajectory met the kink 11 times — 123
//! iterations, 12 restoration calls / 53 restoration iterations, against
//! 0.12.0's 41 iterations / 2 calls. `merge_abs_pow_product` in `pounce-nl`
//! rewrites the product to the single power it is, whose second derivative
//! at 0 is the finite `0`; measured on x86_64 the solve is 28 iterations
//! with no restoration call, at the same optimum.
//!
//! The bounds below are the plan's acceptance bounds (≤ 60 iterations,
//! ≤ 3 restoration calls), not the measured 28 / 0, so an honest
//! cross-architecture trajectory difference does not fail it; reverting the
//! merge does (123 / 12). The non-finite-Hessian restoration exemption is
//! still guarded, on an unmerged copy of this model, by
//! `nonfinite_hessian_restoration_strike.rs`.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("dallass.nl");
    p
}

const DALLASS_OPTIMUM: f64 = -32393.225737;

#[test]
fn dallass_solves_without_restoration_once_the_abs_powers_are_merged() {
    let json = std::env::temp_dir().join(format!(
        "pounce_issue_1042_dallass_{}.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&json);

    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .output()
        .expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("EXIT: Optimal Solution Found."),
        "dallass did not solve:\n{stdout}"
    );
    assert!(
        !stdout.contains("second opinion"),
        "dallass needed the second-opinion ladder:\n{stdout}"
    );

    let text = std::fs::read_to_string(&json).expect("read JSON report");
    let _ = std::fs::remove_file(&json);
    let report: serde_json::Value = serde_json::from_str(&text).expect("parse JSON report");
    // gh#1042 C1: every number below is the *reported* solve's, and on a
    // promoted `mu_strategy_fallback` that is the re-solve alone — the first
    // attempt's verdict and cost appear nowhere else. R2b's 492 → 109 on
    // `pooling_rt2stp` was exactly that (first attempt Acceptable/494), so
    // the pin must establish there was only one attempt before it reads
    // anything.
    assert!(
        report["statistics"]["mu_strategy_fallback"].is_null(),
        "dallass's reported result is a mu_strategy_fallback re-solve, so the \
         bounds below would describe the retry, not the first attempt: {}",
        report["statistics"]["mu_strategy_fallback"]
    );
    let obj = report["statistics"]["final_objective"]
        .as_f64()
        .expect("final_objective");
    assert!(
        (obj - DALLASS_OPTIMUM).abs() <= 1e-6 * DALLASS_OPTIMUM.abs(),
        "objective {obj} is not the optimum {DALLASS_OPTIMUM}"
    );
    let iters = report["statistics"]["iteration_count"]
        .as_u64()
        .expect("iteration_count");
    assert!(
        iters <= 60,
        "dallass took {iters} iterations (0.12.0: 41; 0.13.0 before the merge: 123)"
    );

    // `Number of restoration iterations = N (in K calls)`; the line is
    // absent when restoration never ran.
    let calls = stdout
        .lines()
        .find(|l| l.starts_with("Number of restoration iterations"))
        .and_then(|l| l.split("(in ").nth(1))
        .and_then(|r| r.split_whitespace().next())
        .map(|k| k.parse::<u64>().expect("restoration call count"))
        .unwrap_or(0);
    assert!(
        calls <= 3,
        "dallass entered restoration {calls} times (0.13.0 before the merge: 12, \
         11 of them forced by a non-finite Hessian at the |x|^2*|x|^0.85 kink)"
    );
}
