//! gh#1249 / gh#1250 review: a FERAL refusal under
//! `POUNCE_FERAL_MAX_FACTOR_BYTES` ends the solve.
//!
//! Before the fix, the refusal was a `FatalError` like any other. The convex
//! path then re-solved on the NLP arm (the gh#535 / gh#1183 fallback), the NLP
//! arm entered restoration, and the second-opinion ladder re-solved again.
//! Each of these factored a new pattern and paid for another symbolic
//! analysis, and the run ended under an unrelated status
//! (`Not_Enough_Degrees_Of_Freedom`, `Restoration_Failed`) or ran past
//! `max_wall_time`. Now exactly one analysis runs and the run reports
//! `Insufficient_Memory`, on both the convex and the NLP arm.

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Run the CLI with a 1-byte budget, which refuses every factorization.
fn refused_run(model: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fixture(model))
        .env("POUNCE_FERAL_MAX_FACTOR_BYTES", "1")
        .env_remove("POUNCE_FERAL_ORDERING")
        .env_remove("POUNCE_FERAL_REPORT_SYMBOLIC")
        .env("NO_COLOR", "1")
        .output()
        .expect("pounce runs");
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

fn check(model: &str) {
    let log = refused_run(model);
    let analyses = log.matches("FERAL factorization refused: the").count();
    assert_eq!(analyses, 1, "{model}: one analysis, then stop\n{log}");
    assert!(
        log.contains("Status: Insufficient_Memory"),
        "{model}: the refusal is the reported verdict\n{log}"
    );
    assert!(
        !log.contains("re-solved on the general NLP"),
        "{model}: no LP -> NLP fallback after a refusal\n{log}"
    );
}

#[test]
fn a_refused_lp_reports_insufficient_memory_without_the_nlp_fallback() {
    check("lp_degen2.nl");
}

#[test]
fn a_refused_nlp_reports_insufficient_memory_without_restoration_or_the_ladder() {
    check("hs13_bigstart.nl");
}
