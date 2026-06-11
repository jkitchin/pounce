//! Integration test for issue #49: AMPL imported (external) functions.
//!
//! Checks that pounce parses a `.nl` declaring imported functions and
//! errors cleanly when `AMPLFUNC` is unset — i.e. there is no external
//! function library to bind the referenced funcalls against.
//!
//! Note: the end-to-end *solve* check against the live IDAES Helmholtz
//! dylib was removed. That fixture bakes a machine- and venv-specific
//! absolute path to the EOS parameter data (`h2o_parameters.json`) into
//! the `.nl`, which rots whenever the IDAES install moves (e.g. a Python
//! minor-version bump deletes the old `site-packages`). When the data
//! file can't be found the external functions return garbage and the
//! solve diverges — a fixture/data problem, not a solver or ABI bug
//! (pointing the fixture at a valid parameters directory solves it to
//! optimality in a handful of iterations). The check also only ran
//! locally, skipping in CI whenever the dylib was absent.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures_issue_49");
    p.push("idaes_helmholtz.nl");
    p
}

#[test]
fn pounce_rejects_external_function_problem_without_amplfunc() {
    let out = Command::new(pounce_exe())
        .env_remove("AMPLFUNC")
        .arg(fixture_path())
        .output()
        .expect("spawn pounce binary");

    assert!(!out.status.success(), "pounce should fail without AMPLFUNC");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined.contains("AMPLFUNC") || combined.to_lowercase().contains("external function"),
        "error should mention AMPLFUNC or external functions, got: {combined}"
    );
}
