//! Regression test for the real-AMPL driver conventions (code-review
//! 2026-06 item M15): an extensionless `.nl` stub and the
//! `pounce_options` environment variable.
//!
//! AMPL invokes a solver as `pounce mystub -AMPL`, passing the stub name
//! *without* the `.nl` extension and expecting `mystub.nl` to be read and
//! `mystub.sol` written, and it conveys solver directives through the
//! `<solver>_options` env var (`pounce_options`) rather than the command
//! line. Before the fix neither was honored: the stub failed with "could
//! not read", and `pounce_options` was ignored entirely.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Fresh scratch dir for one test (no `tempfile` dev-dep).
fn scratch(tag: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let seq = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("pounce_m15_{}_{seq}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Copy the shared LP/QP fixture into `dir/mystub.nl` and return the
/// extensionless stub path `dir/mystub`.
fn stub_in(dir: &std::path::Path) -> PathBuf {
    let nl = std::fs::read(fixture("convex_qp.nl")).expect("read fixture");
    std::fs::write(dir.join("mystub.nl"), nl).unwrap();
    dir.join("mystub")
}

#[test]
fn extensionless_stub_resolves_to_nl_and_writes_sol() {
    let dir = scratch("stub");
    let stub = stub_in(&dir);
    assert!(!stub.exists(), "stub must be extensionless / absent");

    // `pounce mystub -AMPL` — AMPL's invocation. No --sol-output: the
    // driver derives `mystub.sol` from the stub.
    let output = Command::new(pounce_exe())
        .arg(&stub)
        .arg("solver_selection=nlp")
        .arg("-AMPL")
        .output()
        .expect("spawn pounce");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        output.status.success(),
        "stub invocation failed (exit {:?}); output:\n{combined}",
        output.status.code(),
    );
    assert!(
        combined.contains("Optimal Solution Found"),
        "expected an optimal solve; output:\n{combined}"
    );
    // The .sol lands next to the stub, named off the stem.
    assert!(
        dir.join("mystub.sol").exists(),
        "expected mystub.sol next to the stub; dir held: {:?}",
        std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.file_name()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn pounce_options_env_var_is_applied() {
    let dir = scratch("env");
    let stub = stub_in(&dir);

    // A bogus option name in `pounce_options` must be *read and rejected*
    // (exit 2, "failed to set ..."), proving the env var is consumed — a
    // deterministic signal that doesn't depend on the solve trajectory.
    let output = Command::new(pounce_exe())
        .arg(format!("{}.nl", stub.display()))
        .arg("--no-sol")
        .env("pounce_options", "definitely_not_an_option=1")
        .output()
        .expect("spawn pounce");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(
        output.status.code(),
        Some(2),
        "bogus pounce_options should fail with exit 2; output:\n{combined}"
    );
    assert!(
        combined.contains("failed to set definitely_not_an_option"),
        "expected the env option to be applied (and rejected); output:\n{combined}"
    );
}

#[test]
fn cli_key_value_overrides_pounce_options_env() {
    let dir = scratch("override");
    let stub = stub_in(&dir);
    let nl = format!("{}.nl", stub.display());

    // env caps iterations at 1 → would stop at "Maximum ... Exceeded".
    // The CLI `max_iter=3000` must win (applied last), letting it converge.
    let output = Command::new(pounce_exe())
        .arg(&nl)
        .arg("solver_selection=nlp")
        .arg("max_iter=3000")
        .arg("--no-sol")
        .env("pounce_options", "max_iter=1")
        .output()
        .expect("spawn pounce");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        output.status.success(),
        "override solve failed (exit {:?}); output:\n{combined}",
        output.status.code(),
    );
    assert!(
        combined.contains("Optimal Solution Found"),
        "CLI max_iter should override the env cap; output:\n{combined}"
    );
    assert!(
        !combined.contains("Maximum Number of Iterations Exceeded"),
        "env max_iter=1 leaked past the CLI override; output:\n{combined}"
    );
}
