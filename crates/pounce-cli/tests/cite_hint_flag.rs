//! Regression test for the `--cite` "wrong file" hint (code review L22).
//!
//! When `--cite <model>.nl` is given a `.nl` model instead of a solve-report
//! JSON, the CLI prints a hint telling the user how to produce a report first.
//! That hint must name the real flag, `--json-output` — an earlier version
//! suggested `--solve-report`, which the CLI does not accept (`cli.rs` parses
//! only `--json-output`), so a user following the hint hit "unknown argument".

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

#[test]
fn cite_nl_hint_names_real_json_output_flag() {
    // A `.nl`-extension file whose contents are not a valid solve-report JSON
    // triggers the model-instead-of-report hint branch.
    let dir = std::env::temp_dir();
    let nl = dir.join("pounce_l22_cite_hint_fixture.nl");
    std::fs::write(&nl, "g3 0 1 0\tnot a solve report\n").expect("write fixture .nl");

    let output = Command::new(pounce_exe())
        .arg("--cite")
        .arg(&nl)
        .output()
        .expect("spawn pounce");

    let _ = std::fs::remove_file(&nl);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--json-output"),
        "hint must point at the real flag --json-output; stderr={stderr}"
    );
    assert!(
        !stderr.contains("--solve-report"),
        "hint must not suggest the nonexistent --solve-report flag; stderr={stderr}"
    );
}
