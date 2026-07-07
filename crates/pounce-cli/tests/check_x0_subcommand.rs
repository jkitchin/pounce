//! End-to-end integration test for the `pounce check-x0` subcommand.
//!
//! Drives the real binary against committed fixtures and checks the
//! contract:
//!
//! * a healthy `.nl` (`parametric.nl`) evaluates cleanly → exit 0, JSON
//!   report with `"fatal": false` and the documented schema fields;
//! * a builtin (`rosenbrock`) works through `--builtin` → exit 0;
//! * `--x0-file` with the wrong length → usage error, exit 2;
//! * `--x0-file` overriding the start is honored (a wildly infeasible
//!   point raises the reported constraint violation but stays exit 0 —
//!   infeasibility is not fatal, only non-evaluability is).

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_nl() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("parametric.nl");
    p
}

fn tmp(suffix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pounce_check_x0_{}_{suffix}", std::process::id()));
    p
}

#[test]
fn healthy_nl_is_clean_and_emits_schema_json() {
    let out = Command::new(pounce_exe())
        .arg("check-x0")
        .arg(fixture_nl())
        .arg("--json")
        .output()
        .expect("spawn pounce check-x0");
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout is a JSON report");
    assert_eq!(v["schema"], "pounce.check-x0/v1");
    assert_eq!(v["fatal"], false);
    assert_eq!(v["problem"]["n_vars"], 5);
    assert_eq!(v["problem"]["n_cons"], 4);
    assert!(v["evaluation"]["objective_finite"].as_bool().unwrap());
    assert!(v["problem"]["sha256"].as_str().is_some());
}

#[test]
fn builtin_is_supported() {
    let out = Command::new(pounce_exe())
        .arg("check-x0")
        .arg("--builtin")
        .arg("rosenbrock")
        .arg("--json")
        .output()
        .expect("spawn pounce check-x0 --builtin");
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("JSON");
    assert_eq!(v["fatal"], false);
    assert_eq!(v["problem"]["source"], "builtin:rosenbrock");
}

#[test]
fn x0_file_wrong_length_is_usage_error() {
    let bad = tmp("short_x0.txt");
    std::fs::write(&bad, "1.0 2.0").unwrap(); // parametric.nl has 5 vars
    let code = Command::new(pounce_exe())
        .arg("check-x0")
        .arg(fixture_nl())
        .arg("--x0-file")
        .arg(&bad)
        .status()
        .expect("spawn")
        .code();
    assert_eq!(code, Some(2));
    let _ = std::fs::remove_file(&bad);
}

#[test]
fn x0_file_overrides_start_and_infeasibility_is_not_fatal() {
    let far = tmp("far_x0.txt");
    std::fs::write(&far, "1e6 1e6 1e6 1e6 1e6").unwrap();
    let report = tmp("far_report.json");
    let code = Command::new(pounce_exe())
        .arg("check-x0")
        .arg(fixture_nl())
        .arg("--x0-file")
        .arg(&far)
        .arg("--json-output")
        .arg(&report)
        .status()
        .expect("spawn")
        .code();
    assert_eq!(
        code,
        Some(0),
        "an infeasible-but-evaluable start is not fatal"
    );
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report).unwrap()).expect("JSON");
    assert_eq!(v["fatal"], false);
    assert!(v["x0"]["source"].as_str().unwrap().contains("far_x0.txt"));
    // A point at 1e6 in every coordinate must violate something in a
    // 5-var/4-con parametric model with finite constraint bounds.
    assert!(v["constraint_violation"]["max_violation"].as_f64().unwrap() > 1.0);
    let _ = std::fs::remove_file(&far);
    let _ = std::fs::remove_file(&report);
}
