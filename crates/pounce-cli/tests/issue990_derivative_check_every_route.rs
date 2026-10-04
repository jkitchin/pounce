//! gh#990 (remaining item j): the derivative checker's verdict
//! (`statistics.derivative_check`) was in the report on the IPM path only. The
//! active-set SQP drain did not copy it, and the CLI's convex route ran the
//! check but hand-built a report without it.

use std::path::PathBuf;
use std::process::Command;

fn report(tag: &str, extra: &[&str]) -> serde_json::Value {
    let dir = std::env::temp_dir().join(format!("pounce_990j_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let nl = dir.join("m.nl");
    let mut src = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src.push("tests/fixtures/convex_qp.nl");
    std::fs::copy(src, &nl).unwrap();
    let json = dir.join("r.json");
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&nl)
        .args(extra)
        .arg("derivative_test=first-order")
        .arg("--json-output")
        .arg(&json)
        .arg("--no-sol")
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|_| {
        panic!(
            "no report; stderr:\n{}",
            String::from_utf8_lossy(&out.stderr)
        )
    });
    serde_json::from_str(&text).unwrap()
}

fn assert_has_check(r: &serde_json::Value, engine: &str) {
    assert_eq!(
        r["solution"]["engine"].as_str(),
        Some(engine),
        "fixture must reach the {engine} arm: {}",
        r["solution"]
    );
    let dc = &r["statistics"]["derivative_check"];
    assert_eq!(dc["mode"].as_str(), Some("first-order"), "{engine}: {dc}");
    assert_eq!(dc["clean"].as_bool(), Some(true), "{engine}: {dc}");
}

#[test]
fn the_convex_route_reports_the_derivative_check() {
    assert_has_check(&report("cvx", &[]), "cvx-qp");
}

#[test]
fn the_active_set_sqp_route_reports_the_derivative_check() {
    assert_has_check(
        &report("sqp", &["solver_selection=nlp", "algorithm=active-set-sqp"]),
        "sqp-active-set",
    );
}

#[test]
fn the_ipm_route_still_reports_it() {
    assert_has_check(&report("ipm", &["solver_selection=nlp"]), "nlp");
}
