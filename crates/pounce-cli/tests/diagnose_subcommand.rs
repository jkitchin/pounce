//! End-to-end tests for `pounce diagnose` (gh#1160) against committed
//! fixtures, through the real binary.
//!
//! Each proof-level check is driven on a fixture known to be infeasible for
//! that reason, and on a fixture that looks similar but is feasible, so both
//! branches of the verdict are exercised:
//!
//! | fixture | expected |
//! |---|---|
//! | `parametric.nl` (healthy) | exit 0, `CLEAN`, every check ran |
//! | `crossed_bound_qp.nl` | `inconsistent_variable_bounds` |
//! | `feasible_x0_sentinel_bound.nl` | no bound error: `g <= -5e20` has no lower bound |
//! | `presolve_overflow_feasible.nl` | no bound error: `g >= 1e300` has no upper bound |
//! | `inconsistent_eq_qp.nl` | `inconsistent_parallel_rows` |
//! | `infeasible_qp.nl` | `inconsistent_single_variable_rows` + a dependent pair |
//! | `issue981_water_main.nl` | `row_cannot_be_satisfied` (the solver agrees: infeasible) |
//! | `gibbs_xlogx_origin.nl` | the NaN at the origin is localized to `log`, and is a warning: the interior clamp moves the start off it |

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

/// Run `pounce diagnose <fixture> --json [extra]`; return (exit code, JSON).
fn diagnose(name: &str, extra: &[&str]) -> (Option<i32>, serde_json::Value) {
    let out = Command::new(pounce_exe())
        .arg("diagnose")
        .arg(fixture(name))
        .arg("--json")
        .args(extra)
        .output()
        .expect("spawn pounce diagnose");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "stdout is not JSON ({e}); stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        )
    });
    (out.status.code(), v)
}

fn codes(v: &serde_json::Value, severity: &str) -> Vec<String> {
    v["findings"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["severity"] == severity)
        .map(|f| f["code"].as_str().unwrap().to_string())
        .collect()
}

fn finding<'a>(v: &'a serde_json::Value, code: &str) -> Option<&'a serde_json::Value> {
    v["findings"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["code"] == code)
}

#[test]
fn healthy_model_is_clean_and_runs_every_check() {
    let (code, v) = diagnose("parametric.nl", &[]);
    assert_eq!(code, Some(0));
    assert_eq!(v["schema"], "pounce.diagnose/v1");
    assert_eq!(v["verdict"], "CLEAN", "{:#}", v["findings"]);
    let checks: Vec<&str> = v["checks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["check"].as_str().unwrap())
        .collect();
    for c in [
        "bounds",
        "linear_rows",
        "incidence",
        "expression_domains",
        "row_enclosures",
        "coefficient_ranges",
        "start_point",
        "nonfinite_localization",
        "structural_rank",
        "numerical_rank",
        "jacobian_scaling",
    ] {
        assert!(checks.contains(&c), "check {c} missing: {checks:?}");
    }
    assert!(v["sections"]["check_x0"]["schema"] == "pounce.check-x0/v1");
    assert!(v["problem"]["sha256"].as_str().is_some());
}

#[test]
fn crossed_variable_bounds_are_an_error() {
    let (code, v) = diagnose("crossed_bound_qp.nl", &[]);
    assert_eq!(code, Some(22));
    assert!(codes(&v, "error").contains(&"inconsistent_variable_bounds".to_string()));
}

#[test]
fn bounds_beyond_the_absent_sentinel_are_not_crossed() {
    for name in [
        "feasible_x0_sentinel_bound.nl",
        "presolve_overflow_feasible.nl",
    ] {
        let (_, v) = diagnose(name, &[]);
        let errs = codes(&v, "error");
        assert!(
            !errs.iter().any(|c| c.starts_with("inconsistent_")),
            "{name} is feasible, but: {errs:?}"
        );
    }
}

#[test]
fn contradictory_parallel_equalities() {
    let (code, v) = diagnose("inconsistent_eq_qp.nl", &[]);
    assert_eq!(code, Some(22));
    let f = finding(&v, "inconsistent_parallel_rows").expect("found");
    assert_eq!(f["basis"], "proof");
}

#[test]
fn contradictory_single_variable_rows_and_their_dependency() {
    let (code, v) = diagnose("infeasible_qp.nl", &[]);
    assert_eq!(code, Some(22));
    assert!(codes(&v, "error").contains(&"inconsistent_single_variable_rows".to_string()));
    let dep =
        finding(&v, "dependent_equality_rows").expect("the two rows are a minimal dependent set");
    assert_eq!(dep["affected"]["constraints"].as_array().unwrap().len(), 2);
    assert!(finding(&v, "structurally_overdetermined_equalities").is_some());
}

#[test]
fn row_that_cannot_hold_over_the_box() {
    let (code, v) = diagnose("issue981_water_main.nl", &[]);
    assert_eq!(code, Some(22));
    let f = finding(&v, "row_cannot_be_satisfied").expect("found");
    assert_eq!(f["basis"], "proof");
}

#[test]
fn nan_at_the_origin_is_localized_and_the_clamp_is_accounted_for() {
    let (_, v) = diagnose("gibbs_xlogx_origin.nl", &[]);
    let f = finding(&v, "nonfinite_from_pole").expect("log(0) localized");
    assert_eq!(f["evidence"]["cause"], "pole");
    assert!(
        f["evidence"]["expression"]
            .as_str()
            .unwrap()
            .contains("log"),
        "{f:#}"
    );
    assert_eq!(
        f["severity"], "warning",
        "the clamp moves the start off x = 0: {f:#}"
    );
}

#[test]
fn x0_file_moves_the_point() {
    let dir = std::env::temp_dir();
    let x0 = dir.join(format!("pounce_diagnose_x0_{}.txt", std::process::id()));
    std::fs::write(&x0, "1 1 1 1 1").unwrap();
    let (code, v) = diagnose("parametric.nl", &["--x0-file", x0.to_str().unwrap()]);
    let _ = std::fs::remove_file(&x0);
    assert_eq!(code, Some(0));
    assert!(v["point"].as_str().unwrap().contains("pounce_diagnose_x0"));
}

#[test]
fn usage_errors_exit_2() {
    let out = Command::new(pounce_exe())
        .arg("diagnose")
        .output()
        .expect("spawn");
    assert_eq!(out.status.code(), Some(2));
    let out = Command::new(pounce_exe())
        .args(["diagnose", "--help"])
        .output()
        .expect("spawn");
    assert_eq!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stdout).contains("Usage: pounce diagnose"));
}
