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

// ---------------------------------------------------------------- phase 2

/// Solve `name` in a fresh temp dir and return the path of its `.sol`.
fn solve_to_sol(name: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "pounce_diagnose_p2_{}_{}",
        std::process::id(),
        name.trim_end_matches(".nl")
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let nl = dir.join(name);
    std::fs::copy(fixture(name), &nl).unwrap();
    let out = Command::new(pounce_exe())
        .arg(&nl)
        .current_dir(&dir)
        .output()
        .expect("solve");
    let sol = nl.with_extension("sol");
    assert!(
        sol.exists(),
        "no .sol for {name}: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    (nl, sol)
}

fn diagnose_sol(name: &str) -> serde_json::Value {
    let (nl, sol) = solve_to_sol(name);
    let out = Command::new(pounce_exe())
        .arg("diagnose")
        .arg(&nl)
        .arg("--sol")
        .arg(&sol)
        .arg("--json")
        .output()
        .expect("spawn");
    let _ = std::fs::remove_dir_all(nl.parent().unwrap());
    serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("not JSON ({e}): {}", String::from_utf8_lossy(&out.stderr)))
}

#[test]
fn converged_convex_qp_is_a_strict_local_minimizer() {
    let v = diagnose_sol("convex_qp.nl");
    assert!(
        finding(&v, "second_order_sufficient").is_some(),
        "{:#}",
        v["findings"]
    );
    for bad in [
        "not_stationary_at_point",
        "multiplier_sign_violation",
        "complementarity_violation",
        "licq_fails",
    ] {
        assert!(finding(&v, bad).is_none(), "{bad}: {:#}", v["findings"]);
    }
    assert_eq!(
        v["sections"]["kkt_audit"]["bound_multipliers"],
        "from the .sol suffixes"
    );
}

#[test]
fn duplicated_equalities_fail_licq_with_nonunique_multipliers() {
    let v = diagnose_sol("rankdef_eq_qp.nl");
    assert!(finding(&v, "licq_fails").is_some(), "{:#}", v["findings"]);
    assert!(
        finding(&v, "nonunique_multipliers").is_some(),
        "{:#}",
        v["findings"]
    );
    assert!(
        finding(&v, "rank_loss_persistent").is_some(),
        "{:#}",
        v["findings"]
    );
}

#[test]
fn hs13_optimum_is_not_a_kkt_point() {
    // HS13's minimizer (1, 0) is a cusp where no constraint qualification
    // holds and no multipliers exist.
    let v = diagnose_sol("hs13_bigstart.nl");
    assert!(
        finding(&v, "not_stationary_at_point").is_some(),
        "{:#}",
        v["findings"]
    );
}

#[test]
fn saddle_point_has_negative_reduced_curvature() {
    let v = diagnose_sol("nonconvex_two_escapes.nl");
    assert!(
        finding(&v, "reduced_hessian_negative_curvature").is_some(),
        "{:#}",
        v["findings"]
    );
}

#[test]
fn without_sol_the_solution_checks_say_why_not() {
    let (_, v) = diagnose("convex_qp.nl", &[]);
    let checks = v["checks"].as_array().unwrap();
    for c in ["kkt_audit", "licq", "mfcq", "reduced_hessian"] {
        let rec = checks.iter().find(|r| r["check"] == c).expect(c);
        assert_eq!(rec["status"], "unavailable", "{c}");
        assert!(rec["reason"].as_str().unwrap().contains("--sol"));
    }
}
