//! gh#987 items 1-2: a MILP `.nl` is solved as its relaxation (now warned
//! about, and `verify` checks integrality), and `verify --feas-tol` is a
//! relative per-row test (documented, with `--abs-feas-tol` as the absolute
//! alternative).

use std::path::PathBuf;
use std::process::{Command, Output};

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn tmp(suffix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pounce_987_{}_{suffix}", std::process::id()));
    p
}

/// max 10x0+13x1+7x2 s.t. 4x0+6x1+3x2<=8, x binary (nbv=3); integer optimum 17,
/// relaxation 19.1667 at (1, 1/6, 1).
const KNAP: &str = "g3 1 1 0\n 3 1 1 0 0\n 0 0\n 0 0\n 0 0 0\n 0 0 0 1\n 3 0 0 0 0\n 3 3\n 0 0\n 0 0 0 0 0\nC0\nn0\nO0 1\nn0\nr\n1 8\nb\n0 0 1\n0 0 1\n0 0 1\nk2\n1\n2\nJ0 3\n0 4\n1 6\n2 3\nG0 3\n0 10\n1 13\n2 7\n";

/// x + y = 2e6 (a single row), min 0; solution (0, 2e6).
const BIG: &str = "g3 1 1 0\n 2 1 1 0 1 0\n 0 0\n 0 0\n 0 0 0\n 0 0 0 1\n 0 0 0 0 0\n 2 1\n 0 0\n 0 0 0 0 0\nC0\nn0\nO0 0\nn0\nr\n4 2000000\nb\n0 0 10000000\n0 0 10000000\nk1\n1\nJ0 2\n0 1\n1 1\nG0 1\n0 1\n";

fn run(args: &[&PathBuf], extra: &[&str]) -> Output {
    Command::new(pounce_exe())
        .args(args)
        .args(extra)
        .output()
        .expect("spawn pounce")
}

fn text(o: &Output) -> (String, String) {
    (
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

#[test]
fn a_milp_nl_is_warned_about_and_verify_rejects_the_fractional_point() {
    let nl = tmp("knap.nl");
    std::fs::write(&nl, KNAP).unwrap();
    let out = run(&[&nl], &["-AMPL", "--no-options-file"]);
    let (_o, e) = text(&out);
    assert!(
        e.contains("RELAXATION") && e.contains("3 binary"),
        "the CLI must warn that the integer declaration was dropped; stderr:\n{e}"
    );
    let sol = nl.with_extension("sol");
    assert!(sol.exists(), "no .sol written");

    let v = run(&[&PathBuf::from("verify"), &nl, &sol], &[]);
    let (o, _e) = text(&v);
    assert!(
        o.contains("integrality") && o.contains("max distance to an integer"),
        "verify must report the integrality check; stdout:\n{o}"
    );
    assert_eq!(
        v.status.code(),
        Some(20),
        "a fractional point must not verify against declared binaries; stdout:\n{o}"
    );
    assert!(o.contains("not integer-feasible"), "stdout:\n{o}");
}

#[test]
fn verify_accepts_an_integral_point_for_the_same_model() {
    let nl = tmp("knap_int.nl");
    std::fs::write(&nl, KNAP).unwrap();
    // x = (1, 0, 1): 4+3 = 7 <= 8, objective 17.
    let sol = tmp("knap_int.sol");
    std::fs::write(
        &sol,
        "POUNCE: handmade\n\nOptions\n3\n1\n1\n0\n1\n1\n3\n3\n0\n1\n0\n1\nobjno 0 0\n",
    )
    .unwrap();
    let v = run(&[&PathBuf::from("verify"), &nl, &sol], &[]);
    let (o, _) = text(&v);
    assert_eq!(v.status.code(), Some(0), "stdout:\n{o}");
    assert!(o.contains("max distance to an integer"), "stdout:\n{o}");
}

#[test]
fn feas_tol_is_relative_and_abs_feas_tol_is_absolute() {
    let nl = tmp("big.nl");
    std::fs::write(&nl, BIG).unwrap();
    let sol = tmp("claim.sol");
    // x + y = 2e6 violated by 1.0.
    std::fs::write(
        &sol,
        "POUNCE: handmade\n\nOptions\n3\n1\n1\n0\n1\n1\n2\n2\n0\n0\n1999999.0\nobjno 0 0\n",
    )
    .unwrap();
    let rel = run(
        &[&PathBuf::from("verify"), &nl, &sol],
        &["--feas-tol", "1e-6"],
    );
    let (o, _) = text(&rel);
    assert!(
        o.contains("relative per row"),
        "label the relative test; stdout:\n{o}"
    );
    assert_eq!(
        rel.status.code(),
        Some(0),
        "documented relative behaviour; stdout:\n{o}"
    );

    let abs = run(
        &[&PathBuf::from("verify"), &nl, &sol],
        &["--abs-feas-tol", "1e-6"],
    );
    let (o, _) = text(&abs);
    assert_eq!(
        abs.status.code(),
        Some(20),
        "absolute test must reject a violation of 1.0; stdout:\n{o}"
    );
}

/// gh#987 item 1, second pass. The relaxation notice is not stderr-only: the
/// JSON report's `statistics.warnings` carries it as `integer_relaxation: ...`,
/// so a consumer that reads the report (and not the console) still learns that
/// `optimal` is a bound on the MIP optimum. The `solve_result_num` stays in
/// the solved band on purpose -- see CHANGELOG.
#[test]
fn the_relaxation_notice_is_in_the_report_warnings() {
    let nl = tmp("knap_report.nl");
    std::fs::write(&nl, KNAP).unwrap();
    let json = tmp("knap_report.json");
    let out = run(
        &[&nl],
        &[
            "-AMPL",
            "--no-options-file",
            "--json-output",
            json.to_str().unwrap(),
        ],
    );
    assert!(out.status.success(), "solve failed: {:?}", text(&out));
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    let warnings = v["statistics"]["warnings"]
        .as_array()
        .expect("warnings array");
    assert!(
        warnings.iter().any(|w| w
            .as_str()
            .is_some_and(|w| w.starts_with("integer_relaxation:") && w.contains("RELAXATION"))),
        "report warnings: {warnings:?}"
    );
}
