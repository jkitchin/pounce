//! `.nl` complementarity rows (AMPL `complements`, `r`-segment kind 5)
//! through the CLI: read as pairs, solved by
//! `pounce_restoration::mpcc_driver::solve_mpcc`, certified.
//!
//! The fixtures live in `tests/fixtures_mpcc/`, outside `tests/fixtures/`, so
//! the fixture sweep's corpus is unchanged by them.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures_mpcc")
        .join(name)
}

struct Run {
    code: Option<i32>,
    stderr: String,
    json: Option<serde_json::Value>,
}

fn run(name: &str, extra: &[&str]) -> Run {
    let dir = std::env::temp_dir().join(format!("pounce-mpcc-{}-{}", name, std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let json_path = dir.join("report.json");
    let out = Command::new(pounce_exe())
        .arg(fixture(name))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json_path)
        .args(extra)
        .output()
        .expect("spawn pounce");
    let json = std::fs::read_to_string(&json_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    Run {
        code: out.status.code(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        json,
    }
}

fn x_of(r: &Run) -> Vec<f64> {
    r.json.as_ref().unwrap()["solution"]["x"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

/// `qpec_small` (two pairs, the second biactive at the answer `(1, 1, 0)`):
/// read from `.nl`, solved, and certified strongly stationary.
#[test]
fn a_qpec_written_with_complements_is_solved_and_certified() {
    let r = run("qpec_small.nl", &[]);
    assert_eq!(r.code, Some(0), "{}", r.stderr);
    // 1e-6, not looser: the exact-product stage's own point,
    // (1.000019, 1.0000095, 0), is within 1e-4 of the answer, and the report
    // once carried it in place of the continuation's, which was the one
    // accepted and summarized.
    let x = x_of(&r);
    for (got, want) in x.iter().zip([1.0, 1.0, 0.0]) {
        assert!((got - want).abs() < 1e-6, "x = {x:?}");
    }
    assert!(
        r.stderr.contains("2 pairs, 1 biactive; stationarity: S"),
        "{}",
        r.stderr
    );
    // The structure classifier reads the rows as a QP; with the pairing it is
    // not one, and the solve must not go to a convex engine.
    assert_eq!(
        r.json.as_ref().unwrap()["solution"]["engine"].as_str(),
        Some("nlp"),
        "{:?}",
        r.json
    );
}

/// `ralph1`: the answer, the origin, admits no S-stationary multiplier. The
/// run converges and says so: W, not S.
#[test]
fn a_converged_but_not_strongly_stationary_answer_is_labelled() {
    let r = run("ralph1.nl", &[]);
    assert_eq!(r.code, Some(0), "{}", r.stderr);
    assert!(x_of(&r).iter().all(|v| v.abs() < 1e-4), "{:?}", x_of(&r));
    assert!(r.stderr.contains("stationarity: W only"), "{}", r.stderr);
}

/// A two-sided (mixed) complementarity is refused with the reason and a
/// load-error exit, never solved as if the pairing were absent.
#[test]
fn a_mixed_pair_is_refused_not_dropped() {
    let r = run("mixed_pair.nl", &[]);
    assert_eq!(r.code, Some(2), "{}", r.stderr);
    assert!(r.stderr.contains("only one-sided pairs"), "{}", r.stderr);
}

/// Presolve renumbers rows and variables, which the pairs index; asking for
/// it is honoured by declining it, out loud.
#[test]
fn presolve_stands_down_for_complementarity() {
    let r = run("qpec_small.nl", &["presolve=yes"]);
    assert_eq!(r.code, Some(0), "{}", r.stderr);
    assert!(r.stderr.contains("disabling presolve"), "{}", r.stderr);
}
