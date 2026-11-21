//! `feral_static_pivoting` reaches the convex IPM's FERAL backend, and an
//! explicit setting wins over `POUNCE_FERAL_STATIC_PIVOTING`.
//!
//! The CLI's convex LP/QP path used to read only the environment variable, so
//! `feral_static_pivoting=no` changed nothing: on `lp_degen2` at `tol=1e-8`
//! (where the convex rule defaults to static pivoting) 22ae155e reported
//! `-1435.1779999844987` with the option set to `no`, which is the default's
//! objective to every digit. Delayed pivoting reports `-1435.177999992415`.
//! The two factorizations differ only in rounding, but the solve is
//! deterministic, so the digits identify which one ran.
//!
//! Every cell of the env × option table is checked, so each override
//! direction is a separate case.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;

const ENV: &str = "POUNCE_FERAL_STATIC_PIVOTING";

fn objective(env: Option<&str>, opt: Option<&str>) -> (f64, String) {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("pounce_feral_sp_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lp_degen2.nl");
    std::fs::copy(&src, dir.join("m.nl")).expect("copy fixture");
    let mut cmd = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")));
    cmd.current_dir(&dir)
        .env_remove(ENV)
        .arg("m.nl")
        .arg("--json-output")
        .arg("r.json")
        .arg("print_level=0")
        .arg("tol=1e-8");
    if let Some(e) = env {
        cmd.env(ENV, e);
    }
    if let Some(o) = opt {
        cmd.arg(format!("feral_static_pivoting={o}"));
    }
    cmd.output().expect("run pounce");
    let json = std::fs::read_to_string(dir.join("r.json")).expect("json report");
    let report: SolveReport = serde_json::from_str(&json).expect("deserialize SolveReport");
    let _ = std::fs::remove_dir_all(&dir);
    (report.solution.objective, report.solution.engine)
}

#[test]
fn the_option_decides_and_beats_the_environment() {
    let (stat, engine) = objective(None, Some("yes"));
    assert_eq!(engine, "cvx-qp");
    let (delayed, _) = objective(None, Some("no"));
    assert_ne!(
        stat.to_bits(),
        delayed.to_bits(),
        "static and delayed pivoting gave the same digits ({stat:e}); the fixture no longer \
         tells them apart, so this test checks nothing"
    );
    // The convex rule's default at tol=1e-8 with no free column is static.
    assert_eq!(objective(None, None).0.to_bits(), stat.to_bits());
    for env in [None, Some("yes"), Some("no")] {
        for (opt, want) in [(Some("yes"), stat), (Some("no"), delayed)] {
            let (got, _) = objective(env, opt);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "env {env:?} option {opt:?}: {got:e}"
            );
        }
    }
    // With no option, the environment variable still decides.
    assert_eq!(objective(Some("no"), None).0.to_bits(), delayed.to_bits());
    assert_eq!(objective(Some("yes"), None).0.to_bits(), stat.to_bits());
}
