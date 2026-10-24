//! gh#1054: `max_wall_time` / `max_cpu_time` bound the active-set SQP arm.
//!
//! Both options were read only by the interior-point convergence check, so
//! `algorithm=active-set-sqp` ran unbounded: the reporter's cold SQP solve
//! of a 15 000-variable model sat in its first QP for 30 minutes and was
//! stopped from outside. The budget is now checked every outer iteration
//! and handed to each step QP as its `time_limit`.

use std::path::PathBuf;
use std::process::Command;

use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_solve_report::SolveReport;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn solve(option: &str) -> SolveReport {
    let json = std::env::temp_dir().join(format!(
        "pounce-issue1054-{}-{}.json",
        std::process::id(),
        option.replace(['=', '.'], "-")
    ));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(pounce_exe())
        .arg(fixture("hs13_bigstart.nl"))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .arg("algorithm=active-set-sqp")
        .arg(option)
        .output()
        .expect("spawn pounce");
    let report = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "no JSON report ({e}); stdout=\n{}\nstderr=\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    let _ = std::fs::remove_file(&json);
    serde_json::from_str(&report).expect("parse JSON report")
}

#[test]
fn sqp_arm_honors_max_wall_time() {
    let r = solve("max_wall_time=1e-12");
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::MaximumWallTimeExceeded
    );
    assert_eq!(r.solution.solve_result_num, 400);
}

#[test]
fn sqp_arm_honors_max_cpu_time() {
    let r = solve("max_cpu_time=1e-12");
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::MaximumCpuTimeExceeded
    );
}
