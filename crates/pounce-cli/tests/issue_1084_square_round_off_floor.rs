//! gh#1084 — tightening `tol` below a square model's round-off floor must not
//! turn a feasible point into `Restoration_Failed`.
//!
//! `square_flowsheet_resto.nl` is square (32 × 32, `f = 0`) and feasible by
//! construction. Its rows carry `3e6` Pa next to `1e-6` mole fractions, so the
//! smallest violation the arithmetic represents is about `3e-10`. With `tol`
//! under that floor, restoration parks at `3.2e-10` and the sub-solve's next
//! step computation fails. Before the fix pounce reported
//! `Restoration_Failed` (AMPL 500) and wrote the point restoration was
//! *entered* at, `1.1e5` off, which `pounce verify` rejects. Ipopt 3.14
//! reports `Feasible point for square problem found` on the same `.nl` at the
//! same `tol`.
//!
//! Two defects, two assertions per `tol`:
//!
//! * **The verdict.** A square model at a point feasible to `constr_viol_tol`
//!   is solved, however the restoration sub-solve came to stop there. The
//!   status must stay out of the AMPL 200 and 500 bands for every `tol`
//!   (`dev-notes/termination-status-invariants.md`: tightening a tolerance
//!   may cost iterations, never manufacture a failure).
//! * **The point.** The `.sol` must hold the point the verdict is about.
//!   `pounce verify` is the oracle: it re-evaluates the canonical `.nl`
//!   without trusting the solver.
//!
//! # What this file is *not* evidence about
//!
//! One square model, the exact-Hessian leg. The failed-restoration hand-back
//! on non-square models (the same change, the `failed_point` path in
//! `min_c_1nrm.rs`) is pinned at unit level in that file.

use std::path::PathBuf;
use std::process::Command;

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/square_flowsheet_resto.nl");
    p
}

/// Solve at `tol` and return `(solve_result_num, status, verify exit code)`.
fn solve(tol: &str) -> (i64, String, i32) {
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_pounce"));
    let dir = std::env::temp_dir();
    let stem = format!("pounce_i1084_{}_{tol}", std::process::id());
    let sol = dir.join(format!("{stem}.sol"));
    let json = dir.join(format!("{stem}.json"));
    let out = Command::new(&bin)
        .arg(fixture())
        .arg(&sol)
        .arg(format!("tol={tol}"))
        .arg("--json-output")
        .arg(&json)
        .output()
        .expect("spawn pounce");
    assert!(
        json.exists(),
        "no JSON report at tol={tol}:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
    let srn = report["solution"]["solve_result_num"].as_i64().unwrap();
    let status = report["solution"]["status"].as_str().unwrap().to_string();
    let verify = Command::new(&bin)
        .arg("verify")
        .arg(fixture())
        .arg(&sol)
        .output()
        .expect("spawn pounce verify");
    let _ = std::fs::remove_file(&sol);
    let _ = std::fs::remove_file(&json);
    (srn, status, verify.status.code().unwrap_or(-1))
}

/// Below the floor: the measured case. Each of these was `Restoration_Failed`
/// (500) at a point `pounce verify` rejected.
#[test]
fn below_the_round_off_floor_the_feasible_point_is_the_answer() {
    for tol in ["1e-10", "1e-11", "1e-12"] {
        let (srn, status, verify) = solve(tol);
        assert_eq!(
            status, "FeasiblePointFound",
            "tol={tol}: a square model at a point feasible to constr_viol_tol is solved"
        );
        assert!(
            srn < 100,
            "tol={tol}: solve_result_num {srn} is outside the solved band"
        );
        assert_eq!(
            verify, 0,
            "tol={tol}: the .sol must hold the point the verdict is about"
        );
    }
}

/// The invariant across the whole sweep, the floor's both sides included.
/// It goes no tighter than 1e-12 because the iteration count is the same
/// below the floor; the sweep is about the verdict, not the cost.
#[test]
fn tightening_tol_never_reaches_the_200_or_500_band() {
    for tol in ["1e-6", "1e-8", "1e-9", "1e-10", "1e-11", "1e-12"] {
        let (srn, status, _) = solve(tol);
        assert!(
            !(200..300).contains(&srn) && srn < 500,
            "tol={tol}: {status} (solve_result_num {srn}) on a feasible square model"
        );
    }
}
