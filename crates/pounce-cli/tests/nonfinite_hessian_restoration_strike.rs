//! A restoration entry forced by a non-finite Hessian is not a gh#274
//! near-feasible strike.
//!
//! Fixture `dallass.nl` is Vanderbei's (CUTE) `dallass`, a 46-variable
//! water-distribution network with 31 linear equality rows. Its objective
//! carries `k·|x|^2·|x|^0.85` terms, whose second derivative is `0·∞` when a
//! flow lands on exactly `x = 0`. The factorization then rejects the KKT
//! matrix as non-finite before any inertia is read, and the step computation
//! hands off to restoration — whose Hessian drops the objective term and moves
//! the flow off zero. Upstream Ipopt + MA57 stops at the same point with
//! `Invalid_Number_Detected`.
//!
//! On feral 0.21 the trajectory meets that kink three times (iterations 4, 19
//! and 29), every time with the constraints satisfied to `tol`. Each rescue was
//! productive, but each one counted toward `resto_near_feasible_count`, which is
//! deliberately never reset (DECONVBNE). The third tripped the gh#274 exit, and
//! the solve ended `Error_In_Step_Computation` at `-32390.34` with the dual
//! infeasibility at `3.6e-3`: one iteration from the optimum. feral 0.20 met
//! the kink twice on its path and solved by luck.
//!
//! The counter exists to catch a restoration *cycle*. An entry the Hessian
//! forced says nothing about one, so it is not counted; restoration still runs,
//! and the static and recovery cycle detectors and `max_iter` still bound a real
//! cycle. Revert the exemption in `IpoptAlgorithm::invoke_restoration` and this
//! test reports `Error_In_Step_Computation`.
//!
//! **Which file it runs (gh#1042 R3).** Since the `.nl` reader merges
//! `|x|^p·|x|^q` into `|x|^(p+q)` (`merge_abs_pow_product` in
//! `pounce-nl`), `dallass.nl` no longer reaches a non-finite Hessian at all
//! — it solves in 28 iterations with no restoration — so it stopped
//! exercising the exemption. This test therefore runs
//! `tests/data/dallass_kink_unmerged.nl`: `dallass.nl` with each
//! `|x|^0.85` factor's operand written `x + 0`, which evaluates identically
//! but is not the same expression to the merge, so the product rule and its
//! `0·∞` Hessian come back. It reproduces the 0.13.0 run bit for bit (123
//! iterations, 12 restoration calls, 11 of them non-finite-`W` entries,
//! objective `-32393.225736201894`). It lives in `tests/data/`, not
//! `tests/fixtures/`, so the fixture sweep does not run it twice.

use std::path::PathBuf;
use std::process::Command;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("data");
    p.push("dallass_kink_unmerged.nl");
    p
}

/// The 0.20 optimum, and the one this run reaches.
const DALLASS_OPTIMUM: f64 = -32393.225737;

#[test]
fn nonfinite_hessian_rescues_do_not_trip_the_near_feasible_exit() {
    let json = std::env::temp_dir().join(format!(
        "pounce_nonfinite_hessian_strike_{}.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&json);

    let out = Command::new(pounce_exe())
        .arg(fixture())
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .output()
        .expect("spawn pounce");
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        stdout.contains("EXIT: Optimal Solution Found."),
        "dallass did not solve; a non-finite-Hessian restoration entry was \
         likely counted as a gh#274 strike:\n{stdout}"
    );
    // Solved by the base run, not rescued by the second-opinion ladder.
    assert!(
        !stdout.contains("second opinion"),
        "dallass needed the second-opinion ladder:\n{stdout}"
    );

    let text = std::fs::read_to_string(&json).expect("read JSON report");
    let _ = std::fs::remove_file(&json);
    let report: serde_json::Value = serde_json::from_str(&text).expect("parse JSON report");
    let obj = report["statistics"]["final_objective"]
        .as_f64()
        .expect("final_objective");
    assert!(
        (obj - DALLASS_OPTIMUM).abs() <= 1e-6 * DALLASS_OPTIMUM.abs(),
        "objective {obj} is not the optimum {DALLASS_OPTIMUM}"
    );
}
