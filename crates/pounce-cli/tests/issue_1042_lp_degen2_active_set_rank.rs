//! gh#1042 — NETLIB `DEGEN2` under `solver_selection=qp-active-set` after
//! FERAL 0.21.
//!
//! Bisected to `97bc0f15` ("deps: feral 0.21.0", #1038): `lp_degen2` went from
//! 8.2 s / 1971 iterations (`8d8ab93c`; FERAL 0.20 at `f1197ef3`: 9.5 s / 2071)
//! to more than 300 s on the CLI, and `issue974_active_set_netlib.rs`'s
//! `lp_degen2` test to about 1205 s.
//!
//! The mechanism. Around iteration 630 `pounce-qp`'s Schur loop holds 474
//! active slots on 473 variables. That set is rank-deficient by counting, and
//! the reset that factors it relied on the backend to *say* so: FERAL 0.20
//! flagged the singular KKT, the rank repair pruned it, and the solve went on.
//! FERAL 0.21 factors it without complaint, so nothing pruned; the next drop's
//! step carried `‖A_W p‖ ≈ 0.4‖p‖`, the dropped row re-entered at `α = 0`, and
//! the drop/re-add two-cycle ran into the gh#974 cycle guard as `MaxIter`.
//! The convex driver then spent three cold recoveries of 8530 iterations each
//! before answering. The fix prunes an over-full set *before* factoring it
//! (`ParametricActiveSetSolver::schur_reset_rank_repaired`); its unit tests
//! (`over_n_rank_repair_tests` in `pounce-qp/src/solver.rs`) pin the invariant
//! on sets FERAL 0.21 also leaves unflagged.
//!
//! What this file pins is the magnitude, because that is the whole symptom:
//! the answer was right either way (`issue974`'s test passed, in 20 minutes).
//! The iteration bound sits about 2× above every measured good run (1971,
//! 2071, 2009) and far below one cold recovery's 8530; the wall cap is set so
//! a regression fails in bounded time rather than taking the suite with it.
//!
//! Mutation-checked before gh#1061: with the pre-factor prune removed, this
//! test failed on the wall cap (the solve was killed) on x86_64. Since gh#1061
//! an LP on the active-set arm is answered by the simplex (0.4 s, 2026
//! pivots, prune or no prune), so this file no longer reaches the prune and
//! guards only the magnitude; `over_n_rank_repair_tests` pins the prune
//! itself.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_solve_report::SolveReport;

const FIXTURE: &str = "lp_degen2";

/// The convex IPM's objective on the same file (`solver_selection=auto`), the
/// oracle `issue974_active_set_netlib.rs` uses.
const OPTIMUM: f64 = -1435.178;

/// Measured 2009 with the fix (x86_64, FERAL 0.21); 1971 and 2071 before
/// FERAL 0.21. One cold recovery alone is 8530.
const MAX_ITERS: usize = 4000;

/// Measured 10–15 s with the fix (release and test builds alike), against
/// more than 300 s without it. Eight times the fix, far short of
/// the regression.
const WALL_CAP: Duration = Duration::from_secs(120);

fn tmp_path(ext: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pounce_gh1042_degen2_{}.{ext}", std::process::id()));
    p
}

#[test]
fn lp_degen2_active_set_does_not_cycle_into_the_cold_recoveries() {
    let mut fx = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fx.push("tests");
    fx.push("fixtures");
    fx.push(format!("{FIXTURE}.nl"));
    let json = tmp_path("json");
    let sol = tmp_path("sol");
    let started = Instant::now();
    let mut child = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(&fx)
        .arg("--sol-output")
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .arg("solver_selection=qp-active-set")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn pounce");
    loop {
        if child.try_wait().expect("wait on pounce").is_some() {
            break;
        }
        if started.elapsed() > WALL_CAP {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "{FIXTURE} under qp-active-set ran past {WALL_CAP:?}: the Schur \
                 loop is cycling on an over-full active set again (gh#1042)"
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let wall = started.elapsed();
    let text = std::fs::read_to_string(&json).expect("JSON report");
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&sol);
    let r: SolveReport = serde_json::from_str(&text).expect("parse SolveReport JSON");
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "{FIXTURE}: must solve"
    );
    assert_eq!(
        r.solution.engine, "qp-active-set",
        "{FIXTURE}: must be answered by the active-set engine it was routed to"
    );
    let rel = ((r.solution.objective - OPTIMUM) / OPTIMUM).abs();
    assert!(rel < 1e-6, "{FIXTURE}: objective {}", r.solution.objective);
    let it = r.statistics.iteration_count as usize;
    assert!(
        it <= MAX_ITERS,
        "{FIXTURE}: {it} iterations in {wall:?} (measured 2009 with the fix, \
         1971–2071 before FERAL 0.21)"
    );
}
