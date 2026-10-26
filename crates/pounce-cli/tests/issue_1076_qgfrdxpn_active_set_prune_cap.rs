//! gh#1076 item 3 — Maros-Meszaros `QGFRDXPN` under
//! `solver_selection=qp-active-set`.
//!
//! #1047 gave `pounce-qp`'s Schur loop a counting prune (an over-full active
//! set is pruned to an independent subset before it is factored) and stood
//! that trigger down for the rest of a pass the first time it removed a slot
//! it had already removed. That cost this model: `SolvedToAcceptableLevel`
//! before #1047, `ErrorInStepComputation` (11445 iterations) after.
//!
//! The mechanism, traced on arm64. The convex driver's first attempt enters
//! the l1-elastic phase-1 (1856 variables after presolve and elastic slacks).
//! Its first pass repeats a pruned slot at iteration 73, and the stand-down
//! switches the counting prune off. The over-full set then factors, its
//! multipliers fail stationarity, and the stationarity prune takes over the
//! same drop / re-add cycle — 992 times, never seen by the gh#974 cycle guard
//! because that branch `continue`s before it — until the rank-repair budget
//! is exhausted at iteration 7779 and the pass ends in a hard singular-KKT
//! error. With the counting prune left on, every phase-1 pass instead ends
//! `MaxIter` at the cycle guard within 24 prunes, and the phase-2 solve from
//! there is optimal in 3553 working-set changes.
//!
//! The fix replaces the stand-down with a per-pass cap of 32 prunes
//! (`COUNT_PRUNE_CAP`). The cap is measured, not derived — see the comment at
//! the prune for the window on this model and on `QFORPLAN`, which is why
//! this file pins the model rather than the constant.
//!
//! Mutation-checked: on `main` at `ab59784a` (the stand-down rule) this file
//! fails on the status (`ErrorInStepComputation`); with the cap at 16 it fails
//! on the status (`MaximumIterationsExceeded`, 23820 iterations).

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_solve_report::SolveReport;

const FIXTURE: &str = "convex_qp_qgfrdxpn";

/// Maros & Meszaros' published optimum, `0.100790585D+12`.
const OPTIMUM: f64 = 1.00790585e11;

/// Measured 3553 with the fix (arm64, FERAL 0.22). The failing runs measured
/// 11445 (stand-down) and 20334–27063 (caps below 24).
const MAX_ITERS: usize = 8000;

/// Measured 7–9 s with the fix in a release build; the stand-down rule took
/// 73–104 s to fail.
const WALL_CAP: Duration = Duration::from_secs(180);

fn tmp_path(ext: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_gh1076_qgfrdxpn_{}.{ext}",
        std::process::id()
    ));
    p
}

#[test]
fn qgfrdxpn_active_set_solves_with_the_per_pass_prune_cap() {
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
            panic!("{FIXTURE} under qp-active-set ran past {WALL_CAP:?} (gh#1076)");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let text = std::fs::read_to_string(&json).expect("JSON report");
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&sol);
    let r: SolveReport = serde_json::from_str(&text).expect("parse SolveReport JSON");
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "{FIXTURE}: must solve under qp-active-set (gh#1076)"
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
        "{FIXTURE}: {it} iterations (measured 3553 with the fix)"
    );
}
