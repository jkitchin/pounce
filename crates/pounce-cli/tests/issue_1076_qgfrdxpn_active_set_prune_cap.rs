//! gh#1076 item 3: the active-set QP's counting prune, on the two
//! Maros-Meszaros models that bound it from either side, under
//! `solver_selection=qp-active-set`.
//!
//! #1047 gave `pounce-qp`'s Schur loop a counting prune: an over-full active
//! set is pruned to an independent subset before it is factored. The prune
//! then stood down for the rest of a pass the first time it removed a slot it
//! had already removed.
//!
//! * **`QGFRDXPN` needs the prune to survive repeats.** Its first elastic
//!   phase-1 pass repeats a slot at iteration 73. With the counting prune off,
//!   the over-full set factors, its multipliers fail stationarity, and the
//!   stationarity prune ran the same cycle 992 times until the rank-repair
//!   budget was gone. The pass then ended in a hard singular-KKT error
//!   (`ErrorInStepComputation`, 11445 iterations).
//! * **`QFORPLAN` needs the prune to stop.** Unguarded, its phase-2 passes
//!   prune a rotating set of slots 73 times a pass. The objective moves
//!   between repeats, so the cycle guard cannot end them, and the solve ends
//!   `MaximumIterationsExceeded` (11776 iterations).
//!
//! The rule that satisfies both is a per-pass cap of 32 prunes
//! (`COUNT_PRUNE_CAP`). The cap is measured, not derived, and the outcome is
//! not monotone in it; the comment at the prune has the grid. That is why
//! both models are pinned rather than the constant.
//!
//! The same PR fixes three defects found on `QGFRDXPN`: a stationarity-prune
//! cycle guard, an exhausted repair budget ending a pass as a limit instead of
//! an error, and phase-1 feasibility checked on the rows as well as the
//! slacks. None of them recovers `QGFRDXPN` under the stand-down rule (it ends
//! `MaximumIterationsExceeded`), so the cap stays.
//!
//! Mutation evidence (arm64):
//!
//! | mutation | `QGFRDXPN` | `QFORPLAN` |
//! |---|---|---|
//! | none | 3553, optimal | 11278, optimal |
//! | `main` `ab59784a` (stand-down rule) | `ErrorInStepComputation` | optimal |
//! | stand-down rule, with this PR's three fixes | `MaximumIterationsExceeded` | optimal |
//! | cap 16 | `MaximumIterationsExceeded` | optimal |
//! | cap 48 or 64 | optimal | `MaximumIterationsExceeded` |
//! | unguarded | optimal | `MaximumIterationsExceeded` |

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_solve_report::SolveReport;

/// Generous for both models: `QGFRDXPN` takes 6–9 s with the fix and
/// `QFORPLAN` 39–50 s, in release builds at low load. The failing runs end on
/// the status, not on this cap.
const WALL_CAP: Duration = Duration::from_secs(400);

fn solve_active_set(fixture: &str) -> SolveReport {
    let mut fx = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fx.push("tests");
    fx.push("fixtures");
    fx.push(format!("{fixture}.nl"));
    let tmp = |ext: &str| {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "pounce_gh1076_{fixture}_{}.{ext}",
            std::process::id()
        ));
        p
    };
    let (json, sol) = (tmp("json"), tmp("sol"));
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
            panic!("{fixture} under qp-active-set ran past {WALL_CAP:?} (gh#1076)");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let text = std::fs::read_to_string(&json).expect("JSON report");
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&sol);
    let r: SolveReport = serde_json::from_str(&text).expect("parse SolveReport JSON");
    assert_eq!(
        r.solution.engine, "qp-active-set",
        "{fixture}: must be answered by the active-set engine it was routed to"
    );
    r
}

fn assert_optimal(fixture: &str, optimum: f64, max_iters: usize) {
    let r = solve_active_set(fixture);
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "{fixture}: must solve under qp-active-set (gh#1076)"
    );
    let rel = ((r.solution.objective - optimum) / optimum).abs();
    assert!(rel < 1e-6, "{fixture}: objective {}", r.solution.objective);
    let it = r.statistics.iteration_count as usize;
    assert!(it <= max_iters, "{fixture}: {it} iterations");
}

/// Published optimum `0.100790585D+12`. Measured 3553 iterations with the fix;
/// the failing runs above measured 11445 to 58118.
#[test]
fn qgfrdxpn_active_set_solves_with_the_per_pass_prune_cap() {
    assert_optimal("convex_qp_qgfrdxpn", 1.00790585e11, 8000);
}

/// Published optimum `7.45663148D+09`. Measured 11278 iterations with the fix.
/// The bound is loose because the failing runs end on the status, near 11 000
/// iterations as well.
#[test]
fn qforplan_active_set_does_not_cycle_under_the_per_pass_prune_cap() {
    assert_optimal("convex_qp_qforplan", 7.45663148e9, 25_000);
}
