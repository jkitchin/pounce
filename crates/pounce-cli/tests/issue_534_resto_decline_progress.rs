//! The acceptable-point restoration decline must not stop a solve that is still
//! converging — and deferring it must never cost the answer (gh #534).
//!
//! THE GUARD. When the line search fails at a point that already passes the
//! acceptable-level tolerances, `IpoptAlgorithm::invoke_restoration` declines to
//! enter restoration and reports that point (upstream
//! `IpBacktrackingLineSearch.cpp`'s `ACCEPTABLE_POINT_REACHED`). The reasoning
//! is sound — restoration reduces the constraint violation, and from an already
//! acceptable point it has nothing to reduce and a reportable solution to lose.
//!
//! WHAT IT WAS MISSING. It read the entry point and nothing about the trajectory
//! that reached it, so it stopped a contracting endgame and a dead stall with
//! equal confidence. On CUTE `eigena2` it fires while the dual infeasibility is
//! quartering every iteration on unit steps (`1.19e-5 → 2.96e-6 → 7.38e-7 →
//! 1.84e-7`), three iterations short of a strict certificate. `#534` adds the
//! missing progress test, plus a floor that makes the resulting bet free.
//!
//! WHAT THIS FILE PINS, and what it does not. `eigena2` is not reproducible
//! here — the benchmark `.nl` archive is gitignored and not in the checkout.
//! `csfi2` is: it is the other model the issue names as reaching this guard, it
//! is small enough to carry as a fixture, and this build reaches the guard on it
//! at `theta 1.501e-7`, matching the value quoted in the issue to four digits.
//! So the two properties tested here are the two that `csfi2` can actually
//! decide:
//!
//!   * a **stall** is still declined, unchanged. `csfi2`'s window at the guard
//!     is `[3.267e0, 1.845e-6, 8.468e-8, 8.524e-8]` — three healthy contractions
//!     and then a flat step — so the progress test refuses and the answer is
//!     bit-identical to `resto_decline_deferrals=0`, the pre-#534 behaviour.
//!   * a **lost bet is free**. Forcing the deferral (`resto_decline_progress_
//!     ratio` large drops the progress requirement) makes `csfi2` take the bet
//!     it should not take: the continuation runs, finds nothing better, the
//!     deadline expires, the floor is restored — and the reported answer is
//!     again bit-identical, at a cost of the ten-iteration budget and no more.
//!
//! That second case is the worst case the issue asks to be guaranteed ("the
//! worst case is the current behaviour plus a few wasted iterations, never a
//! worse reported answer"), exercised on a real model rather than argued.
//!
//! Whether the deferral *converts* an `eigena2`-shaped solve into a strict
//! certificate is NOT tested here, because no model that reaches this guard with
//! a contracting window was reachable from this checkout — every live firing
//! found (the fixture corpus, plus `csfi2` swept over nineteen starting points)
//! is a stall with a flat final step. The progress test's behaviour on the
//! contracting trace is pinned instead as a unit test over the issue's recorded
//! numbers, in `ipopt_alg.rs` (`eigena2_endgame_reads_as_contracting`).
//!
//! FIXTURE PROVENANCE. `fixtures/csfi2.nl` is CUTE `CSFI2` (problem MINLEN in
//! Vasko and Stott, *SIAM Review* 37(1) pp. 82-84, 1995; SIF input A.R. Conn,
//! April 1995), transcribed statement for statement from the AMPL `.mod` in
//! Vanderbei's CUTE-in-AMPL collection and written to `.nl` with Pyomo rather
//! than AMPL. 5 variables, 4 constraints. It is the same problem the benchmark
//! archive carries: this build solves it to `55.017604529660318`, against
//! `55.01760452966031` in `benchmarks/BENCHMARK_REPORT.json`, and reaches the
//! guard at the `theta 1.501e-7` the issue quotes.
//!
//! THE TESTS RUN `csfi2_stall_start.nl` (gh#1098), which is `csfi2.nl` with a
//! different starting point (seed 11 of a random-start scan) and nothing else
//! changed. From `csfi2.nl`'s own start the stall at the guard depends on the
//! inertia correction's `δ_s = δ_x`: at `perturb_slack_fact=0.03` the solve
//! certifies (`Optimal`/37) without ever reaching the guard, so the two
//! "same point" tests below passed vacuously and the bounded-cost test failed
//! its "deferral was taken" check. From `csfi2_stall_start` the guard is
//! reached and declined at iteration 12 (`Solved_To_Acceptable_Level`), and a
//! forced deferral is taken and lost (23 iterations, the same `x`), measured
//! identically at `perturb_slack_fact` 1, 0.3, 0.1, 0.03 and 0.01 and 17/17
//! under the round-off screen (`mu_init = 0.1·(1 ± k·1e-12)`) at each. The
//! window and `theta` numbers quoted above are `csfi2.nl`'s and were not
//! re-measured on this start.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;

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

fn tmp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_issue534_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

fn solve(extra: &[&str]) -> SolveReport {
    let json_path = tmp_path("csfi2.json");
    let sol_path = tmp_path("csfi2.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture("csfi2_stall_start.nl"))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path)
        // This file compares arms that differ only in the #534
        // restoration-decline guard. Two of them name a
        // `TERMINATION_POLICY_OPTIONS` member, which suppresses the
        // default-on μ-strategy retry (gh #757), so leaving the retry
        // to its default would let it fire on the bare arm alone and
        // the comparison would no longer isolate the guard. On this
        // fixture the retry is a real promotion -- Solved_To_Acceptable_Level
        // in 12 iterations becomes Solve_Succeeded in 9 -- which is
        // measured in `issue_757_acceptable_retry.rs`, not here.
        .arg("mu_strategy_fallback=no");
    for o in extra {
        cmd.arg(o);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// Drops the progress requirement, so the decline is deferred on any window —
/// the "bypass the guard and see how far the solve gets" switch. On `csfi2` the
/// bet is a losing one, which is the point: this is the worst case.
const FORCE_DEFERRAL: [&str; 1] = ["resto_decline_progress_ratio=1e20"];
/// The pre-#534 behaviour: decline immediately, always.
const NO_DEFERRAL: [&str; 1] = ["resto_decline_deferrals=0"];

/// `csfi2` stalls at the guard, so the default settings must behave exactly as
/// the pre-#534 build did — same status, same objective, same iteration count.
///
/// That comparison is only evidence if the guard is reached at all: a solve
/// that never gets there is identical on both arms for that reason alone. The
/// forced-deferral arm is the witness — it moves only if the guard fires.
#[test]
fn a_stalled_solve_is_declined_exactly_as_before() {
    let baseline = solve(&NO_DEFERRAL);
    let default = solve(&[]);
    let forced = solve(&FORCE_DEFERRAL);
    assert!(
        forced.statistics.iteration_count > baseline.statistics.iteration_count,
        "the guard was not reached (forcing the deferral changed nothing, \
         {} iterations either way), so this comparison is vacuous",
        baseline.statistics.iteration_count,
    );
    assert_eq!(
        default.solution.status, baseline.solution.status,
        "the progress test changed the status on a stalled solve",
    );
    assert_eq!(
        default.solution.objective, baseline.solution.objective,
        "the progress test moved the reported point on a stalled solve",
    );
    assert_eq!(
        default.statistics.iteration_count, baseline.statistics.iteration_count,
        "the progress test cost iterations on a solve it should not have \
         deferred at all",
    );
}

/// The floor: a deferral that does not pay off returns the same point the guard
/// would have returned. Bit-identical, not merely close — the floor is a
/// restored snapshot of that exact iterate, so anything else means the rollback
/// did not happen.
#[test]
fn a_lost_deferral_returns_the_same_point() {
    let baseline = solve(&NO_DEFERRAL);
    let deferred = solve(&FORCE_DEFERRAL);
    assert!(
        deferred.statistics.iteration_count > baseline.statistics.iteration_count,
        "the deferral was not taken ({} iterations either way); the \
         same-point checks below would pass vacuously",
        baseline.statistics.iteration_count,
    );
    assert_eq!(
        deferred.solution.status, baseline.solution.status,
        "a lost deferral changed the reported status",
    );
    assert_eq!(
        deferred.solution.objective, baseline.solution.objective,
        "a lost deferral returned a different point than declining would have \
         ({} vs {})",
        deferred.solution.objective, baseline.solution.objective,
    );
    assert_eq!(
        deferred.solution.x, baseline.solution.x,
        "a lost deferral returned a different primal iterate than declining \
         would have",
    );
}

/// ...and the cost of losing it is bounded by the continuation budget, not open
/// ended. Pre-#534 the same model that motivated this guard (`qcqp1000-1nc`)
/// ground out 2780 iterations after entering restoration from an acceptable
/// point; the deadline is what makes that impossible now.
#[test]
fn a_lost_deferral_costs_a_bounded_number_of_iterations() {
    let baseline = solve(&NO_DEFERRAL).statistics.iteration_count;
    let deferred = solve(&FORCE_DEFERRAL).statistics.iteration_count;
    assert!(
        deferred > baseline,
        "the deferral was not actually taken (both runs stopped at iteration \
         {baseline}); this test would then pass vacuously",
    );
    // The budget is ten outer iterations past the deferral, plus the iteration
    // that observes the expiry.
    assert!(
        deferred <= baseline + 11,
        "a lost deferral ran {} iterations past the decline; the continuation \
         budget is 10",
        deferred - baseline,
    );
}
