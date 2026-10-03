//! gh#592 — a `Solve_Succeeded` point that a restart from it improves.
//!
//! Reported against `pounce-solver` 0.10.0 on a LyoPRONTO Problem 2 GDP
//! fixed-policy NLP: pounce returned `Solve_Succeeded`, and re-solving
//! the identical NLP from the returned primal point improved the
//! objective by 25.10 s (0.079%), landing on the point Ipopt 3.14.16
//! reaches in a single solve.
//!
//! ## What it was
//!
//! Two compounding faults in how a `Singular` factorization report is
//! produced and answered.
//!
//! 1. `feral_inertia_pivot_floor` (gh#540) reports `Singular` when a
//!    mismatching inertia count was read off a pivot at the noise floor.
//!    Its floor was the constant `1e-12`, which corresponds to `n ≈ 4500`
//!    on the `n · eps` scale the option's own rationale names — more than
//!    an order of magnitude too generous for the few-hundred-order KKTs
//!    an IPM actually factors. It is now `n · eps`.
//! 2. `Singular` means "the constraint Jacobian may be rank-deficient",
//!    so the handler answers it with `δ_c`. On this model the Jacobian
//!    has full rank, `δ_c` could not help, and because it stays switched
//!    on for the rest of the augmented system the `δ_x` ladder had to
//!    climb against a matrix `δ_c` had made *harder* to hit the requested
//!    inertia on: five rungs, ending at `δ_w = 1e2` where Ipopt accepted
//!    the step at `1e-4`. The over-damped step froze the objective for
//!    eight iterations and the solver exited on the loose tolerance.
//!    `perturb_delta_c_max_rungs` now withdraws `δ_c` once the ladder has
//!    demonstrated it is not helping.
//!
//! ## What this file pins
//!
//! The reported model is GPL-3.0 (LyoPRONTO) and pounce is EPL-2.0, so
//! the captured `.nl` is not vendored here. The mechanism is pinned
//! directly and deterministically by unit tests —
//! `pounce_common::pd_perturbation` for the walk-back state machine and
//! `pounce_feral` for the floor — and this file pins the end-to-end
//! consequence on a fixture the repository already carries.
//!
//! `pooling_rt2stp` walked the same detour. gh#544 took it from 206 to 812
//! iterations, recorded at the time as a known cost of the
//! `feral_inertia_pivot_floor` fix (see
//! `issue_250_dual_guard_never_worse.rs`). It was the same `δ_c` spent on
//! the same evidence, and withdrawing it returned the model to 298.
//!
//! ## gh#693: this fixture no longer reproduces the detour
//!
//! Removing the Tikhonov `δ = 1e-8` from the initial multiplier estimate
//! moved iteration 0, and the escalation this file was written around
//! stopped happening on this model:
//!
//! ```text
//!                          walkback on      walkback off (rungs=0)
//!   main (0.10.0)          298 it           812 it        <- the detour
//!   with gh#693            128 it           116 it        <- no detour
//! ```
//!
//! Both gh#693 columns reach `-3273.9549`. The `δ_c` path is simply not
//! taken any more: turning the walk-back off costs nothing because there
//! is nothing to walk back from. This is not an artifact of the
//! dimension-aware floor from fault 1 above — pinning the pre-#592
//! `feral_inertia_pivot_floor=1e-12` explicitly reproduces `main`'s
//! 298/812 exactly and leaves gh#693's 128/116 exactly as they are, so
//! the two faults are independent here and re-arming fault 1 does not
//! bring the witness back.
//!
//! **What that costs, and what the search for a replacement found.**
//! The end-to-end consequence of fault 2 has no witness in this
//! repository any more, and the search for one came back not merely
//! empty but pointing the other way.
//!
//! Searched: all 58 CLI fixtures under `perturb_delta_c_max_rungs=0`
//! against the default; all 58 again with fault 1 re-armed maximally
//! (`feral_inertia_pivot_floor=1e30`, verified live -- it moves 8
//! fixtures, e.g. pooling 128 -> 601); `feral_singular_pivot_floor`
//! scans across 12 decades on 4 models; and 117 problems from the
//! external benchmark corpus.
//!
//! Every apparent hit was then re-measured at 17 values of `mu_init` at
//! `0.1 * (1 +- k*1e-12)` -- round-off scale, where a chaotic model
//! scatters and a real effect does not. That screen is what the rest of
//! this PR's post-mortems turn on, and it disqualified both candidates:
//!
//!   * `deb7` (fixtures) -- fails at rungs=0, converges at the default,
//!     but its ladder is non-monotone (rungs=1..4 converge, 5 fails,
//!     8 fails) and a +-1% nudge to `mu_init` flips the rungs=0 run to
//!     converge.
//!   * `vanderbei/twirism1` (corpus) -- a single draw showed 178 it with
//!     the walk-back on vs 441 off, the same 2.5x shape as gh#544's
//!     298/812. Under the round-off screen both arms converge at all 17
//!     points and the medians are 154 on vs **146 off**. The single draw
//!     was sampling noise.
//!
//! Pinning either would have been the exact mistake the fixture-sweep
//! post-mortem in `dev-notes/` is about.
//!
//! **The corpus produced three robust anti-witnesses instead.** On
//! `vanderbei/steenbrd`, `steenbrf` and `steenbrg` the walk-back *costs*
//! the solve, deterministically -- 17/17 under the same round-off screen,
//! in both directions, and identically on `main`, so this is a property
//! of the gh#592 mechanism and not something gh#693 introduced:
//!
//! ```text
//!                walkback on (default)          walkback off (rungs=0)
//!   steenbrd     16/17 ErrorInStepComputation   17/17 Optimal, ~118 it
//!   steenbrf     17/17 SolvedToAcceptableLevel  17/17 Optimal, ~452 it
//!   steenbrg     14/17 ErrorInStepComputation   17/17 Optimal, ~79 it
//! ```
//!
//! (A fourth corpus candidate, `CVXQP1_L`, was inert under the screen --
//! 17/17 identical on both arms and both builds -- and is recorded here
//! only so the count is not overstated.)
//!
//! So the honest state of the `δ_c` walk-back after gh#693 is: no
//! measured problem is robustly helped by it, and three are robustly hurt.
//! That is an argument for revisiting the `perturb_delta_c_max_rungs`
//! default, which is a trajectory change in its own right and is
//! deliberately **not** made in this PR -- it needs its own fixture sweep,
//! and folding it in here would make gh#693's sweep undiffable. The
//! measurement is recorded so the next reader starts from it rather than
//! from scratch.
//!
//! The *mechanism* remains directly pinned, as it always was, by the
//! `pounce_common::pd_perturbation` unit tests for the walk-back state
//! machine and the `pounce_feral` unit tests for the floor. What is gone
//! is the end-to-end demonstration that the state machine matters to a
//! real solve.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::return_codes::ApplicationReturnStatus;

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
        "pounce_issue592_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

/// Sized against `cargo test` contention, not against this model — see
/// the note in `issue_250_dual_guard_never_worse.rs`.
const HANG_GUARD: &str = "max_wall_time=300";

fn solve(extra: &[&str]) -> SolveReport {
    let json_path = tmp_path("pooling.json");
    let sol_path = tmp_path("pooling.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture("pooling_rt2stp.nl"))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path)
        .arg(HANG_GUARD);
    for o in extra {
        cmd.arg(o);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("deserialize SolveReport")
}

/// The pre-#592 escalation: `δ_c` stays on however far the `δ_x` ladder
/// has to climb against it.
const NO_WALKBACK: [&str; 1] = ["perturb_delta_c_max_rungs=0"];

const POOLING_OPTIMUM: f64 = -3273.9549;

/// The headline, with its scope corrected by gh#693.
///
/// It was written to say "812 iterations is what gh#544 left behind and
/// `issue_250` recorded; 298 is what withdrawing the unhelpful `δ_c`
/// gives back", with the bound set between the two so that ordinary
/// drift would not fail it but the detour coming back would.
///
/// Since gh#693 the default run is 128 iterations and does not reach for
/// `δ_c` at all, so this **no longer pins the walk-back specifically** —
/// the companion test below now measures that directly and finds it
/// inert. What survives is still worth keeping and still fails if gh#544
/// comes back by any route: this model reaches its known optimum with a
/// certificate, in far fewer than 500 iterations. The bound is left where
/// it was, because its job (catch a return to the 812-iteration régime)
/// is unchanged.
#[test]
fn pooling_reaches_its_optimum_without_the_gh544_detour() {
    let r = solve(&[]);
    assert_eq!(
        r.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "pooling_rt2stp lost its certificate (dual inf {:e})",
        r.statistics.final_dual_inf,
    );
    assert!(
        (r.solution.objective - POOLING_OPTIMUM).abs() < 1e-3,
        "pooling_rt2stp did not reach its known optimum: {}",
        r.solution.objective,
    );
    assert!(
        r.statistics.iteration_count < 500,
        "pooling_rt2stp took {} iterations; the delta_c detour gh#592 \
         removes is back (it was 812 before, 298 after)",
        r.statistics.iteration_count,
    );
}

/// This was the guard against a vacuous pass — with the walk-back off,
/// the build had to still reproduce the 812-iteration run, so that if a
/// later change shortened `pooling_rt2stp` by some other route the guard
/// would fail and say so rather than let the headline pass for a reason
/// it did not describe.
///
/// It did exactly that on gh#693, which shortened the model by a
/// different route. Rather than delete the guard, it is inverted: the
/// measured fact is now that the walk-back is **inert on this fixture**,
/// and that fact is pinned so that the header's claim above cannot
/// silently go stale in the other direction either. If a future change
/// makes the walk-back load-bearing here again, this fails and says so —
/// and that would be the replacement witness the header describes
/// searching for and not finding.
#[test]
fn the_walkback_is_inert_on_this_fixture_since_gh693() {
    let with = solve(&[]);
    let without = solve(&NO_WALKBACK);
    assert_eq!(
        without.solution.status,
        ApplicationReturnStatus::SolveSucceeded,
        "walk-back off used to mean 812 iterations and now means {}; if it \
         has started costing the certificate, the header table above is \
         stale and needs re-measuring",
        without.statistics.iteration_count,
    );
    assert!(
        without.statistics.iteration_count < 300,
        "the pre-#592 escalation is reproducing again ({} iterations with \
         the walk-back off, against {} with it on). That is a *witness \
         coming back*, not a regression -- the header above records a \
         search that failed to find one. Restore the `> 600` guard this \
         test replaced and re-point the header table.",
        without.statistics.iteration_count,
        with.statistics.iteration_count,
    );
}

/// The walk-back must not cost the certificate it is meant to reach
/// sooner: same point, either way. Still true after gh#693, though now
/// for the trivial reason that the walk-back does nothing here.
#[test]
fn the_walkback_reaches_the_same_point_it_used_to() {
    let with = solve(&[]).solution.objective;
    let without = solve(&NO_WALKBACK).solution.objective;
    assert!(
        (with - without).abs() < 1e-4,
        "withdrawing delta_c moved the answer: {with} vs {without}",
    );
}
