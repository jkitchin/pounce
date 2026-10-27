//! gh#815 — a restoration failure opens the second-opinion ladder.
//!
//! Before this test's fix, `Restoration_Failed` opened no ladder at all: only
//! `Infeasible_Problem_Detected` and `Invalid_Number_Detected` did
//! (`SecondOpinionTrigger::for_status`). That left the one verdict class
//! whose whole content is "the path went somewhere the restoration
//! sub-problem could not work from" with no second opinion, while the rung
//! that answers exactly that — a displaced start — sat available and unused.
//!
//! `degenerate_start_ladder.rs` is the companion file and covers the two
//! triggers that already opened a ladder. It cannot cover this one: both of
//! its fixtures are HS008 from the origin, which exits
//! `Infeasible_Problem_Detected`, so the `Restoration_Failed` arm of
//! `for_status` is never executed there. That is the rule from
//! `CLAUDE.md` — a green leg is only evidence about the branch its fixture
//! reaches — applied to the status map.
//!
//! # The fixture
//!
//! `square_flowsheet_resto.nl` is a **square** (32 × 32, zero degrees of
//! freedom, `f(x) = 0`) flowsheet-shaped model in the gh#815 family:
//! four constraint blocks (mass, pressure drop, vapour-liquid equilibrium,
//! energy) over eight stages closed into a recycle, with magnitudes spanning
//! `1e-6` mole fractions to `3e6` Pa in the same system — the mixed scaling
//! that makes IDAES flowsheets hard. Every row is written `expr(x) ==
//! expr(x*)`, so the model is **feasible by construction** and the generator
//! (`dev-notes/square_flowsheet_resto_gen.py`) asserts the residual at `x*`
//! is exactly `0`. The start is `x*` displaced by a factor of three in `P`,
//! `1/3` in `F` and `3` in `y` — strictly inside every bound.
//!
//! On FERAL 0.20 that start reached `Restoration Failed!` in 131 iterations
//! (47 outer, then 84 inside restoration), far short of `max_iter`, and
//! rung 3 recovered it in 54. Ipopt 3.13.2 solves the same model in 34.
//!
//! **FERAL 0.21 solves it outright** — `Optimal` in 95 at the default `tol`,
//! with no ladder at all — so the exact-Hessian default run no longer reaches
//! the branch this file is about. Until gh#1084 the witness was a `tol` below
//! the model's round-off floor (`tol=1e-11`): restoration parked at a
//! violation of about `3.1e-10`, round-off on rows that carry `3e6` Pa, and
//! the sub-solve's next step failed. That was **not** a restoration failure.
//! It was a square model at a point feasible to `constr_viol_tol` by six
//! orders, which upstream reports as `Feasible point for square problem
//! found`, and pounce now does too (`issue_1084_square_round_off_floor.rs`).
//!
//! [`LEG`] is the witness now: the L-BFGS leg at the default `tol`. There the
//! base solve fails in restoration for real, 250 iterations in at a violation
//! of order `1e5`, nowhere near feasible. The ladder opens on that verdict
//! with the displaced start first. The displaced start and rung 4 fail, and
//! the square-system stage recovers the model in 8 iterations. That gives
//! this trigger a recovery to pin again, which the round-off witness never
//! had: rung 3 recovering a model is still pinned on the local-infeasibility
//! trigger in `degenerate_start_ladder.rs`, and what is specific to this
//! trigger, and what this file pins, is that `Restoration_Failed` opens the
//! ladder, that the displaced start is its first rung, and that rungs 1 and
//! 2 stay shut.
//!
//! # What this file is *not* evidence about
//!
//! One fixture, one linear solver (the FERAL default), one leg (L-BFGS). It pins that
//! the trigger is wired and that rung 3 is the first rung it opens; it says
//! nothing about how often a displaced start rescues a restoration failure in
//! general — `start_point_retry`'s option text carries that measurement (13
//! of 15 over the KRONOS corpus).

use std::path::PathBuf;
use std::process::Command;

const MODEL: &str = "square_flowsheet_resto.nl";

/// The leg on which this model genuinely fails in restoration; see the module
/// comment. On the exact-Hessian leg FERAL 0.21 solves the fixture and no
/// ladder opens.
const LEG: &str = "hessian_approximation=limited-memory";

fn run(extra: &[&str]) -> String {
    let mut model = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    model.push("tests");
    model.push("fixtures");
    model.push(MODEL);
    let sol = std::env::temp_dir().join(format!("pounce_i815_{}.sol", extra.len()));
    let _ = std::fs::remove_file(&sol);
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .arg(&model)
        .arg(&sol)
        .arg(LEG)
        .args(extra)
        .output()
        .expect("spawn pounce");
    let _ = std::fs::remove_file(&sol);
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    )
}

/// The fix, end to end. The baseline solve fails in restoration; the ladder
/// opens on that verdict, and the displaced start is the first rung it
/// spends. (On FERAL 0.20 it was also the rung that recovered the model, in
/// 54; on this leg the square-system stage recovers it instead.)
#[test]
fn a_restoration_failure_opens_the_ladder_with_the_displaced_start_first() {
    let log = run(&[]);
    // The baseline solve is supposed to fail in restoration — if it stops
    // doing so the fixture no longer reaches the branch under test and the
    // assertions below pass vacuously.
    //
    // The witness is the ladder's own trigger line, not an
    // `EXIT: Restoration Failed!` banner. The verdict is printed once per run
    // rather than once per attempt (`one_verdict_per_run.rs`), so a banner
    // describes whichever attempt is reported, not the base one.
    //
    // The trigger line is the direct witness:
    // `SecondOpinionTrigger::for_status` emits "restoration failure" for
    // `Restoration_Failed` and for nothing else, so the phrase appears only
    // when the base solve ended exactly there.
    assert!(
        log.contains("restoration failure — re-solving along"),
        "the baseline solve must fail in restoration and open the ladder on \
         that trigger (gh#815):\n{log}"
    );
    assert!(
        log.contains(
            "(second-opinion ladder: start_point_perturbation=1e-2, \
             feral_increase_quality=no)"
        ),
        "rung 3 is the rung this trigger opens first; rung 4 follows it only \
         because this base solve escalated (gh#857):\n{log}"
    );
    let first_rung = log
        .lines()
        .find(|l| l.starts_with("pounce: second opinion — re-solving with"))
        .unwrap_or_else(|| panic!("no rung was run:\n{log}"));
    assert!(
        first_rung.contains("start_point_perturbation=1e-2"),
        "the displaced start must be the first re-solve spent: {first_rung}"
    );
    assert!(
        !log.contains("re-solving with feral_scaling=mc64")
            && !log.contains("re-solving with mu_strategy=adaptive"),
        "rungs 1 and 2 stay gated on local infeasibility, so this trigger \
         must not reach them:\n{log}"
    );
    assert!(
        log.contains("start_point_perturbation=1e-2 re-solve did not recover")
            && log.contains("square_system_lm recovered the problem")
            && log.contains("Status: Solve_Succeeded"),
        "the displaced start does not recover this leg and the square-system \
         stage does. If the displaced start now recovers it, restore the \
         recovery assertion this test carried on FERAL 0.20:\n{log}"
    );
}

/// Turning off every rung a restoration failure can reach leaves it with no
/// ladder at all, rather than falling back to the two rungs that are not
/// evidence about it. This is also the escape hatch for a caller who wants
/// upstream's one-solve-one-verdict behaviour.
///
/// "Every rung" became two with gh#857, which added
/// `feral_increase_quality_retry` — it opens on the same
/// `Restoration_Failed` whenever the base solve escalated, as this one does,
/// so leaving it on announces a ladder. Rungs 1 and 2
/// remain gated on local infeasibility, and the assertions below still prove
/// that. A rung added later that catches this trigger belongs in this list;
/// the test is the escape hatch, so it has to name the whole hatch.
///
/// Three as of the square-system stage, which also opens on a restoration
/// failure of a square model with a constant objective — this one.
#[test]
fn the_ladder_can_be_switched_off() {
    let log = run(&[
        "infeasibility_perturbed_start_retry=no",
        "feral_increase_quality_retry=no",
        "square_system_recovery=no",
    ]);
    assert!(
        log.contains("EXIT: Restoration Failed!"),
        "still fails in restoration:\n{log}"
    );
    assert!(
        !log.contains("second-opinion ladder"),
        "no rung is available, so no ladder should be announced:\n{log}"
    );
    assert!(
        log.contains("Status: Restoration_Failed"),
        "and the original verdict ships:\n{log}"
    );
}
