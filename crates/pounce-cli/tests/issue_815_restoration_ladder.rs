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
//! From that start pounce reaches `Restoration Failed!` in 131 iterations
//! (47 outer, then 84 inside restoration), far short of `max_iter`, which is
//! what makes "give it a bigger budget" the wrong answer and a different
//! starting point the right one. Ipopt 3.13.2 solves the same model in 34.
//! The 131 is what the summary reports as of gh#819; before that fix it read
//! 47, because the restoration sub-solve's rows were printed and then left
//! out of the count.
//!
//! # What this file is *not* evidence about
//!
//! One fixture, one linear solver (the FERAL default), one leg. It pins that
//! the trigger is wired and that rung 3 is the only rung it opens; it says
//! nothing about how often a displaced start rescues a restoration failure in
//! general — `start_point_retry`'s option text carries that measurement (13
//! of 15 over the KRONOS corpus).

use std::path::PathBuf;
use std::process::Command;

const MODEL: &str = "square_flowsheet_resto.nl";

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
/// opens on that verdict, spends its one rung, and the displaced start
/// recovers the model that was feasible all along.
#[test]
fn a_restoration_failure_opens_the_ladder_and_the_displaced_start_recovers_it() {
    let log = run(&[]);
    // The baseline solve is supposed to fail in restoration — if it stops
    // doing so the fixture no longer reaches the branch under test and the
    // assertions below pass vacuously.
    //
    // The witness is the ladder's own trigger line, not an
    // `EXIT: Restoration Failed!` banner, which is what this used to match.
    // That banner was the BASE attempt's, and the verdict is now printed once
    // per run rather than once per attempt — this run ends
    // `EXIT: Optimal Solution Found.` because rung 3 promotes, which is the
    // very thing the test is here to demonstrate. See `one_verdict_per_run.rs`.
    //
    // The trigger line is an equally strong witness and a more direct one:
    // `SecondOpinionTrigger::for_status` emits "restoration failure" for
    // `Restoration_Failed` and for nothing else, so the phrase appears only
    // when the base solve ended exactly there. The old banner match was
    // weaker — any attempt's banner could have satisfied it.
    assert!(
        log.contains("restoration failure — re-solving along"),
        "the baseline solve must fail in restoration and open the ladder on \
         that trigger (gh#815):\n{log}"
    );
    assert!(
        log.contains("re-solving with start_point_perturbation=1e-2"),
        "rung 3 is the rung this trigger opens:\n{log}"
    );
    assert!(
        !log.contains("re-solving with feral_scaling=mc64")
            && !log.contains("re-solving with mu_strategy=adaptive"),
        "rungs 1 and 2 stay gated on local infeasibility, so this trigger \
         must cost exactly one extra solve:\n{log}"
    );
    assert!(
        log.contains("Status: Solve_Succeeded"),
        "the displaced start recovers the model:\n{log}"
    );
}

/// Turning off every rung a restoration failure can reach leaves it with no
/// ladder at all, rather than falling back to the two rungs that are not
/// evidence about it. This is also the escape hatch for a caller who wants
/// upstream's one-solve-one-verdict behaviour.
///
/// "Every rung" is two as of gh#857, which added
/// `feral_increase_quality_retry` — it opens on the same
/// `Restoration_Failed` and recovers this fixture by a different route, so
/// leaving it on announces a one-rung ladder and promotes. Rungs 1 and 2
/// remain gated on local infeasibility, and the assertions below still prove
/// that. A rung added later that catches this trigger belongs in this list;
/// the test is the escape hatch, so it has to name the whole hatch.
#[test]
fn the_ladder_can_be_switched_off() {
    let log = run(&[
        "infeasibility_perturbed_start_retry=no",
        "feral_increase_quality_retry=no",
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
