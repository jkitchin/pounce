//! gh #757 — the default-on μ-strategy retry takes
//! `Solved_To_Acceptable_Level` when the caller left the convergence
//! configuration alone.
//!
//! pounce#748 turned `mu_strategy_fallback` on by default but narrowed
//! its trigger to `Maximum_Iterations_Exceeded`, refusing
//! `Solved_To_Acceptable_Level` outright. Two of the three reasons it
//! gave are properties of a *caller-modified* configuration rather than
//! of the status: retrying launders a downgrade the caller armed an
//! option to produce, and it can hand back the other run's point. Both
//! are addressed by deferring to `TERMINATION_POLICY_OPTIONS` instead of
//! to the status alone, which is what these tests pin.
//!
//! The motivating model is `cho_parmest`, which lives in the external
//! benchmark corpus and so cannot be a fixture here; `csfi2` reproduces
//! the same promotion on an in-repo model.
//!
//! **From `csfi2_stall_start.nl`, not `csfi2.nl` (gh#1098).** These tests
//! need a stock run that *stalls* at an acceptable-level exit, so that the
//! retry has something to promote. From `csfi2.nl`'s own start that stall
//! is a property of the inertia correction's `δ_s = δ_x`: at
//! `perturb_slack_fact=0.03` the base solve certifies directly
//! (`Optimal`/37) and every test here would pass or fail without the retry
//! ever being asked. `csfi2_stall_start.nl` is the same model from a
//! different start (seed 11 of a random-start scan), on which the stall is
//! not a property of that knob. Measured identically at
//! `perturb_slack_fact` 1, 0.3, 0.1, 0.03 and 0.01: base
//! `Solved_To_Acceptable_Level`/12, adaptive retry `Optimal`/9, and 17/17
//! the same at each under the round-off screen (`mu_init = 0.1·(1 ± k·1e-12)`).

use std::process::Command;

fn pounce_exe() -> String {
    let mut p = std::path::PathBuf::from(env!("CARGO_BIN_EXE_pounce"));
    p.set_extension(std::env::consts::EXE_EXTENSION);
    p.to_string_lossy().into_owned()
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// `(exit line, iteration count, attempts)` — the exit and iteration count
/// of the *last* solve the run reported, and how many solves it reported
/// (one `Number of Iterations` line each; the retry is the second).
fn solve(extra: &[&str]) -> (String, u32, usize) {
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture("csfi2_stall_start.nl")).arg("--no-sol");
    for o in extra {
        cmd.arg(o);
    }
    let out = cmd.output().expect("spawn pounce");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let exit = text
        .lines()
        .filter(|l| l.contains("EXIT:"))
        .next_back()
        .unwrap_or_default()
        .trim()
        .to_string();
    let iters = text
        .lines()
        .filter(|l| l.contains("Number of Iterations"))
        .next_back()
        .and_then(|l| l.rsplit(' ').next().and_then(|v| v.parse().ok()))
        .unwrap_or(0);
    let attempts = text
        .lines()
        .filter(|l| l.contains("Number of Iterations"))
        .count();
    (exit, iters, attempts)
}

/// Stock options: the acceptable-level exit is POUNCE's own stall, so it
/// earns one retry under the flipped schedule — and that retry certifies.
#[test]
fn stock_options_retry_an_acceptable_level_exit() {
    let (exit, iters, attempts) = solve(&[]);
    assert_eq!(
        attempts, 2,
        "expected the base solve to stall and the retry to run (base \
         acceptable/12, retry optimal/9); a single attempt means the base \
         certified on its own and this test no longer reaches the retry",
    );
    assert!(
        exit.contains("Optimal Solution Found"),
        "expected the retry to promote csfi2 to a certificate, got {exit:?}",
    );
    assert!(
        iters > 0 && iters < 30,
        "expected the promoting (adaptive) attempt's iteration count, got {iters}",
    );
}

/// `mu_strategy_fallback=no` restores upstream's single solve.
#[test]
fn an_explicit_no_still_declines() {
    let (exit, _, attempts) = solve(&["mu_strategy_fallback=no"]);
    assert!(
        exit.contains("Solved To Acceptable Level"),
        "an explicit no must not retry, got {exit:?}",
    );
    assert_eq!(attempts, 1, "an explicit no must not retry");
}

/// A caller-armed termination guard means the downgrade may be the
/// signal the caller asked for, so the retry stands down — this is the
/// pounce#748 laundering objection, preserved.
#[test]
fn a_caller_set_termination_option_declines_the_retry() {
    for opt in [
        "kkt_fidelity_tol=1e-14",
        "acceptable_iter=3",
        "dual_diverging_streak=1",
        "resto_decline_deferrals=0",
        "tol=1e-9",
    ] {
        let (exit, _, attempts) = solve(&[opt]);
        assert!(
            exit.contains("Solved To Acceptable Level"),
            "{opt}: expected the acceptable-level stall the retry would \
             otherwise take, got {exit:?}",
        );
        assert_eq!(
            attempts, 1,
            "{opt} must suppress the default-on retry, got {attempts} attempts",
        );
    }
}

/// The caller can always override the guard back on.
#[test]
fn an_explicit_yes_overrides_a_caller_set_termination_option() {
    let (exit, _, attempts) = solve(&["kkt_fidelity_tol=1e-14", "mu_strategy_fallback=yes"]);
    assert_eq!(attempts, 2, "an explicit yes must run the retry");
    assert!(
        exit.contains("Optimal Solution Found") || exit.contains("Solved To Acceptable Level"),
        "an explicit yes must still run the retry path, got {exit:?}",
    );
}
