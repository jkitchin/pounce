//! pounce#748 — what the default-on μ-strategy retry is allowed to
//! trigger on.
//!
//! #748 turned `mu_strategy_fallback` on by default, to pay for #746's
//! `dirichlet120` casualty. The retry re-runs a solve under the other
//! `mu_strategy` and keeps the second answer only if it reaches
//! `Solve_Succeeded`. Since #138 it had triggered on two statuses:
//! `Maximum_Iterations_Exceeded` and `Solved_To_Acceptable_Level`.
//!
//! Carrying both into the default broke five test targets, all the same
//! way, and the fixture sweep did not see any of it — the sweep runs
//! default options only, and every one of these needs a non-default
//! option to provoke the downgrade the retry then erased:
//!
//! ```text
//!   optimize_hs71::hs071_kkt_fidelity_tol_downgrades_succeeded
//!   masked_certificate_fuzz::the_earliest_refusal_is_the_one_restored_not_the_strictest
//!   issue_616_ls_init_downgrades::the_safeguards_measured_cost_is_now_csfi2_alone
//!   issue_250_dual_guard_never_worse::dual_guard_diversion_does_not_return_a_worse_point
//!   issue_534_resto_decline_progress::a_lost_deferral_costs_a_bounded_number_of_iterations
//! ```
//!
//! `Solved_To_Acceptable_Level` is not a failure — it is a converged
//! answer at the acceptable tolerance — and retrying it by default is
//! wrong three separate ways:
//!
//! 1. It doubles the cost of a solve that already succeeded. "One extra
//!    solve on a run that had already failed" does not describe this.
//! 2. It launders downgrades the caller induced *deliberately* — a tight
//!    `kkt_fidelity_tol`, a certificate veto, `least_square_init_primal`
//!    — so the signal those options exist to produce never arrives.
//! 3. The retry returns the other run's **point**, not just its status,
//!    so it can hand back a different local solution. On
//!    `autocorr_bern55-06` with the dual-divergence guard enabled it
//!    swapped -2304.0000278 for -2320.0000298.
//!
//! So #748 made the default-on retry trigger on
//! `Maximum_Iterations_Exceeded` alone, and an explicit
//! `mu_strategy_fallback=yes` kept the historical pair. `dirichlet120`
//! stalls at max-iterations, so #748's motivating case is recovered
//! either way.
//!
//! The cost of that narrowing, measured at the time: the three
//! fixture-legs the flip had gained went back to where main had them,
//! and they were the only three that had moved.
//!
//! ```text
//!   exact csfi2           SolveSucceeded 21  -> SolvedToAcceptableLevel 35
//!   lbfgs eigenb2         SolvedToAcceptableLevel 41 -> SolvedToAcceptableLevel 69
//!   lbfgs pooling_rt2stp  SolveSucceeded 295 -> SolvedToAcceptableLevel 362
//! ```
//!
//! # pounce#757 supersedes the status-only half of that rule
//!
//! Narrowing by *status* threw out the stock-configuration recoveries to
//! protect the caller-configured ones. Look again at the five broken
//! targets above: every one of them arms a non-default option to provoke
//! the downgrade the retry erased — `kkt_fidelity_tol`, a certificate
//! veto, `dual_diverging_streak`, `resto_decline_deferrals`. Objections 2
//! and 3 are properties of a caller-MODIFIED configuration, not of the
//! exit status, so the condition that actually separates the cases is not
//! "did we end acceptable" but "did the caller tell us what termination
//! means".
//!
//! #757 makes that the rule. The default-on retry now also takes
//! `Solved_To_Acceptable_Level`, but only while the caller has named none
//! of `Application::TERMINATION_POLICY_OPTIONS`. Objection 1 — cost —
//! survives intact and is the accepted price; what it buys is
//! `cho_parmest`, which stalls 5% short of `tol` on the dual term alone
//! with the iterate frozen, and which adaptive certifies in 20 iterations.
//! The three fixture-legs in the table above come back, and the sweep
//! moves nothing else: 3 of 142 lines, both legs, all improvements.
//!
//! What #748 pinned that still holds is below — the opt-in, the
//! `mu_strategy` condition, and the stand-down, which has simply moved
//! from "any acceptable-level exit" to "an acceptable-level exit the
//! caller's own options may have caused". `issue_757_acceptable_retry`
//! owns the positive half.
//!
//! The tests below run `csfi2_stall_start`, not `csfi2` (gh#1098). The
//! sweep numbers above are `csfi2.nl`'s, and its acceptable-level stall
//! depends on the inertia correction's `δ_s = δ_x`: at
//! `perturb_slack_fact=0.03` it certifies directly (`Optimal`/37), so the
//! stand-down tests would pass with no retry to stand down.
//! `csfi2_stall_start.nl` is the same model from a start on which the stall
//! (`Solved_To_Acceptable_Level`/12) and the adaptive promotion
//! (`Solve_Succeeded`/9) are measured identically at `perturb_slack_fact`
//! 1, 0.3, 0.1, 0.03 and 0.01, 17/17 under the round-off screen at each.
//! Every test also reads `statistics.mu_strategy_fallback`, so that it
//! states whether the retry ran rather than inferring it from a status.

use std::path::PathBuf;
use std::process::Command;

use pounce_cli::solve_report::SolveReport;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(format!("{name}.nl"));
    p
}

fn solve(model: &str, opts: &[&str]) -> SolveReport {
    // Keep `=` out of the temp-file names: the option strings are
    // `KEY=VALUE`, and a path carrying one does not survive the round
    // trip through the CLI.
    let slug: String = opts
        .join("_")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let tag = format!("{}_{}_{}", model, std::process::id(), slug);
    let mut json_path = std::env::temp_dir();
    json_path.push(format!("pounce_issue748_{tag}.json"));
    let mut sol_path = std::env::temp_dir();
    sol_path.push(format!("pounce_issue748_{tag}.sol"));

    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture(model))
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path);
    for o in opts {
        cmd.arg(o);
    }
    let _ = cmd.status().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    serde_json::from_str(&text).expect("parse SolveReport JSON")
}

/// The stand-down, in the form #757 left it: an acceptable-level exit is
/// retried by default, *unless* the caller armed one of the options that
/// decides what termination means. `csfi2_stall_start` finishes
/// `Solved_To_Acceptable_Level` in 12 iterations and the opposite μ
/// strategy reaches `Solve_Succeeded` in 9 — a genuine improvement the
/// stock configuration is now allowed to take, and a deliberate downgrade
/// the tuned configuration must still be shown.
///
/// If the second half of this test starts reading `SolveSucceeded`, the
/// trigger has widened past `TERMINATION_POLICY_OPTIONS` and every
/// caller-induced downgrade in the suite is being laundered again — see
/// the five targets in this file's header. Status only; the iteration
/// count is the platform-sensitive half.
#[test]
fn a_caller_set_termination_option_stands_the_retry_down() {
    let stock = solve("csfi2_stall_start", &[]);
    let mf = stock
        .statistics
        .mu_strategy_fallback
        .as_ref()
        .expect("the stock run's first attempt must stall and be retried");
    assert_eq!(mf.base_status, "Solved_To_Acceptable_Level", "{mf:?}");
    assert!(mf.promoted, "{mf:?}");
    assert_eq!(
        format!("{:?}", stock.solution.status),
        "SolveSucceeded",
        "with no options set, csfi2's acceptable-level exit must be retried \
         and promoted (pounce#757). Reading SolvedToAcceptableLevel here means \
         the default trigger has narrowed back to Maximum_Iterations_Exceeded \
         alone and cho_parmest has stopped certifying",
    );

    let tuned = solve("csfi2_stall_start", &["kkt_fidelity_tol=1e-14"]);
    assert!(
        tuned.statistics.mu_strategy_fallback.is_none(),
        "a caller-set termination option must stand the retry down: {:?}",
        tuned.statistics.mu_strategy_fallback,
    );
    assert_eq!(
        format!("{:?}", tuned.solution.status),
        "SolvedToAcceptableLevel",
        "naming a termination-policy option must stand the automatic retry \
         down, so the downgrade that option exists to produce still reaches \
         the caller (pounce#748 objection 2, pounce#757)",
    );
}

/// The historical trigger pair is still there for anyone who names it.
/// The retry keeps the second answer only on `Solve_Succeeded`, so asking
/// for it cannot make the status worse — only the wall clock.
#[test]
fn an_explicit_opt_in_still_retries_and_promotes() {
    let r = solve("csfi2_stall_start", &["mu_strategy_fallback=yes"]);
    let mf = r
        .statistics
        .mu_strategy_fallback
        .as_ref()
        .expect("mu_strategy_fallback=yes must retry the acceptable-level exit");
    assert!(mf.promoted, "{mf:?}");
    assert_eq!(
        format!("{:?}", r.solution.status),
        "SolveSucceeded",
        "mu_strategy_fallback=yes must still retry a Solved_To_Acceptable_Level \
         solve under the other mu_strategy and promote it (pounce#138, #748)",
    );
}

/// An explicit `mu_strategy` stands the automatic retry down — #748's
/// condition, pinned end to end rather than at the predicate. Retrying
/// under the other schedule recovers a solve that stalled on a strategy
/// POUNCE chose; it is not licence to override one the caller named, and
/// without this every controlled comparison that pins `mu_strategy` —
/// this repository's own benchmark arms included — would silently run
/// both arms.
#[test]
fn naming_a_strategy_stands_the_automatic_retry_down() {
    let r = solve("csfi2_stall_start", &["mu_strategy=monotone"]);
    assert!(
        r.statistics.mu_strategy_fallback.is_none(),
        "an explicit mu_strategy must suppress the automatic retry: {:?}",
        r.statistics.mu_strategy_fallback,
    );
    assert_eq!(
        format!("{:?}", r.solution.status),
        "SolvedToAcceptableLevel",
        "an explicit mu_strategy must suppress the automatic retry (pounce#748)",
    );
}
