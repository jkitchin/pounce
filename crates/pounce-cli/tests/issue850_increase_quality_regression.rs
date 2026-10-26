//! The `increase_quality` rung cost two solves on `square_flowsheet_resto`
//! under FERAL 0.20, and `feral_increase_quality=no` was the recovery (gh #850,
//! the underlying regression rather than the reporting one). Under FERAL 0.21
//! neither half holds as filed — see "What FERAL 0.21 did to both lines".
//!
//! # What it is
//!
//! `2c4f25f1` wired `FeralSolverInterface::increase_quality`, which had
//! returned a hard-coded `false`, through to FERAL's escalation ladder. Ipopt
//! calls `IncreaseQuality` when `PdFullSpaceSolver`'s refinement stalls, and
//! every upstream backend that can escalate does, so wiring it looked like
//! restoring a missing rung.
//!
//! The contract it restores is not the one the ladder satisfies. MA57 answers
//! the call by raising `pivtol` toward `pivtolmax`: strictly more conservative
//! each time, so keeping it raised for the rest of the solve can only make the
//! factorization safer. FERAL's ladder changes *which pivots are taken* — a
//! lateral move in trajectory terms — and it persists identically, across every
//! later factorization including a restoration sub-solve's. So it reroutes
//! solves, and on this fixture under FERAL 0.20 it rerouted both legs into
//! failure:
//!
//! ```text
//!   exact   Optimal/99   ->  RestorationFailed/131, shipped only because a
//!                            second-opinion rung rescues it at 185 total
//!   lbfgs   Optimal/178  ->  3000 iterations, at the cap, rescued by nothing
//! ```
//!
//! The lbfgs line reads "rescued by nothing" as of gh#850, and gh#857 changed
//! that: `feral_increase_quality_retry` re-solves it without the escalation
//! and promoted `Optimal`/178 at 3178 total.
//!
//! The lbfgs leg is worse than the reported exact one and was found by the
//! `2nd=` sweep column added alongside this: it showed the leg failing with no
//! ladder behind it at all.
//!
//! # What FERAL 0.21 did to both lines
//!
//! The 0.21 bump moved this fixture on both legs, in opposite directions for
//! the claim this file makes, and the tests below are the measured result
//! rather than the claim as it was filed:
//!
//! ```text
//!                      FERAL 0.20                     FERAL 0.21
//!   exact   rung on    RestorationFailed/131          Optimal/95, no ladder
//!   exact   rung off   Optimal/99                     Optimal/59
//!   lbfgs   rung on    3000, at the cap               3000, at the cap
//!   lbfgs   rung off   Optimal/178                    3000, at the cap
//! ```
//!
//! On the exact leg the regression is **gone**: the rung still costs
//! iterations (95 against 59) but no longer the verdict, so the test that
//! pinned the loss now pins its absence. On the lbfgs leg the rung is **no
//! longer the cause**: the leg runs to the cap with it off as well, so
//! `feral_increase_quality=no` is not a recovery there any more, and rung 4
//! (which is that option, taken automatically) cannot be either. What recovers
//! the leg at the default options on 0.21 is the square-system stage, in 8
//! iterations, pinned in `square_system_recovery.rs`. Both lbfgs runs below
//! switch that stage off, because with it on the second run's last summary
//! block is the stage's and the run reads `Optimal` for a reason that has
//! nothing to do with this option — which is how this file's lbfgs test kept
//! passing across the bump with its premise false.
//!
//! # Why this is an option and not a changed default
//!
//! **The rung also buys things nothing else supplies, so flipping it is a
//! trade, not a fix.** The 12-variable model in
//! `pounce-rs/tests/watchdog_trial_is_not_a_divergence_verdict.rs` ends
//! `SolvedToAcceptableLevel` at `obj = 3.7e-6` with the rung and at
//! `obj = 3.42` against `f* = 0` without it — a wrong-ish answer under a
//! success-shaped status, which is worse than an honest failure. It also buys
//! 15–25% of the iterations on five fixture-legs.
//!
//! And nothing separated the two sides on FERAL 0.20 (the figures in this
//! paragraph and the next are that version's). The rung fired exactly twice in
//! this fixture's **base solve** on the exact leg — once in the main loop, printed
//! as a `q` on the row labelled 26, and once inside restoration at `76r`,
//! printed nowhere — and allowing **only the first** still loses the leg, so
//! scoping it out of restoration would not help. Nor does a count: `deb7` and
//! `square_flowsheet_resto` each fire it exactly twice on their exact legs,
//! one gaining 16% of its iterations and the other losing its verdict.
//!
//! Read the scope on that "twice": it is the base solve. gh#857 added the
//! `quality_escalations` statistic, which reproduces it independently and
//! reports the rest — six process-wide on the exact leg once the
//! second-opinion rung's own solve is counted, twenty-five per solve on the
//! lbfgs leg. It is also what gh#857's recovery rung gates on, and it uses the
//! count only as a `>= 1` admission test, for exactly the reason in the
//! paragraph above.
//!
//! One correction this file owes its own text: FERAL's ladder is documented as
//! scaling-then-pivot-threshold, but the scaling rung is **unreachable** as
//! pounce ships — feral takes it only under `ScalingStrategy::Identity` and
//! pounce defaults to `Auto`. Every escalation here is a `pivot_threshold`
//! bump, the first 1e-8 → 1e-6, and no milder ladder helps: every static
//! `feral_pivtol` in {1e-6, 3.16e-5, 4.2e-4, 1e-2, 0.5} loses the lbfgs leg
//! from iteration 0.
//!
//! So the default stands, and what this file pins is the **trade itself** as
//! the current FERAL leaves it — on 0.20, that the rung cost the two solves and
//! that turning it off recovered them; on 0.21, that the exact leg keeps its
//! verdict with the rung on and pays only iterations, and that the lbfgs leg
//! is lost with the rung off too. Resolving it properly needs a
//! *revertible* escalation, which FERAL's `quality_level` cannot express today.

use std::path::PathBuf;
use std::process::Command;

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

/// `(stdout + stderr, exit banner, base-solve iteration count)` for one solve.
///
/// Both streams, because the two things this file reads live on different
/// ones: the `EXIT:` banner and the iteration count are printed to stdout, and
/// the second-opinion ladder's narration is `eprintln!`.
fn solve(extra: &[&str]) -> (String, String, i64) {
    let tag: String = extra
        .join("_")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let sol = std::env::temp_dir().join(format!("pounce_850iq_{tag}.sol"));
    let out = Command::new(pounce_exe())
        .arg(fixture("square_flowsheet_resto.nl"))
        .arg(&sol)
        .args(extra)
        .output()
        .expect("run pounce");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let s = format!("{stdout}{}", String::from_utf8_lossy(&out.stderr));
    let exit = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("EXIT: "))
        .next_back()
        .unwrap_or("<no EXIT>")
        .to_string();
    // The FIRST block: every claim in this file is about the base solve. A
    // run that ends on a failure can print a further attempt after it
    // ("Re-solving the same problem"), and that block's count is not the base
    // solve's — on ubuntu CI it read 200 where the base solve ran to 3000.
    let iters = stdout
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix("Number of Iterations....:"))
        .next()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(-1);
    (s, exit, iters)
}

/// The exact leg, which gh #850 reports, as FERAL 0.21 leaves it: the rung no
/// longer loses the verdict. With it on, the base solver reaches `Optimal`
/// itself and no second-opinion rung runs; with it off the base solver
/// reaches the same verdict in fewer iterations. So the rung is still a cost
/// here — 95 iterations against 59 — but a cost in iterations, not the
/// verdict.
///
/// This is the inverted form of the test gh#850 wrote, which asserted the
/// exact leg reached `Optimal` *only* through a promotion (131 failing, 54 in
/// the rescue rung, 185 total, on 0.20). If the escalation starts losing the
/// verdict again, the first assertion block fails and the old pin is the one
/// to restore.
#[test]
fn the_exact_leg_is_no_longer_lost_to_the_rung() {
    let (out, exit, iters) = solve(&[]);
    assert!(
        exit.starts_with("Optimal Solution Found"),
        "with the rung on, FERAL 0.21's base solver reaches Optimal on this \
         leg, got {exit:?}"
    );
    assert!(
        !out.contains("second-opinion ladder") && !out.contains("promoting"),
        "and it reaches it without a second opinion — on FERAL 0.20 this leg \
         failed in restoration and only a promoted rung rescued it:\n{out}"
    );

    let (out, exit, iters_off) = solve(&["feral_increase_quality=no"]);
    assert!(
        exit.starts_with("Optimal Solution Found"),
        "with the rung off the base solver should reach it, got {exit:?}"
    );
    assert!(
        !out.contains("promoting"),
        "with the rung off no rescue should be needed:\n{out}"
    );
    assert!(
        iters_off < iters,
        "the rung still costs iterations on this leg: {iters} with it, \
         {iters_off} without (95 and 59 when FERAL 0.21 was taken)"
    );
}

/// The lbfgs leg, which is worse and which no ladder was rescuing when gh#850
/// was filed: with the rung on, the base solve runs to the 3000-iteration cap
/// (since gh#1076 it fails in restoration at 250 instead; see below).
/// The L-BFGS path is not exotic coverage — the Python frontend and the CasADi
/// plugin both select `limited-memory` on their own whenever no exact
/// Lagrangian Hessian is available.
///
/// On FERAL 0.20 turning the rung off recovered this leg (`Optimal`/178). On
/// 0.21 it does not: the leg runs to the cap either way, so the rung is not
/// what loses it any more. That is pinned here as the inverted claim, with the
/// recovery paths switched off on both runs so the verdict read is the base
/// solve's — `feral_increase_quality_retry=no` on the first (rung 4 would
/// re-solve with the option off, which is the second run), and
/// `square_system_recovery=no` on both (the stage recovers the leg in 8, and
/// with it on the second run reads `Optimal` whatever this option does). If
/// the option starts recovering this leg again, the second assertion fails
/// and the 0.20 pin — `Optimal` in under 1000 — is the one to restore.
#[test]
fn the_limited_memory_leg_hits_the_cap_with_or_without_the_rung() {
    let (_out, exit, iters) = solve(&[
        "hessian_approximation=limited-memory",
        "feral_increase_quality_retry=no",
        "square_system_recovery=no",
    ]);
    // gh#1076 moved how this run fails, not whether it does. Until then it
    // ran to the 3000 cap. With a zero pivot read under a reused MC64 scaling
    // now refactored under a fresh one, it stops in restoration at 250
    // instead (arm64). Both are a lost base
    // solve, which is the premise this test needs; the second run, with the
    // rung off, is unmoved.
    assert!(
        !exit.starts_with("Optimal Solution Found"),
        "premise: with the rung on this leg is expected to lose the base \
         solve; it reached {exit:?} at {iters}, so the regression has moved"
    );

    let (_out, exit, iters) = solve(&[
        "hessian_approximation=limited-memory",
        "feral_increase_quality=no",
        "square_system_recovery=no",
    ]);
    assert!(
        exit.starts_with("Maximum Number of Iterations Exceeded"),
        "on FERAL 0.21 this leg runs to the cap with the rung off as well; \
         got {exit:?} at {iters}"
    );
}
