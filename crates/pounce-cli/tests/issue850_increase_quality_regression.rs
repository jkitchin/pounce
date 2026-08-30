//! The `increase_quality` rung costs two solves on `square_flowsheet_resto`,
//! and `feral_increase_quality=no` is the recovery (gh #850, the underlying
//! regression rather than the reporting one).
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
//! solves, and on this fixture it reroutes both legs into failure:
//!
//! ```text
//!   exact   Optimal/99   ->  RestorationFailed/131, shipped only because a
//!                            second-opinion rung rescues it at 185 total
//!   lbfgs   Optimal/178  ->  3000 iterations, at the cap, rescued by nothing
//! ```
//!
//! The lbfgs leg is worse than the reported exact one and was found by the
//! `2nd=` sweep column added alongside this: it showed the leg failing with no
//! ladder behind it at all.
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
//! And nothing separates the two sides. Measured with a process-global firing
//! cap, the rung fires exactly twice here — once in the main solve at iteration
//! 25 and once inside restoration at `76r` — and allowing **only the first**
//! still loses the leg, so scoping it out of restoration would not help. Nor
//! does a count: `deb7` and `square_flowsheet_resto` each fire it exactly twice
//! on their exact legs, one gaining 16% of its iterations and the other losing
//! its verdict.
//!
//! So the default stands, the option is the documented recovery, and what this
//! file pins is the **trade itself** — that the rung is what costs the two
//! solves, and that turning it off recovers them. Resolving it properly needs a
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

/// `(stdout + stderr, exit banner, iteration count)` for one solve.
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
    let iters = stdout
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix("Number of Iterations....:"))
        .next_back()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(-1);
    (s, exit, iters)
}

/// The exact leg, which gh #850 reports: with the rung on, the base solver
/// fails and only a second-opinion rung makes the verdict come out right.
/// Turning the rung off recovers the base solve outright.
#[test]
fn the_exact_leg_is_lost_to_the_rung_and_recovered_by_the_option() {
    let (out, exit, iters) = solve(&[]);
    assert!(
        exit.starts_with("Optimal Solution Found"),
        "the ladder should still deliver a verdict, got {exit:?}"
    );
    assert!(
        out.contains("promoting"),
        "with the rung on, this fixture reaches its verdict only through the \
         second-opinion rescue — if it no longer does, the regression is gone \
         by some other route and this file's subject has moved:\n{out}"
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
        (0..300).contains(&iters_off),
        "expected ~99 iterations with the rung off, got {iters_off} \
         (the rescued path reports {iters})"
    );
}

/// The lbfgs leg, which is worse and which no ladder was rescuing: with the
/// rung on it runs to the 3000-iteration cap. The L-BFGS path is not exotic
/// coverage — the Python frontend and the CasADi plugin both select
/// `limited-memory` on their own whenever no exact Lagrangian Hessian is
/// available.
#[test]
fn the_limited_memory_leg_hits_the_cap_and_the_option_recovers_it() {
    let (_out, exit, iters) = solve(&["hessian_approximation=limited-memory"]);
    assert!(
        iters >= 1000,
        "premise: with the rung on this leg is expected to run to the cap; it \
         stopped at {iters} ({exit:?}), so the regression has moved"
    );

    let (_out, exit, iters) = solve(&[
        "hessian_approximation=limited-memory",
        "feral_increase_quality=no",
    ]);
    assert!(
        exit.starts_with("Optimal Solution Found"),
        "with the rung off this leg should solve, got {exit:?} at {iters}"
    );
    assert!(
        iters < 1000,
        "expected ~178 iterations with the rung off, got {iters}"
    );
}
