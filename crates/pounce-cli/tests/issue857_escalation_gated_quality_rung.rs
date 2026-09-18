//! The losing direction of `feral_increase_quality` now recovers itself
//! (gh#857, rung 4 of the second-opinion ladder).
//!
//! # The shape of the bug
//!
//! `feral_increase_quality` is on by default and is genuinely two-sided: it
//! buys accuracy and 15–25% of the iterations on several fixture-legs, and it
//! loses whole solves on others. Its own option text has said so since gh#850,
//! and the documented remedy for the losing side was for the user to notice,
//! read the option, and re-run. `square_flowsheet_resto` is the fixture that
//! measures both halves:
//!
//! ```text
//!   exact  escalating   Restoration_Failed/131   (rung 3 rescues it: 54, tot 185)
//!   exact  rung off     SolveSucceeded/99
//!   lbfgs  escalating   MaximumIterations/3000   (nothing rescued it)
//!   lbfgs  rung off     SolveSucceeded/178
//! ```
//!
//! The lbfgs leg is the one with no recovery, and it is not a rare path — the
//! Python frontend and the CasADi plugin both select `limited-memory`
//! automatically when no exact Lagrangian Hessian is available.
//!
//! # Why the gate is a measurement
//!
//! The rung opens on `Restoration_Failed` **or**
//! `Maximum_Iterations_Exceeded`, and only when the failing solve's
//! `quality_escalations` count is at least 1.
//!
//! The second half is what makes the first half affordable. A budget exit
//! normally opens no ladder at all, on the sound reasoning that the answer to
//! running out of iterations is more iterations; naming a trigger for every
//! `Maximum_Iterations_Exceeded` would put an extra solve on every capped run
//! in the corpus. The escalation count is what says which capped runs are
//! candidates — and it is not visible anywhere else, which is why gh#857 had
//! to add it before it could add this. `a_budget_exit_that_never_escalated_…`
//! below is the branch that keeps the cost at zero, and it is the test, not a
//! duplicate of the one above it.
//!
//! It is a gate at `>= 1` and deliberately not a threshold: `deb7` escalates
//! exactly as many times as this fixture's base solve and *gains* by it. The
//! count cannot separate the two; the verdict can.
//!
//! # Why the lbfgs arms name an explicit `max_iter`
//!
//! Left to run free, this fixture's lbfgs leg does not reach the same verdict
//! on every platform. On macOS/aarch64 it exhausts the 3000-iteration default
//! and exits `Maximum_Iterations_Exceeded`; on linux/x86_64 it spends the same
//! 3000 iterations with the same 25 escalations and then exits
//! `Infeasible_Problem_Detected` — a **wrong answer on a feasible model**,
//! and one the three pre-existing infeasibility rungs all fail to rescue.
//! That divergence is what the first cut of this file was reading as its
//! trigger, so the tests were green here and red on CI without either
//! platform being wrong about anything the rung is for.
//!
//! Capping at 500 removes the divergence from the test's path: the cap fires
//! long before the point the two platforms part company, the base solve exits
//! `Maximum_Iterations_Exceeded` with `quality_escalations >= 1` on both, and
//! what is left under assertion is the mechanism — the gate opened, the rung
//! promoted, the μ flip stood down — rather than a particular free-run
//! trajectory. The iteration counts are asserted against the cap for the same
//! reason: `< 500` says "this is the recovered solve", `>= 500` says "this one
//! ran out", and neither pins a number that is a property of one machine.
//!
//! The divergence itself is not a test artifact, and rung 4 now names
//! `LocalInfeasibility` in its trigger set because of it: without that, the
//! escalation the rung exists to undo is the direct cause of a wrong answer on
//! linux and the rung never opens to undo it.
//!
//! # Appended, not prepended
//!
//! On a `Restoration_Failed` the gh#815 displacement rung runs first and
//! promotes first. `square_flowsheet_resto`'s exact leg is exactly that case,
//! so it reaches the same answer in the same 185 iterations it did before —
//! `the_exact_leg_is_unchanged_because_the_new_rung_is_last` is the pin.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

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

struct Run {
    out: String,
    /// The reported iteration count, which on a promoted run is the promoted
    /// rung's — the same field the sweep's `it=` column reads.
    iterations: i64,
}

fn solve(model: &str, extra: &[&str]) -> Run {
    // Two pairs of tests below drive the same model with the same options and
    // differ only in what they assert, so the option string is not a unique
    // name. The harness runs them concurrently; without the counter they race
    // on one `.json` and one reads the other's report.
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let tag: String = format!("{model}_{}", extra.join("_"))
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let sol = dir.join(format!("pounce_857g_{pid}_{seq}_{tag}.sol"));
    let json = dir.join(format!("pounce_857g_{pid}_{seq}_{tag}.json"));
    let out = Command::new(pounce_exe())
        .arg(fixture(model))
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .args(extra)
        .output()
        .expect("run pounce");
    let text = std::fs::read_to_string(&json).expect("json report written");
    let key = "\"iteration_count\":";
    let at = text.find(key).expect("iteration_count in report");
    let digits: String = text[at + key.len()..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-')
        .collect();
    Run {
        out: format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        iterations: digits.parse().expect("numeric iteration_count"),
    }
}

/// Every second-opinion narration line, which is what a user actually sees.
///
/// Not simply every `pounce:` line: the CLI prefixes its ordinary chatter the
/// same way (`pounce: wrote …`), so a bare prefix filter is never empty and
/// the "no ladder ran" assertions below would have failed on a correct
/// build — as this file's first draft did.
fn ladder_lines(out: &str) -> Vec<&str> {
    out.lines()
        .filter(|l| l.starts_with("pounce:"))
        .filter(|l| {
            l.contains("re-solving") || l.contains("re-solve") || l.contains("keeping the original")
        })
        .collect()
}

/// The fix. A capped solve that escalated is re-solved without the
/// escalation, and that re-solve converges.
#[test]
fn an_escalating_budget_exit_recovers_by_undoing_the_escalation() {
    let run = solve(
        "square_flowsheet_resto.nl",
        &["hessian_approximation=limited-memory", "max_iter=500"],
    );
    // The base solve must still hit the cap: this rung recovers a failure,
    // it does not prevent one.
    //
    // Read off the FIRST attempt's own statistics block rather than off an
    // `EXIT: Maximum Number of Iterations Exceeded` banner, which is what
    // this used to match. The verdict is now printed once per run rather
    // than once per attempt, so the only `EXIT:` line here is the promoted
    // rung's `Optimal Solution Found` — see
    // `one_verdict_per_run.rs`. The per-attempt statistics block is
    // unchanged, and it is the more direct evidence anyway: it reports the
    // base attempt's own count instead of a status string that a later
    // attempt could also have produced.
    let base_iters: i64 = run
        .out
        .lines()
        .find_map(|l| l.trim_start().strip_prefix("Number of Iterations....:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or_else(|| panic!("no per-attempt iteration count:\n{}", run.out));
    assert!(
        base_iters >= 500,
        "the base solve should still hit the 500-iteration cap; its own \
         summary reports {base_iters}:\n{}",
        run.out
    );
    let lines = ladder_lines(&run.out);
    assert!(
        lines.iter().any(|l| l.contains(
            "iteration limit after a factorization escalation — re-solving \
             along 1 different trajectory"
        )),
        "the ladder should open on the budget exit, with exactly one rung — \
         the displacement rung must not join it:\n{lines:#?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l
                .contains("feral_increase_quality=no re-solve recovered the problem — promoting")),
        "{lines:#?}"
    );
    assert!(
        run.out.contains("EXIT: Optimal Solution Found"),
        "{}",
        run.out
    );
    assert!(
        run.iterations < 500,
        "the promoted rung is the un-escalated solve, which converges well \
         inside the cap the base solve ran out of — the same trajectory a \
         user gets by setting feral_increase_quality=no by hand (178 \
         iterations here at the time of writing):\n{}",
        run.out
    );
}

/// The rung's own enable, and the evidence that the recovery above is this
/// rung and not something else: turn it off and the pre-gh#857 verdict comes
/// straight back.
#[test]
fn the_rung_can_be_turned_off_and_the_old_verdict_returns() {
    let run = solve(
        "square_flowsheet_resto.nl",
        &[
            "hessian_approximation=limited-memory",
            "max_iter=500",
            "feral_increase_quality_retry=no",
        ],
    );
    assert!(
        run.out.contains("Maximum Number of Iterations Exceeded"),
        "{}",
        run.out
    );
    assert!(
        ladder_lines(&run.out).is_empty(),
        "with the rung disabled the ladder has nothing to run, and must not \
         narrate:\n{}",
        run.out
    );
    assert!(
        run.iterations >= 500,
        "and the verdict that stands is the capped one:\n{}",
        run.out
    );
}

/// **The other branch of the gate**, and the reason `for_status` naming a
/// trigger for every `Maximum_Iterations_Exceeded` costs nothing.
///
/// The fixture is `deb7`, capped at 100 iterations. Three things have to be
/// true at once for this to be evidence and they are all measured, not
/// assumed:
///
/// - it reaches the **NLP arm**, so the ladder code actually runs (42 of the
///   79 CLI fixtures never touch it, and a control on the convex arm would
///   pass while proving nothing — the rule from CLAUDE.md);
/// - it exits `Maximum_Iterations_Exceeded`, so `for_status` names the new
///   trigger;
/// - and it has escalated **zero** times by iteration 100, so rung 4's gate
///   is the only thing standing between it and an extra solve.
///
/// `deb7` is the fixture rather than a never-escalating one because on the
/// machine this was measured on it *does* escalate — once by iteration 110
/// and twice by 120 — so the escalation path is reachable on this exact
/// model under these exact options, and the count is provably the operative
/// difference rather than an accident of which model was picked.
///
/// That last paragraph is macOS/aarch64 talking. On linux/x86_64 `deb7`
/// reaches `Optimal` in 143 iterations having escalated **zero** times, so
/// the escalation is not reachable there at all and the fixture is a
/// never-escalating one after all. Only the assertions below are portable —
/// both platforms agree that by iteration 100 nothing has escalated and no
/// rung opens — which is why the count itself is deliberately not asserted
/// anywhere in this file.
/// `the_verdict_and_not_the_count_is_what_excludes_a_winning_solve` in
/// `issue857_quality_escalations_are_reported.rs` measures the other end
/// of the same run, and carries the same warning.
///
/// The matching *one* branch is on `square_flowsheet_resto`, not here, and
/// that is forced rather than chosen: on the machine above, no cap on `deb7`
/// produces a surviving budget exit that escalated, because the μ-strategy
/// stall retry recovers it to `Optimal` at 108 from `max_iter=110` on — past
/// the first escalation and before the cap can hold. Conversely no cap on
/// `square_flowsheet_resto` produces a budget exit that has *not* escalated:
/// even at `max_iter=5` its internal retry escalates twice. One fixture per
/// branch is the most this corpus allows.
///
/// Without this branch the change would add a second full solve to every
/// capped run in the corpus, and
/// `an_escalating_budget_exit_recovers_by_undoing_the_escalation` would pass
/// just the same.
#[test]
fn a_budget_exit_that_never_escalated_is_left_alone() {
    let run = solve("deb7.nl", &["max_iter=100"]);
    assert!(
        run.out.contains("Maximum Number of Iterations Exceeded"),
        "{}",
        run.out
    );
    assert!(
        run.out.contains("Selected solver: NLP filter"),
        "the control must reach the arm the ladder lives on, or it is \
         evidence about nothing:\n{}",
        run.out
    );
    assert!(
        !run.out
            .contains("Number of linear solver quality escalations"),
        "deb7 has not escalated by iteration 100; if this line appeared the \
         cap has drifted past the first escalation and the test is no longer \
         exercising the zero branch:\n{}",
        run.out
    );
    assert!(
        ladder_lines(&run.out).is_empty(),
        "a budget exit with no escalation has no hypothesis for the ladder \
         to test and must not pay for a solve:\n{}",
        run.out
    );
    assert_eq!(run.iterations, 100);
}

/// The rung is dropped when the baseline already ran with the escalation
/// off, where it would re-run the solve that just failed.
#[test]
fn a_baseline_that_already_declined_the_escalation_gets_no_rung() {
    let run = solve(
        "square_flowsheet_resto.nl",
        &[
            "hessian_approximation=limited-memory",
            "feral_increase_quality=no",
        ],
    );
    // This combination happens to converge, so the ladder never opens at all;
    // the assertion that matters is that the answer is the un-escalated one
    // and nothing was re-solved to get it.
    assert!(
        run.out.contains("EXIT: Optimal Solution Found"),
        "{}",
        run.out
    );
    assert_eq!(run.iterations, 178);
    assert!(ladder_lines(&run.out).is_empty(), "{}", run.out);
}

/// Appended, not prepended — the ordering claim, measured.
///
/// The exact leg fails with `Restoration_Failed` after escalating twice, so
/// both rung 3 and rung 4 are open. Rung 3 runs first and promotes, so the
/// leg reaches the same answer at the same cost it did before gh#857: 131 in
/// the base solve plus 54 in the rung. Rung 4 is announced and never run.
#[test]
fn the_exact_leg_is_unchanged_because_the_new_rung_is_last() {
    let run = solve("square_flowsheet_resto.nl", &[]);
    let lines = ladder_lines(&run.out);
    assert!(
        lines.iter().any(|l| l.contains(
            "restoration failure — re-solving along 2 different trajectories \
             before believing it (second-opinion ladder: \
             start_point_perturbation=1e-2, feral_increase_quality=no)"
        )),
        "both rungs should be open, in this order:\n{lines:#?}"
    );
    assert!(
        lines.iter().any(|l| l
            .contains("start_point_perturbation=1e-2 re-solve recovered the problem — promoting")),
        "{lines:#?}"
    );
    assert!(
        !lines
            .iter()
            .any(|l| l.contains("re-solving with feral_increase_quality=no")),
        "rung 3 promoted, so rung 4 must never have run — an extra solve \
         here is the cost this ordering exists to avoid:\n{lines:#?}"
    );
    assert_eq!(
        run.iterations, 54,
        "the promoted rung's own count, unchanged from before gh#857"
    );
}

/// Every `Number of Iterations....:` line — one per solve the run performed,
/// and the only place the cost of a *declined* verdict is visible at all. The
/// JSON report and the sweep both carry the promoted solve's numbers, so a
/// wasted intermediate solve leaves no trace in either.
fn solve_count(out: &str) -> usize {
    out.lines()
        .filter(|l| l.starts_with("Number of Iterations"))
        .count()
}

/// The μ-strategy stall retry stands down for this rung (gh#857 follow-up).
///
/// Before this, an escalating budget exit paid for **two** rescue solves and
/// used one. `run_with_mu_strategy_fallback` fires unconditionally on
/// `Maximum_Iterations_Exceeded`, so this fixture's lbfgs leg ran its 3000
/// iterations, then a second full 3000 with `mu_strategy` flipped — which
/// escalated 25 times all over again and ended no better — and only then
/// reached rung 4, which converges it in 178. 6178 real iterations to produce
/// an answer 3178 of them reach, and the sweep cannot see any of it: `it=`,
/// `q=` and the objective all belong to the promoted solve.
///
/// The flip is a *blind* second opinion — it varies the barrier schedule and
/// hopes. The escalation is a *measured* one: FERAL reroutes which pivots are
/// taken and never steps back down, so flipping μ on top of it holds the knob
/// that is implicated and varies the one that is not. Measured both ways on
/// this leg: `mu_strategy=adaptive` alone still gives 3000 with 25
/// escalations, and `feral_increase_quality=no` gives 178 under *either* μ
/// strategy.
///
/// Declined rather than folded into the flip because the FERAL backend factory
/// is minted from an options snapshot the caller takes *before* `solve()`:
/// writing `feral_increase_quality` from inside the fallback is too late to
/// reach the retry's linear solver, while `mu_strategy` is read per-solve and
/// is not. That layer can only choose whether to spend the solve, not what to
/// spend it on.
#[test]
fn the_mu_flip_stands_down_when_the_escalation_rung_is_open() {
    let run = solve(
        "square_flowsheet_resto.nl",
        &["hessian_approximation=limited-memory", "max_iter=500"],
    );
    assert_eq!(
        solve_count(&run.out),
        2,
        "the capped base solve and rung 4, and nothing between them — a third \
         block here is the μ flip back, burning a full budget on a trajectory \
         the escalation still governs:\n{}",
        run.out
    );
    assert!(
        run.iterations < 500,
        "and the promoted answer is the rung's, not a capped one:\n{}",
        run.out
    );
}

/// The switch is one switch. `feral_increase_quality_retry=no` removes rung 4
/// **and** the stand-down above, so a user who turns it off gets exactly the
/// pre-gh#857 behaviour on both sides: the μ flip runs, and the escalating
/// verdict stands.
///
/// This is the branch that keeps the stand-down from being a silent removal of
/// the flip. It is also the arm that measures what the flip was worth here:
/// two solves, 6000 iterations, and the same `Maximum_Iterations_Exceeded`.
#[test]
fn turning_the_rung_off_gives_the_mu_flip_back() {
    let run = solve(
        "square_flowsheet_resto.nl",
        &[
            "hessian_approximation=limited-memory",
            "max_iter=500",
            "feral_increase_quality_retry=no",
        ],
    );
    assert_eq!(
        solve_count(&run.out),
        2,
        "with the rung off the base solve is followed by the μ flip, which is \
         what this option's `no` has always meant:\n{}",
        run.out
    );
    assert!(
        ladder_lines(&run.out).is_empty(),
        "and the second solve is the flip, not a ladder rung:\n{}",
        run.out
    );
    assert!(
        run.iterations >= 500,
        "both solves ran the budget out, and the second one's verdict is what \
         is reported:\n{}",
        run.out
    );
}

/// The stand-down is scoped to the status rung 4 opens on.
///
/// `run_with_mu_strategy_fallback` also retries `Solved_To_Acceptable_Level`,
/// and `SecondOpinionTrigger::for_status` maps that status to **no trigger at
/// all** — so declining the flip there would drop a retry with nothing in its
/// place. The exact leg is the control for the other half: it exits
/// `Restoration_Failed`, which is not retry-worthy for the flip in the first
/// place, so the ladder is reached exactly as before and rung 3 still
/// promotes at 54.
#[test]
fn the_stand_down_does_not_reach_the_exact_leg() {
    let run = solve("square_flowsheet_resto.nl", &[]);
    assert!(
        run.out.contains("EXIT: Optimal Solution Found"),
        "{}",
        run.out
    );
    assert_eq!(
        run.iterations, 54,
        "rung 3 still rescues the exact leg, unchanged:\n{}",
        run.out
    );
}

/// The `LocalInfeasibility` trigger, and both branches of the gate on it.
///
/// A false infeasibility verdict is the worst thing the escalation can cause —
/// it is a wrong answer on a feasible model, reported as a verdict rather than
/// a failure — and it is the shape `square_flowsheet_resto`'s lbfgs leg takes
/// on linux/x86_64 (see the module header). So rung 4 opens on it, under the
/// same `quality_escalations >= 1` gate as the other two triggers.
///
/// The cost is real and is not hypothetical: on a model that is *genuinely*
/// infeasible the rung cannot recover anything, and re-solving to find that
/// out costs roughly what the base solve cost. Across the fixture corpus that
/// is six moved lines, `infeasible_square_scaled_1em4` 61 → 78 total
/// iterations on the exact leg and `issue_508_infeasible_gap_1em4` 982 →
/// 1423 — no status, objective or engine moves anywhere. That is the price of
/// not believing an infeasibility verdict the escalation may have manufactured,
/// and it is paid on purpose.
///
/// The gate is what keeps it from being paid everywhere. Of the eight NLP-arm
/// infeasibility fixture-legs in the corpus, four escalated and take the rung;
/// four never escalated and are untouched at three rungs. The second test
/// below is that branch, and it is the test — not a duplicate of the first.
#[test]
fn an_escalating_infeasibility_verdict_opens_the_quality_rung() {
    let run = solve("infeasible_square_scaled_1em4.nl", &[]);
    let lines = ladder_lines(&run.out);
    assert!(
        lines
            .iter()
            .any(|l| l.contains("local infeasibility — re-solving along 4 different trajectories")),
        "the escalating infeasibility verdict should open all four rungs:\n{lines:#?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("feral_increase_quality=no re-solve did not recover")),
        "{lines:#?}"
    );
    assert!(
        run.out.contains("it survived 4 independent re-solve(s)"),
        "this model really is infeasible, so the rung confirms the verdict \
         rather than overturning it — the verdict itself must not move:\n{}",
        run.out
    );
}

/// The other branch: an infeasibility verdict reached without a single
/// factorization escalation has nothing for this rung to undo, and pays
/// nothing for it.
#[test]
fn an_infeasibility_verdict_that_never_escalated_keeps_three_rungs() {
    let run = solve("issue_372_infeasible_bounds.nl", &[]);
    let lines = ladder_lines(&run.out);
    assert!(
        lines
            .iter()
            .any(|l| l.contains("local infeasibility — re-solving along 3 different trajectories")),
        "no escalation, so the ladder is the pre-gh#857 three:\n{lines:#?}"
    );
    assert!(
        !lines
            .iter()
            .any(|l| l.contains("feral_increase_quality=no")),
        "{lines:#?}"
    );
}
