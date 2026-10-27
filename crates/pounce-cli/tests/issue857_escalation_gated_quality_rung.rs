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
//! measured both halves when this rung was written, on FERAL 0.20:
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
//! # What FERAL 0.21 moved, and where the rung is measured now
//!
//! The 0.21 bump took both of those witnesses away. The exact leg now reaches
//! `Optimal` in 95 with the escalation on (59 with it off), so it opens no
//! ladder at all; the lbfgs leg still runs to the cap with the escalation on,
//! but now runs to it with the escalation **off** too, so rung 4 re-solves it
//! and fails — the square-system stage is what recovers it, in 8. Across the
//! whole fixture sweep on 0.21 no default-options run is promoted by rung 4.
//!
//! The rung is not wrong for that; the trade it answers moved from the verdict
//! to the iteration count. And that is what still reaches it: the exact leg's
//! escalated trajectory takes 95 iterations and its un-escalated one 59, so a
//! budget between the two — [`EXACT_CAP`], `max_iter=75` — is exhausted by the
//! escalating solve (three escalations by then) and not by the re-solve, which
//! is the gh#857 situation exactly: a capped run the escalation walked into the
//! wall, recovered by undoing it. The cap sits sixteen iterations clear of
//! each side. The tests that pin the rung, the stand-down and the switch all
//! run there.
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
//! It is a gate at `>= 1` and deliberately not a threshold: on FERAL 0.20
//! `deb7` escalated exactly as many times as this fixture's base solve and
//! *gained* by it, and on 0.21 this fixture's own exact leg escalates four
//! times and reaches `Optimal`. The count cannot separate the two; the
//! verdict can.
//!
//! # Why the lbfgs arms named an explicit `max_iter`
//!
//! (FERAL 0.20 history; the lbfgs arms moved to the exact leg at
//! [`EXACT_CAP`] above, and the one lbfgs arm left caps at 500 for the same
//! reason.) Left to run free, this fixture's lbfgs leg did not reach the same verdict
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
//! An earlier rung that recovers the model promotes before rung 4 runs. On
//! FERAL 0.20 the witness was `square_flowsheet_resto`'s exact leg, where the
//! gh#815 displacement rung promoted at the same 185 total as before gh#857.
//! On 0.21 that leg opens no ladder, and the witness is `circle_parabola_t0p5`
//! on the lbfgs leg: an escalating local-infeasibility verdict, all four rungs
//! open, and rung 1 promotes —
//! `the_new_rung_is_last_so_an_earlier_promotion_is_unchanged` is the pin.

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

/// `square_flowsheet_resto`'s exact leg capped between its escalated (95)
/// and un-escalated (59) trajectories; see the module comment.
const EXACT_CAP: &str = "max_iter=75";

struct Run {
    out: String,
    /// The reported iteration count, which on a promoted run is the promoted
    /// rung's — the same field the sweep's `it=` column reads.
    iterations: i64,
    /// The JSON report, for the `second_opinion` block: which rungs ran and
    /// which one promoted, read from the report rather than the narration.
    report: serde_json::Value,
}

impl Run {
    /// `second_opinion.tried`, empty when no ladder ran.
    fn tried(&self) -> Vec<String> {
        self.report["second_opinion"]["tried"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn promoted_by(&self) -> Option<&str> {
        self.report["second_opinion"]["promoted_by"].as_str()
    }
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
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&sol);
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
        report: serde_json::from_str(&text).expect("report parses"),
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
///
/// On FERAL 0.20 this ran on the lbfgs leg at `max_iter=500` (500 capped,
/// then rung 4 at 178). On 0.21 rung 4 cannot recover that leg at any budget,
/// so it runs on the exact leg at [`EXACT_CAP`]: 75 capped with three
/// escalations, then rung 4 at 59.
#[test]
fn an_escalating_budget_exit_recovers_by_undoing_the_escalation() {
    let run = solve("square_flowsheet_resto.nl", &[EXACT_CAP]);
    // The base solve must still hit the cap: this rung recovers a failure,
    // it does not prevent one.
    //
    // Read off the FIRST attempt's own statistics block rather than off an
    // `EXIT: Maximum Number of Iterations Exceeded` banner. The verdict is
    // printed once per run rather than once per attempt, so the only `EXIT:`
    // line here is the promoted rung's `Optimal Solution Found` — see
    // `one_verdict_per_run.rs`. The per-attempt statistics block reports the
    // base attempt's own count instead of a status string that a later
    // attempt could also have produced.
    let base_iters: i64 = run
        .out
        .lines()
        .find_map(|l| l.trim_start().strip_prefix("Number of Iterations....:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or_else(|| panic!("no per-attempt iteration count:\n{}", run.out));
    assert!(
        base_iters >= 75,
        "the base solve should still hit the 75-iteration cap; its own \
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
    assert_eq!(
        (run.tried(), run.promoted_by()),
        (
            vec!["feral_increase_quality=no".to_string()],
            Some("feral_increase_quality=no")
        ),
        "the report must name rung 4 as the one rung tried and the one that \
         promoted"
    );
    assert!(
        run.out.contains("EXIT: Optimal Solution Found"),
        "{}",
        run.out
    );
    assert!(
        run.iterations < 75,
        "the promoted rung is the un-escalated solve, which converges inside \
         the cap the base solve ran out of — the same trajectory a user gets \
         by setting feral_increase_quality=no by hand (59 iterations when \
         FERAL 0.21 was taken):\n{}",
        run.out
    );
}

/// The rung's own enable, and the evidence that the recovery above is this
/// rung and not something else: turn it off and the pre-gh#857 verdict comes
/// straight back. Same fixture, leg and cap as the recovery test, so the only
/// difference between the two runs is the switch. (The square-system stage
/// is switched off too: it opens on the same budget exit of this square model
/// and would otherwise answer instead.)
#[test]
fn the_rung_can_be_turned_off_and_the_old_verdict_returns() {
    let run = solve(
        "square_flowsheet_resto.nl",
        &[
            EXACT_CAP,
            "feral_increase_quality_retry=no",
            "square_system_recovery=no",
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
        run.iterations >= 75,
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
    // gh#983 item 5: the returned point is the first attempt's (the gh#884
    // retry is declined on the answer), so the report's iteration count is the
    // first attempt's own -- the *first* "Number of Iterations" line, which
    // counts the restoration iterations it spent (84 outer + 32 restoration
    // = 116 here). This used to read 100, the declined retry's count, which
    // was the mixed-run report the issue describes: objective and point from
    // one run, iteration count from the other.
    let first_pass: i64 = run
        .out
        .lines()
        .find_map(|l| l.strip_prefix("Number of Iterations....:"))
        .and_then(|v| v.trim().parse().ok())
        .expect("console iteration count");
    assert!(first_pass >= 100, "the cap was reached: {first_pass}");
    assert_eq!(
        run.iterations, first_pass,
        "the report must describe the run that produced the returned point"
    );
}

/// The rung is dropped when the baseline already ran with the escalation
/// off, where it would re-run the solve that just failed.
///
/// On FERAL 0.20 this combination converged (`Optimal`/178), so the test
/// could only show that nothing was re-solved on a run that never failed. On
/// 0.21 the lbfgs leg runs to the cap with the escalation off as well, which
/// is the case the drop is for: a budget exit, from a baseline that already
/// declined the escalation, gets no rung 4. The cap is 500 for the reason the
/// module header gives, and the square-system stage is switched off because
/// it would otherwise answer this budget exit of a square model.
///
/// Read what this can and cannot catch. With the escalation off the solve
/// cannot escalate, so `quality_escalations` is 0 and the `>= 1` gate drops
/// the rung on its own; removing `already_no_increase_quality` alone leaves
/// this green. That branch is pinned directly, with a nonzero count, by
/// `the_quality_rung_is_droppable_and_never_a_no_op` in
/// `pounce-algorithm/src/second_opinion.rs`. What this pins end to end is the
/// observable contract: the failing verdict stands, and no rung is narrated or
/// recorded.
#[test]
fn a_baseline_that_already_declined_the_escalation_gets_no_rung() {
    let run = solve(
        "square_flowsheet_resto.nl",
        &[
            "hessian_approximation=limited-memory",
            "feral_increase_quality=no",
            "max_iter=500",
            "square_system_recovery=no",
        ],
    );
    assert!(
        run.out
            .contains("EXIT: Maximum Number of Iterations Exceeded"),
        "premise: with the escalation off this leg still runs to the cap on \
         FERAL 0.21; if it now converges the test has lost its failing \
         baseline:\n{}",
        run.out
    );
    assert!(ladder_lines(&run.out).is_empty(), "{}", run.out);
    assert!(
        run.tried().is_empty(),
        "no rung may be recorded either: {:?}",
        run.tried()
    );
    assert!(run.iterations >= 500, "{}", run.out);
}

/// Appended, not prepended — the ordering claim, measured.
///
/// `circle_parabola_t0p5` on the lbfgs leg exits `Infeasible_Problem_Detected`
/// at 48 having escalated, so every rung that is a different trajectory is
/// open, rung 4 last. That is three, not four: under L-BFGS an unset
/// `mu_strategy` already resolves to adaptive (gh#746), so the
/// `mu_strategy=adaptive` rung would re-run the base solve byte-for-byte and
/// is not offered (gh#1042 review). Rung 1
/// (`feral_scaling=mc64`) runs first and promotes at 77, so the leg reaches
/// the same answer at the same cost it would without rung 4: 48 in the base
/// solve plus 77 in the rung, 125 total — identical on FERAL 0.20 and 0.21.
/// Rung 4 is announced and never run.
///
/// On 0.20 this was `square_flowsheet_resto`'s exact leg (`Restoration_Failed`,
/// rung 3 promoting at 54); on 0.21 that leg opens no ladder at all, and no
/// restoration failure in the corpus is recovered by any rung. The claim is
/// the same one either way: rung 4 cannot cost an earlier rung's promotion
/// anything.
#[test]
fn the_new_rung_is_last_so_an_earlier_promotion_is_unchanged() {
    let run = solve(
        "circle_parabola_t0p5.nl",
        &["hessian_approximation=limited-memory"],
    );
    let lines = ladder_lines(&run.out);
    assert!(
        lines.iter().any(|l| l.contains(
            "local infeasibility — re-solving along 3 different trajectories \
             before believing it (second-opinion ladder: feral_scaling=mc64, \
             start_point_perturbation=1e-2, feral_increase_quality=no)"
        )),
        "every distinct rung should be open, rung 4 last:\n{lines:#?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("feral_scaling=mc64 re-solve recovered the problem — promoting")),
        "{lines:#?}"
    );
    assert!(
        !lines
            .iter()
            .any(|l| l.contains("re-solving with feral_increase_quality=no")),
        "rung 1 promoted, so rung 4 must never have run — an extra solve \
         here is the cost this ordering exists to avoid:\n{lines:#?}"
    );
    assert_eq!(
        (run.tried(), run.promoted_by()),
        (
            vec!["feral_scaling=mc64".to_string()],
            Some("feral_scaling=mc64")
        ),
        "the report must record one rung tried, the first"
    );
    assert_eq!(
        run.iterations, 77,
        "the promoted rung's own count, unchanged by gh#857 and by FERAL 0.21"
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
/// Measured where the rung is measured now, the exact leg at [`EXACT_CAP`]:
/// two summary blocks, 75 capped and 59 in rung 4. The figures in the next
/// two paragraphs are the lbfgs leg's on FERAL 0.20, where this was found.
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
    let run = solve("square_flowsheet_resto.nl", &[EXACT_CAP]);
    assert_eq!(
        solve_count(&run.out),
        2,
        "the capped base solve and rung 4, and nothing between them — a third \
         block here is the μ flip back, burning a full budget on a trajectory \
         the escalation still governs:\n{}",
        run.out
    );
    assert_eq!(
        run.tried(),
        vec!["feral_increase_quality=no".to_string()],
        "the second block is rung 4's"
    );
    assert!(
        run.iterations < 75,
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
/// two solves, 150 iterations at [`EXACT_CAP`], and the same
/// `Maximum_Iterations_Exceeded` (on FERAL 0.20's lbfgs leg, 6000 and the
/// same verdict).
#[test]
fn turning_the_rung_off_gives_the_mu_flip_back() {
    let run = solve(
        "square_flowsheet_resto.nl",
        &[
            EXACT_CAP,
            "feral_increase_quality_retry=no",
            "square_system_recovery=no",
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
        run.iterations >= 75,
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
/// place. This is the control for the other half: a `Restoration_Failed` exit
/// from a solve that escalated (four times), which is not retry-worthy for the
/// flip in the first place, so the ladder must be reached exactly as it would
/// be without the stand-down — rung 3, then rung 4, one solve each.
///
/// On FERAL 0.20 the control was `square_flowsheet_resto`'s exact leg at the
/// default `tol`, which failed in restoration and was rescued by rung 3 at 54.
/// On 0.21 that leg solves. The control then moved below the model's
/// round-off floor (`tol=1e-11`), until gh#1084 showed that exit was a square
/// model at a feasible point, now reported as `Feasible_Point_Found`. It is
/// the lbfgs leg now, where the base solve fails in restoration for real at
/// 250 (see `issue_815_restoration_ladder.rs`). `square_system_recovery=no`
/// keeps the last stage, which recovers this leg, out of a test about the two
/// rungs. No `max_iter` cap: on linux/x86_64 this leg's restoration failure
/// comes after iteration 500 (it hit a 500 cap first in CI), while on
/// macOS/aarch64 it comes at 250. Rung 4's own re-solve ends `Maximum_Iterations_Exceeded`
/// and takes *its* μ flip, a fourth summary block. That flip belongs to the
/// rung's solve and is not the one this test is about, so the assertion is on
/// the report's `mu_strategy_fallback`, which describes the base solve because
/// its verdict is the one kept.
///
/// What this is not evidence about: the acceptable-level half. No fixture-leg
/// in the corpus reaches the μ flip with `Solved_To_Acceptable_Level` *and* an
/// escalation on 0.21 — `deb7`'s exact leg reports that status with two
/// escalations, but as the final-verdict downgrade gh#983 applies after every
/// retry has decided, so the flip only ever sees its `Solve_Succeeded`. A
/// stand-down widened to every retry-worthy status would leave this test
/// green; the scope is held by the `matches!` in
/// `run_with_mu_strategy_fallback` and by this comment, not by a fixture.
#[test]
fn the_stand_down_does_not_reach_a_restoration_failure() {
    let run = solve(
        "square_flowsheet_resto.nl",
        &[
            "hessian_approximation=limited-memory",
            "square_system_recovery=no",
        ],
    );
    let lines = ladder_lines(&run.out);
    let trigger = "restoration failure — re-solving along 2 different trajectories \
                   before believing it (second-opinion ladder: \
                   start_point_perturbation=1e-2, feral_increase_quality=no)";
    assert!(
        lines.iter().any(|l| l.contains(trigger)),
        "both rungs should be open, in this order:\n{lines:#?}"
    );
    assert!(
        run.report["statistics"]["mu_strategy_fallback"].is_null(),
        "the reported solve is the base one (its verdict ships), and it took \
         no μ flip, which a restoration failure never buys:\n{}",
        run.out
    );
    assert_eq!(
        &run.tried()[..2],
        &[
            "start_point_perturbation=1e-2".to_string(),
            "feral_increase_quality=no".to_string()
        ],
        "the report must record both rungs, in ladder order"
    );
    assert!(
        run.out.contains("Status: Restoration_Failed"),
        "with the last stage off, nothing on the ladder recovers it:\n{}",
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
