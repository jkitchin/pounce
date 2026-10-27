//! A solve that was rerouted by a linear-solver quality escalation now says so
//! (gh#857).
//!
//! # Why a counter
//!
//! `increase_quality` is how the IPM answers a factorization that will not
//! deliver, and with the FERAL backend it does not merely retry: the ladder
//! changes *which pivots are taken* and never steps back down, so one
//! escalation governs every later factorization in the solve — a restoration
//! sub-solve's included. It is therefore a trajectory change, and it is one
//! that produces the *right* answer often enough that nothing else flags it.
//!
//! Before this counter no report carried a trace of it. Two runs could agree
//! on status, objective, iteration count and engine and still have factorized
//! the KKT systems differently, and gh#857's regression had to be found by
//! instrumenting a build — which is the same reporting gap gh#850 closed for
//! second-opinion verdicts and gh#760 for engine routing.
//!
//! # What the numbers are
//!
//! On `square_flowsheet_resto`, measured when the counter landed (FERAL 0.20):
//!
//! ```text
//!   exact  default   base solve   Restoration_Failed/131   q=2
//!                    rescue rung  SolveSucceeded/54        q=4
//!   exact  rung off  base solve   SolveSucceeded/99        q=0
//!   lbfgs  default   main solve   MaxIterations/3000       q=25
//!   lbfgs  rung off  base solve   SolveSucceeded/178       q=0
//! ```
//!
//! The exact leg's `q=2` is the figure the `feral_increase_quality` option
//! text has carried all along ("fires exactly twice"), derived originally with
//! a process-global firing cap. This counter reproduces it independently,
//! which is the cross-check that says the counter is measuring what it claims
//! to. The two are *base-solve* counts: process-wide the exact leg escalates
//! six times, because the rescue rung is a whole second solve. `q=` is
//! per-solve, like every other field beside it.
//!
//! On FERAL 0.21 the exact leg no longer fails: one solve, `Optimal`/95,
//! `q=4`, no rung. The lbfgs leg still reports heavy escalation at the cap.
//! So the per-solve-versus-reported test moved to `circle_parabola_t0p5`'s
//! lbfgs leg, which is a promotion in which **both** solves escalate — base
//! `q=8`, the promoted `feral_scaling=mc64` rung `q=4` — and the restoration
//! test stayed on this fixture, where the one remaining solve still shows the
//! gap between what prints and what happened (two printed `q`s, four
//! counted).
//!
//! # The restoration half is the part that needed the plumbing
//!
//! `PdFullSpaceSolver` already appended a `q` to the info-string column on
//! every successful escalation, so a reader might reasonably ask why counting
//! the `q`s was not enough. Because the restoration sub-solve runs its own
//! `PdFullSpaceSolver` against its own `IpoptData`, and its info strings never
//! reach the printed table: on FERAL 0.20 the base solve above printed **one**
//! `q`, at iteration 26, while two escalations happened — the second at
//! `76r`, inside restoration, invisible. On 0.21 the solve prints two (on rows
//! 2 and 3) and counts four.
//!
//! That is why the counter is a shared cell handed to every builder the
//! application mints rather than a field on one solver.
//! `the_restoration_escalation_is_counted_though_it_never_prints` is the test
//! that fails if someone later "simplifies" it back to the main loop's own.
//!
//! # What this file does not claim
//!
//! Nothing here asserts that escalating is bad. On FERAL 0.20 `deb7` escalated
//! exactly as many times as `square_flowsheet_resto`'s base solve and *gained*
//! 15–25% of its iterations by it; that pair was the evidence that a count
//! alone cannot separate the two, which is why `feral_increase_quality` is an
//! option and not a changed default. On 0.21 the same fixture is the
//! evidence on its own: it escalates four times and reaches `Optimal`, which
//! is the case pinned below.

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

/// `(combined output, reported escalation count from the JSON report)`.
///
/// The JSON is the surface a consumer reads; the console output is what the
/// per-solve assertions below need, because on a promoted fixture the report
/// carries the *promoted rung's* statistics — as `it=` does — and the base
/// solve's count is only visible in its own summary block.
fn solve(model: &str, extra: &[&str]) -> (String, i64) {
    let tag: String = format!("{model}_{}", extra.join("_"))
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let dir = std::env::temp_dir();
    let sol = dir.join(format!("pounce_857q_{tag}.sol"));
    let json = dir.join(format!("pounce_857q_{tag}.json"));
    let out = Command::new(pounce_exe())
        .arg(fixture(model))
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .args(extra)
        .output()
        .expect("run pounce");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&json).expect("json report written");
    // Deliberately a string scan rather than a serde dependency: this test is
    // about the field reaching the file, and a typed read would happily
    // deserialize a report that had lost it (the field is `serde(default)`,
    // so a missing one reads as 0 — exactly the failure this must catch).
    let key = "\"quality_escalations\":";
    let at = text
        .find(key)
        .unwrap_or_else(|| panic!("no {key} in the JSON report:\n{text}"));
    let rest = &text[at + key.len()..];
    let digits: String = rest
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let reported = digits
        .parse()
        .unwrap_or_else(|_| panic!("unparseable quality_escalations near: {}", &rest[..40]));
    (combined, reported)
}

/// Every `Number of linear solver quality escalations = N` line, in order —
/// one per solve the run performed.
fn per_solve_counts(out: &str) -> Vec<i64> {
    out.lines()
        .filter_map(|l| l.split_once("Number of linear solver quality escalations"))
        .filter_map(|(_, v)| v.split('=').nth(1))
        .filter_map(|v| v.trim().parse().ok())
        .collect()
}

/// A promotion in which both solves escalate: the console reports each solve's
/// own count, and the JSON report carries the promoted solve's.
///
/// `circle_parabola_t0p5` on the lbfgs leg: the base solve escalates 8 times
/// and exits `Infeasible_Problem_Detected` at 48; the ladder's first rung,
/// `feral_scaling=mc64`, escalates 4 times and promotes. On FERAL 0.20 the
/// test ran on `square_flowsheet_resto`'s exact leg (`[2, 4]`, rung 3
/// promoting), which on 0.21 is a single solve with nothing to tell apart.
/// The ladder record is asserted first, so a run that stops promoting fails
/// here rather than as a puzzling count.
#[test]
fn a_promoted_run_reports_each_solves_escalations_and_ships_the_last() {
    let (out, reported) = solve(
        "circle_parabola_t0p5.nl",
        &["hessian_approximation=limited-memory"],
    );
    assert!(
        out.contains("feral_scaling=mc64 re-solve recovered the problem — promoting"),
        "premise: the base solve fails and rung 1 promotes:\n{out}"
    );
    let counts = per_solve_counts(&out);
    assert_eq!(
        counts,
        vec![8, 4],
        "expected the base solve to report 8 escalations and the promoted \
         rung 4 — one line per solve, each its own:\n{out}"
    );
    assert_eq!(
        reported, 4,
        "the JSON report carries the promoted rung's statistics, as `it=` \
         does, so it should report that solve's 4 — not the base solve's 8, \
         and not their sum"
    );
}

/// The recovery. `feral_increase_quality=no` is the documented way out of this
/// fixture, and with it there is nothing to report — including no console
/// line, which keeps a non-escalating summary byte-identical to what it was
/// before this field existed.
#[test]
fn with_the_rung_off_there_is_nothing_to_report() {
    let (out, reported) = solve("square_flowsheet_resto.nl", &["feral_increase_quality=no"]);
    assert_eq!(
        reported, 0,
        "the rung is off, so nothing can escalate:\n{out}"
    );
    assert!(
        per_solve_counts(&out).is_empty(),
        "the summary line is printed only when the count is nonzero, so a \
         solve that never escalated must not carry one:\n{out}"
    );
}

/// The lbfgs leg, which is the worse half of gh#857. Pinned as a floor rather
/// than an exact number: it runs to the 3000-iteration cap, so the count rides
/// on 3000 iterations of trajectory and an exact pin would fail on changes that
/// have nothing to do with this field. The floor still separates "escalating
/// heavily" from "not escalating", which is the whole claim.
///
/// `feral_increase_quality_retry=no` is load-bearing here, and for a reason
/// worth stating rather than working around silently: the statistic this file
/// is about belongs to **the solve that produced it**, and the JSON reports the
/// *last* solve's. gh#857's own recovery rung catches this exact capped run,
/// re-solves it with the escalation off, and promotes — so with the rung on,
/// the reported count is the promoted solve's `0`, which is correct and is not
/// what this test is measuring. The base solve is the subject; the rung has its
/// own file (`issue857_escalation_gated_quality_rung.rs`).
#[test]
fn the_lbfgs_leg_reports_heavy_escalation_and_the_option_stops_it() {
    let (out, reported) = solve(
        "square_flowsheet_resto.nl",
        &[
            "hessian_approximation=limited-memory",
            "feral_increase_quality_retry=no",
            "square_system_recovery=no",
        ],
    );
    assert!(
        reported >= 10,
        "expected heavy escalation on the capped lbfgs leg (25 when this was \
         written), got {reported}:\n{out}"
    );

    let (out, reported) = solve(
        "square_flowsheet_resto.nl",
        &[
            "hessian_approximation=limited-memory",
            "feral_increase_quality=no",
        ],
    );
    assert_eq!(
        reported, 0,
        "with the rung off the lbfgs leg solves without escalating:\n{out}"
    );
}

/// The reason the counter is shared rather than per-solver.
///
/// The solve escalates three times (four before gh#1076), but only **two** `q`s reach the printed
/// iteration table, on rows 2 and 3: the other two happen inside the
/// restoration sub-solve (82 `r` rows, one call), which owns a separate
/// `PdFullSpaceSolver` and a separate `IpoptData`, and its info strings are
/// never printed. Counting the printed `q`s — the obvious simplification —
/// would silently report half. (On FERAL 0.20 this was the base solve of a
/// promoted run, one printed against two counted.)
#[test]
fn the_restoration_escalation_is_counted_though_it_never_prints() {
    let (out, _) = solve("square_flowsheet_resto.nl", &["print_info_string=yes"]);
    // The first solve's block: everything up to its own summary line.
    let base = out
        .split_once("Number of linear solver quality escalations")
        .expect("a summary line")
        .0;
    assert!(
        base.lines()
            .any(|l| l.trim_start().starts_with(|c: char| c.is_ascii_digit())
                && l.split_whitespace()
                    .next()
                    .is_some_and(|t| t.ends_with('r'))),
        "premise: the solve must spend iterations in restoration, or there is \
         nowhere for an unprinted escalation to happen:\n{base}"
    );
    let printed = base
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with(|c: char| c.is_ascii_digit()) && t.split_whitespace().count() > 9
        })
        .filter_map(|l| l.split_whitespace().next_back())
        .filter(|tail| !tail.chars().all(|c| c.is_ascii_digit()))
        .map(|tail| tail.matches('q').count())
        .sum::<usize>();
    assert_eq!(
        printed, 2,
        "expected exactly two `q`s in the printed table — the main loop's, on \
         rows 2 and 3:\n{base}"
    );
    let counts = per_solve_counts(&out);
    // gh#1076 moved the total from 4 to 3 (one restoration escalation fewer
    // since a zero pivot under a reused MC64 scaling is re-checked under a
    // fresh one); what this test pins is that the count exceeds the two the
    // table prints.
    let counted = counts.first().copied().unwrap_or(0);
    assert!(
        counted > printed as i64,
        "the solve escalated {counted} times against {printed} printed; if \
         these are equal the counter has been narrowed to the main loop and \
         the restoration escalations are being dropped again"
    );
}

/// The control: a fixture that never escalates reports nothing and prints
/// nothing, so this field cannot be read as "POUNCE now escalates everywhere".
#[test]
fn a_solve_that_never_escalates_says_nothing() {
    let (out, reported) = solve("airport.nl", &[]);
    assert_eq!(reported, 0, "airport does not escalate:\n{out}");
    assert!(
        per_solve_counts(&out).is_empty(),
        "and so prints no line:\n{out}"
    );
}

/// The other side of the trade, and the reason the gh#857 gate consults the
/// *verdict* rather than the count.
///
/// On FERAL 0.21 `square_flowsheet_resto`'s exact leg escalates four times
/// and reaches `Optimal` in 95 — the fixture whose escalations used to *lose*
/// its verdict now escalates more and wins it. So the count says nothing
/// about the outcome, which is the measured reason `feral_increase_quality` is
/// an option rather than a flipped default, and the claim the gate rests on is
/// asserted below: an `Optimal` verdict opens no rung whatever its count. A
/// count-only gate could not say that.
///
/// The count is asserted only as `>= 1`, because without an escalation the
/// test is about nothing. On FERAL 0.20 this was `deb7`, at `Optimal`/147
/// with two escalations on macOS/aarch64 and zero on linux/x86_64 — which is
/// why the count was not asserted there at all. On 0.21 `deb7`'s exact leg
/// ends `Solved_To_Acceptable_Level` (the gh#983 downgrade of a run that
/// showed the gh#884 signature), so it is no longer the winning side; and this
/// fixture's count was pinned exactly across both CI platforms by this file's
/// tests on 0.20, so `>= 1` is the portable floor.
#[test]
fn the_verdict_and_not_the_count_is_what_excludes_a_winning_solve() {
    let (out, reported) = solve("square_flowsheet_resto.nl", &[]);
    assert!(
        out.contains("EXIT: Optimal Solution Found"),
        "the exact leg is the winning side of the trade on FERAL 0.21 and \
         should solve:\n{out}"
    );
    assert!(
        reported >= 1,
        "premise: the winning solve must have escalated, or a gate that ignored \
         the count would pass here for the wrong reason:\n{out}"
    );
    assert!(
        !out.lines()
            .any(|l| l.starts_with("pounce:")
                && (l.contains("re-solving") || l.contains("re-solve"))),
        "and an `Optimal` verdict opens no second-opinion rung, however many \
         times the solve escalated ({reported} here) — which is precisely what \
         a gate on the count alone could not deliver:\n{out}"
    );
}
