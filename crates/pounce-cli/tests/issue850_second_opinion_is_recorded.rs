//! A promoted second-opinion re-solve must leave a trace (gh #850).
//!
//! # What was invisible
//!
//! When the base solve fails and a ladder rung recovers it, the report's
//! `status` and `statistics.iteration_count` both become the **promoted rung's**
//! and nothing else says the base solver failed. So a fixture that *lost* its
//! baseline solve and is now only rescued by a retry reads in
//! `scripts/sweep-fixtures.sh` as a large improvement:
//!
//! ```text
//!                                          status             iters
//!   v0.10.0, defaults                      SolveSucceeded      116
//!   HEAD, defaults                         SolveSucceeded       54
//!   HEAD, infeasibility_perturbed_start_retry=no   RestorationFailed   131
//! ```
//!
//! `v0.10.0` does not have `infeasibility_perturbed_start_retry` — it rejects
//! the option outright — so that 116 is the *base solver* converging, and
//! HEAD's base solver no longer does. The only thing between the user and a
//! `RestorationFailed` is a rung added in the same release. The sweep read it
//! as `116 -> 54`, **a 2× improvement**.
//!
//! That is worse than a gap in the evidence: `scripts/sweep-fixtures.sh` is the
//! repo's primary trajectory guard and CLAUDE.md makes it the required evidence
//! for a trajectory change, so a guard that converts a lost solve into a
//! recorded win produces positive evidence for the wrong conclusion.
//!
//! Three things had to line up, and they did: the narration is on stdout at the
//! default `print_level`, but `sweep_leg` captures that stdout only to scrape
//! the engine out of it and then deletes it; and the JSON report had no field
//! for any of it. This file pins the JSON half — the sweep column is built from
//! it.
//!
//! # The cost was understated too
//!
//! `statistics.iteration_count` is the promoted rung's **alone**. The 131
//! iterations that failed first are not in it, so that fixture's true cost was
//! `131 + 54 = 185`, 3.4× what the report said.
//!
//! # What recording it made possible
//!
//! With the loss legible it was traced to the `increase_quality` rung, and
//! `feral_increase_quality=no` recovers both of this fixture's legs — see
//! `issue850_increase_quality_regression.rs`, which also records why that is
//! an option rather than a changed default. The column also exposed a second
//! instance the issue never mentions, `degenerate_start_hs008`, whose base
//! solve returns `Infeasible_Problem_Detected` at 9 iterations and which is
//! reported `SolveSucceeded` at `it=5` against a true cost of 30.
//!
//! # Which fixture the tests use
//!
//! `degenerate_start_hs008`, as of the FERAL 0.21 bump. The tests were written
//! against `square_flowsheet_resto`, and on 0.21 that fixture's exact leg no
//! longer loses its base solve — it reaches `Optimal` in 95 with no ladder —
//! so it stopped being a promotion and the premise test below read
//! `SolveSucceeded`. `degenerate_start_hs008` has the property this file needs
//! at the default options on the exact leg, on both versions: the base solve
//! fails (`Infeasible_Problem_Detected`/9), three rungs run (9, 7 and 5), and
//! the third, `start_point_perturbation=1e-2`, promotes. That makes it the
//! stronger witness for the cost test too, because a total that summed only
//! the promoted rung would be right on a one-rung ladder and wrong here.

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

/// The `.sol` and `.json` paths for ONE `report` call, unique per call
/// rather than per (fixture, options).
///
/// Three tests here solve `PROMOTING` — with the rung on, with it off, and
/// for the cost check — and cargo runs them on parallel threads. Keying the
/// name on the options was the first attempt at separating them, and it is
/// one case short: `a_promotion_records_what_the_base_solve_did` and
/// `the_true_cost_is_recorded_not_just_the_promoted_rungs` both pass no
/// options at all, so both resolved to `pounce_850_square_flowsheet_resto_nl`
/// and raced anyway. The symptom is not a wrong verdict but a missing one —
/// the loser reads a half-written file, `string(&json, "status")` finds no
/// key, and the assertion reads `left: None`. Seen on CI 2026-08-30.
///
/// The counter makes it per call by construction, so a fourth test on the
/// same fixture and options cannot reintroduce it. The pid is for the other
/// half: these files live in the shared temp dir and outlive the run, so a
/// name that repeats across runs lets `json.exists()` be satisfied by a
/// previous run's leftover while this run's solve is still writing.
///
/// Split out of `report` so the property can be asserted directly rather
/// than waited for — see `two_calls_with_the_same_arguments_get_own_paths`.
/// A timing bug that only opens under load is not something a green test run
/// is evidence about.
fn out_paths(name: &str, extra: &[&str]) -> (PathBuf, PathBuf) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(0);

    let dir = std::env::temp_dir();
    let tag: String = format!("{name}{}", extra.join("_"))
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let unique = format!(
        "{}_{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    );
    (
        dir.join(format!("pounce_850_{tag}_{unique}.sol")),
        dir.join(format!("pounce_850_{tag}_{unique}.json")),
    )
}

/// Two `report` calls that agree on every argument must still write to
/// different files. This is the race above, stated as a property instead of
/// a symptom: on the scheme this replaces the paths were a pure function of
/// `(name, extra)`, so this assertion fails there by construction, which a
/// re-run of the flaky test could never establish.
#[test]
fn two_calls_with_the_same_arguments_get_own_paths() {
    let (sol_a, json_a) = out_paths(PROMOTING, &[]);
    let (sol_b, json_b) = out_paths(PROMOTING, &[]);
    assert_ne!(json_a, json_b, "two calls shared a JSON path");
    assert_ne!(sol_a, sol_b, "two calls shared a .sol path");
}

/// Solve `name` and return its JSON report as untyped text.
fn report(name: &str, extra: &[&str]) -> String {
    let (sol, json) = out_paths(name, extra);
    let out = Command::new(pounce_exe())
        .arg(fixture(name))
        .arg(&sol)
        .arg("--json-output")
        .arg(&json)
        .args(extra)
        .output()
        .expect("run pounce");
    assert!(
        json.exists(),
        "no JSON written for {name}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&json).expect("read json");
    // Unique names would otherwise leave one pair per test per run behind.
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&sol);
    text
}

/// Pull `"key": <number>` out of the JSON text. Enough for this file, and it
/// keeps the test free of a serde dependency on the report crate's shape.
fn number(json: &str, key: &str) -> Option<i64> {
    let pat = format!("\"{key}\":");
    let at = json.find(&pat)? + pat.len();
    let rest = json[at..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-')?;
    rest[..end].parse().ok()
}

/// The integers of a `"key": [a, b, c]` array.
fn numbers(json: &str, key: &str) -> Option<Vec<i64>> {
    let pat = format!("\"{key}\":");
    let at = json.find(&pat)? + pat.len();
    let rest = json[at..].trim_start().strip_prefix('[')?;
    let end = rest.find(']')?;
    Some(
        rest[..end]
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect(),
    )
}

fn string(json: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":");
    let at = json.find(&pat)? + pat.len();
    let rest = json[at..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// A fixture whose base solve fails and whose ladder promotes, at the default
/// options. See the module comment for why it is not `square_flowsheet_resto`.
const PROMOTING: &str = "degenerate_start_hs008.nl";

/// The premise, asserted rather than assumed: this fixture's base solver really
/// does fail, and the ladder really is what rescues it. Turning the rungs off
/// is how the report separates the two.
///
/// Note the plural. This test disables **every** rung a local-infeasibility
/// verdict can reach: the two infeasibility rungs, the displacement rung that
/// promotes here, `feral_increase_quality_retry` (closed on this fixture
/// anyway, since it never escalates, but named so the premise does not rest on
/// a count), and the square-system stage — HS008 is square with a constant
/// objective, so the stage would otherwise run once the rungs are gone. Leaving
/// any one on turns "the base solver alone" into "the base solver plus a
/// rung". A rung added later that catches `Infeasible_Problem_Detected`
/// belongs in this list for the same reason.
///
/// On FERAL 0.20 this test ran on `square_flowsheet_resto`, whose base solve
/// failed `RestorationFailed`/131; on 0.21 that solve converges, which is
/// why the fixture moved (module comment).
#[test]
fn the_base_solver_alone_does_not_solve_this_fixture() {
    let json = report(
        PROMOTING,
        &[
            "feral_infeasibility_scaling_retry=no",
            "infeasibility_mu_strategy_retry=no",
            "infeasibility_perturbed_start_retry=no",
            "feral_increase_quality_retry=no",
            // Rung 5: HS008 from the origin is a saddle of the violation, and
            // this rung recovers it with every other rung off.
            "infeasibility_saddle_escape_retry=no",
            "square_system_recovery=no",
        ],
    );
    let status = string(&json, "status").expect("a status");
    assert_eq!(
        status, "InfeasibleProblemDetected",
        "premise: without the rung this fixture fails; if it now solves, this \
         file's subject has been fixed and the fixture needs replacing"
    );
}

/// The fix: a promotion is recorded, and it names the base solve's verdict —
/// the fact the report used to lose entirely.
#[test]
fn a_promotion_records_what_the_base_solve_did() {
    let json = report(PROMOTING, &[]);
    assert_eq!(string(&json, "status").as_deref(), Some("SolveSucceeded"));
    assert!(
        json.contains("\"second_opinion\""),
        "a promoted solve must carry a second_opinion block:\n{json}"
    );
    assert_eq!(
        string(&json, "base_status").as_deref(),
        Some("Infeasible_Problem_Detected"),
        "the base solve's own verdict is the thing that was invisible"
    );
    assert_eq!(
        string(&json, "promoted_by").as_deref(),
        Some("start_point_perturbation=1e-2"),
    );
}

/// The cost, which is understated on the same line. `iteration_count` is the
/// promoted rung's alone; `total_iteration_count` is what the solve spent.
#[test]
fn the_true_cost_is_recorded_not_just_the_promoted_rungs() {
    let json = report(PROMOTING, &[]);
    let reported = number(&json, "iteration_count").expect("iteration_count");
    let base = number(&json, "base_iteration_count").expect("base_iteration_count");
    let total = number(&json, "total_iteration_count").expect("total_iteration_count");
    assert!(
        base > 0 && total > base,
        "base {base} and total {total} must both be real and total must \
         include the base solve"
    );
    assert!(
        total > reported,
        "the reported {reported} iterations understate the true cost {total}, \
         which is the point of recording it"
    );
    // `total` is the base solve plus *every* rung, not just the promoted one.
    // This fixture runs three (9 + 7 + 5 against a base of 9), so an assertion
    // of `base + reported` would only have been right for a single-rung
    // fixture -- which is what it was originally written against.
    let rungs = numbers(&json, "rung_iteration_counts").expect("rung_iteration_counts");
    assert_eq!(
        total,
        base + rungs.iter().sum::<i64>(),
        "total must be the base solve plus every rung: base {base}, rungs {rungs:?}"
    );
    assert_eq!(
        rungs.last().copied(),
        Some(reported),
        "the promoted rung is the last one tried, and its count is what \
         `statistics.iteration_count` reports"
    );
}

/// The other direction, which "always emit the block" would not distinguish: a
/// solve the ladder never opens must carry no `second_opinion` at all, so the
/// field's *presence* is itself the signal.
#[test]
fn an_ordinary_solve_carries_no_second_opinion_block() {
    let json = report("boxed_qp_min.nl", &["solver_selection=nlp"]);
    assert_eq!(string(&json, "status").as_deref(), Some("SolveSucceeded"));
    assert!(
        !json.contains("\"second_opinion\""),
        "a solve that opened no ladder must not carry the block:\n{json}"
    );
}

/// A ladder that runs and promotes nothing is a third case, and it is not the
/// same as no ladder: `it=` is then the base solve's, but the rungs still cost
/// something and that cost is otherwise invisible too.
#[test]
fn a_ladder_that_promotes_nothing_still_records_what_it_spent() {
    let json = report("infeasible_equalities.nl", &[]);
    assert!(
        json.contains("\"second_opinion\""),
        "the ladder ran here, so it must be recorded:\n{json}"
    );
    assert!(
        !json.contains("\"promoted_by\""),
        "nothing was promoted, so the field is absent rather than null"
    );
    let reported = number(&json, "iteration_count").expect("iteration_count");
    let total = number(&json, "total_iteration_count").expect("total_iteration_count");
    assert!(
        total > reported,
        "three rungs ran; their cost ({total} total against a reported \
         {reported}) is exactly what a reader cannot otherwise see"
    );
}
