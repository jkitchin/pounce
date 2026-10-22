//! gh#1042 — a converged-to-feasible point restoration never released, which
//! the outer phase then cannot leave.
//!
//! Restoration's layer 2 (`IpRestoConvCheck.cpp:227-234`) renders
//! `ConvergedToFeasiblePoint` when its sub-problem has converged at a point
//! feasible for the original NLP that layer 1 did not release: the outer
//! filter rejected it, or it did not cut the violation by `kappa_resto`.
//! Upstream fails restoration there. Pounce hands the point back instead, so
//! the outer phase gets one shot from it, and that shot pays on the `dallas*`
//! family: every hand-back there is followed by accepted steps, soft-
//! restoration ones included (`dallasl` takes `s s S` and converges).
//!
//! What nothing bounded was the hand-back the outer *cannot* take. On
//! Vanderbei `brainpc0`/`brainpc1` (FERAL 0.21, whose inertia is correct
//! where 0.18's was not, gh#1042) the outer asks for restoration again from
//! exactly the point it was handed, without one accepted step. The only exits
//! were cycle detectors keyed on other shapes, and they fired after 2117 and
//! 1073 iterations, 121 and 52 restoration calls, before the
//! `mu_strategy=adaptive` rung solved each in 24 and 52.
//!
//! `IpoptAlgorithm::invoke_restoration` now ends that cycle on its first
//! strike, with the same status and acceptable-point rollback as the other
//! restoration-cycle exits.
//!
//! The two fixtures reach the two branches of the rule — that is the point of
//! having two, since a fixture that only takes one says nothing about the
//! other:
//!
//! * `ssebnln` (Vanderbei, 194 variables): a hand-back at outer iteration 86,
//!   then a re-entry from the same point at 117. Before this change the base
//!   solve ran on to 138 and ended at the near-feasible re-entry detector.
//! * `dallass`: two hand-backs, each followed by accepted steps, so the rule
//!   must not fire and the solve converges where it always did.
//! * `degenerate_start_infeasible`: the outer re-enters restoration from a
//!   recovered point with no step in between — the same shape — but every
//!   recovery there was *released* by layer 1. Those belong to the 10-strike
//!   counter (`resto_no_outer_progress_count`), and the one-strike rule must
//!   not claim them.
//!
//! Both decisions are `tracing` lines at debug level; the JSON report carries
//! only their downstream effect.

use pounce_solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

const HANDBACK: &str = "restoration converged to a feasible point it did not release";
const CYCLE_EXIT: &str = "and the outer re-entered restoration from it without an accepted step";
const NEAR_FEASIBLE_EXIT: &str = "near-feasible restoration re-entry";

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

struct Run {
    report: SolveReport,
    log: String,
}

impl Run {
    fn count(&self, needle: &str) -> usize {
        self.log.lines().filter(|l| l.contains(needle)).count()
    }
}

fn solve(name: &str, tag: &str) -> Run {
    solve_with(name, tag, &[])
}

fn solve_with(name: &str, tag: &str, env: &[(&str, &str)]) -> Run {
    let json = std::env::temp_dir().join(format!("pounce_issue_1042_{tag}.json"));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fixture(name))
        .arg("--no-sol")
        .arg("--no-options-file")
        .arg("--json-output")
        .arg(&json)
        .env("RUST_LOG", "pounce::algorithm=debug")
        .env("NO_COLOR", "1")
        .envs(env.iter().copied())
        .output()
        .unwrap_or_else(|e| panic!("spawn pounce for {name}: {e}"));
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&json)
        .unwrap_or_else(|e| panic!("JSON report should be written for {name}: {e}\n{log}"));
    let _ = std::fs::remove_file(&json);
    Run {
        report: serde_json::from_str(&text).expect("deserialize report"),
        log,
    }
}

/// The branch the change adds. The base solve must end on the new exit — not
/// on the near-feasible detector that used to end it 21 iterations later —
/// and the second-opinion ladder must still recover the model from there.
#[test]
fn a_handback_the_outer_cannot_leave_ends_the_base_solve() {
    let run = solve("ssebnln.nl", "ssebnln");
    assert!(
        run.count(HANDBACK) >= 1,
        "ssebnln must reach a converged-to-feasible hand-back for this test to \
         say anything about the rule\n{}",
        run.log
    );
    assert_eq!(
        run.count(CYCLE_EXIT),
        1,
        "the base solve must end on the gh#1042 exit, exactly once\n{}",
        run.log
    );
    assert_eq!(
        run.count(NEAR_FEASIBLE_EXIT),
        0,
        "the near-feasible detector ended this solve before gh#1042; it must \
         not be what ends it now\n{}",
        run.log
    );
    let so = run
        .report
        .second_opinion
        .as_ref()
        .expect("the cycle exit opens the second-opinion ladder");
    assert_eq!(so.base_status, "Error_In_Step_Computation");
    assert!(
        so.base_iteration_count < 138,
        "the exit fires at the first re-entry from the handed-back point, ahead \
         of where the near-feasible detector used to stop the solve (138); got {}",
        so.base_iteration_count
    );
    assert_eq!(so.promoted_by.as_deref(), Some("mu_strategy=adaptive"));
    assert!(matches!(
        run.report.solution.status,
        pounce_nlp::return_codes::ApplicationReturnStatus::SolvedToAcceptableLevel
            | pounce_nlp::return_codes::ApplicationReturnStatus::SolveSucceeded
    ));
}

/// The branch the change must leave alone: a hand-back the outer *does* take.
/// `dallass` is handed back a converged-to-feasible point and accepts steps
/// from it every time, so the exit must never fire and the solve converges.
#[test]
fn a_handback_followed_by_accepted_steps_is_kept() {
    let run = solve("dallass.nl", "dallass");
    assert!(
        run.count(HANDBACK) >= 1,
        "dallass must reach a converged-to-feasible hand-back for this test to \
         say anything about the rule\n{}",
        run.log
    );
    assert_eq!(
        run.count(CYCLE_EXIT),
        0,
        "every dallass hand-back is followed by accepted steps; the exit must \
         not fire\n{}",
        run.log
    );
    assert!(
        matches!(
            run.report.solution.status,
            pounce_nlp::return_codes::ApplicationReturnStatus::SolveSucceeded
        ),
        "expected Optimal, got {:?}",
        run.report.solution.status
    );
    assert!(run.report.second_opinion.is_none());
}

/// The shape without the hand-back: immediate re-entries from *released*
/// recoveries. `POUNCE_DBG_RESTO_CYCLE` prints the entry-vs-recovery
/// distance, which is how this test shows the shape is reached at all.
#[test]
fn a_released_recovery_the_outer_cannot_leave_is_not_claimed() {
    let run = solve_with(
        "degenerate_start_infeasible.nl",
        "degenerate",
        &[("POUNCE_DBG_RESTO_CYCLE", "1")],
    );
    let immediate_reentries = run
        .log
        .lines()
        .filter(|l| l.contains("entry-vs-recovery dx_rel=0.000000e0 ds_rel=0.000000e0"))
        .count();
    assert!(
        immediate_reentries >= 1,
        "the fixture must re-enter restoration from a recovered point for this \
         test to say anything about the rule\n{}",
        run.log
    );
    assert_eq!(
        run.count(HANDBACK),
        0,
        "no hand-back expected here\n{}",
        run.log
    );
    assert_eq!(
        run.count(CYCLE_EXIT),
        0,
        "a released recovery is the 10-strike counter's, not this exit's\n{}",
        run.log
    );
    assert!(matches!(
        run.report.solution.status,
        pounce_nlp::return_codes::ApplicationReturnStatus::InfeasibleProblemDetected
    ));
}
