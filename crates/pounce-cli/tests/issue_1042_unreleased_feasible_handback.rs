//! gh#1042 — a converged-to-feasible point restoration never released, which
//! the outer phase then cannot leave.
//!
//! Upstream fails restoration with `RESTORATION_CONVERGED_TO_FEASIBLE_POINT`
//! when its sub-problem converges at a point feasible for the original NLP
//! that layer 1 did not release: the outer filter rejected it, or it did not
//! cut the violation by `kappa_resto` (`IpRestoConvCheck.cpp:227-234`, and the
//! acceptable-stop throw at `IpRestoMinC_1Nrm.cpp:278-287`). `IpIpoptAlg.cpp`
//! reports that as `Restoration_Failed` on a non-square problem. Pounce hands
//! the point back instead, so the outer phase gets one shot from it. That shot
//! pays on the `dallas*` family: every hand-back there is followed by steps
//! that move the iterate, soft-restoration ones included (`dallasl` takes
//! `s s S` and converges).
//!
//! Nothing bounded the hand-back the outer *cannot* take. On Vanderbei
//! `brainpc0`/`brainpc1` (FERAL 0.21, whose inertia is correct where 0.18's
//! was not, gh#1042), the outer asks for restoration again from exactly the
//! point it was handed. The only exits were cycle detectors keyed on other
//! shapes. They fired after 2117 and 1073 iterations, 121 and 52 restoration
//! calls.
//!
//! `IpoptAlgorithm::invoke_restoration` now ends that on the first re-entry
//! from the handed-back point and reports upstream's verdict,
//! `Restoration_Failed`, through the acceptable-point rollback.
//!
//! Each fixture reaches a different branch of the rule. A fixture that only
//! takes one says nothing about the others:
//!
//! * `ssebnln` (Vanderbei, 194 variables): a hand-back at outer iteration 86,
//!   then a re-entry from the same point at 117. Before this change the base
//!   solve ran on to 138 and ended at the near-feasible re-entry detector.
//! * `dallass`: hand-backs followed by steps that leave the point, so the
//!   rule must not fire and the solve converges where it always did. Its
//!   re-entries are all forced by a non-finite Hessian (`|x|^2.85` at 0),
//!   which is exempt on its own.
//! * `eigena2` on L-BFGS: hand-backs, then steps that leave the point, then
//!   ordinary line-search re-entries. This is the fixture that shows the
//!   position test is what keeps the rule quiet.
//! * `degenerate_start_infeasible`: immediate re-entries from *released*
//!   recoveries. Those belong to the 10-strike counter, not this rule.
//!
//! The decisions are `tracing` lines at debug level. The JSON report carries
//! only their downstream effect.

use pounce_solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

const HANDBACK: &str = "restoration converged to a feasible point it did not release";
const EXIT: &str = "re-entered restoration from it without leaving it";
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
    solve_with(name, tag, &[], &[])
}

fn solve_with(name: &str, tag: &str, env: &[(&str, &str)], opts: &[&str]) -> Run {
    let json = std::env::temp_dir().join(format!(
        "pounce_issue_1042_{tag}_{}.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fixture(name))
        .arg("--no-sol")
        .arg("--no-options-file")
        .arg("--json-output")
        .arg(&json)
        .args(opts)
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

/// The branch the change adds. The base solve must end on the new exit, with
/// upstream's verdict, and not on the near-feasible detector that used to end
/// it 21 iterations later. The second-opinion ladder must still recover the
/// model from there.
#[test]
fn a_handback_the_outer_cannot_leave_fails_restoration() {
    let run = solve("ssebnln.nl", "ssebnln");
    assert!(
        run.count(HANDBACK) >= 1,
        "ssebnln must reach a converged-to-feasible hand-back for this test to \
         say anything about the rule\n{}",
        run.log
    );
    assert_eq!(
        run.count(EXIT),
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
        .expect("a restoration failure opens the second-opinion ladder");
    assert_eq!(
        so.base_status, "Restoration_Failed",
        "upstream reports RESTORATION_CONVERGED_TO_FEASIBLE_POINT as a \
         restoration failure on a non-square problem"
    );
    assert!(
        so.base_iteration_count < 138,
        "the exit fires at the first re-entry from the handed-back point, ahead \
         of where the near-feasible detector used to stop the solve (138); got {}",
        so.base_iteration_count
    );
    assert!(
        so.promoted_by.is_some(),
        "the ladder must still recover ssebnln"
    );
    assert!(matches!(
        run.report.solution.status,
        pounce_nlp::return_codes::ApplicationReturnStatus::SolvedToAcceptableLevel
            | pounce_nlp::return_codes::ApplicationReturnStatus::SolveSucceeded
    ));
}

/// The branch the change must leave alone: a hand-back the outer *does*
/// leave. `dallass` (exact) converges after its hand-backs, but every
/// re-entry there is forced by a non-finite Hessian and so exempt on its own.
/// `eigena2` on L-BFGS re-enters through ordinary line-search failures after
/// leaving each handed-back point, so it is the case that only the position
/// test keeps quiet. It fails either way (`Error_In_Step_Computation`, as on
/// main), and the exit must not be what ends it.
#[test]
fn a_handback_the_outer_leaves_is_kept() {
    let dallass = solve("dallass.nl", "dallass");
    let eigena2 = solve_with(
        "eigena2.nl",
        "eigena2_lbfgs",
        &[],
        &["hessian_approximation=limited-memory"],
    );
    for (name, run) in [("dallass", &dallass), ("eigena2/lbfgs", &eigena2)] {
        assert!(
            run.count(HANDBACK) >= 1,
            "{name} must reach a converged-to-feasible hand-back for this test \
             to say anything about the rule\n{}",
            run.log
        );
        assert_eq!(
            run.count(EXIT),
            0,
            "every {name} hand-back is followed by steps that leave it; the \
             exit must not fire\n{}",
            run.log
        );
    }
    assert!(
        matches!(
            dallass.report.solution.status,
            pounce_nlp::return_codes::ApplicationReturnStatus::SolveSucceeded
        ),
        "dallass: expected Optimal, got {:?}",
        dallass.report.solution.status
    );
    assert!(dallass.report.second_opinion.is_none());
    assert!(matches!(
        eigena2.report.solution.status,
        pounce_nlp::return_codes::ApplicationReturnStatus::ErrorInStepComputation
    ));
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
        &[],
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
        run.count(EXIT),
        0,
        "a released recovery is the 10-strike counter's, not this exit's\n{}",
        run.log
    );
    assert!(matches!(
        run.report.solution.status,
        pounce_nlp::return_codes::ApplicationReturnStatus::InfeasibleProblemDetected
    ));
}
