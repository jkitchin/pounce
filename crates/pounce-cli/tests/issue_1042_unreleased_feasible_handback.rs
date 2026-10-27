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
//! `Restoration_Failed`, through the acceptable-point rollback. The solve
//! statistics flag the exit (`unreleased_handback_exit`), and the
//! second-opinion ladder opens `mu_strategy=adaptive` on that one
//! `Restoration_Failed` ahead of the displaced start every restoration
//! failure gets (`SecondOpinionTrigger::UnreleasedHandback`).
//!
//! Each fixture reaches a different branch of the rule. A fixture that only
//! takes one says nothing about the others:
//!
//! * `ssebnln` (Vanderbei, 194 variables): a hand-back at outer iteration 86,
//!   then a re-entry from the same point at 117. Before this change the base
//!   solve ran on to 138 and ended at the near-feasible re-entry detector.
//! * `dallass_kink_unmerged` (`tests/data/`): hand-backs followed by steps
//!   that leave the point, so the rule must not fire and the solve converges
//!   where it always did. Its re-entries are all forced by a non-finite
//!   Hessian (`|x|^2 * |x|^0.85` at 0), which is exempt on its own. It is
//!   `dallass.nl` written so the `.nl` reader's `|e|^p * |e|^q` merge (gh#1042
//!   R3) declines; `dallass.nl` itself now has a finite Hessian at the kink
//!   and never enters restoration on the exact leg, so it reaches no
//!   hand-back at all.
//! * `eigena2` on L-BFGS: hand-backs, then steps that leave the point, then
//!   ordinary line-search re-entries. This is the fixture that shows the
//!   position test is what keeps the rule quiet.
//! * `issue1042_nonfinite_w_handback` (`issue1042.py`): re-entries from the
//!   handed-back point, every one forced by a non-finite Hessian — the
//!   exemption, which nothing else in the corpus reaches.
//! * `degenerate_start_infeasible`: immediate re-entries from *released*
//!   recoveries. Those belong to the 10-strike counter, not this rule.
//! * Vanderbei `brainpc0` (not in the repository; the ignored test below runs
//!   it when `POUNCE_VANDERBEI_NL` names a directory holding it): the gh#1076
//!   arm. The outer leaves each hand-back on soft-restoration `s` steps alone
//!   and re-enters restoration from elsewhere, so the position test never
//!   fires. No fixture in the corpus reaches that arm. `dallass_kink_unmerged`
//!   above also pins gh#1076's filter restart: without it the base solve ends
//!   `Error_In_Step_Computation` at 116 and needs the ladder.
//!
//! Upstream's second throw site — an inner acceptable stop at a feasible
//! point — is reached by `deb7` on L-BFGS and by nothing else in the corpus,
//! and `deb7` is not a portable witness (gh#887), so it has no test here.
//!
//! The decisions are `tracing` lines at debug level. The JSON report carries
//! only their downstream effect.

use pounce_solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

const HANDBACK: &str = "restoration converged to a feasible point it did not release";
const EXIT: &str = "re-entered restoration from it without leaving it";
const NEAR_FEASIBLE_EXIT: &str = "near-feasible restoration re-entry";
const SHOT_EXIT: &str = "without the original globalization accepting a step from it";

fn fixture(name: &str) -> PathBuf {
    test_file("fixtures", name)
}

fn test_file(dir: &str, name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push(dir);
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
    solve_path(fixture(name), name, tag, env, opts)
}

fn solve_path(path: PathBuf, name: &str, tag: &str, env: &[(&str, &str)], opts: &[&str]) -> Run {
    let json = std::env::temp_dir().join(format!(
        "pounce_issue_1042_{tag}_{}.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(&path)
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
/// model from there, and must reach `mu_strategy=adaptive` to do it.
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
    // The ladder routing (gh#1042 "ladder routing"). `Restoration_Failed`
    // alone opens only the displaced start; this exit's opens
    // `mu_strategy=adaptive` ahead of it, which is the rung that recovered
    // brainpc0/1/2 on arm64 while the exit still reported
    // `Error_In_Step_Computation`. On ssebnln it recovers the model in 41
    // (158 in total), where the displaced start took 122 (239).
    // Mutation-checked: reading `for_status` instead of `for_solve` in the
    // ladder driver promotes `start_point_perturbation=1e-2` here.
    assert_eq!(
        so.tried.first().map(String::as_str),
        Some("mu_strategy=adaptive"),
        "this exit's Restoration_Failed must open the barrier rung first; \
         ladder tried {:?}",
        so.tried
    );
    assert_eq!(
        so.promoted_by.as_deref(),
        Some("mu_strategy=adaptive"),
        "the ladder must still recover ssebnln, through the barrier rung"
    );
    assert!(matches!(
        run.report.solution.status,
        pounce_nlp::return_codes::ApplicationReturnStatus::SolvedToAcceptableLevel
            | pounce_nlp::return_codes::ApplicationReturnStatus::SolveSucceeded
    ));
}

/// The branch the change must leave alone: a hand-back the outer *does*
/// leave. `dallass_kink_unmerged` (exact) converges after its hand-backs, but
/// every re-entry there is forced by a non-finite Hessian and so exempt on its
/// own.
/// `eigena2` on L-BFGS re-enters through ordinary line-search failures after
/// leaving each handed-back point, so it is the case that only the position
/// test keeps quiet. It fails either way (`Error_In_Step_Computation`, as on
/// main), and the exit must not be what ends it.
#[test]
fn a_handback_the_outer_leaves_is_kept() {
    let dallass = solve_path(
        test_file("data", "dallass_kink_unmerged.nl"),
        "dallass_kink_unmerged.nl",
        "dallass",
        &[],
        &[],
    );
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

/// The exemption: a re-entry from the handed-back point that a non-finite
/// Hessian forced. `issue1042_nonfinite_w_handback` (see `issue1042.py`) is
/// handed back an unreleased point with `y = 0`, where `W` holds `0·∞`, so
/// every re-entry from it is the gh#1042 shape *and* exempt. The exit must
/// not claim it; the solve ends where it does on main (78 + 39 iterations,
/// `Error_In_Step_Computation` after the `mu_strategy=adaptive` rung).
/// `dallass_kink_unmerged` cannot witness this on its own: the position test already keeps
/// it quiet, so dropping the exemption leaves it green.
#[test]
fn a_reentry_forced_by_a_nonfinite_hessian_is_exempt() {
    let run = solve("issue1042_nonfinite_w_handback.nl", "nonfinite_w");
    assert!(
        run.count(HANDBACK) >= 1,
        "the fixture must reach an unreleased hand-back\n{}",
        run.log
    );
    assert!(
        run.count("exempt from the gh#1042 exit") >= 1,
        "the fixture must re-enter from the handed-back point on a non-finite \
         Hessian for this test to say anything about the exemption\n{}",
        run.log
    );
    assert_eq!(
        run.count(EXIT),
        0,
        "a non-finite-W entry says nothing about the outer being stuck\n{}",
        run.log
    );
    assert!(matches!(
        run.report.solution.status,
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

/// The gh#1076 arm, on the model it was built for. Vanderbei `brainpc0` is
/// 6905 variables and not in the repository, so this runs only with
/// `POUNCE_VANDERBEI_NL` naming a directory that holds `brainpc0.nl`
/// (`cargo test -- --ignored`).
///
/// Its restorations are entered at infeasible points and hand back points
/// feasible to ~1e-7 at f ≈ 8e5, which the outer filter, holding f ≈ 0.38,
/// rejects. The outer leaves each on `s` steps alone and re-enters from
/// elsewhere, so the position test never fires; without the arm that cycle ran
/// 45 restoration calls to a false `Infeasible_Problem_Detected` at 1678 (108
/// s in total on arm64). The arm must end the base solve with upstream's
/// verdict, `Restoration_Failed`, judged on the feasible hand-back rather than
/// the re-entry point, so the ladder opens `mu_strategy=adaptive` first, which
/// recovers it (365 + 24 iterations). Judging it on the re-entry point reads
/// `Infeasible_Problem_Detected` and opens `feral_scaling=mc64` first (+403).
#[test]
#[ignore = "needs Vanderbei brainpc0.nl via POUNCE_VANDERBEI_NL"]
fn a_handback_left_on_soft_restoration_steps_alone_fails_restoration() {
    let Some(dir) = std::env::var_os("POUNCE_VANDERBEI_NL") else {
        eprintln!("POUNCE_VANDERBEI_NL is not set; skipping");
        return;
    };
    let run = solve_path(
        PathBuf::from(dir).join("brainpc0.nl"),
        "brainpc0.nl",
        "brainpc0",
        &[],
        &[],
    );
    assert!(run.count(HANDBACK) >= 1, "{}", run.log);
    assert_eq!(
        run.count(SHOT_EXIT),
        1,
        "the base solve must end on the gh#1076 arm, exactly once\n{}",
        run.log
    );
    let so = run
        .report
        .second_opinion
        .as_ref()
        .expect("a restoration failure opens the second-opinion ladder");
    assert_eq!(so.base_status, "Restoration_Failed");
    assert!(
        so.base_iteration_count < 1678,
        "the arm must end the cycle ahead of the false infeasibility verdict \
         at 1678; got {}",
        so.base_iteration_count
    );
    assert_eq!(
        so.tried.first().map(String::as_str),
        Some("mu_strategy=adaptive")
    );
    assert_eq!(so.promoted_by.as_deref(), Some("mu_strategy=adaptive"));
}
