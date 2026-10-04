//! pounce#870 — a `mu_strategy_fallback` retry that does not promote must give
//! back the *answer* it displaced, not only the status.
//!
//! `run_with_mu_strategy_fallback` runs the solve, and on a stall re-runs it
//! under the flipped barrier schedule. The promote-only-on-`Solve_Succeeded`
//! rule then returns `first_status` unless the retry earned better. That rule
//! floored the **status** and not the **point**: `optimize_constrained` calls
//! the user TNLP's `finalize_solution` once per attempt, so a losing retry
//! still overwrote the reported solution with its own iterate, and the
//! statistics with its own residuals.
//!
//! What the caller then received was a status describing one solve attached to
//! a point from another.
//!
//! `mu_fallback_point_floor.nl` is a 19-variable nonconvex QP with two equality
//! rows and `cond(P) = 1.2e3` — not a pathological model. Its two attempts are
//! far apart, which is what makes it a discriminator rather than a smoke check:
//!
//! | attempt              | objective   | exit                                  |
//! |----------------------|-------------|---------------------------------------|
//! | 1 (base schedule)    | `-3.83e7`   | `Solved To Acceptable Level`          |
//! | 2 (flipped, retried) | `+3.41e5`   | `Converged to a point of local infeasibility` |
//!
//! Before the fix, stock POUNCE reported attempt 1's **status** with attempt
//! 2's **point** — an objective 38.7 million worse, carrying a
//! `final_kkt_error` of `2.85e-4`, which is 285x the `acceptable_tol` that the
//! reported status names. The report contradicted itself.
//!
//! Measured prevalence on a random corpus of 1200 nonconvex models: 20 (1.7%)
//! returned a materially worse point under an unchanged status, worst case a
//! sign flip (`-2.38e7` to `+7.89e7`). The example on record in the option help
//! — `autocorr_bern55-06`, `-2304.0000278` for `-2320.0000298` — is the same
//! defect three orders of magnitude smaller.
//!
//! **This file covers one half of the fix and is blind to the other.** The
//! answer reaches consumers through two independent sinks — the CLI's report,
//! written from `self.statistics`, and `TNLP::finalize_solution`, which
//! `pounce-py`, the C interface and any Rust caller read. The floor restores
//! both, and each half is caught by exactly one test file.
//! `pounce-algorithm/tests/issue_870_fallback_finalize_floor.rs` owns the
//! other. Do not read a green run here as covering `FinalizeSnapshot::replay`:
//! with `replay` deleted, every test below still passes while
//! `pounce.minimize` goes back to returning the losing retry's point.
//!
//! MUTATION TABLE — measured, both directions. Three sinks, and this file sees
//! exactly one of them:
//!
//! | change                                    | this file | the pounce-algorithm file |
//! |-------------------------------------------|-----------|---------------------------|
//! | drop `floor.replay(&tnlp)`                | **green** | 2 of 4 fail               |
//! | drop the `SolutionCertificate` restore    | 3 of 5 fail | **green**               |
//! | drop the trace re-emit                    | **green** | 1 of 4 fails              |
//! | floor unconditionally (ignore promotion)  | `a_promoting_retry_is_still_allowed_to_win` fails | green |

use pounce_solve_report::SolveReport;
use std::path::PathBuf;
use std::process::Command;

/// Attempt 1's objective — the run that earns the reported status.
const FIRST_ATTEMPT_OBJ: f64 = -3.8315032248004645e7;
/// Attempt 2's objective — the locally-infeasible point that used to be reported.
const LOSING_RETRY_OBJ: f64 = 3.4099319647241163e5;
/// The default `acceptable_tol`, which a `SolvedToAcceptableLevel` claim names.
const ACCEPTABLE_TOL: f64 = 1e-6;

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture_named(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

fn solve_named(fixture: &str, tag: &str, opts: &[&str]) -> SolveReport {
    let json = std::env::temp_dir().join(format!("pounce_870_{tag}.json"));
    let _ = std::fs::remove_file(&json);
    let out = Command::new(pounce_exe())
        .arg(fixture_named(fixture))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .args(opts)
        .output()
        .expect("spawn pounce");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.status.code(), Some(0), "must solve:\n{combined}");
    let text = std::fs::read_to_string(&json).expect("JSON report should be written");
    let _ = std::fs::remove_file(&json);
    serde_json::from_str(&text).expect("deserialize report")
}

const FIXTURE: &str = "mu_fallback_point_floor.nl";

/// The defect itself: stock must report the point whose solve earned the status.
#[test]
fn the_reported_point_is_the_one_that_earned_the_status() {
    let stock = solve_named(FIXTURE, "stock", &[]);
    let obj = stock.solution.objective;
    assert!(
        (obj - FIRST_ATTEMPT_OBJ).abs() / FIRST_ATTEMPT_OBJ.abs() < 1e-6,
        "expected the first attempt's objective {FIRST_ATTEMPT_OBJ:e}, got {obj:e}. \
         {LOSING_RETRY_OBJ:e} is the losing retry's locally-infeasible point, which \
         is what a build without the solution floor reports (pounce#870)."
    );
}

/// The floor is only meaningful if the retry actually runs and actually loses;
/// without this the test above could pass on a build where the retry never
/// fired at all.
#[test]
fn disabling_the_retry_gives_the_same_answer_as_flooring_it() {
    let stock = solve_named(FIXTURE, "stock2", &[]);
    let no_retry = solve_named(FIXTURE, "noretry", &["mu_strategy_fallback=no"]);
    assert_eq!(
        stock.solution.status, no_retry.solution.status,
        "the losing retry must not change the status"
    );
    let (a, b) = (stock.solution.objective, no_retry.solution.objective);
    assert!(
        (a - b).abs() / b.abs().max(1.0) < 1e-9,
        "a retry that does not promote must leave the answer where the base \
         solve left it: stock {a:e} vs mu_strategy_fallback=no {b:e}"
    );
}

/// The statistics must describe the same solve as the status, not the retry's.
/// A `SolvedToAcceptableLevel` claim whose own reported error exceeds
/// `acceptable_tol` is self-contradictory.
#[test]
fn the_reported_statistics_belong_to_the_reported_status() {
    let stock = solve_named(FIXTURE, "stats", &[]);
    let err = stock.statistics.final_kkt_error;
    assert!(
        err <= ACCEPTABLE_TOL,
        "the report claims {:?} but carries final_kkt_error {err:e}, which is \
         above acceptable_tol {ACCEPTABLE_TOL:e} — the statistics came from a \
         different attempt than the status (pounce#870)",
        stock.solution.status
    );
}

/// The whole per-run tally is floored with the certificate (gh#983 item 5):
/// iteration count, iteration rows, evaluation counts. `eigenb2` under limited
/// memory is the in-tree case -- its base solve takes 47 iterations and 159
/// objective evaluations, the losing retry 41 and 86 -- and a report that
/// carried the base attempt's point and residuals beside the retry's 86
/// evaluations described no run that happened. Only the wall clock is the
/// whole invocation's.
#[test]
fn the_floor_restores_the_run_tally_with_the_certificate() {
    let opts = ["hessian_approximation=limited-memory"];
    let stock = solve_named("eigenb2.nl", "eig", &opts);
    let no_retry = solve_named(
        "eigenb2.nl",
        "eig_noretry",
        &[
            "hessian_approximation=limited-memory",
            "mu_strategy_fallback=no",
        ],
    );
    assert_eq!(
        stock.statistics.iteration_count, no_retry.statistics.iteration_count,
        "the retry lost, so the reported iteration count is the returned \
         run's, the same one `mu_strategy_fallback=no` reports"
    );
    assert_eq!(
        stock.statistics.num_obj_evals, no_retry.statistics.num_obj_evals,
        "evaluation counts belong to the returned run as well"
    );
}

/// The floor must not swallow a retry that genuinely wins. `csfi2` is the
/// recorded promotion (`Solved_To_Acceptable_Level`/35 -> `Solve_Succeeded`/21),
/// so a build that restores unconditionally fails here.
#[test]
fn a_promoting_retry_is_still_allowed_to_win() {
    let stock = solve_named("csfi2.nl", "csfi2", &[]);
    assert_eq!(
        format!("{:?}", stock.solution.status),
        "SolveSucceeded",
        "csfi2's retry promotes; the floor must not undo a won bet"
    );
}
