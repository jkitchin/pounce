//! The LP / convex-QP → NLP fallback (gh #535) is not spent on a convex
//! answer that already meets `tol` on the ruler the fallback compares answers
//! by (gh #1183) -- and when it is spent, the NLP arm runs on what is left of
//! `max_wall_time`, not a fresh one.
//!
//! Maros-Meszaros BOYD2 at `tol=1e-8`, `max_wall_time=300`, single thread:
//! the convex arm stopped `OptimalInaccurate` after 44.5 s at a relative KKT
//! error of `1.12e-10`. Its absolute stopping rule read `1.96e-7`, all of it
//! on one 10992-term equality row of magnitude `3.2e7` (`6.3e-14` of the
//! row's own terms, inside the rounding of evaluating it). The fallback then
//! ran the NLP arm to `Maximum_WallTime_Exceeded` at `5.9e-6`, the convex
//! answer reported anyway, and the solve took 312-324 s. With the skip it
//! takes 49.7 s and reports the same point (`kkt 1.1e-10` on the benchmark's
//! verifier). BOYD2 is 93 263 variables, too large for the test suite; the
//! fixtures here reach the same two branches of `convex_answer_meets_tol`.
//!
//! | fixture | convex relative error | branch |
//! |---|---|---|
//! | `issue880_sigma_uncertified.nl` | `1.6e-15`, acceptable, `tol=1e-8` | skip |
//! | `issue880_sigma_uncertified.nl` at `tol=1e-15` | `5.7e-13`, acceptable | fallback |
//! | `lp_tuff.nl` at `tol=1e-12`, `relative_gap_tol=0` | `1.4e-12`, acceptable | fallback |
//!
//! The rule's absolute-band arm (`abs <= 1e3·tol`) and its status arm are
//! pinned by the unit tests in `main.rs` (`convex_in_tol_tests`): no small
//! fixture is known that ends `OptimalInaccurate` in tol relatively with an
//! absolute error outside the band.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::return_codes::ApplicationReturnStatus;

const LBFGS: &str = "hessian_approximation=limited-memory";

struct Run {
    stdout: String,
    stderr: String,
    report: SolveReport,
}

fn run(fixture: &str, opts: &[&str]) -> Run {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("pounce_in_tol_skip_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    std::fs::copy(&src, dir.join("m.nl")).expect("copy fixture");
    let out = Command::new(PathBuf::from(env!("CARGO_BIN_EXE_pounce")))
        .current_dir(&dir)
        .arg("m.nl")
        .arg("--json-output")
        .arg("r.json")
        .args(opts)
        .output()
        .expect("run pounce");
    let json = std::fs::read_to_string(dir.join(Path::new("r.json"))).unwrap_or_default();
    let report: SolveReport = serde_json::from_str(&json).expect("deserialize SolveReport");
    let _ = std::fs::remove_dir_all(&dir);
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        report,
    }
}

const SKIP_NOTE: &str = "already meets tol, so it is reported without an NLP re-solve";
const FALLBACK_NOTE: &str = "LP/QP fallback (gh #1183)";

/// The skip: a reduced-accuracy convex answer at `1.6e-15` against
/// `tol=1e-8` reports from the convex arm, and the NLP arm never runs. Before
/// the skip this ran the fallback and reported the NLP arm's
/// `Solve_Succeeded` at `1.5e-16` -- the status this gives up.
#[test]
fn an_in_tol_acceptable_convex_answer_skips_the_fallback() {
    for leg in [&[][..], &[LBFGS][..]] {
        let r = run("issue880_sigma_uncertified.nl", leg);
        assert!(
            r.report.convex_fallback.is_none(),
            "{leg:?}: the fallback ran; {:?}",
            r.report.convex_fallback
        );
        assert_eq!(r.report.solution.engine, "cvx-qp", "{leg:?}");
        assert_eq!(
            r.report.solution.status,
            ApplicationReturnStatus::SolvedToAcceptableLevel,
            "{leg:?}"
        );
        assert!(
            r.stderr.contains(SKIP_NOTE),
            "{leg:?}: the skip is not announced; stderr=\n{}",
            r.stderr
        );
        assert!(!r.stderr.contains(FALLBACK_NOTE), "{leg:?}");
        assert!(
            r.stdout.contains("Status: Solved_To_Acceptable_Level"),
            "{leg:?}: stdout=\n{}",
            r.stdout
        );
    }
}

/// The other branch, twice: a convex answer whose relative error is above
/// `tol` still goes to the NLP arm, and the comparison decides.
#[test]
fn an_out_of_tol_acceptable_convex_answer_still_falls_back() {
    let cases: [(&str, &[&str]); 2] = [
        ("issue880_sigma_uncertified.nl", &["tol=1e-15"]),
        ("lp_tuff.nl", &["tol=1e-12", "relative_gap_tol=0"]),
    ];
    for (fixture, opts) in cases {
        for leg in [&[][..], &[LBFGS][..]] {
            let mut args = opts.to_vec();
            args.extend_from_slice(leg);
            let r = run(fixture, &args);
            let fb =
                r.report.convex_fallback.as_ref().unwrap_or_else(|| {
                    panic!("{fixture} {leg:?}: no fallback; stderr=\n{}", r.stderr)
                });
            assert_eq!(
                fb.convex_status, "Solved_To_Acceptable_Level",
                "{fixture} {leg:?}"
            );
            // The precondition that makes this the other branch.
            let tol: f64 = opts[0].trim_start_matches("tol=").parse().unwrap();
            assert!(
                fb.convex_kkt_error > tol,
                "{fixture} {leg:?}: precondition -- convex relative error above tol; {fb:?}"
            );
            assert!(!r.stderr.contains(SKIP_NOTE), "{fixture} {leg:?}");
            assert!(r.stderr.contains(FALLBACK_NOTE), "{fixture} {leg:?}");
        }
    }
}

/// The wall budget is handed down: the NLP arm of a fallback runs on
/// `max_wall_time` minus what the convex arm spent (`charge_wall_budget`),
/// read here off the option value the NLP arm was given.
#[test]
fn the_fallback_runs_on_the_remaining_wall_budget() {
    const LIMIT: f64 = 50.0;
    let r = run(
        "issue880_sigma_uncertified.nl",
        &["tol=1e-15", "max_wall_time=50", "print_user_options=yes"],
    );
    let fb = r.report.convex_fallback.as_ref().expect("fallback taken");
    let given: f64 = r
        .stdout
        .lines()
        .find_map(|l| l.strip_prefix("max_wall_time "))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no max_wall_time in the option dump; stdout=\n{}", r.stdout));
    let charged = LIMIT - given;
    assert!(
        charged > 0.0,
        "the NLP arm got the whole budget ({given}) after a {:.3}s convex attempt",
        fb.convex_wallclock_time_secs
    );
    // The charge is the convex attempt's wall clock (plus model extraction),
    // not more: the NLP arm is not short-changed either.
    assert!(
        charged >= 0.5 * fb.convex_wallclock_time_secs
            && charged <= r.report.statistics.total_wallclock_time_secs + 1.0,
        "charged {charged}s against a {:.3}s convex attempt; {fb:?}",
        fb.convex_wallclock_time_secs
    );
}
