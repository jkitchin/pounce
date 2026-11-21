//! The LP / convex-QP → NLP fallback (gh #535) is not spent on a convex
//! answer that already meets `tol` on the ruler the fallback compares answers
//! by (gh #1183) -- unless the convex arm demoted it for an uncertified cost
//! normalization (`σ`, gh #880 / gh #414), which the KKT ruler cannot see --
//! and when it is spent, the NLP arm runs on what is left of `max_wall_time`,
//! not a fresh one.
//!
//! Maros-Meszaros BOYD2 at `tol=1e-8`, `max_wall_time=300`, single thread:
//! the convex arm stopped `OptimalInaccurate` after 44.5 s at a relative KKT
//! error of `1.12e-10`. Its absolute stopping rule read `1.96e-7`, all of it
//! on one 10992-term equality row of magnitude `3.2e7` (`6.3e-14` of the
//! row's own terms, inside the rounding of evaluating it). The fallback then
//! ran the NLP arm to `Maximum_WallTime_Exceeded` at `5.9e-6`, the convex
//! answer reported anyway, and the solve took 312-324 s. With the skip it
//! takes 49.7 s and reports the same point (`kkt 1.1e-10` on the benchmark's
//! verifier).
//!
//! | fixture | convex relative error | branch |
//! |---|---|---|
//! | `issue880_sigma_uncertified.nl` | `1.6e-15`, acceptable (`σ`-demoted) | fallback, NLP wins |
//! | `issue745_netlib_problem.nl` at `tol=1e-12`, `relative_gap_tol=0` | `3.5e-8` (`1.2e-8` L-BFGS), acceptable | fallback |
//! | `lp_tuff.nl` at `tol=3e-13`, `relative_gap_tol=0` | `1.2e-14`, acceptable; stopping error `2.8e-13` | skip |
//!
//! TUFF reaches the skip because the HSDE stop holds the complementarity sum
//! to the gap tolerance (the `traininstance6` fix): it declines only in a
//! narrow window (`1.5e-13`–`6e-13`), and every point there is far inside
//! `tol` on the relative ruler. Re-measure the window if the convex
//! trajectory moves. The absolute band, the status arm and the `σ` arm are
//! pinned by the unit tests in `main.rs` (`convex_in_tol_tests`).

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

/// The `σ` exclusion: `issue880_sigma_uncertified` ends `OptimalInaccurate`
/// at a relative KKT error of `1.6e-15` against `tol=1e-8` -- in tol on every
/// KKT reading -- but its demotion is the uncertified cost normalization, so
/// it still goes to the NLP arm, which wins with `Solve_Succeeded`.
#[test]
fn a_sigma_uncertified_convex_answer_still_falls_back() {
    for leg in [&[][..], &[LBFGS][..]] {
        let r = run("issue880_sigma_uncertified.nl", leg);
        let fb = r
            .report
            .convex_fallback
            .as_ref()
            .unwrap_or_else(|| panic!("{leg:?}: no fallback; stderr=\n{}", r.stderr));
        assert_eq!(fb.convex_status, "Solved_To_Acceptable_Level", "{leg:?}");
        // The precondition that puts this on the σ arm rather than the
        // relative-error one: the KKT ruler reads it in tol.
        assert!(
            fb.convex_kkt_error <= 1e-8,
            "{leg:?}: precondition -- convex relative error in tol; {fb:?}"
        );
        assert_eq!(fb.winner, "nlp", "{leg:?}: {fb:?}");
        assert_eq!(r.report.solution.engine, "nlp", "{leg:?}");
        assert_eq!(
            r.report.solution.status,
            ApplicationReturnStatus::SolveSucceeded,
            "{leg:?}"
        );
        assert!(!r.stderr.contains(SKIP_NOTE), "{leg:?}");
        assert!(r.stderr.contains(FALLBACK_NOTE), "{leg:?}");
        assert!(
            r.stdout.contains("Status: Solve_Succeeded"),
            "{leg:?}: stdout=\n{}",
            r.stdout
        );
    }
}

/// The relative-error arm: a convex answer (not `σ`-demoted) whose relative
/// error is above `tol` still goes to the NLP arm, and the comparison decides.
#[test]
fn an_out_of_tol_convex_answer_still_falls_back() {
    let cases: [(&str, &[&str]); 1] = [(
        "issue745_netlib_problem.nl",
        &["tol=1e-12", "relative_gap_tol=0"],
    )];
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

/// The skip itself: a convex answer that stops `OptimalInaccurate` but
/// already meets `tol` on the relative ruler reports without an NLP re-solve.
#[test]
fn an_in_tol_convex_answer_skips_the_fallback() {
    for leg in [&[][..], &[LBFGS][..]] {
        let mut args = vec!["tol=3e-13", "relative_gap_tol=0"];
        args.extend_from_slice(leg);
        let r = run("lp_tuff.nl", &args);
        assert!(
            r.stderr.contains(SKIP_NOTE),
            "{leg:?}: stderr=\n{}",
            r.stderr
        );
        assert!(!r.stderr.contains(FALLBACK_NOTE), "{leg:?}");
        assert!(r.report.convex_fallback.is_none(), "{leg:?}");
        assert_eq!(r.report.solution.engine, "cvx-qp", "{leg:?}");
        assert_eq!(
            r.report.solution.status,
            ApplicationReturnStatus::SolvedToAcceptableLevel,
            "{leg:?}"
        );
        // NETLIB TUFF's published optimum.
        let obj = r.report.solution.objective;
        assert!(
            ((obj - 0.29214776509) / 0.29214776509).abs() < 1e-6,
            "{leg:?}: objective {obj}"
        );
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
        &["max_wall_time=50", "print_user_options=yes"],
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
