//! gh #1183: the LP / convex-QP → NLP fallback (gh #535) reports the more
//! accurate of its two answers, not the NLP one unconditionally.
//!
//! When the convex arm finishes an LP without certifying it, `auto` re-solves
//! the model on the NLP arm. Until gh #1183 the NLP answer then replaced the
//! convex one whatever it was, and on NETLIB WOOD1P / WOODW and MIPLIB leo2 it
//! was the less accurate of the two: the NLP arm's `Solve_Succeeded` is judged
//! on Ipopt's scaled criteria and certified points with a relative duality
//! gap of `1.2e-6`, while the convex candidate it discarded verified at
//! `1e-7`. So the CLI now measures both on one ruler — the five relative KKT
//! residuals of `pounce_convex::RelativeKktResiduals`, on the model as
//! declared — and the smaller error reports.
//!
//! One fixture per branch of the rule (`main.rs`, `convex_candidate_wins`):
//!
//! | fixture | convex | NLP | reports | decided by |
//! |---|---|---|---|---|
//! | `lp_etamacro.nl` (NETLIB ETAMACRO) at `tol=5e-13`, `relative_gap_tol=0` | `1.5e-12`, acceptable | `4.2e-11`, succeeded (`8.6e-11`–`1.0e-10`, acceptable, under L-BFGS) | convex | accuracy |
//! | `issue880_sigma_uncertified.nl` | `1.6e-15`, acceptable | `1.5e-16`, succeeded | NLP | accuracy |
//! | `lp_israel.nl` at `tol=1e-16` | `2.0e-16`, iteration limit | `5.2e-14`, acceptable | NLP | the status guard |
//! | `lp_afiro.nl` at `tol=1e-20` | `2.4e-16`, iteration limit | `5.5e-13`, search direction too small | convex, its own failing status | accuracy |
//!
//! The `lp_israel` row is the gh #535 shape — a convex failure against an NLP
//! success — and it is a test of the guard and nothing else: the convex point
//! is the more accurate by 250×, so without the guard the convex answer
//! would report. (NETLIB GEN/GEN1, the models gh #535 was built for, no
//! longer reach the fallback at all: the convex arm certifies both in 20
//! iterations.) The fourth branch, a convex `NumericalFailure`, is pinned in
//! `issue_535_lp_falls_back_to_nlp.rs` on `lp_afiro` with `qp_tau=0.99`.
//!
//! The margins are a decade or more on purpose; NETLIB CAPRI reaches the
//! fallback at `3.9e-10` against `2.7e-10` and is not used. Since gh#1180's
//! unreduced retry the convex arm certifies most of the NETLIB models that
//! used to reach the fallback at the default `tol` (ETAMACRO, SHARE2B, and
//! the issue's WOOD1P / WOODW among them), so the convex-wins fixture runs at
//! a tightened `tol`. The fixture was NETLIB TUFF until two changes met on
//! it: the HSDE stop began holding the complementarity sum `ŝᵀẑ` to the gap
//! tolerance (the `traininstance6` fix), after which TUFF declines only in a
//! narrow `tol` window (`1.5e-13`–`6e-13`) and certifies outside it; and the
//! fallback is now skipped when the convex answer already meets `tol` on this
//! ruler (BOYD2), which every point in that window does (`1.2e-14` at
//! `3e-13`). The convex-wins branch therefore needs a convex answer *outside*
//! `tol` that still beats the NLP one, which ETAMACRO at `tol=5e-13` gives on
//! both legs by 27x and 55x (67x on macOS). The `tol` is chosen for margin on every
//! platform, not just one: at `tol=1e-12` the convex arm stopped at
//! `1.56e-12` on aarch64 macOS (declined, 1.6x over) but certified on x86-64
//! and aarch64 Linux, so the precondition held on one machine only. At
//! `5e-13` it stops at `1.55e-12` (3x over) on all three, and every number
//! in this row is bit-identical across them (measured in Docker). Re-measure
//! it if the convex trajectory moves.
//! `lp_tuff.nl` and `lp_etamacro.nl` were produced by
//! `benchmarks/lp/generate_nl.py`
//! from the NETLIB MPS file. Every case takes well under a second.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;
use pounce_nlp::return_codes::ApplicationReturnStatus;

struct Run {
    stdout: String,
    stderr: String,
    code: Option<i32>,
    report: SolveReport,
    sol: String,
}

fn run(fixture: &str, opts: &[&str]) -> Run {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("pounce_issue1183_{}_{n}", std::process::id()));
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
    let read = |p: &Path| std::fs::read_to_string(dir.join(p)).unwrap_or_default();
    let report: SolveReport =
        serde_json::from_str(&read(Path::new("r.json"))).expect("deserialize SolveReport");
    let sol = read(Path::new("m.sol"));
    let _ = std::fs::remove_dir_all(&dir);
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        code: out.status.code(),
        report,
        sol,
    }
}

const LBFGS: &str = "hessian_approximation=limited-memory";

fn status_lines(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .filter(|l| l.starts_with("Status:"))
        .collect()
}

/// The issue's case: the convex answer is the more accurate, so it reports —
/// its point, its status, its `.sol`, and a record of what it beat.
#[test]
fn a_more_accurate_convex_answer_is_kept() {
    for leg in [&[][..], &[LBFGS][..]] {
        // `relative_gap_tol=0`: the fallback otherwise holds the NLP arm to the
        // convex arm's duality gap, which lifts it to 2.4e-13 here and makes it
        // the winner — that branch is pinned in
        // `relative_gap_tol_closes_the_nlp_gap.rs`. This test is about the
        // comparison, so it keeps the NLP arm on Ipopt's own test.
        let mut args = vec!["tol=5e-13", "relative_gap_tol=0"];
        args.extend_from_slice(leg);
        let r = run("lp_etamacro.nl", &args);
        assert!(
            r.stderr.to_lowercase().contains("did not certify"),
            "{leg:?}: precondition — the convex solve must decline; stderr=\n{}",
            r.stderr
        );
        let fb = r
            .report
            .convex_fallback
            .as_ref()
            .expect("fallback recorded");
        assert_eq!(
            (fb.winner.as_str(), fb.loser.as_str()),
            ("cvx-qp", "nlp"),
            "{leg:?}: {fb:?}"
        );
        let nlp = fb.nlp_kkt_error.expect("the NLP arm produced a point");
        assert!(
            fb.convex_kkt_error * 10.0 < nlp,
            "{leg:?}: precondition — a clear convex win; {fb:?}"
        );
        // The NLP arm *succeeded* by its own criteria (strictly on the exact
        // leg, at acceptable level under L-BFGS); the convex answer replaces
        // it on accuracy, which is the whole point of the issue.
        assert!(
            matches!(
                fb.nlp_status.as_str(),
                "Solve_Succeeded" | "Solved_To_Acceptable_Level"
            ),
            "{leg:?}: {fb:?}"
        );

        assert_eq!(r.report.solution.engine, "cvx-qp", "{leg:?}");
        assert_eq!(
            r.report.solution.status,
            ApplicationReturnStatus::SolvedToAcceptableLevel,
            "{leg:?}: the convex answer reports with its own status"
        );
        assert_eq!(r.report.solution.solve_result_num, 1, "{leg:?}");
        assert_eq!(r.code, Some(0), "{leg:?}: acceptable is a successful exit");
        assert_eq!(
            status_lines(&r.stdout),
            ["Status: Solved_To_Acceptable_Level"],
            "{leg:?}: one verdict, the convex one; stdout=\n{}",
            r.stdout
        );
        assert!(
            r.sol
                .lines()
                .next()
                .is_some_and(|l| l.contains("IPM (pounce-convex)")),
            "{leg:?}: the .sol is the convex arm's; head=\n{}",
            r.sol.lines().take(2).collect::<Vec<_>>().join("\n")
        );
        // NETLIB ETAMACRO's published optimum.
        let obj = r.report.solution.objective;
        assert!(
            ((obj - -755.71523) / 755.71523).abs() < 1e-6,
            "{leg:?}: objective {obj}"
        );
        // The run paid for both solves: its wall time is the sum, and each
        // arm's share is on the record.
        assert!(fb.convex_wallclock_time_secs > 0.0 && fb.nlp_wallclock_time_secs > 0.0);
        let total = r.report.statistics.total_wallclock_time_secs;
        assert!(
            (total - (fb.convex_wallclock_time_secs + fb.nlp_wallclock_time_secs)).abs() < 1e-9,
            "{leg:?}: total {total} is not the sum of {fb:?}"
        );
    }
}

/// The other side of accuracy: when the NLP answer is the more accurate, it
/// still reports — and the convex candidate it beat is recorded, not printed.
#[test]
fn a_more_accurate_nlp_answer_still_reports() {
    for leg in [&[][..], &[LBFGS][..]] {
        let r = run("issue880_sigma_uncertified.nl", leg);
        let fb = r
            .report
            .convex_fallback
            .as_ref()
            .expect("fallback recorded");
        assert_eq!(
            (fb.winner.as_str(), fb.loser.as_str()),
            ("nlp", "cvx-qp"),
            "{leg:?}: {fb:?}"
        );
        // Both usable, so accuracy decides, and the NLP point is a decade
        // ahead.
        assert_eq!(fb.convex_status, "Solved_To_Acceptable_Level", "{leg:?}");
        let nlp = fb.nlp_kkt_error.expect("the NLP arm produced a point");
        assert!(
            nlp * 5.0 < fb.convex_kkt_error,
            "{leg:?}: precondition — a clear NLP win; {fb:?}"
        );
        assert_eq!(r.report.solution.engine, "nlp", "{leg:?}");
        assert_eq!(
            r.report.solution.status,
            ApplicationReturnStatus::SolveSucceeded,
            "{leg:?}"
        );
        assert_eq!(
            status_lines(&r.stdout),
            ["Status: Solve_Succeeded"],
            "{leg:?}: one verdict, the NLP one; stdout=\n{}",
            r.stdout
        );
        assert!(
            !r.stdout.contains("IPM, pounce-convex"),
            "{leg:?}: the losing convex candidate must not print a verdict"
        );
        // The run paid for the convex attempt too.
        let total = r.report.statistics.total_wallclock_time_secs;
        assert!(
            total >= fb.convex_wallclock_time_secs,
            "{leg:?}: total {total} omits the convex attempt; {fb:?}"
        );
    }
}

/// The gh #535 shape and the status guard: the convex arm runs out of
/// iterations, the NLP arm succeeds, and the NLP answer reports — even though
/// the convex point is the more accurate by two orders. A failed convex
/// answer never turns a successful exit into a failed one. Remove the guard
/// in `convex_candidate_wins` and this reports `cvx-qp` /
/// `MaximumIterationsExceeded`.
#[test]
fn a_failed_convex_answer_never_displaces_an_nlp_success() {
    for leg in [&[][..], &[LBFGS][..]] {
        let mut args = vec!["tol=1e-16"];
        args.extend_from_slice(leg);
        let r = run("lp_israel.nl", &args);
        let fb = r
            .report
            .convex_fallback
            .as_ref()
            .expect("fallback recorded");
        assert_eq!(fb.convex_status, "Maximum_Iterations_Exceeded", "{leg:?}");
        assert_eq!(fb.nlp_status, "Solved_To_Acceptable_Level", "{leg:?}");
        let nlp = fb.nlp_kkt_error.expect("the NLP arm produced a point");
        assert!(
            fb.convex_kkt_error * 10.0 < nlp,
            "{leg:?}: precondition — the convex point must be the more \
             accurate, or this test does not isolate the guard; {fb:?}"
        );
        assert_eq!(fb.winner, "nlp", "{leg:?}: {fb:?}");
        assert_eq!(r.report.solution.engine, "nlp", "{leg:?}");
        assert_eq!(
            r.report.solution.status,
            ApplicationReturnStatus::SolvedToAcceptableLevel,
            "{leg:?}"
        );
        assert_eq!(r.code, Some(0), "{leg:?}: the run still exits successfully");
    }
}

/// Both candidates fail (the MIPLIB toguru shape): accuracy still decides, and
/// the winner reports under its *own* failing status — never borrowing the
/// other arm's, never upgraded to a success.
#[test]
fn when_both_fail_the_verdict_stays_honest() {
    let r = run("lp_afiro.nl", &["tol=1e-20"]);
    let fb = r
        .report
        .convex_fallback
        .as_ref()
        .expect("fallback recorded");
    assert_eq!(fb.convex_status, "Maximum_Iterations_Exceeded", "{fb:?}");
    assert_ne!(fb.nlp_status, "Solve_Succeeded", "precondition: {fb:?}");
    assert_ne!(
        fb.nlp_status, "Solved_To_Acceptable_Level",
        "precondition: {fb:?}"
    );
    assert_eq!(fb.winner, "cvx-qp", "{fb:?}");

    assert_eq!(r.report.solution.engine, "cvx-qp");
    assert_eq!(
        r.report.solution.status,
        ApplicationReturnStatus::MaximumIterationsExceeded,
        "the winner's own status, not the other arm's"
    );
    assert_eq!(r.report.solution.solve_result_num, 400);
    assert_eq!(
        status_lines(&r.stdout),
        ["Status: Maximum_Iterations_Exceeded"],
        "stdout=\n{}",
        r.stdout
    );
}

/// Every run that reaches the fallback says, on stderr, what it compared and
/// which answer it kept — the console counterpart of `convex_fallback`.
#[test]
fn the_comparison_is_announced() {
    let r = run("lp_etamacro.nl", &["tol=5e-13", "relative_gap_tol=0"]);
    let line = r
        .stderr
        .lines()
        .find(|l| l.contains("LP/QP fallback (gh #1183)"))
        .unwrap_or_else(|| panic!("no comparison note; stderr=\n{}", r.stderr));
    assert!(line.contains("reporting the convex answer"), "{line}");
    assert!(
        line.contains("Solve_Succeeded"),
        "names the NLP status: {line}"
    );
}

/// A solve that never reaches the fallback carries no `convex_fallback`
/// section at all, so its absence reads as "no comparison ran".
#[test]
fn an_ordinary_solve_records_no_comparison() {
    let r = run("lp_afiro.nl", &[]);
    assert_eq!(r.report.solution.engine, "cvx-qp");
    assert!(r.report.convex_fallback.is_none());
}
