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
//! | fixture | convex | NLP | reports |
//! |---|---|---|---|
//! | `lp_etamacro.nl` (NETLIB ETAMACRO) | `7.6e-11`, acceptable | `1.1e-8`, succeeded | convex |
//! | `lp_capri.nl` (NETLIB CAPRI) | `7.6e-9`, acceptable | `2.7e-10`, succeeded | NLP — the gh #535 guard |
//! | `lp_afiro.nl` at `tol=1e-20` | `2.4e-16`, iteration limit | `5.5e-13`, search direction too small | convex, with its own failing status |
//!
//! The margins are a decade or more on purpose (NETLIB SHARE2B also reaches
//! the fallback, at `4.82e-10` against `4.88e-10`, and is not used). The
//! remaining branch — a convex answer whose own status is a failure never
//! displaces an NLP *success* — has no fixture that reaches it and is pinned
//! by the unit tests beside `convex_candidate_wins`.
//!
//! Both fixtures were produced by `benchmarks/lp/generate_nl.py` from the
//! NETLIB MPS files and are the only two of 42 small NETLIB models swept that
//! reach the fallback with a clear winner. Each takes well under a second.

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
        let r = run("lp_etamacro.nl", leg);
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
        // The NLP arm *succeeded* by its own criteria; the convex answer
        // replaces it on accuracy, which is the whole point of the issue.
        assert_eq!(fb.nlp_status, "Solve_Succeeded", "{leg:?}");

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
            ((obj - -755.71521774) / 755.71521774).abs() < 1e-6,
            "{leg:?}: objective {obj}"
        );
    }
}

/// gh #535's case, and the guard against breaking it: when the NLP answer is
/// the more accurate, it still reports — and the convex candidate it beat is
/// recorded, not printed.
#[test]
fn a_more_accurate_nlp_answer_still_reports() {
    for leg in [&[][..], &[LBFGS][..]] {
        let r = run("lp_capri.nl", leg);
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
        let nlp = fb.nlp_kkt_error.expect("the NLP arm produced a point");
        assert!(
            nlp * 10.0 < fb.convex_kkt_error,
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
        // NETLIB CAPRI's published optimum.
        let obj = r.report.solution.objective;
        assert!(
            ((obj - 2690.0129138) / 2690.0129138).abs() < 1e-6,
            "{leg:?}: objective {obj}"
        );
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
    let r = run("lp_etamacro.nl", &[]);
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
