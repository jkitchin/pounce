//! gh #767: the convex path must present the same end-of-run contract as the
//! NLP path — timing statistics when asked for them, an `EXIT:` banner and a
//! machine-readable `Status:` line, and a JSON `solution` a consumer can
//! compare against Ipopt's own enumerator spelling.
//!
//! All three gaps were found while instrumenting the Mittelmann suite, where
//! two of 47 instances route convex and the other 45 do not:
//!
//! 1. `print_timing_statistics=yes` was accepted, reported `(used)` by
//!    `print_user_options`, and emitted nothing. A tool attributing solve cost
//!    by phase then read 0% for *every* phase of a convex-routed instance,
//!    which reads as "already fast" rather than "not measured" — `bearing_400`
//!    ran 9.8 s and printed no timer at all.
//! 2. The log ended after the residual block: no `EXIT:`, no `Status:`. Every
//!    consumer of the CLI other than `benchmarks/scripts/run_nl_bench.sh` — which
//!    carries its own ladder of convex-specific stdout scrapes — had to
//!    reimplement that ladder.
//! 3. The JSON report's `solution.status` is the Rust variant name
//!    (`SolveSucceeded`) on *both* paths, while the spelling every Ipopt-facing
//!    consumer keys off is `Solve_Succeeded`. A consumer comparing against the
//!    latter matched nothing and silently classified every solve as a failure,
//!    and only noticed on the convex path because that path printed no
//!    `Status:` line to compare with. The report now carries both spellings.
//!
//! The tests run every convex arm the router can reach — the QP/LP
//! interior-point engine, the parametric active-set engine, and the conic
//! (QCQP) engine — because each has its own driver and its own linear algebra,
//! and a green leg on one says nothing about the others.

use std::path::PathBuf;
use std::process::Command;

use pounce_cli::solve_report::SolveReport;

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

fn run(name: &str, args: &[&str]) -> String {
    let out = Command::new(pounce_exe())
        .arg(fixture(name))
        .arg("--no-sol")
        .args(args)
        .output()
        .expect("spawn pounce");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Solve `name` and return `(stdout, report)`.
fn run_with_report(name: &str, args: &[&str]) -> (String, SolveReport) {
    let json = std::env::temp_dir().join(format!(
        "pounce_767_{}_{}_{}.json",
        std::process::id(),
        name.replace('.', "-"),
        args.join("_").replace(['=', '.', '/'], "-")
    ));
    let out = Command::new(pounce_exe())
        .arg(fixture(name))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .args(args)
        .output()
        .expect("spawn pounce");
    let report: SolveReport =
        serde_json::from_str(&std::fs::read_to_string(&json).expect("read JSON report"))
            .expect("parse JSON report");
    let _ = std::fs::remove_file(&json);
    (String::from_utf8_lossy(&out.stdout).into_owned(), report)
}

fn status_lines(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .filter(|l| l.starts_with("Status:"))
        .collect()
}

/// The convex arms, one fixture each. `solver_selection` is pinned rather than
/// left at `auto` so a routing change cannot quietly turn one of these legs
/// into a second copy of another.
const CONVEX_ARMS: &[(&str, &str, &str)] = &[
    (
        "lp_afiro.nl",
        "solver_selection=qp-ipm",
        "LP, interior-point",
    ),
    (
        "convex_qp_share1b.nl",
        "solver_selection=qp-ipm",
        "convex QP, interior-point",
    ),
    (
        "boxed_qp_min.nl",
        "solver_selection=qp-active-set",
        "convex QP, active-set",
    ),
    (
        "qcqp_ball.nl",
        "solver_selection=socp",
        "convex QCQP, conic",
    ),
];

/// Item 2. Every convex arm ends its log the way the NLP path does.
#[test]
fn every_convex_arm_ends_with_an_exit_banner_and_a_status_line() {
    for (fixture, selection, arm) in CONVEX_ARMS {
        let stdout = run(fixture, &[selection]);
        assert!(
            stdout.contains("EXIT: Optimal Solution Found."),
            "{arm}: no EXIT: banner; stdout=\n{stdout}"
        );
        assert_eq!(
            status_lines(&stdout),
            vec!["Status: Solve_Succeeded"],
            "{arm}: expected exactly one machine-readable status line; stdout=\n{stdout}"
        );
    }
}

/// Item 3, and the half of item 2 that matters to a machine: the spelling on
/// the `Status:` line and the spelling in the JSON report agree, on both
/// paths, so a consumer can compare one literal against either surface.
#[test]
fn the_printed_status_and_the_json_report_agree_on_both_paths() {
    // `hs13_bigstart.nl` is an NLP; the others route convex.
    for (fixture, selection) in [
        ("hs13_bigstart.nl", "solver_selection=nlp"),
        ("convex_qp_share1b.nl", "solver_selection=qp-ipm"),
        ("qcqp_ball.nl", "solver_selection=socp"),
    ] {
        let (stdout, report) = run_with_report(fixture, &[selection]);
        let printed = status_lines(&stdout)
            .first()
            .map(|l| l.trim_start_matches("Status:").trim().to_string())
            .unwrap_or_else(|| panic!("{fixture}: no Status: line; stdout=\n{stdout}"));
        assert_eq!(
            report.solution.status_upstream, printed,
            "{fixture}: the report's upstream status must be the printed one"
        );
        assert_eq!(
            printed, "Solve_Succeeded",
            "{fixture}: expected the upstream spelling, not the Rust variant name"
        );
        // The pre-existing field keeps its documented meaning — the Rust
        // variant name — so a consumer reading it does not break.
        assert_eq!(format!("{:?}", report.solution.status), "SolveSucceeded");
    }
}

/// Item 1. `print_timing_statistics=yes` produces a phase breakdown on every
/// convex arm, and the phases nest the way the labels claim — a block of
/// structural zeros, or one whose parts exceed their whole, would be the same
/// defect with more output.
///
/// The magnitude claim is made only where the report can resolve it: rows
/// print three decimals, so a sub-millisecond fixture legitimately shows
/// `0.000s` everywhere and asserting otherwise would flake on a fast machine.
/// That the instrumentation exists at all is pinned deterministically, per
/// engine, by `pounce-linsol`'s `a_convex_timing_scope_charges_each_phase_to_
/// its_own_row` and `pounce-qp`'s `tests/convex_timing_rows.rs`, which read
/// the raw nanosecond totals instead of the printed rows.
#[test]
fn print_timing_statistics_attributes_the_solve_on_every_convex_arm() {
    /// Ten times the report's display quantum: below this a zero row says
    /// "too fast to print", above it a zero row says "never measured".
    const RESOLVABLE: f64 = 0.005;

    for (fixture, selection, arm) in CONVEX_ARMS {
        let stdout = run(fixture, &[selection, "print_timing_statistics=yes"]);
        assert!(
            stdout.contains("\nTiming Statistics:\n"),
            "{arm}: no timing block; stdout=\n{stdout}"
        );

        // The dot leader is part of the match: `Presolve` alone also prefixes
        // the reduction line presolve logs above the summary.
        let seconds = |label: &str| -> f64 {
            let leader = format!("{label}.");
            let row = stdout
                .lines()
                .find(|l| l.trim_start().starts_with(&leader))
                .unwrap_or_else(|| panic!("{arm}: no `{label}` row; stdout=\n{stdout}"));
            row.rsplit_once(' ')
                .and_then(|(_, v)| v.trim().trim_end_matches('s').parse::<f64>().ok())
                .unwrap_or_else(|| panic!("{arm}: unparsable row {row:?}"))
        };

        let overall = seconds("OverallAlgorithm");
        let solve = seconds("ConvexSolve");
        // Every driver stage has a row, whether or not it did anything.
        for stage in ["ProblemExtraction", "Presolve", "SolutionRecovery"] {
            let _ = seconds(stage);
        }
        // The linear-algebra split is the reason a phase report is worth
        // printing, and it is engine-specific: the interior-point drivers
        // factor through `pounce_linsol::Factorization`, the active-set engine
        // through `pounce_qp`'s own `LinearSolver`.
        let linear = seconds("LinearSystemSymbolicFactorization")
            + seconds("LinearSystemFactorization")
            + seconds("LinearSystemBackSolve");

        // Rounding can only move a row by half a quantum in each direction.
        const SLACK: f64 = 0.001;
        assert!(
            solve <= overall + SLACK,
            "{arm}: ConvexSolve {solve}s exceeds OverallAlgorithm {overall}s"
        );
        // The linear algebra happens inside the solve or, on the conic QCQP
        // arm, inside SolutionRecovery too: the gh#1114 polish (PR #1120)
        // factors the original KKT system after the conic solve, under the
        // recovery timer. Its factorizations are charged to the LinearSystem*
        // rows correctly, so on a slow machine (CI: 0.004s against a 0.002s
        // solve) they exceed `ConvexSolve` alone — the bound is the two
        // stages they can happen in.
        let recovery = seconds("SolutionRecovery");
        assert!(
            linear <= solve + recovery + SLACK,
            "{arm}: the LinearSystem* rows ({linear}s) exceed the stages they \
             happen inside (ConvexSolve {solve}s + SolutionRecovery \
             {recovery}s)"
        );

        if solve >= RESOLVABLE {
            assert!(
                overall > 0.0,
                "{arm}: OverallAlgorithm read {overall}s — the option must \
                 measure something"
            );
            assert!(
                linear > 0.0,
                "{arm}: the LinearSystem* rows sum to {linear}s beside a \
                 {solve}s solve — an uninstrumented engine reports 0% for every \
                 phase, which reads as `already fast` rather than `not \
                 measured` (gh #767)"
            );
        }
    }
}

/// The block is opt-in: a default solve keeps its old output.
#[test]
fn the_timing_block_is_absent_without_the_option() {
    let stdout = run("convex_qp_share1b.nl", &["solver_selection=qp-ipm"]);
    assert!(
        !stdout.contains("Timing Statistics:"),
        "timing must stay opt-in; stdout=\n{stdout}"
    );
    // …but the verdict is not opt-in.
    assert!(stdout.contains("EXIT:"), "stdout=\n{stdout}");
}

/// `timing_statistics=yes` alone arms the detailed timers without printing
/// them, exactly as it does on the NLP path (`print_timing_statistics`
/// implies it, not the other way round).
#[test]
fn timing_statistics_alone_does_not_print() {
    let stdout = run(
        "convex_qp_share1b.nl",
        &["solver_selection=qp-ipm", "timing_statistics=yes"],
    );
    assert!(
        !stdout.contains("Timing Statistics:"),
        "only print_timing_statistics prints; stdout=\n{stdout}"
    );
}

/// The verdict block is gated on `print_level` the same way
/// `Application::emit_end_summary` gates the NLP path's: at print_level 0 the
/// console is silent by request, and adding a status line would break that.
#[test]
fn print_level_zero_suppresses_the_convex_verdict() {
    let stdout = run(
        "convex_qp_share1b.nl",
        &["solver_selection=qp-ipm", "print_level=0"],
    );
    assert!(
        !stdout.contains("EXIT:") && status_lines(&stdout).is_empty(),
        "print_level=0 must not print a verdict; stdout=\n{stdout}"
    );
}

/// `--debug-json` makes stdout a pure protocol channel, so the machine-readable
/// status line stays off it — the same carve-out the NLP path makes.
#[test]
fn json_debug_keeps_the_status_line_off_stdout() {
    let stdout = run("convex_qp.nl", &["--debug-json"]);
    assert!(
        status_lines(&stdout).is_empty(),
        "--debug-json owns stdout; stdout=\n{stdout}"
    );
}

/// One run, one verdict. gh #535 hands an uncertified LP back to the NLP path,
/// and the convex attempt that declined must not have left a status line
/// behind — a log with two `Status:` lines is exactly the ambiguity the
/// machine-readable line exists to remove.
#[test]
fn a_convex_attempt_that_declines_leaves_no_status_line() {
    let stdout = run("lp_afiro.nl", &["solver_selection=auto", "tol=1e-20"]);
    assert_eq!(
        status_lines(&stdout).len(),
        1,
        "a rerouted run must print exactly one status line; stdout=\n{stdout}"
    );
}
