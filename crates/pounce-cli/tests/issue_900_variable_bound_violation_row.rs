//! gh#900: the `Variable bound violation` row reports a measurement, and the
//! POUNCE-only declared-model line is styled so it is not read past.
//!
//! Both console printers hardcoded `0.0` on that row
//! (`pounce-solve-report/src/console.rs`), so it read `0.00e+00` on every
//! solve — including the ones where the returned point genuinely sits outside
//! the box the caller wrote. It is the row a reader is told to check, and
//! upstream Ipopt is the only place the number was available.
//!
//! The fixture is the smallest model on which the answer is not a matter of
//! opinion:
//!
//! ```text
//! min 1e8*x + 0.5*x^2   s.t.  x >= 0
//! ```
//!
//! `f'(x) = 1e8 + x > 0` everywhere feasible, so `x* = 0` and `f* = 0`,
//! uniquely — the objective is strictly convex, so there is no second
//! minimum to have found instead. Widen the bound by `1e-8` and the solve
//! returns `f ≈ -1`: a value for a quantity that cannot go below zero, under
//! `EXIT: Optimal Solution Found`. The multiplier at the bound is `1e8` and
//! the shift is `δ · λ`, which is the LISWET/YAO family of
//! `benchmarks/BENCHMARK_REPORT.md` in one variable.
//!
//! Ipopt 3.14.20/MA57 on this model returns the same objective and prints
//! `9.9999090909090909e-09` on that row; POUNCE's NLP arm now prints the same
//! number to every digit the assertions below can portably pin. That is not
//! asserted exactly: the value is `δ` minus the barrier's last
//! fraction-to-boundary step, so it is trajectory-dependent and a bit-for-bit
//! pin would fail for a reason that has nothing to do with this row.
//!
//! **Which branch each case takes.** The row is computed by two independent
//! code paths — `IpoptCalculatedQuantities::curr_declared_box_violation_max`
//! on the NLP arm, `QpResiduals::bound_violation` against a
//! `BoundRelax::NONE` re-extraction on the convex one — so a test that
//! exercised one would say nothing about the other. Every case below is run
//! on both, at both settings of the widening, which is four corners: a
//! nonzero that must be reported and a zero that must not be fabricated, per
//! arm.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use pounce_cli::solve_report::SolveReport;

/// `bound_relax_factor` at Ipopt's default. The convex arm no longer applies
/// it unless asked (gh#744/#745), so a test about the widening names it.
const RELAX: &str = "bound_relax_factor=1e-8";

fn pounce_exe() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pounce"))
}

fn fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/bound_relax_cliff.nl");
    p
}

fn tmp_path(suffix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pounce_gh900_{}_{}_{suffix}",
        std::process::id(),
        n
    ));
    p
}

struct Run {
    stdout: String,
    report: SolveReport,
}

/// Solve the cliff fixture on `engine`, capturing both the console block and
/// the JSON report so the two can be held against each other.
fn run(engine: &str, extra: &[&str], color: bool) -> Run {
    let json_path = tmp_path("report.json");
    let sol_path = tmp_path("out.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(fixture())
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path)
        .arg(format!("solver_selection={engine}"));
    for o in extra {
        cmd.arg(o);
    }
    if color {
        // The test harness captures stdout through a pipe, so `anstream`
        // strips the styling by default — which is the behaviour the plain
        // cases below rely on. Force it back on to see the styled bytes, and
        // clear `NO_COLOR`, which wins over the force and may be set in CI.
        cmd.env("CLICOLOR_FORCE", "1");
        cmd.env_remove("NO_COLOR");
    } else {
        cmd.env_remove("CLICOLOR_FORCE");
    }
    let out = cmd.output().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        report: serde_json::from_str(&text).expect("deserialize SolveReport"),
    }
}

/// The two columns of a summary row, by label.
fn row(stdout: &str, label: &str) -> (f64, f64) {
    let line = stdout
        .lines()
        .find(|l| l.starts_with(label))
        .unwrap_or_else(|| panic!("no `{label}` row in:\n{stdout}"));
    let rest = line.split_once(':').expect("labelled row").1;
    let mut it = rest.split_whitespace();
    let parse = |s: Option<&str>| {
        s.unwrap_or_else(|| panic!("missing column in `{line}`"))
            .parse::<f64>()
            .unwrap_or_else(|e| panic!("unparsable column in `{line}`: {e}"))
    };
    (parse(it.next()), parse(it.next()))
}

fn bound_violation_row(stdout: &str) -> (f64, f64) {
    row(stdout, "Variable bound violation")
}

/// One widening, `min(1e-8 * max(|0|, 1), 1e-4)`.
const DELTA: f64 = 1e-8;

// ── the widened corner: a number, where there used to be a zero ──────────────

#[test]
fn the_nlp_arm_reports_the_box_violation_it_incurred() {
    // This arm widens by default: a feasible-iterate log-barrier needs `x`
    // strictly inside its bounds, so the widening is not opt-in here.
    let r = run("nlp", &[], false);
    let (scaled, unscaled) = bound_violation_row(&r.stdout);
    assert!(
        (scaled - DELTA).abs() <= 0.01 * DELTA,
        "the point sits one widening outside the declared bound `x >= 0`, so \
         the row should read ~{DELTA:e}; got {scaled:e}. A `0.0` here is the \
         gh#900 defect."
    );
    // Variable bounds carry no scaling — POUNCE scales the objective and the
    // constraint rows only — so the one measurement is right in both columns
    // rather than one of them being a placeholder.
    assert_eq!(
        scaled, unscaled,
        "the box violation has no scaled/unscaled distinction"
    );
    // What the row is evidence *of*: an objective below a floor of zero.
    let obj = r.report.solution.objective;
    assert!(
        obj < -0.9,
        "the widened bound should buy `δ · λ = 1e-8 · 1e8 ≈ 1` of objective \
         on a problem whose true minimum is 0; got {obj:e}. Without that this \
         fixture has stopped exercising the gap."
    );
}

#[test]
fn the_convex_arm_reports_the_box_violation_it_incurred() {
    let r = run("auto", &[RELAX], false);
    assert_eq!(
        r.report.solution.engine, "cvx-qp",
        "this case exists to cover the convex printer's own path to the row; \
         if `auto` stops routing here it is covering the NLP one twice"
    );
    let (scaled, unscaled) = bound_violation_row(&r.stdout);
    assert!(
        (scaled - DELTA).abs() <= 0.01 * DELTA,
        "expected ~{DELTA:e} on the convex arm too; got {scaled:e}"
    );
    assert_eq!(scaled, unscaled);
    let obj = r.report.solution.objective;
    assert!(obj < -0.9, "expected an objective near -1; got {obj:e}");
}

// ── the unwidened corner: a zero, and not a fabricated one ───────────────────
//
// The opposite failure to gh#900 and just as bad: a row that always reports
// something would be as uninformative as one that always reports nothing.

#[test]
fn the_nlp_arm_reports_zero_when_it_did_not_widen() {
    let r = run("nlp", &["bound_relax_factor=0"], false);
    let (scaled, _) = bound_violation_row(&r.stdout);
    assert_eq!(
        scaled, 0.0,
        "with no widening the point is inside its declared box and the row \
         must say so; got {scaled:e}"
    );
    // The answer is the analytic one from the correct side. A converged
    // barrier iterate stops strictly inside the box, so `x > 0` and hence
    // `f > 0` by a barrier-sized margin — measured `9.1e-6`, i.e. `x` about
    // `9.1e-14` times the `1e8` slope. Pinning `|f| < 1e-6` would be pinning
    // the final `mu`, which is trajectory-dependent and not what this test is
    // about. What is about this test is the *sign*: `f* = 0` is a floor the
    // unwidened solve respects and the widened one is five orders below.
    let obj = r.report.solution.objective;
    assert!(
        (0.0..1e-3).contains(&obj),
        "the unwidened answer must approach `f* = 0` from inside the box, so \
         `0 <= f << 1`; got {obj:e}"
    );
}

#[test]
fn the_convex_arm_reports_zero_at_its_default() {
    // The convex arm's default is no widening (gh#744/#745), so this is what
    // an ordinary `pounce model.nl` prints.
    let r = run("auto", &[], false);
    let (scaled, _) = bound_violation_row(&r.stdout);
    assert_eq!(scaled, 0.0, "expected an unviolated box; got {scaled:e}");
    // Same floor as the NLP arm above, reached far more tightly: the convex
    // interior-point run lands at `f = 2.8e-12`.
    let obj = r.report.solution.objective;
    assert!(
        (0.0..1e-3).contains(&obj),
        "the analytic answer `f* = 0`, approached from inside; got {obj:e}"
    );
}

// ── the console and the JSON report agree ────────────────────────────────────

#[test]
fn the_json_report_carries_the_same_number_as_the_row() {
    for (engine, extra) in [("nlp", &[][..]), ("auto", &[RELAX][..])] {
        let r = run(engine, extra, false);
        let (printed, _) = bound_violation_row(&r.stdout);
        let reported = r.report.statistics.final_declared_box_viol;
        // Not `assert_eq!`. The two agree bit-for-bit in the file — the
        // console renders the `f64` at 17 significant digits and `serde_json`
        // writes the shortest round-tripping form of the same bits — but
        // `serde_json` is built here without its `float_roundtrip` feature, so
        // *parsing* the report back can land an ulp away. That is the test
        // harness's arithmetic, not the solver's, and pinning equality here
        // would be pinning a dependency's default. What the row and the field
        // must be is one measurement, which a few ulps says and an exact
        // compare would overstate.
        let tol = 8.0 * f64::EPSILON * reported.abs().max(printed.abs());
        assert!(
            (printed - reported).abs() <= tol,
            "`final_declared_box_viol` and the printed row must be one \
             measurement, not two, on the {engine} arm: {reported:e} vs \
             {printed:e}"
        );
        assert!(
            reported > 0.0,
            "and both must be the widening, not a zero, on the {engine} arm"
        );
    }
}

// ── the styling ──────────────────────────────────────────────────────────────

/// The declared-model line is bold red on a terminal.
#[test]
fn the_declared_violation_line_is_styled_when_color_is_on() {
    let r = run("nlp", &[], true);
    let line = r
        .stdout
        .lines()
        .find(|l| l.contains("Violation of the model as declared"))
        .unwrap_or_else(|| panic!("no declared-violation line in:\n{}", r.stdout));
    assert!(
        line.contains("\u{1b}[1m") && line.contains("\u{1b}[31m"),
        "expected bold red; got {line:?}"
    );
    assert!(
        line.ends_with("\u{1b}[0m"),
        "and a reset at the end of the line; got {line:?}"
    );
}

/// ...and plain text everywhere else, so redirected logs, the benchmark
/// harness's stdout scrapes and every `assert!(stdout.contains(..))` in the
/// suite are byte-for-byte unaffected. `anstream` owes this to `NO_COLOR` and
/// to a non-TTY sink; the test harness's pipe is the non-TTY sink.
#[test]
fn nothing_is_styled_when_stdout_is_not_a_terminal() {
    let r = run("nlp", &[], false);
    assert!(
        r.stdout.contains(
            "Violation of the model as declared (before the bound_relax_factor widening):"
        ),
        "the line itself must still be there, unstyled:\n{}",
        r.stdout
    );
    assert!(
        !r.stdout.contains('\u{1b}'),
        "no escape byte may reach a redirected stdout"
    );
}

// ── the active-set SQP arm ───────────────────────────────────────────────────
//
// The third code path, and the one this file's header did not know about: it
// printed `nan` on that row for every solve while the two arms above printed
// measurements. `IpoptApplication::optimize_sqp_tnlp` never called
// `relax_bounds` — correctly, because this arm applies no widening — and the
// declared-box snapshot that the row is computed from was taken inside it.
//
// The cliff fixture cannot carry these: it is a convex QP, so `auto` routes it
// to pounce-convex and `algorithm=active-set-sqp` never reaches the SQP driver
// at all. A genuine NLP is needed to exercise this arm.

/// `hs71_obj1e8.nl` — a real NLP (so it reaches the SQP driver) with a finite
/// lower and upper bound on every variable, so the box is not vacuous.
fn nlp_fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/hs71_obj1e8.nl");
    p
}

fn run_sqp(extra: &[&str]) -> Run {
    let json_path = tmp_path("sqp_report.json");
    let sol_path = tmp_path("sqp_out.sol");
    let mut cmd = Command::new(pounce_exe());
    cmd.arg(nlp_fixture())
        .arg(&sol_path)
        .arg("--json-output")
        .arg(&json_path)
        .arg("algorithm=active-set-sqp");
    for o in extra {
        cmd.arg(o);
    }
    cmd.env_remove("CLICOLOR_FORCE");
    let out = cmd.output().expect("spawn pounce");
    let text = std::fs::read_to_string(&json_path).expect("read json report");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&sol_path);
    Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        report: serde_json::from_str(&text).expect("deserialize SolveReport"),
    }
}

/// The row carries a number on this arm too. `nan` is the defect: it is not a
/// measurement, and it is what the row read on every active-set SQP solve.
#[test]
fn the_sqp_arm_reports_a_number_on_the_bound_violation_row() {
    let r = run_sqp(&[]);
    // Confirm the SQP driver actually ran — this fixture must not quietly
    // route elsewhere, or the test measures the wrong arm.
    assert_eq!(
        r.report.solution.engine, "sqp-active-set",
        "expected the active-set SQP arm; got {:?}",
        r.report.solution.engine
    );
    let (scaled, unscaled) = bound_violation_row(&r.stdout);
    assert!(
        scaled.is_finite() && unscaled.is_finite(),
        "the row must be a measurement, not `nan`: {scaled} / {unscaled}"
    );
    assert!(
        r.report.statistics.final_declared_box_viol.is_finite(),
        "and the JSON field too: {}",
        r.report.statistics.final_declared_box_viol
    );
    let tol = 8.0 * f64::EPSILON * scaled.abs().max(1.0);
    assert!(
        (scaled - r.report.statistics.final_declared_box_viol).abs() <= tol,
        "row and field must be one measurement: {scaled:e} vs {:e}",
        r.report.statistics.final_declared_box_viol
    );
}

/// Why that number is zero, stated as a contrast rather than left to be
/// assumed: this arm applies no `bound_relax_factor` widening, so its declared
/// box IS the box it solves against and the answer cannot land outside it.
///
/// The same option on the interior-point arm moves the row off zero — that is
/// `the_row_reports_the_widening_on_the_nlp_arm` above. Running both here
/// makes the SQP zero a statement about this arm rather than a number nobody
/// checked. That the zero is *measured* and not hardcoded is pinned where the
/// point can be placed by hand, in `pounce-nlp`'s
/// `the_box_violation_measures_the_distance_outside_the_declared_box`.
#[test]
fn the_sqp_arm_does_not_widen_so_its_bound_violation_stays_zero() {
    for extra in [&[][..], &[RELAX][..]] {
        let r = run_sqp(extra);
        let (scaled, _) = bound_violation_row(&r.stdout);
        assert_eq!(
            scaled, 0.0,
            "the active-set arm applies no widening, at {extra:?}, so the \
             returned point is inside the box the caller wrote; got {scaled:e}"
        );
    }
}

/// The residual table stays free of styling even with color forced on. It is
/// diffed against `ipopt`'s own output byte-for-byte, which is the reason the
/// styling went on the POUNCE-only line instead of the row that would most
/// obviously carry it.
#[test]
fn the_upstream_compatible_residual_table_is_never_styled() {
    let r = run("nlp", &[], true);
    for label in [
        "Objective...............",
        "Dual infeasibility......",
        "Constraint violation....",
        "Variable bound violation",
        "Complementarity.........",
        "Overall NLP error.......",
    ] {
        let line = r
            .stdout
            .lines()
            .find(|l| l.starts_with(label))
            .unwrap_or_else(|| panic!("no `{label}` row in:\n{}", r.stdout));
        assert!(
            !line.contains('\u{1b}'),
            "`{label}` must stay byte-compatible with upstream's block; got \
             {line:?}"
        );
    }
}
