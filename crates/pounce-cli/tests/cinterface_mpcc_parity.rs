//! CLI parity of declared complementarity through the C API (gh#1147 S1-I8).
//!
//! Every model in `tests/fixtures_mpcc/` is solved three ways and the three
//! are held to one another:
//!
//! * the CLI binary on the `.nl` file (`pounce <fixture> --json-output`);
//! * the C API (`CreateIpoptProblem` + `SetPounceComplementarityPairs` +
//!   `IpoptSolve` + `IpoptWriteSolveReport`) with the `.nl` model's own
//!   evaluator, `NlTnlp`, behind its five callbacks (`common/nl_c_harness.rs`);
//! * Python's `Problem(..., complementarity=...)` on the same evaluator
//!   through `pounce.read_nl` (`common/cinterface_mpcc_python.json`, from
//!   `common/gen_cinterface_mpcc_python.py`).
//!
//! **What is compared.** `solution.status`, `solution.objective` and
//! `solution.x` **to the bit**, `statistics.iteration_count`, and the whole
//! `complementarity` report block except the wall-clock `seconds` of each
//! lowered solve — strategy, reported, accepted_by, fallback,
//! total_iterations, every lowered solve's stage / tau / status / iterations
//! / product, the certificate, the B-check, escapes, lifted, polish and the
//! summary text. The block is one `MpccOutcome::report_info()` in all three
//! frontends, so equal blocks mean the same driver run.
//!
//! **Why exact iteration parity holds, and the one trajectory it does not.**
//! Before this test the C API had only been compared with the CLI through
//! hand-written callbacks (`pounce-cinterface/tests/common/qpec.rs`, the
//! `QpecSmall` of `python/tests/test_complementarity.py`), and on
//! `qpec_small` those took 139 iterations against the CLI's 191 while C and
//! Python agreed bit for bit. Bisected (§9 S1-I8 of
//! `dev-notes/issue-1147-c-api-capabilities-plan.md`), the cause is the
//! model, not the frontend: AMPL writes a row's constant into its bound, so
//! the `.nl` reader hands the solver `2 y1 - x >= 1`, where the hand-written
//! model says `2 y1 - 1 - x >= 0`. The two are the same set; their `g`, and
//! the products the driver forms from it, differ in the last bits, and the
//! third Scholtes solve of this degenerate model goes 47 iterations one way
//! and 24 the other. With the `.nl` evaluator behind the callbacks the C API
//! reproduces the CLI exactly on every fixture and option set below; no
//! option the CLI sets, and no optional TNLP answer (scaling suffixes,
//! linearity, derivative proofs, names), moves an exact-Hessian run.
//!
//! The exception is the **limited-memory Hessian**, which is restricted to
//! the variables the model declares nonlinear. `NlTnlp` reports that list
//! from the `.nl` header; a callback model has no way to know it unless the
//! caller says so with `IpoptSetNonlinearVariables`. The `lbfgs_masked_*`
//! rows pass the `.nl` list through and are exact against the CLI; the
//! `lbfgs_unmasked_*` rows do not, and are held to the CLI on the answer only
//! (and to Python — which has no such declaration — exactly). `qpec_small`
//! is nonlinear in every variable, so its limited-memory row is exact
//! without the list.
//!
//! **The other difference the C side cannot express: the objective sense.**
//! `NlTnlp` negates a maximize objective, so its callbacks — and therefore
//! the C API and Python — minimize `-f` and report `-f*`; the CLI reports in
//! the model's sense. `design_cent_3` is a maximize model; its C objective is
//! the CLI's negated, bit for bit.
//!
//! **Python parity is pinned on Linux x86_64 only.** The CLI and the C API
//! run on the same machine in this test, so `assert_cli_exact` holds to the
//! bit on every platform. The Python side is a checked-in reference,
//! generated on Linux x86_64, and the last bits of a real depend on the
//! platform's math library: on macOS arm64 four rows differ only in
//! `certificate.residual` at round-off (`2.3276935934291032e-12` against
//! `2.327249504219253e-12`), and the unmasked limited-memory
//! `opf_vv_lower_knee` row ends `SolvedToAcceptableLevel` instead of
//! `SolveSucceeded`. That is the same sensitivity that keeps the unmasked
//! limited-memory rows off exact parity with the CLI, showing up across
//! platforms instead of across frontends. So `assert_python_exact` compares
//! only where the reference was made, and says so on stderr elsewhere; to pin
//! another platform, regenerate the reference there
//! (`common/gen_cinterface_mpcc_python.py`) rather than loosening the check.
//!
//! **Branches reached** (CLAUDE.md, "A test is only evidence about the branch
//! it reaches"): certificate classes `S` (`qpec_small`, `design_cent_3`,
//! `opf_vv_lower_knee_held`), `W` (`ralph1`, `mixed_pair`,
//! `opf_vv_lower_knee`) and `none` (`infeasible_pair`); reported `answer`,
//! `weak_fallback` and `verdict`; polish taken (`qpec_small`,
//! `opf_vv_lower_knee`), declined with a reason (`opf_vv_lower_knee_held`)
//! and not run (`ralph1`); the B-check (`ralph1`); a lifted row at the
//! default (`design_cent_3`) and on request (`qpec_small`, `lift=yes`);
//! strategies `both`, `exact`, `continuation`; the coupled continuation and
//! the schedule. `not searched` is reached by no fixture (it needs more
//! sign-constrained columns than `MAX_SEARCHED_COLUMNS`); the C getters'
//! test reaches it through callbacks
//! (`pounce-cinterface/tests/complementarity_getters.rs`). Each row asserts
//! the branch it is listed for, so a fixture that drifts off its branch
//! fails here rather than silently testing another.

// Test code: a panicking unwrap/expect is the test failing, as intended.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/nl_c_harness.rs"]
mod nl_c_harness;

use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

const LBFGS: &str = "hessian_approximation=limited-memory";

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures_mpcc")
        .join(format!("{name}.nl"))
}

/// A fresh directory per call: the tests run in parallel.
fn scratch(name: &str) -> PathBuf {
    static CALL: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "pounce-cc-parity-{name}-{}-{}",
        std::process::id(),
        CALL.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The CLI's report on `name` with `options` (`key=value`).
fn cli(name: &str, options: &[&str], dir: &std::path::Path) -> Value {
    let json = dir.join("cli-report.json");
    let out = Command::new(env!("CARGO_BIN_EXE_pounce"))
        .arg(fixture(name))
        .arg("--no-sol")
        .arg("--json-output")
        .arg(&json)
        .args(options)
        .output()
        .expect("spawn pounce");
    let text = std::fs::read_to_string(&json).unwrap_or_else(|e| {
        panic!(
            "{name}: no CLI report ({e}); stderr:\n{}",
            String::from_utf8_lossy(&out.stderr)
        )
    });
    serde_json::from_str(&text).unwrap()
}

/// `true` when the `.nl` objective is minimized.
fn minimizes(name: &str) -> bool {
    pounce_nl::nl_reader::read_nl_file(&fixture(name))
        .unwrap()
        .minimize
}

/// The report's `complementarity` block without the wall-clock `seconds`.
fn block(report: &Value) -> Value {
    let mut b = report["complementarity"].clone();
    assert!(b.is_object(), "no complementarity block in the report");
    for s in b["solves"].as_array_mut().unwrap() {
        s.as_object_mut().unwrap().remove("seconds");
    }
    b
}

fn reals(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// One row: the C run, and the CLI's report on the same fixture and options.
struct Row {
    name: &'static str,
    c: nl_c_harness::CRun,
    cli: Value,
}

fn row(name: &'static str, options: &[&str], forward_nonlinear: bool) -> Row {
    let dir = scratch(name);
    let cli = cli(name, options, &dir);
    let c = nl_c_harness::solve_with(&fixture(name), options, forward_nonlinear, &dir);
    // The C side's own outputs agree with its report. (Within an ulp, not
    // to the bit: serde_json without `float_roundtrip` can read a printed
    // real back one ulp off. Every bitwise comparison below is therefore
    // between two reports, parsed by the same parser from text the same
    // writer printed — equal text iff equal bits.)
    let xr = reals(&c.report["solution"]["x"]);
    assert_eq!(xr.len(), c.x.len(), "{name}: report x length");
    for (a, b) in c.x.iter().zip(&xr) {
        assert!(
            (a - b).abs() <= 4.0 * f64::EPSILON * a.abs(),
            "{name}: IpoptSolve's x {a:e} differs from its report's {b:e}"
        );
    }
    let fr = c.report["solution"]["objective"].as_f64().unwrap();
    assert!(
        (c.obj - fr).abs() <= 4.0 * f64::EPSILON * c.obj.abs(),
        "{name}: IpoptSolve's objective differs from its report"
    );
    assert_eq!(
        c.report["statistics"]["iteration_count"].as_i64(),
        Some(i64::from(c.iter_count)),
        "{name}: GetIpoptIterCount differs from the report"
    );
    Row { name, c, cli }
}

impl Row {
    /// Exact parity with the CLI: status, objective (in the CLI's sense),
    /// point, iteration count and the whole block (bar `seconds`).
    fn assert_cli_exact(&self) -> &Self {
        let (c, cli, name) = (&self.c.report, &self.cli, self.name);
        assert_eq!(
            c["solution"]["status"], cli["solution"]["status"],
            "{name}: status"
        );
        let sense = if minimizes(name) { 1.0 } else { -1.0 };
        let c_obj = c["solution"]["objective"].as_f64().unwrap();
        let cli_obj = cli["solution"]["objective"].as_f64().unwrap();
        assert_eq!(
            (sense * c_obj).to_bits(),
            cli_obj.to_bits(),
            "{name}: objective {c_obj} (C, sense {sense}) vs {cli_obj} (CLI)"
        );
        assert_eq!(
            bits(&reals(&c["solution"]["x"])),
            bits(&reals(&cli["solution"]["x"])),
            "{name}: point"
        );
        assert_eq!(
            c["statistics"]["iteration_count"], cli["statistics"]["iteration_count"],
            "{name}: iteration_count"
        );
        assert_eq!(block(c), block(cli), "{name}: complementarity block");
        self
    }

    /// The answer only: same certificate class and reported kind, statuses
    /// both converged or equal, and for an answer the objective within
    /// `1e-4` relative and the point within `1e-5`. For the rows whose
    /// trajectory legitimately differs (the limited-memory Hessian without
    /// the `.nl` model's nonlinear-variable list).
    fn assert_cli_answer(&self) -> &Self {
        let (c, cli, name) = (&self.c.report, &self.cli, self.name);
        let (bc, bl) = (block(c), block(cli));
        assert_eq!(
            bc["certificate"]["class"], bl["certificate"]["class"],
            "{name}: class"
        );
        assert_eq!(bc["reported"], bl["reported"], "{name}: reported");
        let converged = |s: &Value| {
            matches!(
                s.as_str(),
                Some("SolveSucceeded") | Some("SolvedToAcceptableLevel")
            )
        };
        let (sc, sl) = (&c["solution"]["status"], &cli["solution"]["status"]);
        assert!(
            sc == sl || (converged(sc) && converged(sl)),
            "{name}: status {sc} (C) vs {sl} (CLI)"
        );
        if bc["reported"] != "verdict" {
            let sense = if minimizes(name) { 1.0 } else { -1.0 };
            let (fc, fl) = (
                sense * self.c.obj,
                cli["solution"]["objective"].as_f64().unwrap(),
            );
            assert!(
                (fc - fl).abs() <= 1e-4 * fl.abs().max(1.0),
                "{name}: objective {fc} (C) vs {fl} (CLI)"
            );
            let xl = reals(&cli["solution"]["x"]);
            let dx = self
                .c
                .x
                .iter()
                .zip(&xl)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(dx <= 1e-5, "{name}: point differs by {dx:e}");
        }
        self
    }

    /// Exact parity with Python's run of the same row
    /// (`common/cinterface_mpcc_python.json`), on the platform the reference
    /// was generated on (module docs, "Python parity is pinned on Linux
    /// x86_64 only"). Elsewhere it is a no-op that says so.
    fn assert_python_exact(&self, key: &str) -> &Self {
        if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            eprintln!(
                "{key}: Python parity not compared: the reference was generated on \
                 Linux x86_64 and is exact only there (C against the CLI still is)"
            );
            return self;
        }
        let all: Value = serde_json::from_str(include_str!("common/cinterface_mpcc_python.json"))
            .expect("parse the Python reference");
        let py = &all["rows"][key];
        assert!(py.is_object(), "no Python row {key}");
        assert_eq!(py["fixture"], self.name, "{key}: fixture");
        let c = &self.c.report;
        assert_eq!(c["solution"]["status"], py["status"], "{key}: status");
        assert_eq!(
            c["solution"]["objective"].as_f64().unwrap().to_bits(),
            py["objective"].as_f64().unwrap().to_bits(),
            "{key}: objective"
        );
        assert_eq!(
            bits(&reals(&c["solution"]["x"])),
            bits(&reals(&py["x"])),
            "{key}: point"
        );
        assert_eq!(
            c["statistics"]["iteration_count"], py["iteration_count"],
            "{key}: iteration_count"
        );
        assert_eq!(
            block(c),
            py["complementarity"],
            "{key}: complementarity block"
        );
        self
    }

    /// The C block's value at `path` (`/`-separated), for the branch checks.
    fn at(&self, path: &str) -> Value {
        block(&self.c.report)
            .pointer(path)
            .cloned()
            .unwrap_or(Value::Null)
    }

    fn total_iterations(&self, report: &Value) -> u64 {
        report["complementarity"]["total_iterations"]
            .as_u64()
            .unwrap()
    }
}

// ---------------------------------------------------------------------------
// Every fixture at the default options
// ---------------------------------------------------------------------------

#[test]
fn qpec_small_s_stationary_polish_taken() {
    let r = row("qpec_small", &[], false);
    r.assert_cli_exact().assert_python_exact("qpec_small");
    assert_eq!(r.at("/certificate/class"), "S");
    assert_eq!(r.at("/reported"), "answer");
    assert_eq!(r.at("/strategy"), "both");
    assert_eq!(r.at("/polish/taken"), true);
    // The trajectory the hand-written callbacks could not reproduce (139).
    assert_eq!(r.at("/total_iterations"), 191);
}

#[test]
fn ralph1_weak_fallback_with_b_check() {
    let r = row("ralph1", &[], false);
    r.assert_cli_exact().assert_python_exact("ralph1");
    assert_eq!(r.at("/certificate/class"), "W");
    assert_eq!(r.at("/reported"), "weak_fallback");
    assert_eq!(r.at("/accepted_by"), Value::Null);
    assert_eq!(r.at("/b_stationarity/verdict"), "B");
    assert_eq!(r.at("/polish"), Value::Null, "polish not run");
}

#[test]
fn mixed_pair_mixed_shape() {
    let r = row("mixed_pair", &[], false);
    r.assert_cli_exact().assert_python_exact("mixed_pair");
    assert_eq!(r.at("/certificate/class"), "W");
    assert_eq!(r.at("/reported"), "weak_fallback");
}

#[test]
fn design_cent_3_lifted_at_the_default_maximize() {
    let r = row("design_cent_3", &[], false);
    r.assert_cli_exact().assert_python_exact("design_cent_3");
    assert!(!minimizes("design_cent_3"), "the objective-sense branch");
    assert_eq!(r.at("/certificate/class"), "S");
    assert!(r.at("/lifted").as_u64().unwrap() > 0, "lifted path");
}

#[test]
fn infeasible_pair_verdict() {
    let r = row("infeasible_pair", &[], false);
    r.assert_cli_exact().assert_python_exact("infeasible_pair");
    assert_eq!(r.at("/certificate/class"), "none");
    assert_eq!(r.at("/reported"), "verdict");
    assert_eq!(
        r.c.report["solution"]["status"],
        "InfeasibleProblemDetected"
    );
}

#[test]
fn opf_vv_lower_knee_polish_taken() {
    let r = row("opf_vv_lower_knee", &[], false);
    r.assert_cli_exact()
        .assert_python_exact("opf_vv_lower_knee");
    assert_eq!(r.at("/certificate/class"), "W");
    assert_eq!(r.at("/polish/taken"), true);
}

#[test]
fn opf_vv_lower_knee_held_polish_declined_with_reason() {
    let r = row("opf_vv_lower_knee_held", &[], false);
    r.assert_cli_exact()
        .assert_python_exact("opf_vv_lower_knee_held");
    assert_eq!(r.at("/certificate/class"), "S");
    assert_eq!(r.at("/polish/taken"), false);
    assert!(r.at("/polish/reason").is_string(), "reason present");
}

// ---------------------------------------------------------------------------
// qpec_small under each option set
// ---------------------------------------------------------------------------

#[test]
fn qpec_small_strategy_exact() {
    let r = row("qpec_small", &["complementarity_strategy=exact"], false);
    r.assert_cli_exact().assert_python_exact("qpec_small_exact");
    assert_eq!(r.at("/strategy"), "exact");
    assert_eq!(r.at("/accepted_by"), "exact product");
}

#[test]
fn qpec_small_strategy_continuation() {
    let r = row(
        "qpec_small",
        &["complementarity_strategy=continuation"],
        false,
    );
    r.assert_cli_exact()
        .assert_python_exact("qpec_small_continuation");
    assert_eq!(r.at("/strategy"), "continuation");
}

#[test]
fn qpec_small_continuation_schedule() {
    let r = row(
        "qpec_small",
        &["complementarity_continuation=schedule"],
        false,
    );
    r.assert_cli_exact()
        .assert_python_exact("qpec_small_schedule");
    assert_eq!(r.at("/certificate/class"), "S");
}

#[test]
fn qpec_small_lift_yes() {
    let r = row("qpec_small", &["complementarity_lift=yes"], false);
    r.assert_cli_exact()
        .assert_python_exact("qpec_small_lift_yes");
    assert!(r.at("/lifted").as_u64().unwrap() > 0, "lifted on request");
}

#[test]
fn qpec_small_lift_no() {
    let r = row("qpec_small", &["complementarity_lift=no"], false);
    r.assert_cli_exact()
        .assert_python_exact("qpec_small_lift_no");
    assert_eq!(r.at("/lifted"), 0);
}

/// Limited-memory on `qpec_small`: every variable is nonlinear, so the
/// `.nl` list says nothing a callback model does not assume — exact with
/// and without passing it.
#[test]
fn qpec_small_limited_memory() {
    let r = row("qpec_small", &[LBFGS], false);
    r.assert_cli_exact().assert_python_exact("qpec_small_lbfgs");
    row("qpec_small", &[LBFGS], true).assert_cli_exact();
}

// ---------------------------------------------------------------------------
// The limited-memory Hessian and the nonlinear-variable list
// ---------------------------------------------------------------------------

/// With the `.nl` model's nonlinear-variable list passed through
/// `IpoptSetNonlinearVariables`, the C API is exact against the CLI; without
/// it, the C API and Python (which cannot declare one) agree exactly with
/// each other, and with the CLI on the answer only. The trajectory must
/// differ without the list, or the masked row would be no evidence that the
/// list is what restores parity.
fn limited_memory_masked_and_unmasked(name: &'static str, python_key: &str) {
    let masked = row(name, &[LBFGS], true);
    masked.assert_cli_exact();
    let unmasked = row(name, &[LBFGS], false);
    unmasked.assert_python_exact(python_key).assert_cli_answer();
    assert_ne!(
        block(&unmasked.c.report),
        block(&unmasked.cli),
        "{name}: without the nonlinear-variable list the trajectory was \
         measured to differ from the CLI's; if it no longer does, the masked \
         row no longer tests the list"
    );
    eprintln!(
        "{name}: limited-memory total iterations: CLI {}, C with the .nl list {}, C without {}",
        unmasked.total_iterations(&unmasked.cli),
        masked.total_iterations(&masked.c.report),
        unmasked.total_iterations(&unmasked.c.report),
    );
}

#[test]
fn infeasible_pair_limited_memory() {
    limited_memory_masked_and_unmasked("infeasible_pair", "infeasible_pair_lbfgs_unmasked");
}

#[test]
fn opf_vv_lower_knee_limited_memory() {
    limited_memory_masked_and_unmasked("opf_vv_lower_knee", "opf_vv_lower_knee_lbfgs_unmasked");
}

#[test]
fn opf_vv_lower_knee_held_limited_memory() {
    limited_memory_masked_and_unmasked(
        "opf_vv_lower_knee_held",
        "opf_vv_lower_knee_held_lbfgs_unmasked",
    );
}
