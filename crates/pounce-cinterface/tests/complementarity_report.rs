//! gh#1147 — `IpoptWriteSolveReport` carries the `complementarity` block
//! after a solve that ran the complementarity driver.
//!
//! Three things are held here, on every certificate class a real model
//! reaches (the models of `common/cc_models.rs`: `S`, `W`, `none`,
//! `not searched`, and the taken-polish branch of `S`):
//!
//! 1. **The getters and the block agree field for field on the same solve.**
//!    The getters (S1-I6) encode `MpccOutcome` from the driver's enums; the
//!    block is `MpccOutcome::report_info()`, which encodes the same fields
//!    to strings. Every field of `PounceComplementarityStats`, of each
//!    `PounceComplementaritySolve` and of each text is compared with its
//!    key in the block — the reals bitwise, `seconds` included (both are
//!    read from the same record) — and every key of the block must be one
//!    this test compares, so a field added to the block without a getter
//!    fails here.
//! 2. **The block is Python's.** C and Python run the same driver on the same
//!    callbacks and their trajectories agree to the bit (S1-I5, S1-I6), so
//!    the block, less the wall-clock `seconds`, must equal the one Python's
//!    `Problem.solve(report_path=...)` wrote for the same model, start and
//!    options. `common/cc_python_report_blocks.json` holds those blocks
//!    (Python 0.16.0 extension built from this branch, `seconds` removed;
//!    Python's `"full"` and `"summary"` blocks were identical). The CLI
//!    writes the block through the same `report_info()`; its trajectory on
//!    `qpec_small.nl` differs (191 iterations against 139, S1-I5), which is
//!    S1-I8's question, not this file's.
//! 3. **The block is not detail-gated** (`ReportBuilder::finish` copies it
//!    unconditionally): `"summary"` and `"full"` carry the same block.
//!
//! The absent branch: no block before any solve (no report at all), after a
//! solve without pairs, after the pairs are cleared and the problem
//! re-solved, and no report after a refused solve.

#[path = "common/cc_models.rs"]
mod cc_models;
#[path = "common/qpec.rs"]
mod qpec;

use cc_models::*;
use pounce_cinterface::*;
use pounce_nlp::ApplicationReturnStatus;
use serde_json::Value;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Python's `complementarity` blocks, keyed by model.
const PYTHON_BLOCKS: &str = include_str!("common/cc_python_report_blocks.json");

fn python_block(name: &str) -> Value {
    let all: Value = serde_json::from_str(PYTHON_BLOCKS).expect("reference JSON");
    all.get(name)
        .unwrap_or_else(|| panic!("no Python block {name}"))
        .clone()
}

fn report_path(detail: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "pounce-cc-report-{}-{}-{detail}.json",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Write the report at `detail` and parse it; `None` when the write
/// returned `FALSE` (and then no file was written).
fn write_report(prob: IpoptProblem, detail: &str) -> Option<Value> {
    let path = report_path(detail);
    let _ = std::fs::remove_file(&path);
    let cpath = CString::new(path.to_str().unwrap()).unwrap();
    let cdetail = CString::new(detail).unwrap();
    let ok = unsafe { IpoptWriteSolveReport(prob, cpath.as_ptr(), cdetail.as_ptr()) };
    if ok == 0 {
        assert!(!path.exists(), "a FALSE write left a file");
        return None;
    }
    assert_eq!(ok, 1);
    let text = std::fs::read_to_string(&path).expect("report written");
    let _ = std::fs::remove_file(&path);
    Some(serde_json::from_str(&text).expect("report is JSON"))
}

/// The block with every lowered solve's wall-clock `seconds` removed.
fn without_seconds(block: &Value) -> Value {
    let mut b = block.clone();
    for s in b["solves"].as_array_mut().expect("solves") {
        s.as_object_mut().unwrap().remove("seconds");
    }
    b
}

// ---------------------------------------------------------------------------
// The block read the way the getters encode it
// ---------------------------------------------------------------------------

/// A real in the block: absent or `null` reads as the getters' NaN.
fn real(v: Option<&Value>) -> Number {
    match v {
        None | Some(Value::Null) => Number::NAN,
        Some(x) => x.as_f64().expect("a number"),
    }
}

/// A count in the block.
fn count(v: &Value) -> Index {
    Index::try_from(v.as_u64().expect("a count")).unwrap()
}

/// A flag in the block, as the getters' 0/1.
fn flag(v: &Value) -> Index {
    Index::from(v.as_bool().expect("a bool"))
}

/// A label in the block: absent or `null` reads as the getters' `""`.
fn label_of(v: Option<&Value>) -> &str {
    match v {
        None | Some(Value::Null) => "",
        Some(s) => s.as_str().expect("a string"),
    }
}

/// An optional text in the block: absent or `null` is the text getter's `-1`.
fn text_of(v: Option<&Value>) -> Option<&str> {
    match v {
        None | Some(Value::Null) => None,
        Some(s) => Some(s.as_str().expect("a string")),
    }
}

fn reported_code(name: &str) -> Index {
    match name {
        "answer" => POUNCE_CC_REPORTED_ANSWER,
        "weak_fallback" => POUNCE_CC_REPORTED_WEAK_FALLBACK,
        "feasible_uncertified" => POUNCE_CC_REPORTED_FEASIBLE_UNCERTIFIED,
        "verdict" => POUNCE_CC_REPORTED_VERDICT,
        other => panic!("unknown reported {other}"),
    }
}

/// The `ApplicationReturnStatus` code of an upstream status name.
fn status_code(name: &str) -> Index {
    use ApplicationReturnStatus::*;
    const ALL: [ApplicationReturnStatus; 20] = [
        SolveSucceeded,
        SolvedToAcceptableLevel,
        InfeasibleProblemDetected,
        SearchDirectionBecomesTooSmall,
        DivergingIterates,
        UserRequestedStop,
        FeasiblePointFound,
        MaximumIterationsExceeded,
        RestorationFailed,
        ErrorInStepComputation,
        MaximumCpuTimeExceeded,
        MaximumWallTimeExceeded,
        NotEnoughDegreesOfFreedom,
        InvalidProblemDefinition,
        InvalidOption,
        InvalidNumberDetected,
        UnrecoverableException,
        NonIpoptExceptionThrown,
        InsufficientMemory,
        InternalError,
    ];
    ALL.iter()
        .find(|s| s.upstream_name() == name)
        .unwrap_or_else(|| panic!("unknown status {name}"))
        .as_int()
}

#[track_caller]
fn keys_known(what: &str, obj: &Value, known: &[&str]) {
    for k in obj.as_object().expect("an object").keys() {
        assert!(
            known.contains(&k.as_str()),
            "{what}.{k}: a block field this test (and the getters) do not cover"
        );
    }
}

#[track_caller]
fn same_real(what: &str, got: Number, block: Option<&Value>) {
    let want = real(block);
    assert!(same(got, want), "{what}: getter {got:e}, block {want:e}");
}

/// Hold every getter on `prob` to the block its report wrote.
fn assert_getters_match_block(prob: IpoptProblem, b: &Value) {
    keys_known(
        "complementarity",
        b,
        &[
            "strategy",
            "accepted_by",
            "fallback",
            "reported",
            "summary",
            "solves",
            "total_iterations",
            "max_abs_product",
            "certificate",
            "b_stationarity",
            "escapes",
            "lifted",
            "natural_residual",
            "polish",
            "polish_threshold",
        ],
    );
    let s = stats(prob).expect("stats after a driver solve");

    assert_eq!(
        label(&s.strategy),
        b["strategy"].as_str().unwrap(),
        "strategy"
    );
    assert_eq!(s.reported, reported_code(b["reported"].as_str().unwrap()));
    assert_eq!(label(&s.accepted_by), label_of(b.get("accepted_by")));
    assert_eq!(label(&s.fallback), label_of(b.get("fallback")));
    let solves = b["solves"].as_array().expect("solves");
    assert_eq!(s.n_solves as usize, solves.len(), "n_solves");
    assert_eq!(s.total_iterations, count(&b["total_iterations"]));
    same_real(
        "max_abs_product",
        s.max_abs_product,
        b.get("max_abs_product"),
    );

    match b.get("certificate") {
        None => {
            assert_eq!(s.certificate_class, POUNCE_CC_CLASS_ABSENT);
            for v in [
                s.n_pairs,
                s.n_biactive,
                s.n_neither,
                s.multipliers_unique,
                s.tight,
                s.feasible,
            ] {
                assert_eq!(v, -1);
            }
            assert!(s.residual.is_nan() && s.resid_tol.is_nan() && s.primal_violation.is_nan());
        }
        Some(c) => {
            keys_known(
                "certificate",
                c,
                &[
                    "class",
                    "n_pairs",
                    "n_biactive",
                    "n_neither",
                    "multipliers_unique",
                    "residual",
                    "resid_tol",
                    "tight",
                    "primal_violation",
                    "feasible",
                ],
            );
            assert_eq!(
                s.certificate_class,
                class_code(c["class"].as_str().unwrap())
            );
            assert_eq!(s.n_pairs, count(&c["n_pairs"]));
            assert_eq!(s.n_biactive, count(&c["n_biactive"]));
            assert_eq!(s.n_neither, count(&c["n_neither"]));
            assert_eq!(s.multipliers_unique, flag(&c["multipliers_unique"]));
            same_real("residual", s.residual, c.get("residual"));
            same_real("resid_tol", s.resid_tol, c.get("resid_tol"));
            assert_eq!(s.tight, flag(&c["tight"]));
            assert_eq!(s.feasible, flag(&c["feasible"]));
            same_real(
                "primal_violation",
                s.primal_violation,
                c.get("primal_violation"),
            );
        }
    }

    match b.get("b_stationarity") {
        None => {
            assert_eq!(s.b_verdict, POUNCE_CC_B_VERDICT_ABSENT);
            assert_eq!(
                (s.b_n_biactive, s.b_branches_solved, s.b_enumerated),
                (-1, -1, -1)
            );
            assert_eq!(text(prob, POUNCE_CC_TEXT_B_REASON), None);
        }
        Some(bc) => {
            keys_known(
                "b_stationarity",
                bc,
                &[
                    "verdict",
                    "reason",
                    "n_biactive",
                    "branches_solved",
                    "enumerated",
                ],
            );
            assert_eq!(s.b_verdict, b_code(bc["verdict"].as_str().unwrap()));
            assert_eq!(s.b_n_biactive, count(&bc["n_biactive"]));
            assert_eq!(s.b_branches_solved, count(&bc["branches_solved"]));
            assert_eq!(s.b_enumerated, flag(&bc["enumerated"]));
            assert_eq!(
                text(prob, POUNCE_CC_TEXT_B_REASON).as_deref(),
                text_of(bc.get("reason"))
            );
        }
    }

    assert_eq!(s.escapes, count(&b["escapes"]));
    assert_eq!(s.lifted, count(&b["lifted"]));
    same_real(
        "natural_residual",
        s.natural_residual,
        b.get("natural_residual"),
    );

    match b.get("polish") {
        None => {
            assert_eq!((s.polish_ran, s.polish_taken), (0, 0));
            assert!(s.polish_residual_before.is_nan() && s.polish_residual_after.is_nan());
            assert_eq!(text(prob, POUNCE_CC_TEXT_POLISH_REASON), None);
        }
        Some(p) => {
            keys_known(
                "polish",
                p,
                &["taken", "residual_before", "residual_after", "reason"],
            );
            assert_eq!(s.polish_ran, 1);
            assert_eq!(s.polish_taken, flag(&p["taken"]));
            same_real(
                "polish residual_before",
                s.polish_residual_before,
                p.get("residual_before"),
            );
            same_real(
                "polish residual_after",
                s.polish_residual_after,
                p.get("residual_after"),
            );
            assert_eq!(
                text(prob, POUNCE_CC_TEXT_POLISH_REASON).as_deref(),
                text_of(p.get("reason"))
            );
        }
    }
    same_real(
        "polish_threshold",
        s.polish_threshold,
        b.get("polish_threshold"),
    );
    assert_eq!(
        text(prob, POUNCE_CC_TEXT_SUMMARY).as_deref(),
        Some(b["summary"].as_str().unwrap()),
        "summary"
    );

    for (k, j) in solves.iter().enumerate() {
        keys_known(
            &format!("solves[{k}]"),
            j,
            &[
                "stage",
                "tau",
                "status",
                "iterations",
                "max_abs_product",
                "seconds",
            ],
        );
        let r = solve_record(prob, k as Index).expect("a record for every listed solve");
        assert_eq!(label(&r.stage), j["stage"].as_str().unwrap(), "solve {k}");
        match j.get("tau") {
            None => {
                assert_eq!(r.relaxed, 0, "solve {k}");
                assert!(r.tau.is_nan(), "solve {k}");
            }
            Some(t) => {
                assert_eq!(r.relaxed, 1, "solve {k}");
                same_real(&format!("solve {k} tau"), r.tau, Some(t));
            }
        }
        assert_eq!(
            r.status,
            status_code(j["status"].as_str().unwrap()),
            "solve {k}"
        );
        assert_eq!(r.iterations, count(&j["iterations"]), "solve {k}");
        same_real(
            &format!("solve {k} max_abs_product"),
            r.max_abs_product,
            j.get("max_abs_product"),
        );
        same_real(&format!("solve {k} seconds"), r.seconds, j.get("seconds"));
    }
    assert!(solve_record(prob, s.n_solves).is_none());
}

/// Solve `m`, write its report at both detail levels, and hold the block to
/// the getters, to the other detail level, and (when given) to Python's.
/// Returns the block.
fn solve_and_check(m: &Model, python: Option<&str>) -> Value {
    let (status, _) = m.solve();
    let full = write_report(m.prob, "full").expect("report after a driver solve");
    let block = full
        .get("complementarity")
        .expect("a driver solve's report carries the block")
        .clone();
    assert_getters_match_block(m.prob, &block);

    // The rest of the report describes the reported lowered solve.
    assert_eq!(
        status_code(full["solution"]["status_upstream"].as_str().unwrap()),
        status
    );
    let reported_iters = unsafe { GetIpoptIterCount(m.prob) };
    assert_eq!(
        full["statistics"]["iteration_count"].as_i64(),
        Some(i64::from(reported_iters))
    );
    assert!(count(&block["total_iterations"]) >= reported_iters);

    let summary = write_report(m.prob, "summary").expect("summary report");
    assert_eq!(
        summary.get("complementarity"),
        Some(&block),
        "the block is not detail-gated"
    );

    if let Some(name) = python {
        assert_eq!(
            without_seconds(&block),
            python_block(name),
            "C block differs from Python's for {name}"
        );
    }
    block
}

// ---------------------------------------------------------------------------
// One test per certificate class (and the polish branch)
// ---------------------------------------------------------------------------

#[test]
fn class_s_block_matches_getters_and_python() {
    let m = Model::qpec([0.0, 0.5, 0.5]);
    let b = solve_and_check(&m, Some("qpec_s"));
    assert_eq!(b["certificate"]["class"], "S");
    assert_eq!(b["reported"], "answer");
    assert!(b.get("polish").is_none());
}

#[test]
fn polish_taken_block_matches_getters_and_python() {
    let m = Model::qpec([0.0, 0.0, 0.0]);
    let b = solve_and_check(&m, Some("qpec_polish"));
    assert_eq!(b["polish"]["taken"], true);
    assert_eq!(b["polish"]["reason"], Value::Null);
}

#[test]
fn strategy_exact_block_matches_getters() {
    let m = Model::qpec([0.0, 0.5, 0.5]);
    m.option("complementarity_strategy", "exact");
    let b = solve_and_check(&m, None);
    assert_eq!(b["strategy"], "exact");
}

#[test]
fn class_w_block_matches_getters_and_python() {
    let m = Model::ralph1();
    let b = solve_and_check(&m, Some("ralph1_w"));
    assert_eq!(b["certificate"]["class"], "W");
    assert_eq!(b["reported"], "weak_fallback");
    assert_eq!(b["b_stationarity"]["verdict"], "B");
}

#[test]
fn class_none_block_matches_getters_and_python() {
    let m = Model::infeasible_pair();
    let b = solve_and_check(&m, Some("infeasible_pair_none"));
    assert_eq!(b["certificate"]["class"], "none");
    assert_eq!(b["reported"], "verdict");
    // The getters' NaN residual is the block's null.
    assert_eq!(b["certificate"]["residual"], Value::Null);
}

#[test]
fn class_not_searched_block_matches_getters_and_python() {
    let m = Model::not_searched();
    let b = solve_and_check(&m, Some("not_searched"));
    assert_eq!(b["certificate"]["class"], "not_searched");
    assert_eq!(b["reported"], "answer");
}

// ---------------------------------------------------------------------------
// The driver did not run
// ---------------------------------------------------------------------------

#[track_caller]
fn assert_no_block(prob: IpoptProblem, when: &str) {
    for detail in ["full", "summary"] {
        let r = write_report(prob, detail).unwrap_or_else(|| panic!("a report {when}"));
        assert!(
            r.get("complementarity").is_none(),
            "{detail} report {when} carries a complementarity block"
        );
    }
}

#[test]
fn no_report_before_any_solve() {
    let m = Model::qpec([0.0, 0.5, 0.5]);
    assert!(write_report(m.prob, "full").is_none());
}

/// A solve without pairs writes no block, as the CLI (`if let Some(out)`)
/// and Python (the same) do for a model without declared pairs.
#[test]
fn no_block_after_a_solve_without_pairs() {
    let m = Model {
        prob: qpec::create_qpec(0),
        data: None,
        x0: vec![0.0, 0.5, 0.5],
    };
    let (status, _) = m.solve();
    assert_eq!(status, 0);
    assert_no_block(m.prob, "after a solve without pairs");
}

/// The report, like the getters, describes the most recent solve: clearing
/// the pairs keeps the block of the solve that used them; the next solve,
/// without pairs, writes none.
#[test]
fn clearing_the_pairs_then_solving_drops_the_block() {
    let m = Model::qpec([0.0, 0.5, 0.5]);
    let before = solve_and_check(&m, Some("qpec_s"));
    assert_eq!(unsafe { ClearPounceComplementarityPairs(m.prob) }, 1);
    let kept = write_report(m.prob, "full").unwrap();
    assert_eq!(kept.get("complementarity"), Some(&before));
    let (status, _) = m.solve();
    assert_eq!(status, 0);
    assert_no_block(
        m.prob,
        "after the pairs were cleared and the problem re-solved",
    );
}

/// A refused solve (pairs with a staged SQP working set) leaves no solve to
/// report, even right after a driver solve on the same handle.
#[test]
fn no_report_after_a_refused_solve() {
    let m = Model::qpec([0.0, 0.5, 0.5]);
    solve_and_check(&m, Some("qpec_s"));
    let xs = [0 as IpoptBoundStatus; qpec::N];
    let gs = [0 as IpoptConsStatus; qpec::M];
    assert_eq!(
        unsafe { IpoptSetWarmStartWorkingSet(m.prob, xs.as_ptr(), gs.as_ptr()) },
        1
    );
    let (status, _) = m.solve();
    assert_eq!(status, -11, "Invalid_Problem_Definition");
    for detail in ["full", "summary"] {
        assert!(
            write_report(m.prob, detail).is_none(),
            "{detail}: a report after a refused solve"
        );
    }
}
