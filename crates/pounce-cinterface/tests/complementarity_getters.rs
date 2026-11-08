//! gh#1147 — the complementarity outcome getters:
//! `GetPounceComplementarityStats`, `GetPounceComplementaritySolve` and
//! `GetPounceComplementarityText`.
//!
//! The certificate class branches four ways, and each branch has its own
//! model here (CLAUDE.md, "A test is only evidence about the branch it
//! reaches"), all reached by a real solve through `IpoptSolve`:
//!
//! | class          | model                                   | reported        |
//! |----------------|-----------------------------------------|-----------------|
//! | `S`            | `QpecSmall` (`common/qpec.rs`)          | answer          |
//! | `W`            | `ralph1` (`fixtures_mpcc/ralph1.nl`)    | weak fallback   |
//! | `none`         | `infeasible_pair` (`fixtures_mpcc/`)    | verdict         |
//! | `not searched` | 605 active bounds, a duplicate row      | answer          |
//!
//! `ralph1` and `infeasible_pair` are the CLI fixtures' models written as
//! callbacks (`infeasible_pair.nl` carries Pyomo's auxiliary variable; here
//! the pair's row is `y` itself, the same model). The `not searched` model
//! makes the stationarity multipliers a set (row 0 duplicates `x_0 >= 0`)
//! with more than `MAX_SEARCHED_COLUMNS = 600` sign-constrained columns.
//! What no in-repo model reaches — `POUNCE_CC_CLASS_ABSENT` (the model not
//! evaluable at the reported point), `FEASIBLE_UNCERTIFIED`, the `not_B` and
//! `not_checked` B verdicts and a polish that was not taken (with its
//! reason) — is held by unit tests of the encoding in `src/lib.rs`
//! (`complementarity_encodings_*`).
//!
//! Every value is held, to the bit, against Python's `info["complementarity"]`
//! for the same model, start and options (`pounce.Problem(complementarity=)`
//! on the same callbacks; Python 0.16.0 extension). The `PY_*` constants
//! were generated from it; the callback frontends share the driver, so they
//! agree exactly (S1-I5 measured the same for the solve itself). Seconds
//! are wall time and are only checked to be finite and non-negative.
//!
//! The other branch of every getter is "the driver did not run": before any
//! solve, after a solve without pairs, after the pairs are cleared and the
//! problem re-solved, after a refused solve, and on NULL arguments.

// Test code: a panicking unwrap/expect is the test failing, as intended.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/cc_models.rs"]
mod cc_models;
#[path = "common/qpec.rs"]
mod qpec;
#[path = "common/struct_size.rs"]
mod struct_size;

use cc_models::*;
use struct_size::check_struct_size_contract;

use pounce_cinterface::*;
use std::ffi::c_char;
use std::ptr;

/// What Python reported for one solve.
struct PyRef {
    status: Index,
    iter_count: Index,
    /// The reported point (a prefix of it, for the large model).
    x: &'static [Number],
    accepted_by: Option<&'static str>,
    fallback: Option<&'static str>,
    total_iterations: Index,
    max_abs_product: Number,
    /// `(stage, tau, status, iterations, max_abs_product)` per lowered solve.
    solves: &'static [(&'static str, Option<Number>, Index, Index, Number)],
    class: &'static str,
    /// `(n_pairs, n_biactive, n_neither, multipliers_unique, tight, feasible)`.
    cert: (Index, Index, Index, bool, bool, bool),
    residual: Number,
    resid_tol: Number,
    primal_violation: Number,
    /// `(verdict, n_biactive, branches_solved, enumerated)`.
    b: Option<(&'static str, Index, Index, bool)>,
    escapes: Index,
    lifted: Index,
    natural_residual: Number,
    /// `(taken, residual_before, residual_after)`.
    polish: Option<(bool, Number, Number)>,
    polish_threshold: Number,
    summary: &'static str,
}

// ---------------------------------------------------------------------------
// Python's reports (generated; see the module docs)
// ---------------------------------------------------------------------------

/// Python, `PY_QPEC`.
const PY_QPEC: PyRef = PyRef {
    status: 0,
    iter_count: 12,
    x: &[
        1.0000000061040115,
        1.0000000030553011,
        5.5115660435885625e-12,
    ],
    accepted_by: Some("Scholtes continuation"),
    fallback: None,
    total_iterations: 139,
    max_abs_product: 6.590727983521438e-12,
    solves: &[
        (
            "Scholtes continuation",
            Some(1.0),
            0,
            15,
            8.257983751122374e-05,
        ),
        (
            "Scholtes continuation",
            Some(1e-08),
            0,
            36,
            4.995737420965263e-09,
        ),
        ("Scholtes continuation", None, 0, 24, 6.590727983521438e-12),
        ("exact product", Some(1e-06), 0, 32, 4.997549451896263e-07),
        ("exact product", Some(1e-08), 0, 12, 4.9944261837507575e-09),
        ("exact product", None, 0, 20, 2.2900878390410384e-23),
    ],
    class: "S",
    cert: (2, 1, 0, true, true, true),
    residual: 9.035978839861492e-24,
    resid_tol: 0.000202,
    primal_violation: 0.0,
    b: None,
    escapes: 0,
    lifted: 0,
    natural_residual: 6.590727963384779e-12,
    polish: None,
    polish_threshold: 1e-07,
    summary: "pounce: complementarity: answered by the Scholtes continuation formulation (6 lowered solves, 139 iterations in all, violation = 6.6e-12); 2 pairs, 1 biactive; stationarity: S (strongly stationary).",
};

/// Python, `PY_QPEC_ZERO`.
const PY_QPEC_ZERO: PyRef = PyRef {
    status: 0,
    iter_count: 1,
    x: &[1.0000000039876262, 1.0000000019938131, 0.0],
    accepted_by: Some("Scholtes continuation"),
    fallback: None,
    total_iterations: 145,
    max_abs_product: 0.0,
    solves: &[
        (
            "Scholtes continuation",
            Some(1.0),
            0,
            16,
            8.216223207532317e-05,
        ),
        (
            "Scholtes continuation",
            Some(1e-08),
            0,
            37,
            4.995563324117214e-09,
        ),
        ("Scholtes continuation", None, 1, 3, 4.81122317973673e-11),
        ("exact product", Some(1e-06), 0, 25, 4.99799254750961e-07),
        ("exact product", Some(1e-08), 0, 13, 4.9947914513732744e-09),
        ("exact product", None, 1, 49, 2.22044604925538e-16),
        ("branch polish", None, 0, 2, 0.0),
    ],
    class: "S",
    cert: (2, 1, 0, true, true, true),
    residual: 4.1359030627651384e-24,
    resid_tol: 0.000202,
    primal_violation: 0.0,
    b: None,
    escapes: 0,
    lifted: 0,
    natural_residual: 0.0,
    polish: Some((true, 9.64671485001639e-07, 0.0)),
    polish_threshold: 1e-07,
    summary: "pounce: complementarity: answered by the Scholtes continuation formulation (7 lowered solves, 145 iterations in all, violation = 0.0e0); 2 pairs, 1 biactive; stationarity: S (strongly stationary).",
};

/// Python, `PY_QPEC_EXACT`.
const PY_QPEC_EXACT: PyRef = PyRef {
    status: 0,
    iter_count: 16,
    x: &[
        1.0000000000018847,
        1.0000000000009424,
        2.945323339678135e-12,
    ],
    accepted_by: Some("exact product"),
    fallback: None,
    total_iterations: 91,
    max_abs_product: 2.2900878390410384e-23,
    solves: &[
        ("exact product", Some(1e-06), 0, 32, 4.997549451896263e-07),
        ("exact product", Some(1e-08), 0, 12, 4.9944261837507575e-09),
        ("exact product", None, 0, 20, 2.2900878390410384e-23),
        ("exact product", None, 1, 27, 0.0),
    ],
    class: "S",
    cert: (2, 1, 0, true, true, true),
    residual: 2.4233807008389483e-27,
    resid_tol: 0.000202,
    primal_violation: 0.0,
    b: None,
    escapes: 0,
    lifted: 0,
    natural_residual: 2.945323339678135e-12,
    polish: None,
    polish_threshold: 1e-07,
    summary: "pounce: complementarity: answered by the exact product formulation (4 lowered solves, 91 iterations in all, violation = 2.3e-23); 2 pairs, 1 biactive; stationarity: S (strongly stationary).",
};

/// Python, PY_RALPH1: status 0, iter_count 26.
const PY_RALPH1: PyRef = PyRef {
    status: 0,
    iter_count: 26,
    x: &[2.0000000001531745e-09, 2.000000000153011e-09],
    accepted_by: None,
    fallback: Some("Scholtes continuation"),
    total_iterations: 303,
    max_abs_product: 3.267363419834431e-31,
    solves: &[
        ("Scholtes continuation", Some(1.0), 0, 6, 0.9999999949704398),
        (
            "Scholtes continuation",
            Some(1e-08),
            0,
            29,
            1.0001317801711096e-08,
        ),
        ("Scholtes continuation", None, 1, 18, 0.0),
        (
            "Scholtes continuation",
            Some(0.1),
            0,
            6,
            0.09999999841512747,
        ),
        (
            "Scholtes continuation",
            Some(0.01),
            0,
            6,
            0.009999999498819327,
        ),
        (
            "Scholtes continuation",
            Some(0.001),
            0,
            6,
            0.0009999998415127808,
        ),
        (
            "Scholtes continuation",
            Some(0.0001),
            0,
            6,
            9.999994988196722e-05,
        ),
        (
            "Scholtes continuation",
            Some(1e-05),
            0,
            7,
            9.999984151312216e-06,
        ),
        (
            "Scholtes continuation",
            Some(1e-06),
            0,
            7,
            9.999981734797942e-07,
        ),
        (
            "Scholtes continuation",
            Some(1e-07),
            0,
            6,
            9.999942541896481e-08,
        ),
        (
            "Scholtes continuation",
            Some(1e-08),
            0,
            6,
            9.999818225903267e-09,
        ),
        ("Scholtes continuation", None, 0, 26, 3.267363419834431e-31),
        ("exact product", Some(1e-06), 0, 28, 9.99994988349828e-07),
        ("exact product", Some(1e-08), 0, 9, 9.999818186776793e-09),
        ("exact product", None, 1, 25, 3.877822696628012e-30),
        ("exact product", None, 0, 29, 7.519823629714613e-34),
        ("l1 penalty", None, 0, 83, 2.5995325169830677e-09),
    ],
    class: "W",
    cert: (1, 1, 0, false, true, true),
    residual: 7.771561172376096e-16,
    resid_tol: 0.000202,
    primal_violation: 1.6336817097922297e-22,
    b: Some(("B", 1, 2, true)),
    escapes: 0,
    lifted: 0,
    natural_residual: 1.6336817097922297e-22,
    polish: None,
    polish_threshold: 1e-07,
    summary: "pounce: complementarity: no formulation produced a strongly stationary answer (17 lowered solves, 303 iterations in all); reporting the Scholtes continuation formulation's weakly stationary point (violation = 3.3e-31); 1 pair, 1 biactive; stationarity: W only (a biactive pair has a wrong-sign multiplier: a branch may still descend).",
};

/// Python, `PY_INFEASIBLE`.
const PY_INFEASIBLE: PyRef = PyRef {
    status: 2,
    iter_count: 16,
    x: &[0.9999314524019693, 0.9999993740040607],
    accepted_by: None,
    fallback: None,
    total_iterations: 177,
    max_abs_product: 0.9999308264489406,
    solves: &[
        (
            "Scholtes continuation",
            Some(1.0),
            0,
            36,
            0.9999999999991596,
        ),
        ("Scholtes continuation", Some(0.1), 2, 7, 0.9966336342418958),
        ("Scholtes continuation", Some(0.1), 2, 7, 0.9966752508799669),
        (
            "Scholtes continuation",
            Some(0.1),
            2,
            25,
            0.6991503359108203,
        ),
        ("Scholtes continuation", None, 2, 11, 0.9999183334051946),
        ("exact product", Some(1e-06), 2, 13, 0.9999999973697551),
        ("exact product", None, 2, 16, 0.9999308264489406),
        ("l1 penalty", None, 2, 62, 1.000000000050117),
    ],
    class: "none",
    cert: (1, 0, 1, true, false, true),
    residual: f64::NAN,
    resid_tol: 0.000101,
    primal_violation: 6.854759803065491e-05,
    b: None,
    escapes: 0,
    lifted: 0,
    natural_residual: 0.9999314524019693,
    polish: None,
    polish_threshold: 1e-07,
    summary: "pounce: complementarity: no formulation produced an answer (8 lowered solves, 177 iterations in all); reporting the exact-product solve's verdict; 1 pair, 0 biactive; stationarity: none (not a stationary point of the model).",
};

/// Python, `PY_NOT_SEARCHED`.
const PY_NOT_SEARCHED: PyRef = PyRef {
    status: 1,
    iter_count: 19,
    x: &[
        1.8181818181807997e-09,
        1.8181818181812233e-09,
        0.0,
        9.090909090908234e-10,
    ],
    accepted_by: Some("Scholtes continuation"),
    fallback: None,
    total_iterations: 61,
    max_abs_product: 0.0,
    solves: &[
        (
            "Scholtes continuation",
            Some(1.0),
            0,
            7,
            1.2558589689073487e-17,
        ),
        ("Scholtes continuation", Some(1e-08), 0, 25, 0.0),
        ("Scholtes continuation", None, 1, 19, 0.0),
        ("exact product", Some(1e-06), 0, 9, 1.2559104188439193e-17),
        ("exact product", Some(1e-08), 0, 1, 1.2559105277173815e-17),
        ("exact product", None, 0, 0, 1.2559105277173815e-17),
    ],
    class: "not_searched",
    cert: (1, 1, 0, false, true, true),
    residual: 1.1102230246251565e-16,
    resid_tol: 0.000101,
    primal_violation: 0.0,
    b: Some(("B", 1, 2, true)),
    escapes: 0,
    lifted: 0,
    natural_residual: 0.0,
    polish: None,
    polish_threshold: 1e-07,
    summary: "pounce: complementarity: answered by the Scholtes continuation formulation (6 lowered solves, 61 iterations in all, violation = 0.0e0); 1 pair, 1 biactive; stationarity: not determined (the model or its multiplier set is too large to certify).",
};

/// Solve `m` and hold every getter to Python's report of the same run.
/// Returns the stats for the branch-specific assertions of the caller.
fn solve_and_match(m: &Model, py: &PyRef) -> PounceComplementarityStats {
    let (status, x) = m.solve();
    assert_eq!(status, py.status, "status");
    for (k, (&got, &want)) in x.iter().zip(py.x).enumerate() {
        assert_same(&format!("x[{k}]"), got, want);
    }
    assert_eq!(
        unsafe { GetIpoptIterCount(m.prob) },
        py.iter_count,
        "iter_count"
    );

    let s = stats(m.prob).expect("stats after a driver solve");
    assert_eq!(label(&s.accepted_by), py.accepted_by.unwrap_or(""));
    assert_eq!(label(&s.fallback), py.fallback.unwrap_or(""));
    assert_eq!(s.n_solves, py.solves.len() as Index);
    assert_eq!(s.total_iterations, py.total_iterations);
    assert_eq!(
        s.total_iterations,
        py.solves.iter().map(|r| r.3).sum::<Index>(),
        "total_iterations is the sum over the lowered solves"
    );
    assert_same("max_abs_product", s.max_abs_product, py.max_abs_product);
    assert_eq!(s.certificate_class, class_code(py.class));
    let (n_pairs, n_biactive, n_neither, unique, tight, feasible) = py.cert;
    assert_eq!(
        (s.n_pairs, s.n_biactive, s.n_neither),
        (n_pairs, n_biactive, n_neither)
    );
    assert_eq!(s.multipliers_unique, Index::from(unique));
    assert_eq!(s.tight, Index::from(tight));
    assert_eq!(s.feasible, Index::from(feasible));
    assert_same("residual", s.residual, py.residual);
    assert_same("resid_tol", s.resid_tol, py.resid_tol);
    assert_same("primal_violation", s.primal_violation, py.primal_violation);
    match py.b {
        None => {
            assert_eq!(s.b_verdict, POUNCE_CC_B_VERDICT_ABSENT);
            assert_eq!(
                (s.b_n_biactive, s.b_branches_solved, s.b_enumerated),
                (-1, -1, -1)
            );
        }
        Some((verdict, nb, solved, enumerated)) => {
            assert_eq!(s.b_verdict, b_code(verdict));
            assert_eq!(
                (s.b_n_biactive, s.b_branches_solved, s.b_enumerated),
                (nb, solved, Index::from(enumerated))
            );
        }
    }
    assert_eq!((s.escapes, s.lifted), (py.escapes, py.lifted));
    assert_same("natural_residual", s.natural_residual, py.natural_residual);
    match py.polish {
        None => {
            assert_eq!((s.polish_ran, s.polish_taken), (0, 0));
            assert!(s.polish_residual_before.is_nan() && s.polish_residual_after.is_nan());
        }
        Some((taken, before, after)) => {
            assert_eq!((s.polish_ran, s.polish_taken), (1, Index::from(taken)));
            assert_same("polish before", s.polish_residual_before, before);
            assert_same("polish after", s.polish_residual_after, after);
        }
    }
    assert_same("polish_threshold", s.polish_threshold, py.polish_threshold);

    for (k, &(stage, tau, status, iterations, product)) in py.solves.iter().enumerate() {
        let r = solve_record(m.prob, k as Index).expect("every k < n_solves has a record");
        assert_eq!(label(&r.stage), stage, "solve {k}");
        assert!(
            STAGES.contains(&label(&r.stage)),
            "solve {k}: unknown stage"
        );
        match tau {
            Some(t) => {
                assert_eq!(r.relaxed, 1, "solve {k}");
                assert_same(&format!("solve {k} tau"), r.tau, t);
            }
            None => {
                assert_eq!(r.relaxed, 0, "solve {k}");
                assert!(r.tau.is_nan(), "solve {k}: tau without relaxation");
            }
        }
        assert_eq!((r.status, r.iterations), (status, iterations), "solve {k}");
        assert_same(&format!("solve {k} product"), r.max_abs_product, product);
        assert!(
            r.seconds.is_finite() && r.seconds >= 0.0,
            "solve {k} seconds"
        );
    }
    assert!(solve_record(m.prob, s.n_solves).is_none(), "k == n_solves");

    assert_eq!(
        text(m.prob, POUNCE_CC_TEXT_SUMMARY).as_deref(),
        Some(py.summary)
    );
    s
}

// ---------------------------------------------------------------------------
// One test per certificate class (and per other branch a real model reaches)
// ---------------------------------------------------------------------------

/// `S`: `QpecSmall` from `(0, 0.5, 0.5)`. An answer, so no B-check runs; its
/// residual is within `10·tol`, so no polish runs either.
#[test]
fn class_s_answer_matches_python() {
    let m = Model::qpec([0.0, 0.5, 0.5]);
    let s = solve_and_match(&m, &PY_QPEC);
    assert_eq!(label(&s.strategy), "both");
    assert_eq!(s.reported, POUNCE_CC_REPORTED_ANSWER);
    assert_eq!(s.certificate_class, POUNCE_CC_CLASS_S);
    assert!(s.n_solves > 0);
    assert!(s.total_iterations > unsafe { GetIpoptIterCount(m.prob) });
    assert!(s.natural_residual <= s.polish_threshold);
    // Absent texts on this run: the polish did not run, no B-check ran.
    assert_eq!(text(m.prob, POUNCE_CC_TEXT_POLISH_REASON), None);
    assert_eq!(text(m.prob, POUNCE_CC_TEXT_B_REASON), None);
}

/// The polish branch: from the origin `QpecSmall`'s exact stage ends at
/// `min(G, H) ≈ 1e-6`, above `10·tol`, so the branch re-solve runs and is
/// taken (`python/tests/test_complementarity.py::
/// test_the_solve_report_carries_the_complementarity_block`). A taken polish
/// has no reason.
#[test]
fn polish_taken_matches_python() {
    let m = Model::qpec([0.0, 0.0, 0.0]);
    let s = solve_and_match(&m, &PY_QPEC_ZERO);
    assert_eq!((s.polish_ran, s.polish_taken), (1, 1));
    assert!(s.natural_residual <= 1e-7, "{:e}", s.natural_residual);
    assert!(s.polish_residual_before > s.polish_threshold);
    let last = solve_record(m.prob, s.n_solves - 1).unwrap();
    assert_eq!(label(&last.stage), "branch polish");
    assert_eq!(text(m.prob, POUNCE_CC_TEXT_POLISH_REASON), None);
}

/// The strategy field's other branch: `complementarity_strategy=exact` runs
/// only the exact-product stage (the penalty is the fallback and is not
/// needed here).
#[test]
fn strategy_exact_is_reported() {
    let m = Model::qpec([0.0, 0.5, 0.5]);
    m.option("complementarity_strategy", "exact");
    let s = solve_and_match(&m, &PY_QPEC_EXACT);
    assert_eq!(label(&s.strategy), "exact");
    assert_eq!(label(&s.accepted_by), "exact product");
    for k in 0..s.n_solves {
        assert_eq!(
            label(&solve_record(m.prob, k).unwrap().stage),
            "exact product"
        );
    }
}

/// `W`: `ralph1`'s origin admits no S multiplier. Reported as a weak
/// fallback (no `accepted_by`); the B-check runs and finds it B-stationary.
#[test]
fn class_w_weak_fallback_matches_python() {
    let m = Model::ralph1();
    let s = solve_and_match(&m, &PY_RALPH1);
    assert_eq!(s.certificate_class, POUNCE_CC_CLASS_W);
    assert_eq!(s.reported, POUNCE_CC_REPORTED_WEAK_FALLBACK);
    assert_eq!(label(&s.accepted_by), "");
    assert_ne!(label(&s.fallback), "");
    assert_eq!(s.b_verdict, POUNCE_CC_B_VERDICT_B);
    // Decided, so no reason.
    assert_eq!(text(m.prob, POUNCE_CC_TEXT_B_REASON), None);
}

/// `none`: no point satisfies `infeasible_pair`'s pair, every stage runs and
/// the exact-product verdict is reported. The residual is not finite (no
/// multipliers fit), which the C struct reads as NaN where the report
/// writes `null`; no B-check, no polish.
#[test]
fn class_none_verdict_matches_python() {
    let m = Model::infeasible_pair();
    let s = solve_and_match(&m, &PY_INFEASIBLE);
    assert_eq!(s.certificate_class, POUNCE_CC_CLASS_NONE);
    assert_eq!(s.reported, POUNCE_CC_REPORTED_VERDICT);
    assert_eq!((label(&s.accepted_by), label(&s.fallback)), ("", ""));
    assert!(s.residual.is_nan());
    assert_eq!(s.b_verdict, POUNCE_CC_B_VERDICT_ABSENT);
    let last = solve_record(m.prob, s.n_solves - 1).unwrap();
    assert_eq!(label(&last.stage), "l1 penalty");
}

/// `not searched`: the multipliers are a set and there are more
/// sign-constrained columns than the search takes, so nothing is claimed;
/// the driver accepts such a point as an answer.
#[test]
fn class_not_searched_matches_python() {
    let m = Model::not_searched();
    let s = solve_and_match(&m, &PY_NOT_SEARCHED);
    assert_eq!(s.certificate_class, POUNCE_CC_CLASS_NOT_SEARCHED);
    assert_eq!(s.reported, POUNCE_CC_REPORTED_ANSWER);
    assert_eq!(s.multipliers_unique, 0);
}

// ---------------------------------------------------------------------------
// The driver did not run
// ---------------------------------------------------------------------------

#[track_caller]
fn assert_nothing(prob: IpoptProblem, when: &str) {
    assert!(stats(prob).is_none(), "stats {when}");
    for k in [0, 1, -1] {
        assert!(solve_record(prob, k).is_none(), "solve {k} {when}");
    }
    for which in [
        POUNCE_CC_TEXT_SUMMARY,
        POUNCE_CC_TEXT_POLISH_REASON,
        POUNCE_CC_TEXT_B_REASON,
    ] {
        assert_eq!(text(prob, which), None, "text {which} {when}");
    }
}

#[test]
fn nothing_before_any_solve() {
    // error-path: GetPounceComplementarityStats
    // error-path: GetPounceComplementaritySolve
    // error-path: GetPounceComplementarityText
    let m = Model::qpec([0.0, 0.5, 0.5]);
    assert_nothing(m.prob, "before any solve");
}

#[test]
fn nothing_after_a_solve_without_pairs() {
    let prob = qpec::create_qpec(0);
    let m = Model {
        prob,
        data: None,
        x0: vec![0.0, 0.5, 0.5],
    };
    let (status, _) = m.solve();
    assert_eq!(status, 0);
    // The solve ran (its statistics are there); the driver did not.
    assert!(unsafe { GetIpoptIterCount(m.prob) } > 0);
    assert_nothing(m.prob, "after a solve without pairs");
}

/// Clearing the pairs does not erase the record of the solve that used
/// them (the getters describe the most recent solve, as every post-solve
/// accessor does); the next solve, without pairs, does.
#[test]
fn clearing_the_pairs_then_solving_ends_the_record() {
    let m = Model::qpec([0.0, 0.5, 0.5]);
    m.solve();
    assert!(stats(m.prob).is_some());
    assert_eq!(unsafe { ClearPounceComplementarityPairs(m.prob) }, 1);
    let s = stats(m.prob).expect("the most recent solve still ran the driver");
    assert_eq!(s.n_solves, PY_QPEC.solves.len() as Index);
    assert_eq!(
        text(m.prob, POUNCE_CC_TEXT_SUMMARY).as_deref(),
        Some(PY_QPEC.summary)
    );
    let (status, _) = m.solve();
    assert_eq!(status, 0);
    assert_nothing(
        m.prob,
        "after the pairs were cleared and the problem re-solved",
    );
}

/// A refused solve (pairs with a staged SQP working set) leaves no record,
/// even right after a driver solve on the same handle.
#[test]
fn nothing_after_a_refused_solve() {
    let m = Model::qpec([0.0, 0.5, 0.5]);
    m.solve();
    assert!(stats(m.prob).is_some());
    let xs = [0 as IpoptBoundStatus; qpec::N];
    let gs = [0 as IpoptConsStatus; qpec::M];
    assert_eq!(
        unsafe { IpoptSetWarmStartWorkingSet(m.prob, xs.as_ptr(), gs.as_ptr()) },
        1
    );
    let (status, _) = m.solve();
    assert_eq!(status, -11, "Invalid_Problem_Definition");
    assert_nothing(m.prob, "after a refused solve");
}

#[test]
fn null_arguments_answer_nothing() {
    unsafe {
        let mut s: PounceComplementarityStats = sentinel();
        s.struct_size = size_of::<PounceComplementarityStats>() as Index;
        assert_eq!(GetPounceComplementarityStats(ptr::null_mut(), &mut s), 0);
        let mut r: PounceComplementaritySolve = sentinel();
        r.struct_size = size_of::<PounceComplementaritySolve>() as Index;
        assert_eq!(GetPounceComplementaritySolve(ptr::null_mut(), 0, &mut r), 0);
        let mut buf = [0 as c_char; 8];
        assert_eq!(
            GetPounceComplementarityText(
                ptr::null_mut(),
                POUNCE_CC_TEXT_SUMMARY,
                buf.as_mut_ptr(),
                8
            ),
            -1
        );
        assert_eq!(
            GetPounceComplementarityText(
                ptr::null_mut(),
                POUNCE_CC_TEXT_SUMMARY,
                ptr::null_mut(),
                0
            ),
            -1
        );
    }
    // On a handle that has a record: NULL output pointers are refused, and
    // the record is still there afterwards.
    let m = Model::qpec([0.0, 0.5, 0.5]);
    m.solve();
    unsafe {
        assert_eq!(GetPounceComplementarityStats(m.prob, ptr::null_mut()), 0);
        assert_eq!(GetPounceComplementaritySolve(m.prob, 0, ptr::null_mut()), 0);
    }
    assert!(stats(m.prob).is_some());
    assert!(solve_record(m.prob, 0).is_some());
}

#[test]
fn solve_index_out_of_range_is_refused() {
    let m = Model::qpec([0.0, 0.5, 0.5]);
    m.solve();
    let n = stats(m.prob).unwrap().n_solves;
    assert!(n > 1);
    assert!(solve_record(m.prob, 0).is_some());
    assert!(solve_record(m.prob, n - 1).is_some());
    for k in [n, n + 1, -1, Index::MIN, Index::MAX] {
        assert!(solve_record(m.prob, k).is_none(), "k = {k}");
    }
}

/// `snprintf`'s contract, every branch: the size query, an exact buffer, a
/// truncating one, a one-byte one (just the NUL), `buflen == 0` and
/// negative with a real buffer (nothing written), and an unknown `which`.
#[test]
fn text_follows_the_snprintf_contract() {
    let m = Model::qpec([0.0, 0.5, 0.5]);
    m.solve();
    let full = PY_QPEC.summary.as_bytes();
    let len = full.len() as Index;
    let call = |buf: &mut [c_char], buflen: Index| unsafe {
        GetPounceComplementarityText(m.prob, POUNCE_CC_TEXT_SUMMARY, buf.as_mut_ptr(), buflen)
    };
    unsafe {
        assert_eq!(
            GetPounceComplementarityText(m.prob, POUNCE_CC_TEXT_SUMMARY, ptr::null_mut(), 0),
            len
        );
        // NULL buffer with a positive length: still only a size query.
        assert_eq!(
            GetPounceComplementarityText(m.prob, POUNCE_CC_TEXT_SUMMARY, ptr::null_mut(), 64),
            len
        );
    }
    const PAT: c_char = 0x7f;
    // Exact fit and roomier: the whole text and a NUL, nothing past it.
    for room in [len + 1, len + 10] {
        let mut buf = vec![PAT; len as usize + 16];
        assert_eq!(call(&mut buf, room), len);
        let got: Vec<u8> = buf[..len as usize].iter().map(|&c| c as u8).collect();
        assert_eq!(got, full);
        assert_eq!(buf[len as usize], 0);
        assert!(
            buf[len as usize + 1..].iter().all(|&c| c == PAT),
            "wrote past the NUL"
        );
    }
    // Truncating: buflen-1 bytes and a NUL, the full length returned.
    for buflen in [len, 5, 2] {
        let mut buf = vec![PAT; len as usize + 16];
        assert_eq!(call(&mut buf, buflen), len, "buflen {buflen}");
        let keep = buflen as usize - 1;
        let got: Vec<u8> = buf[..keep].iter().map(|&c| c as u8).collect();
        assert_eq!(got, &full[..keep], "buflen {buflen}");
        assert_eq!(buf[keep], 0, "buflen {buflen}");
        assert!(
            buf[keep + 1..].iter().all(|&c| c == PAT),
            "buflen {buflen}: wrote past buflen"
        );
    }
    // One byte: just the NUL.
    let mut buf = vec![PAT; 4];
    assert_eq!(call(&mut buf, 1), len);
    assert_eq!(buf, [0, PAT, PAT, PAT]);
    // Zero or negative: nothing written, the length returned.
    for buflen in [0, -1, Index::MIN] {
        let mut buf = vec![PAT; 4];
        assert_eq!(call(&mut buf, buflen), len, "buflen {buflen}");
        assert!(buf.iter().all(|&c| c == PAT), "buflen {buflen} wrote");
    }
    // Unknown `which`: no such text, nothing written.
    for which in [3, -1, 99] {
        let mut buf = vec![PAT; 4];
        assert_eq!(
            unsafe { GetPounceComplementarityText(m.prob, which, buf.as_mut_ptr(), 4) },
            -1,
            "which {which}"
        );
        assert!(buf.iter().all(|&c| c == PAT));
    }
}

/// The `struct_size` contract (`pounce.h`, "Extensible output structs"),
/// on a handle that has a record, so the size check is the branch that
/// decides: refused untouched below `sizeof`, filled at `sizeof`, and never
/// a byte past `sizeof` when the caller's struct is larger. Case (c)'s fill
/// is the same one `stats()` reads for every value test.
#[test]
fn stats_struct_size_contract() {
    // error-path: GetPounceComplementarityStats
    let m = Model::qpec([0.0, 0.5, 0.5]);
    m.solve();
    let s = check_struct_size_contract("PounceComplementarityStats", |p| unsafe {
        GetPounceComplementarityStats(m.prob, p)
    });
    assert_eq!(
        format!("{s:?}"),
        format!("{:?}", stats(m.prob).unwrap()),
        "case (c) is the ordinary fill"
    );
}

/// The same for every lowered-solve record (`solve_record()`'s fill).
#[test]
fn solve_struct_size_contract() {
    // error-path: GetPounceComplementaritySolve
    let m = Model::qpec([0.0, 0.5, 0.5]);
    m.solve();
    let n = stats(m.prob).unwrap().n_solves;
    assert!(n >= 1);
    for k in 0..n {
        let r = check_struct_size_contract("PounceComplementaritySolve", |p| unsafe {
            GetPounceComplementaritySolve(m.prob, k, p)
        });
        assert_eq!(
            format!("{r:?}"),
            format!("{:?}", solve_record(m.prob, k).unwrap()),
            "case (c) is the ordinary fill"
        );
    }
}
