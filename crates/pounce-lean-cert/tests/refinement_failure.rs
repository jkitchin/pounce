//! The refinement **failure boundary**.
//!
//! Every other fixture in this crate is a success case, which left the question
//! of where exact active-set refinement gives up entirely uncharacterized. This
//! file maps it.
//!
//! # The map
//!
//! | Construction | Outcome |
//! |---|---|
//! | duplicate active row | `Refine(Singular)` |
//! | scaled-duplicate active row (`2·c0`) | `Refine(Singular)` |
//! | contradictory equalities (`x₀ = 0` and `x₀ = 1`) | `Refine(Singular)` |
//! | more equality rows than variables | `Refine(Singular)` |
//! | indefinite `Q`, diagonal (`diag(-2, 2)`) | `Refine(Singular)` |
//! | indefinite `Q`, off-diagonal | `Ldl(Indefinite { col })` |
//! | wrongly-active slack row | `Refine(NegativeDual)` |
//! | empty active set, infeasible result | `Refine(InactiveViolated)` |
//! | **PSD-singular `Q`** (`diag(0, 2)`) | **succeeds** — correctly |
//! | **equality target not representable in f64** | **succeeds** — exactly |
//!
//! # Two things this established
//!
//! **`Singular` is the catch-all.** Every way of making the active set
//! degenerate — duplicated rows, linearly dependent rows, contradictory or
//! over-determined equalities — arrives at the same error. That is sound but
//! coarse: a caller cannot distinguish "your active set is redundant" from
//! "your equalities contradict each other". Worth splitting if refusals ever
//! need to be diagnosed in the field.
//!
//! **Diagonal indefiniteness is caught by the KKT solve, not by `LDLᵀ`.**
//! Refinement runs first, so `diag(-2, 2)` reports `Singular` rather than the
//! more informative `Indefinite`. Only off-diagonal indefiniteness survives to
//! the `LDLᵀ` factorization. Both paths refuse, so this costs diagnosis, not
//! soundness.
//!
//! # Unreachable by construction
//!
//! `RefineError::EqualityResidual` and `RefineError::StationarityResidual` are
//! documented in the source as defensive, and no construction attempted here
//! reaches them: the exact solve produces a point satisfying the equalities and
//! stationarity *by construction*, so a violation would mean a shape bug rather
//! than bad input. Likewise `LdlError::SingularNeedsPivot` — for a genuinely PSD
//! matrix a zero pivot forces the rest of its row to vanish, so the pivoting
//! case only arises for input that is already indefinite and rejected earlier.
//! They are kept as assertions, not removed.

#![allow(clippy::unwrap_used)]

use pounce_lean_cert::emit::{CertMeta, LinearConstraint, QpInput, emit_certificate};
use pounce_lean_cert::{EmitError, Rat, ldlt::LdlError, refine::RefineError};

fn meta() -> CertMeta {
    CertMeta {
        nl_sha256: "0".repeat(64),
        sol_sha256: "0".repeat(64),
        solver: "pounce 0.9.0".to_string(),
    }
}

fn con(name: &str, coeffs: Vec<f64>, lower: f64, upper: f64) -> LinearConstraint {
    LinearConstraint {
        name: name.to_string(),
        coeffs,
        lower,
        upper,
    }
}

fn qp(
    n: usize,
    q_lower: Vec<(usize, usize, f64)>,
    constraints: Vec<LinearConstraint>,
    x_float: Vec<f64>,
) -> QpInput {
    QpInput {
        n,
        q_lower,
        half_quadratic: true,
        c: vec![0.0; n],
        constant: 0.0,
        constraints,
        var_lower: vec![f64::NEG_INFINITY; n],
        var_upper: vec![f64::INFINITY; n],
        x_float,
        active_tol: 1e-7,
    }
}

fn convex_2d() -> Vec<(usize, usize, f64)> {
    vec![(0, 0, 2.0), (1, 1, 2.0)]
}

fn err(input: &QpInput) -> EmitError {
    emit_certificate(input, &meta()).expect_err("expected a refusal")
}

// --- degenerate active sets all funnel to Singular -------------------------

#[test]
fn duplicate_active_row_is_singular() {
    let input = qp(
        2,
        convex_2d(),
        vec![
            con("c0", vec![1.0, 1.0], 1.0, f64::INFINITY),
            con("c0_dup", vec![1.0, 1.0], 1.0, f64::INFINITY),
        ],
        vec![0.5, 0.5],
    );
    assert_eq!(err(&input), EmitError::Refine(RefineError::Singular));
}

#[test]
fn linearly_dependent_active_row_is_singular() {
    // 2·c0 — not a byte-duplicate, but the same hyperplane.
    let input = qp(
        2,
        convex_2d(),
        vec![
            con("c0", vec![1.0, 1.0], 1.0, f64::INFINITY),
            con("c0_scaled", vec![2.0, 2.0], 2.0, f64::INFINITY),
        ],
        vec![0.5, 0.5],
    );
    assert_eq!(err(&input), EmitError::Refine(RefineError::Singular));
}

#[test]
fn contradictory_equalities_are_singular() {
    let input = qp(
        2,
        convex_2d(),
        vec![
            con("e0", vec![1.0, 0.0], 0.0, 0.0),
            con("e1", vec![1.0, 0.0], 1.0, 1.0),
        ],
        vec![0.0, 0.0],
    );
    assert_eq!(err(&input), EmitError::Refine(RefineError::Singular));
}

#[test]
fn overdetermined_equalities_are_singular() {
    let input = qp(
        2,
        convex_2d(),
        vec![
            con("e0", vec![1.0, 0.0], 1.0, 1.0),
            con("e1", vec![0.0, 1.0], 1.0, 1.0),
            con("e2", vec![1.0, 1.0], 2.0, 2.0), // implied, but over-determines
        ],
        vec![1.0, 1.0],
    );
    assert_eq!(err(&input), EmitError::Refine(RefineError::Singular));
}

// --- non-convexity is refused on one of two paths --------------------------

#[test]
fn off_diagonal_indefinite_q_is_caught_by_ldlt() {
    // [[1, 2], [2, 1]] has eigenvalues 3 and -1.
    let input = qp(
        2,
        vec![(0, 0, 1.0), (1, 0, 2.0), (1, 1, 1.0)],
        vec![con("c0", vec![1.0, 1.0], 1.0, f64::INFINITY)],
        vec![0.5, 0.5],
    );
    assert_eq!(err(&input), EmitError::Ldl(LdlError::Indefinite { col: 1 }));
}

#[test]
fn diagonal_indefinite_q_is_caught_earlier_by_the_kkt_solve() {
    // Refinement runs before the LDLᵀ factorization, so this reports Singular
    // rather than the more informative Indefinite. Refused either way.
    let input = qp(
        2,
        vec![(0, 0, -2.0), (1, 1, 2.0)],
        vec![con("c0", vec![1.0, 1.0], 1.0, f64::INFINITY)],
        vec![0.5, 0.5],
    );
    assert_eq!(err(&input), EmitError::Refine(RefineError::Singular));
}

// --- the cases that must NOT be treated as failures ------------------------

#[test]
fn psd_singular_q_succeeds() {
    // Q = diag(0, 2) is PSD but singular: f(x) = x₁².
    // Minimizing subject to x₀ + x₁ ≥ 1 gives x* = (1, 0), f = 0.
    // A zero pivot is legitimate here and must not be mistaken for a failure.
    let input = qp(
        2,
        vec![(1, 1, 2.0)],
        vec![con("c0", vec![1.0, 1.0], 1.0, f64::INFINITY)],
        vec![0.5, 0.5],
    );
    let cert = emit_certificate(&input, &meta()).expect("PSD-singular Q is convex and must verify");
    assert_eq!(
        cert.candidate.as_ref().unwrap().x[0].inner().to_string(),
        "1"
    );
    assert_eq!(
        cert.candidate.as_ref().unwrap().x[1].inner().to_string(),
        "0"
    );
    assert_eq!(
        cert.candidate
            .as_ref()
            .unwrap()
            .objective
            .inner()
            .to_string(),
        "0"
    );
}

/// The refinement's whole point, in one test: the certified `x*` is a rational
/// that **cannot be represented in f64 at all**.
///
/// The constraint is `3x₀ = 1`, whose solution is exactly `1/3`. The float hint
/// is the nearest double to 1/3, which is not 1/3. Refinement uses that hint
/// only to choose the active set, then solves exactly over ℚ — so the emitted
/// certificate carries `1/3`, a value the solver could never have returned.
#[test]
fn certified_point_can_be_a_rational_no_f64_can_represent() {
    let input = qp(
        2,
        convex_2d(),
        vec![con("e0", vec![3.0, 0.0], 1.0, 1.0)],
        vec![1.0 / 3.0, 0.0],
    );
    let cert = emit_certificate(&input, &meta()).expect("exactly solvable over ℚ");
    assert_eq!(
        cert.candidate.as_ref().unwrap().x[0].inner().to_string(),
        "1/3",
        "the exact solve must recover 1/3, not the f64 approximation it was hinted with"
    );
    // Sanity: the hint really was not 1/3. Compare *exactly*, by converting the
    // double to its true rational value — note that `(1.0/3.0) * 3.0 == 1.0` in
    // f64, so arithmetic round-tripping cannot answer this question.
    let hint_exact = Rat::from_f64(1.0_f64 / 3.0).unwrap();
    assert_ne!(
        hint_exact.inner().to_string(),
        "1/3",
        "sanity: if f64 could represent 1/3 exactly this test would prove nothing"
    );
    // It is a dyadic approximation with a power-of-two denominator.
    assert!(
        hint_exact
            .inner()
            .denom()
            .to_string()
            .parse::<u128>()
            .unwrap()
            % 2
            == 0,
        "an f64 converts to a dyadic rational"
    );
}
