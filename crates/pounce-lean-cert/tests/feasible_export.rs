//! The `feasible` verdict: what a solve supports when it does not support
//! optimality.
//!
//! Every other verdict in the schema is exact — `tolerance` is `0` and the
//! emitted point satisfies its claims with zero residual. `feasible` is the one
//! that carries a nonzero ε, and this file pins down why that is honest rather
//! than a loophole.
//!
//! The certificate makes two claims, both decided by exact ℚ arithmetic on the
//! consumer side:
//!
//! 1. the reported `x*` violates no constraint by more than ε; and
//! 2. a **genuinely** feasible point exists within ∞-distance ε of `x*`.
//!
//! Claim 2 is the load-bearing one. Claim 1 alone would be satisfiable by a
//! point sitting just outside an empty feasible region — ε-feasibility of a
//! problem with no solutions is not a contradiction. The witness `x̂` closes
//! that gap constructively, and because the constraints are linear the
//! projection producing it is exact, so no interval arithmetic is involved.

#![allow(clippy::unwrap_used)]

use num_rational::BigRational;
use num_traits::Zero;
use pounce_lean_cert::emit::{CertMeta, LinearConstraint, QpInput, emit_feasible_certificate};
use pounce_lean_cert::refine::RefineError;
use pounce_lean_cert::{EmitError, Rat};

fn meta() -> CertMeta {
    CertMeta {
        nl_sha256: "0".repeat(64),
        sol_sha256: "0".repeat(64),
        solver: "pounce 0.9.0".to_string(),
    }
}

fn rat(r: &Rat) -> BigRational {
    r.inner().clone()
}

/// `min x₀ + x₁  s.t.  x₀ + 2x₁ ≥ 4,  2x₀ + x₁ ≥ 4`.
///
/// The same instance `lp.rs` certifies as a global min — reused deliberately.
/// Its optimum `(4/3, 4/3)` is not representable in f64, so the returned float
/// is *infeasible*, which is exactly the situation this verdict exists for.
fn vertex_lp(x_float: Vec<f64>) -> QpInput {
    QpInput {
        n: 2,
        q_lower: vec![],
        half_quadratic: true,
        c: vec![1.0, 1.0],
        constant: 0.0,
        constraints: vec![
            LinearConstraint {
                name: "c0".to_string(),
                coeffs: vec![1.0, 2.0],
                lower: 4.0,
                upper: f64::INFINITY,
            },
            LinearConstraint {
                name: "c1".to_string(),
                coeffs: vec![2.0, 1.0],
                lower: 4.0,
                upper: f64::INFINITY,
            },
        ],
        var_lower: vec![f64::NEG_INFINITY; 2],
        var_upper: vec![f64::INFINITY; 2],
        x_float,
        active_tol: 1e-7,
    }
}

/// The float POUNCE actually returns here is infeasible by ~7e-10 in both rows.
/// The certificate reports it verbatim as the candidate and ships the exact
/// vertex as the witness.
#[test]
fn a_float_that_misses_feasibility_still_certifies_feasible() {
    let x_float = vec![1.33333333261541820, 1.33333333261541842];
    let cert = emit_feasible_certificate(&vertex_lp(x_float.clone()), &meta()).unwrap();

    assert_eq!(cert.verdict, "feasible");
    assert_eq!(cert.problem_class, "qp-convex");

    // The candidate is the solver's point untouched — not a refined one. The
    // verdict is about what the solver reported.
    let cand = cert.candidate.as_ref().unwrap();
    let expect: Vec<BigRational> = x_float
        .iter()
        .map(|&v| Rat::from_f64(v).unwrap().inner().clone())
        .collect();
    assert_eq!(cand.x.iter().map(rat).collect::<Vec<_>>(), expect);

    // The witness is the exact vertex (4/3, 4/3) — a point no f64 can hold.
    let xhat = cert.witnesses.feasible_witness.as_ref().unwrap();
    let four_thirds = BigRational::new(4.into(), 3.into());
    assert_eq!(
        xhat.xhat.iter().map(rat).collect::<Vec<_>>(),
        vec![four_thirds.clone(), four_thirds],
        "both rows are active, so the projection lands on the vertex exactly"
    );

    // No optimality witnesses: this verdict claims nothing about optimality,
    // and shipping duals or a PSD factor would suggest otherwise.
    assert!(cert.witnesses.duals.is_none());
    assert!(cert.witnesses.hessian_psd.is_none());
    assert!(cert.witnesses.active_set.is_none());
}

/// ε bounds both claims and is a round number, so the generated Lean reads as a
/// declared tolerance rather than a 60-digit artifact of binary floating point.
#[test]
fn the_tolerance_is_a_rounded_up_bound_on_both_claims() {
    let x_float = vec![1.33333333261541820, 1.33333333261541842];
    let cert = emit_feasible_certificate(&vertex_lp(x_float), &meta()).unwrap();

    let eps = rat(&cert.tolerance);
    assert!(
        eps > BigRational::zero(),
        "the float is genuinely infeasible"
    );

    // One significant digit: numerator 1..9 over a power of ten.
    let num = eps.numer().clone();
    assert!(
        num >= 1.into() && num <= 9.into(),
        "ε = {eps} should carry a single significant digit"
    );

    let cand: Vec<BigRational> = cert.candidate.as_ref().unwrap().x.iter().map(rat).collect();
    let xhat: Vec<BigRational> = cert
        .witnesses
        .feasible_witness
        .as_ref()
        .unwrap()
        .xhat
        .iter()
        .map(rat)
        .collect();

    // Claim 2's side condition: x̂ is within ε of x*.
    for (h, s) in xhat.iter().zip(&cand) {
        let dist = if h > s { h - s } else { s - h };
        assert!(
            dist <= eps,
            "witness is {dist} from the candidate, ε = {eps}"
        );
    }

    // Claim 1: every row is violated by at most ε at x*.
    for con in cert.problem.constraint_rows() {
        let ax: BigRational = con
            .coeffs
            .iter()
            .map(|c| c.inner().clone())
            .zip(&cand)
            .map(|(a, x)| a * x)
            .sum();
        let lo = con.lower.finite().unwrap().inner().clone();
        assert!(
            &ax + &eps >= lo,
            "row violated by more than ε at the candidate"
        );
    }
}

/// An exactly-feasible float gets ε = 0 — the tolerance is measured, not a
/// constant the emitter carries around.
#[test]
fn an_exactly_feasible_float_certifies_with_zero_tolerance() {
    // Strictly interior, and exactly representable, so nothing is active and
    // the projection is the identity.
    let cert = emit_feasible_certificate(&vertex_lp(vec![4.0, 4.0]), &meta()).unwrap();
    assert!(rat(&cert.tolerance).is_zero());

    let cand: Vec<BigRational> = cert.candidate.as_ref().unwrap().x.iter().map(rat).collect();
    let xhat: Vec<BigRational> = cert
        .witnesses
        .feasible_witness
        .as_ref()
        .unwrap()
        .xhat
        .iter()
        .map(rat)
        .collect();
    assert_eq!(cand, xhat, "an already-feasible point needs no correction");
}

/// The verdict is weak, not free. When no exactly-feasible point can be
/// produced, the answer is a refusal — publishing claim 1 alone would be
/// asserting ε-feasibility of a problem that has no feasible points at all.
#[test]
fn an_infeasible_problem_is_refused_rather_than_declared_eps_feasible() {
    let input = QpInput {
        n: 1,
        q_lower: vec![],
        half_quadratic: true,
        c: vec![1.0],
        constant: 0.0,
        constraints: vec![
            LinearConstraint {
                name: "lo".to_string(),
                coeffs: vec![1.0],
                lower: 1.0,
                upper: f64::INFINITY,
            },
            LinearConstraint {
                name: "hi".to_string(),
                coeffs: vec![1.0],
                lower: f64::NEG_INFINITY,
                upper: 0.0,
            },
        ],
        var_lower: vec![f64::NEG_INFINITY],
        var_upper: vec![f64::INFINITY],
        // Splitting the difference: within 1/2 of both rows, satisfying neither.
        x_float: vec![0.5],
        active_tol: 1e-7,
    };
    let err = emit_feasible_certificate(&input, &meta()).unwrap_err();
    assert!(
        matches!(err, EmitError::Refine(RefineError::InactiveViolated { .. })),
        "expected a refusal naming the violated row, got {err:?}"
    );
}

/// Equality rows are the case the float is guaranteed to miss, and the one the
/// projection was written for.
#[test]
fn an_equality_residual_is_corrected_exactly() {
    let input = QpInput {
        n: 2,
        q_lower: vec![],
        half_quadratic: true,
        c: vec![1.0, 1.0],
        constant: 0.0,
        constraints: vec![LinearConstraint {
            name: "sum".to_string(),
            coeffs: vec![3.0, 0.0],
            lower: 1.0,
            upper: 1.0,
        }],
        var_lower: vec![f64::NEG_INFINITY; 2],
        var_upper: vec![f64::INFINITY; 2],
        // 3 · (1/3 in f64) ≠ 1 exactly, which is the whole difficulty.
        x_float: vec![1.0 / 3.0, 2.0],
        active_tol: 1e-7,
    };
    let cert = emit_feasible_certificate(&input, &meta()).unwrap();
    let xhat: Vec<BigRational> = cert
        .witnesses
        .feasible_witness
        .as_ref()
        .unwrap()
        .xhat
        .iter()
        .map(rat)
        .collect();
    assert_eq!(
        xhat,
        vec![
            BigRational::new(1.into(), 3.into()),
            BigRational::from_integer(2.into())
        ],
        "the equality is met exactly at 1/3, and the free coordinate does not move"
    );
    assert!(
        rat(&cert.tolerance) > BigRational::zero(),
        "the reported point still carries the float residual"
    );
}
