//! SOS end to end: solve the SDP in floating point, then recover an **exact**
//! rational certificate from its output.
//!
//! The two halves were built separately — `sos_constrained_lower_bound_gram`
//! extracts the Gram blocks, `round_gram` makes them exact — and each was
//! tested against data the other did not produce. This is the join: the actual
//! solver output, not a hand-placed float near the answer.
//!
//! That distinction matters. A rounding routine tested only on inputs already
//! close to the target demonstrates arithmetic, not viability; the question is
//! whether a real interior-point solution lands close enough to round.

#![allow(clippy::unwrap_used)]

use num_rational::BigRational;
use pounce_convex::sos::{PolyProblem, Polynomial, sos_constrained_lower_bound_gram, sos_opts};
use pounce_feral::FeralSolverInterface;
use pounce_lean_cert::emit::CertMeta;
use pounce_lean_cert::emit_sos::{SosEmitError, SosInput, emit_sos_certificate};
use pounce_lean_cert::round_gram::{RoundError, round_gram};

fn backend() -> Box<dyn pounce_linsol::SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn r(n: i64) -> BigRational {
    BigRational::from_integer(n.into())
}

/// `p(x) = x⁴ − 2x² + 2`, global minimum 1 at `x = ±1`.
fn quartic() -> Polynomial {
    Polynomial::new(1, vec![(vec![4], 1.0), (vec![2], -2.0), (vec![0], 2.0)])
}

#[test]
fn solver_output_rounds_to_an_exact_sos_certificate() {
    let (bound, gram) =
        sos_constrained_lower_bound_gram(&PolyProblem::new(quartic()), None, &sos_opts(), backend);
    assert!(
        (bound.lower_bound - 1.0).abs() < 1e-6,
        "SDP bound should be ≈ 1, got {}",
        bound.lower_bound
    );
    assert_eq!(gram.len(), 1);
    let block = &gram[0];

    // The exact bound is chosen, not read off the float: γ = 1.
    let p_terms = vec![
        (vec![4usize], r(1)),
        (vec![2usize], r(-2)),
        (vec![0usize], r(2)),
    ];
    let g = round_gram(&p_terms, &r(1), &block.basis, &block.matrix, 1)
        .expect("the solver's Gram must round to an exact certificate");

    // (x² − 1)² is the unique PSD solution; anything else means the pipeline
    // produced a different — and therefore wrong — decomposition.
    let bn = block.basis.len();
    assert_eq!(bn, 3, "basis (1, x, x²)");
    assert_eq!(g[0][0], r(1));
    assert_eq!(g[0][2], r(-1));
    assert_eq!(g[1][1], r(0));
    assert_eq!(g[2][2], r(1));

    // And the identity holds exactly, as a polynomial rather than at samples.
    for x in [-4i64, -2, -1, 0, 1, 3, 7] {
        let xv = BigRational::from_integer(x.into());
        let m: Vec<BigRational> = (0..bn).map(|k| xv.pow(k as i32)).collect();
        let quad: BigRational = (0..bn)
            .flat_map(|i| (0..bn).map(move |j| (i, j)))
            .map(|(i, j)| &m[i] * &g[i][j] * &m[j])
            .sum();
        assert_eq!(quad, xv.pow(4) - r(2) * xv.pow(2) + r(1), "at x = {x}");
    }
}

/// The pipeline must not certify a bound the polynomial does not satisfy, even
/// when handed a perfectly good Gram from the solver.
#[test]
fn a_bound_above_the_true_minimum_is_still_refused_end_to_end() {
    let (_b, gram) =
        sos_constrained_lower_bound_gram(&PolyProblem::new(quartic()), None, &sos_opts(), backend);
    let p_terms = vec![
        (vec![4usize], r(1)),
        (vec![2usize], r(-2)),
        (vec![0usize], r(2)),
    ];
    // The minimum is 1; 3/2 is not a valid lower bound.
    let err = round_gram(
        &p_terms,
        &BigRational::new(3.into(), 2.into()),
        &gram[0].basis,
        &gram[0].matrix,
        1,
    )
    .unwrap_err();
    assert_eq!(err, RoundError::NotPsd { block: 0 });
}

fn meta() -> CertMeta {
    CertMeta {
        nl_sha256: "0".repeat(64),
        sol_sha256: "1".repeat(64),
        solver: "pounce test".to_string(),
    }
}

/// The whole producer chain, from the SDP to a writable certificate: the
/// emitter is handed *only* the solver's own float output — no hand-chosen γ,
/// no hand-placed Gram — and must find the exact certificate itself.
///
/// The unit tests for `emit_sos_certificate` feed it a plausible interior-point
/// matrix; this feeds it the real one. That is the claim worth testing, because
/// the ladder search is exactly the step whose viability depends on where an
/// actual solver lands.
#[test]
fn the_solver_output_alone_produces_a_writable_certificate() {
    let (bound, gram) =
        sos_constrained_lower_bound_gram(&PolyProblem::new(quartic()), None, &sos_opts(), backend);
    let cert = emit_sos_certificate(
        &SosInput {
            n: 1,
            terms: vec![(vec![4], 1.0), (vec![2], -2.0), (vec![0], 2.0)],
            basis: gram[0].basis.clone(),
            gram_float: gram[0].matrix.clone(),
            bound_float: bound.lower_bound,
            // No iterate, so this stays the bound path; the upgrade to a
            // minimum is exercised below.
            x_float: vec![],
            constraints: Vec::new(),
            neighborhood: None,
        },
        &meta(),
    )
    .expect("the solver's own output must yield an exact certificate");

    assert_eq!(cert.verdict, "global-lower-bound");
    // The emitted bound is exactly 1 — not the solver's float, which is merely
    // near it. Recovering the *tight* bound is the point: γ = 0 would also be
    // provable and would say nothing.
    assert_eq!(
        cert.bound.as_ref().unwrap().inner(),
        &BigRational::from_integer(1.into())
    );
    assert!(cert.candidate.is_none());
    assert_eq!(cert.witnesses.sos.as_ref().unwrap().len(), 1);

    // And it survives a round trip through the on-disk JSON, since that — not
    // the in-memory struct — is what the Lean codegen reads.
    let json = pounce_lean_cert::to_canonical_json(&cert).unwrap();
    let back: pounce_lean_cert::Certificate = serde_json::from_str(&json).unwrap();
    assert_eq!(back.verdict, cert.verdict);
    assert_eq!(back.problem.polynomial.as_ref().unwrap().terms.len(), 3);
    assert!(
        back.problem.objective.is_none(),
        "the QP half must stay absent through serialization"
    );
}

/// A polynomial that is nonnegative but **not** a sum of squares: Motzkin's
/// `x⁴y² + x²y⁴ − 3x²y² + 1`, which has minimum 0 yet no SOS decomposition at
/// any degree in these variables. The degree-3 relaxation cannot certify γ = 0,
/// and the emitter must say so rather than manufacture a witness.
#[test]
fn a_nonnegative_but_not_sos_polynomial_is_refused_at_the_tight_bound() {
    let motzkin = vec![
        (vec![4usize, 2usize], 1.0),
        (vec![2, 4], 1.0),
        (vec![2, 2], -3.0),
        (vec![0, 0], 1.0),
    ];
    let err = emit_sos_certificate(
        &SosInput {
            n: 2,
            terms: motzkin.clone(),
            // The natural degree-3 basis for a degree-6 polynomial.
            basis: vec![
                vec![0, 0],
                vec![1, 0],
                vec![0, 1],
                vec![2, 0],
                vec![1, 1],
                vec![0, 2],
                vec![3, 0],
                vec![2, 1],
                vec![1, 2],
                vec![0, 3],
            ],
            gram_float: vec![vec![0.0; 10]; 10],
            bound_float: 0.0,
            x_float: vec![],
            constraints: Vec::new(),
            neighborhood: None,
        },
        &meta(),
    )
    .unwrap_err();
    assert!(
        matches!(err, SosEmitError::NoExactCertificate { .. }),
        "Motzkin is nonnegative but not SOS; got {err:?}"
    );
}
