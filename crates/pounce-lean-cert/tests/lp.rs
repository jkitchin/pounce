//! Linear programs certify through the quadratic path unchanged.
//!
//! An LP is just a QP with `Q = 0`. The zero matrix is positive semidefinite,
//! its `LDLᵀ` witness is `L = I`, `D = 0`, and the convex-KKT theorem applies
//! verbatim — so no new mathematics, no new schema, and no new Lean lemma is
//! needed. This file confirms that rather than assuming it.
//!
//! The end-to-end counterpart is the `certify_lp` fixture, which goes
//! solve → certify → codegen → `lake build` and kernel-checks with axioms
//! `{propext, Classical.choice, Quot.sound}`.

#![allow(clippy::unwrap_used)]

use pounce_lean_cert::emit::{CertMeta, LinearConstraint, QpInput, emit_certificate};

fn meta() -> CertMeta {
    CertMeta {
        nl_sha256: "0".repeat(64),
        sol_sha256: "0".repeat(64),
        solver: "pounce 0.9.0".to_string(),
    }
}

/// `min x₀ + x₁  s.t.  x₀ + 2x₁ ≥ 4,  2x₀ + x₁ ≥ 4`.
///
/// Both rows are active at the vertex `x* = (4/3, 4/3)`, with `f = 8/3` and
/// multipliers `λ = (1/3, 1/3)`. Chosen so the optimum is **not representable
/// in f64**: the solver returns roughly `1.33333333261`, off by about 7e-10.
fn vertex_lp(x_float: Vec<f64>) -> QpInput {
    QpInput {
        n: 2,
        q_lower: vec![], // Q = 0 — the whole point
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

#[test]
fn lp_certifies_through_the_quadratic_path() {
    // The float the LP IPM actually returns for this instance.
    let cert = emit_certificate(
        &vertex_lp(vec![1.33333333261541820, 1.33333333261541842]),
        &meta(),
    )
    .expect("an LP is a convex QP with Q = 0 and must certify");

    assert_eq!(cert.verdict, "global-min");
    assert_eq!(cert.problem_class, "qp-convex");

    // Q is present but empty: the all-zero matrix.
    assert!(
        cert.problem.objective.q.entries.is_empty(),
        "Q should have no entries for an LP"
    );

    // The vertex is recovered exactly, from a float that was ~7e-10 away.
    assert_eq!(cert.candidate.x[0].inner().to_string(), "4/3");
    assert_eq!(cert.candidate.x[1].inner().to_string(), "4/3");
    assert_eq!(cert.candidate.objective.inner().to_string(), "8/3");

    // Both rows active, both multipliers 1/3 ≥ 0.
    assert_eq!(cert.witnesses.duals[0].inner().to_string(), "1/3");
    assert_eq!(cert.witnesses.duals[1].inner().to_string(), "1/3");
    assert_eq!(cert.witnesses.active_set, vec![0, 1]);

    // Exact: no tolerance is claimed.
    assert_eq!(cert.tolerance.inner().to_string(), "0");
}

/// The zero Hessian must produce a usable PSD witness, not a degenerate one:
/// `L = I`, `D = 0`, which satisfies `Q = L·diag(D)·Lᵀ` and `D ≥ 0`.
#[test]
fn zero_hessian_yields_a_valid_psd_witness() {
    let cert =
        emit_certificate(&vertex_lp(vec![4.0 / 3.0, 4.0 / 3.0]), &meta()).expect("must certify");
    let psd = &cert.witnesses.hessian_psd;
    assert_eq!(psd.d.len(), 2, "one pivot per variable");
    for (i, di) in psd.d.iter().enumerate() {
        assert_eq!(di.inner().to_string(), "0", "D[{i}] should be 0 for an LP");
    }
    // Unit-lower L with no strictly-below-diagonal entries is the identity.
    assert!(
        psd.l.entries.is_empty(),
        "L should be the identity for a zero Hessian"
    );
}

/// An LP whose optimum is unbounded below has no KKT point, so no multipliers
/// can satisfy stationarity — it must be refused, not certified.
#[test]
fn unbounded_lp_is_refused() {
    let mut input = vertex_lp(vec![1.0, 1.0]);
    // Flip the objective so minimizing drives x to -∞ along the feasible cone.
    input.c = vec![-1.0, -1.0];
    assert!(
        emit_certificate(&input, &meta()).is_err(),
        "an unbounded LP has no KKT certificate and must be refused"
    );
}
