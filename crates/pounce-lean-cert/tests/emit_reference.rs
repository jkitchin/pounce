//! The driver, end to end: build the canonical convex QP from plain `f64`
//! numbers (as POUNCE would hand it over) and assert the emitted certificate is
//! byte-for-byte the reference cert, modulo nothing. This is the golden the CI
//! drift guard will pin against `pounce-lean`'s codegen.

#![allow(clippy::unwrap_used)]

use pounce_lean_cert::emit::{emit_certificate, CertMeta, LinearConstraint, QpInput};
use pounce_lean_cert::{Certificate, EmitError};

const ZERO_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const REFERENCE: &str = include_str!("fixtures/qp.cert.json");

/// The reference QP: min x₁²+x₂² s.t. x₁+x₂ ≥ 1, solved (float) at (0.5, 0.5).
fn reference_input() -> QpInput {
    QpInput {
        n: 2,
        q_lower: vec![(0, 0, 2.0), (1, 1, 2.0)],
        half_quadratic: true,
        c: vec![0.0, 0.0],
        constant: 0.0,
        constraints: vec![LinearConstraint {
            name: "c0".to_string(),
            coeffs: vec![1.0, 1.0],
            lower: 1.0,
            upper: f64::INFINITY,
        }],
        var_lower: vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
        var_upper: vec![f64::INFINITY, f64::INFINITY],
        x_float: vec![0.5, 0.5],
        active_tol: 1e-7,
    }
}

fn reference_meta() -> CertMeta {
    CertMeta {
        nl_sha256: ZERO_HASH.to_string(),
        sol_sha256: ZERO_HASH.to_string(),
        solver: "pounce 0.5.0".to_string(),
    }
}

#[test]
fn emits_the_reference_certificate() {
    let cert = emit_certificate(&reference_input(), &reference_meta()).unwrap();

    let emitted: serde_json::Value = serde_json::to_value(&cert).unwrap();
    let expected: serde_json::Value = serde_json::from_str(REFERENCE).unwrap();
    assert_eq!(
        emitted, expected,
        "emitted certificate diverged from the validated reference cert"
    );
}

#[test]
fn emitted_cert_reparses() {
    let cert = emit_certificate(&reference_input(), &reference_meta()).unwrap();
    let json = pounce_lean_cert::to_canonical_json(&cert).unwrap();
    let _: Certificate = serde_json::from_str(&json).unwrap();
}

#[test]
fn off_slice_inputs_are_refused() {
    // Indefinite objective (Q = [[1,2],[2,1]]).
    let mut bad = reference_input();
    bad.q_lower = vec![(0, 0, 1.0), (1, 0, 2.0), (1, 1, 1.0)];
    assert!(matches!(
        emit_certificate(&bad, &reference_meta()).unwrap_err(),
        EmitError::Ldl(_)
    ));
}

#[test]
fn finite_variable_bound_is_folded_and_certified() {
    // min x₀²+x₁²  s.t.  x₀ ≥ 1  (a *variable lower bound*, no general row).
    // Exact optimum x* = (1, 0) with the bound active, bound dual λ = 2.
    let input = QpInput {
        n: 2,
        q_lower: vec![(0, 0, 2.0), (1, 1, 2.0)],
        half_quadratic: true,
        c: vec![0.0, 0.0],
        constant: 0.0,
        constraints: vec![], // no general constraints — only the bound
        var_lower: vec![1.0, f64::NEG_INFINITY],
        var_upper: vec![f64::INFINITY, f64::INFINITY],
        x_float: vec![1.0 + 1e-10, 1e-10],
        active_tol: 1e-7,
    };
    let cert = emit_certificate(&input, &reference_meta()).unwrap();
    let v = serde_json::to_value(&cert).unwrap();

    // Exact optimum, recovered from the off-by-1e-10 float point.
    assert_eq!(
        v["candidate"]["x"][0],
        serde_json::json!({"num":"1","den":"1"})
    );
    assert_eq!(
        v["candidate"]["x"][1],
        serde_json::json!({"num":"0","den":"1"})
    );
    assert_eq!(
        v["candidate"]["objective"],
        serde_json::json!({"num":"1","den":"1"})
    );

    // The bound became a one-sided constraint row with its own dual (= 2).
    assert_eq!(v["problem"]["constraints"][0]["name"], "var0_lb");
    assert_eq!(
        v["problem"]["constraints"][0]["coeffs"],
        serde_json::json!([{"num":"1","den":"1"}, {"num":"0","den":"1"}])
    );
    assert_eq!(
        v["problem"]["constraints"][0]["lower"],
        serde_json::json!({"num":"1","den":"1"})
    );
    assert_eq!(v["problem"]["constraints"][0]["upper"], "+inf");
    assert_eq!(
        v["witnesses"]["duals"][0],
        serde_json::json!({"num":"2","den":"1"})
    );

    // var_bounds is emitted as all-infinite (bounds live in constraints in v1).
    assert_eq!(v["problem"]["var_bounds"]["lower"][0], "-inf");
    assert_eq!(v["problem"]["var_bounds"]["upper"][0], "+inf");
}

#[test]
fn equality_constraint_is_certified() {
    // min x₀²+x₁²  s.t.  x₀+x₁ = 1 (equality), x₀ ≥ 0.
    // x* = (1/2, 1/2); the equality multiplier μ = 1 (free-sign), the
    // inequality x₀ ≥ 0 is inactive (λ = 0). Matches pounce-lean's qp-eq fixture.
    let input = QpInput {
        n: 2,
        q_lower: vec![(0, 0, 2.0), (1, 1, 2.0)],
        half_quadratic: true,
        c: vec![0.0, 0.0],
        constant: 0.0,
        constraints: vec![
            LinearConstraint {
                name: "c0".to_string(),
                coeffs: vec![1.0, 1.0],
                lower: 1.0,
                upper: 1.0, // equality x₀+x₁ = 1
            },
            LinearConstraint {
                name: "c1".to_string(),
                coeffs: vec![1.0, 0.0],
                lower: 0.0,
                upper: f64::INFINITY, // x₀ ≥ 0
            },
        ],
        var_lower: vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
        var_upper: vec![f64::INFINITY, f64::INFINITY],
        x_float: vec![0.5, 0.5],
        active_tol: 1e-7,
    };
    let cert = emit_certificate(&input, &reference_meta()).unwrap();
    let v = serde_json::to_value(&cert).unwrap();

    // Equality kept as lower == upper; inequality stays one-sided.
    assert_eq!(
        v["problem"]["constraints"][0]["lower"],
        serde_json::json!({"num":"1","den":"1"})
    );
    assert_eq!(
        v["problem"]["constraints"][0]["upper"],
        serde_json::json!({"num":"1","den":"1"})
    );
    assert_eq!(v["problem"]["constraints"][1]["upper"], "+inf");

    // Exact optimum and the per-constraint duals: μ=1 (equality), λ=0 (inactive).
    assert_eq!(
        v["candidate"]["x"][0],
        serde_json::json!({"num":"1","den":"2"})
    );
    assert_eq!(
        v["candidate"]["objective"],
        serde_json::json!({"num":"1","den":"2"})
    );
    assert_eq!(
        v["witnesses"]["duals"][0],
        serde_json::json!({"num":"1","den":"1"})
    );
    assert_eq!(
        v["witnesses"]["duals"][1],
        serde_json::json!({"num":"0","den":"1"})
    );
    assert_eq!(v["witnesses"]["active_set"], serde_json::json!([0])); // equality always active
}

#[test]
fn equality_with_negative_multiplier_is_certified() {
    // The equality multiplier may be negative — must not be rejected.
    // min x₀²+x₁²  s.t.  x₀+x₁ = −2  →  x* = (−1,−1), μ = −2.
    let input = QpInput {
        n: 2,
        q_lower: vec![(0, 0, 2.0), (1, 1, 2.0)],
        half_quadratic: true,
        c: vec![0.0, 0.0],
        constant: 0.0,
        constraints: vec![LinearConstraint {
            name: "c0".to_string(),
            coeffs: vec![1.0, 1.0],
            lower: -2.0,
            upper: -2.0,
        }],
        var_lower: vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
        var_upper: vec![f64::INFINITY, f64::INFINITY],
        x_float: vec![-1.0, -1.0],
        active_tol: 1e-7,
    };
    let cert = emit_certificate(&input, &reference_meta()).unwrap();
    let v = serde_json::to_value(&cert).unwrap();
    assert_eq!(
        v["candidate"]["x"][0],
        serde_json::json!({"num":"-1","den":"1"})
    );
    assert_eq!(
        v["witnesses"]["duals"][0],
        serde_json::json!({"num":"-2","den":"1"})
    );
}

#[test]
fn two_sided_range_is_split_and_certified() {
    // min (x₀−3)²+(x₁−3)²  s.t.  1 ≤ x₀+x₁ ≤ 4.
    // Q=diag(2,2), c=[−6,−6], k=18. Optimum x*=(2,2) with the UPPER side
    // (x₀+x₁ ≤ 4) active, dual 2; the lower side is inactive (dual 0).
    let input = QpInput {
        n: 2,
        q_lower: vec![(0, 0, 2.0), (1, 1, 2.0)],
        half_quadratic: true,
        c: vec![-6.0, -6.0],
        constant: 18.0,
        constraints: vec![LinearConstraint {
            name: "c0".to_string(),
            coeffs: vec![1.0, 1.0],
            lower: 1.0,
            upper: 4.0,
        }],
        var_lower: vec![f64::NEG_INFINITY, f64::NEG_INFINITY],
        var_upper: vec![f64::INFINITY, f64::INFINITY],
        x_float: vec![2.0 + 1e-10, 2.0],
        active_tol: 1e-7,
    };
    let cert = emit_certificate(&input, &reference_meta()).unwrap();
    let v = serde_json::to_value(&cert).unwrap();

    // The range split into two one-sided rows, in order.
    assert_eq!(v["problem"]["constraints"][0]["name"], "c0_lo");
    assert_eq!(
        v["problem"]["constraints"][0]["lower"],
        serde_json::json!({"num":"1","den":"1"})
    );
    assert_eq!(v["problem"]["constraints"][0]["upper"], "+inf");
    assert_eq!(v["problem"]["constraints"][1]["name"], "c0_hi");
    assert_eq!(v["problem"]["constraints"][1]["lower"], "-inf");
    assert_eq!(
        v["problem"]["constraints"][1]["upper"],
        serde_json::json!({"num":"4","den":"1"})
    );

    // Exact optimum, objective, and the active upper-side dual.
    assert_eq!(
        v["candidate"]["x"][0],
        serde_json::json!({"num":"2","den":"1"})
    );
    assert_eq!(
        v["candidate"]["x"][1],
        serde_json::json!({"num":"2","den":"1"})
    );
    assert_eq!(
        v["candidate"]["objective"],
        serde_json::json!({"num":"2","den":"1"})
    );
    assert_eq!(
        v["witnesses"]["duals"][0],
        serde_json::json!({"num":"0","den":"1"})
    ); // lower inactive
    assert_eq!(
        v["witnesses"]["duals"][1],
        serde_json::json!({"num":"2","den":"1"})
    ); // upper active
    assert_eq!(v["witnesses"]["active_set"], serde_json::json!([1]));
}

#[test]
fn upper_sided_constraint_is_normalized() {
    // min x₁²+x₂² s.t. −x₁−x₂ ≤ −1  (i.e. x₁+x₂ ≥ 1 written as a ≤ row).
    // Normalization negates it to x₁+x₂ ≥ 1, so the exact optimum is the same.
    let mut input = reference_input();
    input.constraints = vec![LinearConstraint {
        name: "c0".to_string(),
        coeffs: vec![-1.0, -1.0],
        lower: f64::NEG_INFINITY,
        upper: -1.0,
    }];
    let cert = emit_certificate(&input, &reference_meta()).unwrap();
    // x* = (1/2, 1/2), dual = 1, objective = 1/2 — same as the canonical form.
    let v = serde_json::to_value(&cert).unwrap();
    assert_eq!(
        v["candidate"]["x"][0],
        serde_json::json!({"num":"1","den":"2"})
    );
    assert_eq!(
        v["witnesses"]["duals"][0],
        serde_json::json!({"num":"1","den":"1"})
    );
    assert_eq!(
        v["candidate"]["objective"],
        serde_json::json!({"num":"1","den":"2"})
    );
}
