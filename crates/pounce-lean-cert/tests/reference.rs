//! The reference certificate (`certs/qp.cert.json` in `pounce-lean`, mirrored
//! here as a fixture) must model losslessly through the schema structs: parse,
//! re-serialize, and the JSON is semantically identical. Spot-checks confirm the
//! exact-rational values match what `f64 → ℚ` produces for the canonical QP.

#![allow(clippy::unwrap_used)]

use pounce_lean_cert::rational::Rat;
use pounce_lean_cert::schema::{Certificate, SCHEMA_TAG};

const REFERENCE: &str = include_str!("fixtures/qp.cert.json");

#[test]
fn reference_cert_roundtrips_semantically() {
    // The raw fixture as free-form JSON.
    let original: serde_json::Value = serde_json::from_str(REFERENCE).unwrap();

    // Through the typed schema and back.
    let cert: Certificate = serde_json::from_str(REFERENCE).unwrap();
    let reserialized = serde_json::to_string(&cert).unwrap();
    let roundtripped: serde_json::Value = serde_json::from_str(&reserialized).unwrap();

    assert_eq!(
        original, roundtripped,
        "schema structs lost or reshaped a field relative to the reference cert"
    );
}

#[test]
fn reference_cert_key_values() {
    let cert: Certificate = serde_json::from_str(REFERENCE).unwrap();

    assert_eq!(cert.schema, SCHEMA_TAG);
    assert_eq!(cert.verdict, "global-min");
    assert_eq!(cert.problem_class, "qp-convex");
    assert_eq!(cert.tolerance, Rat::zero());

    // Objective: ½xᵀQx with Q = diag(2,2), so half_quadratic and Q[0,0]=Q[1,1]=2.
    assert!(cert.problem.objective.half_quadratic);
    assert_eq!(cert.problem.objective.q.symmetric, Some(true));
    assert_eq!(cert.problem.objective.q.entries.len(), 2);
    assert_eq!(
        cert.problem.objective.q.entries[0].val,
        Rat::from_f64(2.0).unwrap()
    );

    // Candidate x* = (1/2, 1/2), objective 1/2 — all lossless from f64.
    assert_eq!(
        cert.candidate.as_ref().unwrap().x[0],
        Rat::from_f64(0.5).unwrap()
    );
    assert_eq!(
        cert.candidate.as_ref().unwrap().x[1],
        Rat::from_f64(0.5).unwrap()
    );
    assert_eq!(
        cert.candidate.as_ref().unwrap().objective,
        Rat::from_f64(0.5).unwrap()
    );

    // Witnesses: one dual per constraint; LDLᵀ of Q is L = I (no entries), D = (2,2).
    assert_eq!(
        cert.witnesses.duals.as_ref().unwrap().len(),
        cert.problem.constraints.len()
    );
    assert_eq!(
        cert.witnesses.duals.as_ref().unwrap()[0],
        Rat::from_f64(1.0).unwrap()
    );
    assert!(
        cert.witnesses
            .hessian_psd
            .as_ref()
            .unwrap()
            .l
            .unit_lower
            .unwrap()
    );
    assert!(
        cert.witnesses
            .hessian_psd
            .as_ref()
            .unwrap()
            .l
            .entries
            .is_empty()
    );
    assert_eq!(
        cert.witnesses.hessian_psd.as_ref().unwrap().d[0],
        Rat::from_f64(2.0).unwrap()
    );
}
