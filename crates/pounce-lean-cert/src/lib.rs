//! Exact-rational `pounce.lean-cert/v1` certificate emitter.
//!
//! POUNCE solves in `f64`; this crate converts a convex-QP global-min solve into
//! an **exact-rational** certificate that the external `pounce-lean` repo can
//! turn into a kernel-checked Lean 4 proof — with no floating point in the
//! trusted path.
//!
//! The witnesses POUNCE emits (KKT duals, the `LDLᵀ` PSD factorization) are
//! *untrusted*: wrong data only makes the Lean proof fail to typecheck, never
//! pass falsely. The emitter therefore self-checks every witness **exactly over
//! ℚ** before writing, and errors out rather than emit a cert that will not
//! verify (see the schema's supported-slice rule).
//!
//! Layers (built bottom-up):
//! * [`rational`] — exact ℚ, lossless `f64 → ℚ`, `±inf` bound sentinels.
//! * [`schema`] — serde structs for the on-disk `pounce.lean-cert/v1` shape.
//! * [`linalg`] — exact dense rational solve (the KKT system).
//! * [`ldlt`] — exact `LDLᵀ` PSD factorization of `Q`.
//! * [`nullspace`] — verifiable null-space basis `Z` of the active Jacobian.
//! * [`refine`] — exact rational active-set KKT solve (Mode B).
//! * [`refine_farkas`] — exact rational Farkas ray for the `infeasible` verdict.
//! * [`round_gram`] — exact rational SOS Gram matrix from the SDP's float one.
//!
//! * [`emit`] — the neutral-`f64` QP → certificate driver + exact self-check gate.

pub mod emit;
pub mod ldlt;
pub mod linalg;
pub mod nullspace;
pub mod rational;
pub mod refine;
pub mod refine_farkas;
pub mod round_gram;
pub mod schema;

pub use emit::{
    CertMeta, EmitError, LinearConstraint, QpInput, canonical_problem, emit_certificate,
    emit_infeasible_certificate, emit_unbounded_certificate, problem_block,
};
pub use rational::{Bound, Rat, RatError};
pub use schema::{Certificate, SCHEMA_TAG};

/// Canonical serialization of a certificate (pretty JSON). Used for on-disk
/// output and golden-fixture diffing so the emitter and `pounce-lean`'s codegen
/// can never silently drift.
pub fn to_canonical_json(cert: &Certificate) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(cert)
}
