//! Serde structs for `pounce.lean-cert/v1`.
//!
//! The field layout mirrors the validated consumer-side spec (the `pounce-lean`
//! repo's `docs/lean-cert-v1.md`) and the canonical worked example
//! `certs/qp.cert.json`. Serialization order is chosen to match the reference
//! cert so a golden byte-diff in CI is meaningful.
//!
//! v1 codegen only consumes the `qp-convex` / `global-min` slice; the emitter
//! refuses to produce anything else (see [`crate::emit`]).

use crate::rational::{Bound, Rat};
use serde::{Deserialize, Serialize};

/// The schema tag every v1 certificate carries.
pub const SCHEMA_TAG: &str = "pounce.lean-cert/v1";

/// The Lean toolchain the `qp-convex`/`global-min` slice is validated against.
/// A proof reproduces only under this exact pin (schema rule §2).
pub const VALIDATED_LEAN: &str = "leanprover/lean4:v4.31.0";
/// The Mathlib revision paired with [`VALIDATED_LEAN`].
pub const VALIDATED_MATHLIB: &str = "fabf563a7c95a166b8d7b6efca11c8b4dc9d911f";

/// Top-level certificate. Serializes to the shape of `certs/qp.cert.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Certificate {
    pub schema: String,
    pub verdict: String,
    pub problem_class: String,
    pub tolerance: Rat,
    pub binding: Binding,
    pub toolchain: Toolchain,
    pub problem: Problem,
    /// Absent for `verdict = "infeasible"`, which is not a claim about a point.
    /// `skip_serializing_if` keeps existing certificates byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<Candidate>,
    pub witnesses: Witnesses,
}

/// Content-addressing + provenance. `nl_sha256`/`sol_sha256` bind the proof to
/// the exact problem and claimed solution bytes (the same hashes `pounce verify`
/// computes). `statement_sha256` is deliberately absent — it belongs to the
/// post-codegen verification receipt, not the cert.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Binding {
    pub nl_sha256: String,
    pub sol_sha256: String,
    pub solver: String,
}

/// Reproducibility pin (not load-bearing for trust).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Toolchain {
    pub lean: String,
    pub mathlib: String,
}

/// The problem over ℚ.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Problem {
    pub n_vars: usize,
    pub objective: Objective,
    pub var_bounds: VarBounds,
    pub constraints: Vec<Constraint>,
}

/// `f(x) = ½·xᵀQx + cᵀx + constant` when `half_quadratic`, else `xᵀQx + …`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Objective {
    pub kind: String,
    pub half_quadratic: bool,
    #[serde(rename = "Q")]
    pub q: SparseMatrix,
    pub c: Vec<Rat>,
    pub constant: Rat,
}

/// Length-`n_vars` arrays of bounds (rationals or `±inf` sentinels).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VarBounds {
    pub lower: Vec<Bound>,
    pub upper: Vec<Bound>,
}

/// One linear row, meaning `lower ≤ coeffs·x ≤ upper`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Constraint {
    pub name: String,
    pub coeffs: Vec<Rat>,
    pub lower: Bound,
    pub upper: Bound,
}

/// Candidate `x*` and its (informational) objective value.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub x: Vec<Rat>,
    pub objective: Rat,
}

/// Untrusted proof hints. Wrong data only makes the proof fail to typecheck.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Witnesses {
    /// KKT multipliers — present for `global-min`, absent for `infeasible`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duals: Option<Vec<Rat>>,
    /// PSD factorization of `Q` — `global-min` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hessian_psd: Option<HessianPsd>,
    /// Active constraint indices (informational) — `global-min` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_set: Option<Vec<usize>>,
    /// Farkas ray proving `A x ≥ b` has no solution — `infeasible` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub farkas: Option<Farkas>,
}

/// Farkas certificate: `y ≥ 0` with `Aᵀy = 0` and `b·y > 0`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Farkas {
    pub y: Vec<Rat>,
}

/// `LDLᵀ` factorization of the cert's `Q`: unit-lower `L`, nonnegative diagonal
/// `D`, with `Q = L·diag(D)·Lᵀ`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HessianPsd {
    pub of: String,
    #[serde(rename = "L")]
    pub l: SparseMatrix,
    #[serde(rename = "D")]
    pub d: Vec<Rat>,
}

/// Sparse rational matrix as a triplet list with explicit shape. `symmetric`
/// matrices store the lower triangle only; `unit_lower` matrices omit the
/// implied unit diagonal and carry strictly-below-diagonal entries.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub symmetric: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub unit_lower: Option<bool>,
    pub entries: Vec<Entry>,
}

/// One `{i, j, val}` triplet of a [`SparseMatrix`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub i: usize,
    pub j: usize,
    pub val: Rat,
}

impl SparseMatrix {
    /// A symmetric matrix (lower triangle stored).
    pub fn symmetric(rows: usize, cols: usize, entries: Vec<Entry>) -> SparseMatrix {
        SparseMatrix {
            rows,
            cols,
            symmetric: Some(true),
            unit_lower: None,
            entries,
        }
    }

    /// A unit-lower-triangular matrix (strictly-below-diagonal entries only).
    pub fn unit_lower(rows: usize, cols: usize, entries: Vec<Entry>) -> SparseMatrix {
        SparseMatrix {
            rows,
            cols,
            symmetric: None,
            unit_lower: Some(true),
            entries,
        }
    }
}
