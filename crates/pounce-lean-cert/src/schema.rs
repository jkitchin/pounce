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
    /// The certified bound `γ` — `global-lower-bound` only, where the claim is
    /// `γ ≤ p(x)` for every `x` rather than anything about a point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound: Option<Rat>,
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
///
/// Two mutually exclusive shapes share this block, distinguished by
/// `problem_class`:
///
/// * `qp-convex` — `objective` / `var_bounds` / `constraints`, all present.
/// * `sos-poly` — `polynomial`, plus `poly_constraints` when the problem is
///   constrained: term lists for a possibly nonconvex polynomial and for the
///   `gₖ(x) ≥ 0` its bound is claimed over.
///
/// The unused half is *absent*, not zero-filled. A zeroed `Q` would not be a
/// harmless placeholder: [`crate::canonical_problem`] compares problem blocks to
/// decide whether two certificates concern the same problem, so a quartic
/// carrying `Q = 0` would assert it is a linear program. Absence is the only
/// encoding that says nothing rather than something false.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Problem {
    pub n_vars: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<Objective>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub var_bounds: Option<VarBounds>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraints: Option<Vec<Constraint>>,
    /// The objective as an exact term list — `sos-poly` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polynomial: Option<PolynomialSpec>,
    /// The feasible set as `gₖ(x) ≥ 0` term lists — `sos-poly` only, and absent
    /// on an unconstrained problem.
    ///
    /// Absence and `[]` would mean the same thing mathematically, but absence is
    /// what an unconstrained certificate has always serialized, and
    /// [`crate::canonical_problem`] compares these blocks byte-for-byte.
    ///
    /// This field changes what the bound *means*: with it, `γ ≤ p(x)` is claimed
    /// only where every `gₖ(x) ≥ 0`. A consumer that ignored it would read a
    /// constrained bound as a global one, which is strictly stronger and
    /// generally false — so the codegen keys the theorem it emits off this
    /// field's presence rather than off the verdict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub poly_constraints: Option<Vec<PolynomialSpec>>,
}

impl Problem {
    /// Constraint rows of a `qp-convex` problem. An `sos-poly` problem carries
    /// none — its feasible set, if any, is in `poly_constraints` — so it reads
    /// as empty here rather than being a case every caller must handle.
    pub fn constraint_rows(&self) -> &[Constraint] {
        self.constraints.as_deref().unwrap_or(&[])
    }

    /// The `gₖ(x) ≥ 0` of an `sos-poly` problem; empty when unconstrained.
    pub fn poly_constraint_specs(&self) -> &[PolynomialSpec] {
        self.poly_constraints.as_deref().unwrap_or(&[])
    }
}

/// A polynomial as `Σ coeff · x^exponents`, exactly over ℚ.
///
/// A *term list*, not an expression tree: it is what the producer already has
/// (`pounce_convex::sos::Polynomial`), what [`crate::round_gram`] consumes, and
/// what renders directly to a Lean `def`. An expression tree buys nothing for a
/// polynomial.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolynomialSpec {
    pub terms: Vec<PolyTerm>,
}

/// One monomial: `coeff · Π xᵢ^exponents[i]`. `exponents` has length `n_vars`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolyTerm {
    pub exponents: Vec<usize>,
    pub coeff: Rat,
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
    /// Recession certificate proving the objective is unbounded below —
    /// `unbounded` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recession: Option<Recession>,
    /// Sum-of-squares blocks witnessing `p(x) − γ = Σ σᵢ(x)` —
    /// `global-lower-bound` only. v1 emits exactly one (the unconstrained case).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sos: Option<Vec<SosBlock>>,
    /// An exactly-feasible point near the candidate — `feasible` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feasible_witness: Option<FeasibleWitness>,
}

/// A point that satisfies every constraint **exactly over ℚ**, close to the
/// certificate's candidate.
///
/// This is what makes `feasible` a claim worth certifying. The candidate is the
/// solver's float point, so its rational image misses the constraints by a
/// residual and only ε-feasibility can be asserted about it — a statement with a
/// knob in it. `xhat` turns that into an existence theorem with no knob: a
/// genuine feasible point exists, within ε of what the solver reported.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeasibleWitness {
    pub xhat: Vec<Rat>,
}

/// One SOS block: `σ(x) = m(x)ᵀ G m(x)` with `G = L·diag(D)·Lᵀ ⪰ 0`.
///
/// `gram` is redundant given `L` and `D` — deliberately. The Lean side checks
/// the polynomial identity against `gram` and PSD-ness against `L`/`D`, and
/// their agreement is itself a proof obligation, so shipping both lets a
/// transcription error fail loudly instead of propagating.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SosBlock {
    /// The monomial lift `m(x)`, as exponent vectors of length `n_vars`.
    pub monomials: Vec<Vec<usize>>,
    pub gram: SparseMatrix,
    #[serde(rename = "L")]
    pub l: SparseMatrix,
    #[serde(rename = "D")]
    pub d: Vec<Rat>,
    /// Which `problem.poly_constraints[k]` this block multiplies. Absent for
    /// `σ₀`, whose multiplier is the constant `1`.
    ///
    /// Named rather than positional because the identity is only true for the
    /// *right* pairing: swap two localizing blocks and the coefficients no
    /// longer match, but nothing in the block itself would say so. The index is
    /// what the codegen reads to build the `Σᵢ σᵢ·gᵢ` sum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multiplier: Option<usize>,
}

/// Recession certificate: a feasible `x0` together with a direction `d`
/// satisfying `Q d = 0`, `A d ≥ 0`, `c·d < 0`.
///
/// Both witnesses are required. A direction alone proves nothing — a problem
/// can be primal *and* dual infeasible, in which case such a `d` exists but
/// there is no feasible point to travel from.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Recession {
    pub x0: Vec<Rat>,
    pub d: Vec<Rat>,
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
