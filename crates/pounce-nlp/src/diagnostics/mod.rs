//! Model diagnostics that need only a `TNLP` — no `.nl` file, no CLI.
//!
//! # Why this is a library module
//!
//! These checks were written for the `pounce` CLI (`pounce check-x0`,
//! `pounce verify`) and lived in `crates/pounce-cli/src/`, which meant the
//! only way to preflight a starting point or to independently check a
//! claimed solution was to be the `pounce` binary. Both are pure functions
//! over a model and a point: the preflight evaluates the model once at `x0`
//! and reports what iteration 0 will see, and the feasibility check is one
//! constraint evaluation against the declared bounds. Nothing about either
//! requires a file on disk.
//!
//! An embedder driving `IpoptApplication` directly — `pounce-rs`,
//! `pounce-cinterface`, `pounce-wasm`, the Python `Problem` API — has the
//! same reasons to want them, so they live here and the CLI renders their
//! results rather than owning them.
//!
//! Argument parsing, file loading, console formatting and JSON rendering
//! stay in the CLI: this module returns data.

use pounce_common::tolerance::is_negligible;
use pounce_common::types::{Number, lower_bound_present, upper_bound_present};

pub mod preflight;

/// One row (a variable or a constraint) reported against its declared box.
#[derive(Debug, Clone)]
pub struct RowReport {
    pub index: usize,
    pub name: String,
    pub value: Number,
    pub lo: Number,
    pub hi: Number,
    pub violation: Number,
}

// `is_finite_bound(b) = b > NLP_LOWER_BOUND_INF && b < NLP_UPPER_BOUND_INF`
// used to live here — a *band* membership test applied to lower and upper
// bounds alike (gh #403). A real upper bound of `-5e20` failed it, so
// `box_violation` scored `0.0` against it and `verify` reported ACCEPTED for a
// `.sol` that violates a declared bound. Presence is directional; use
// `lower_bound_present` / `upper_bound_present` from `pounce_common::types`,
// picking the one that matches the side you hold.

/// `lo ≤ v ≤ hi` violation: how far `v` is outside the box, 0 if inside.
///
/// A non-finite `v` (NaN or ±∞) is treated as an infinite violation, never
/// as feasible: `NaN`-laden arithmetic would otherwise collapse to `0.0`
/// through `f64::max` (which drops NaN operands) and let a fabricated `.sol`
/// slip past the feasibility gate — the exact threat this checker defends
/// against. An unbounded variable pinned at ±∞ is likewise not a real point.
pub fn box_violation(v: Number, lo: Number, hi: Number) -> Number {
    if !v.is_finite() {
        return Number::INFINITY;
    }
    let below = if lower_bound_present(lo) {
        lo - v
    } else {
        Number::NEG_INFINITY
    };
    let above = if upper_bound_present(hi) {
        v - hi
    } else {
        Number::NEG_INFINITY
    };
    below.max(above).max(0.0)
}

/// The natural magnitude of a row, for a scale-relative feasibility test.
///
/// A checker holding only a model and a point has had no solver scaling
/// applied, so the magnitude has to come from the row's own numbers — the
/// evaluated value and whichever bounds are finite. Infinite bounds carry no
/// magnitude information and are skipped.
pub fn row_magnitude(value: Number, lo: Number, hi: Number) -> Number {
    let mut m = if value.is_finite() { value.abs() } else { 0.0 };
    if lower_bound_present(lo) {
        m = m.max(lo.abs());
    }
    if upper_bound_present(hi) {
        m = m.max(hi.abs());
    }
    m
}

/// Whether a row's violation is real, judged relative to the row's own
/// magnitude.
///
/// An absolute tolerance is meaningless against a row evaluating near `1e13`:
/// `feas_tol = 1e-6` is unreachable there, so a solution correct to eleven
/// relative digits was reported REJECTED. Scaling the tolerance by the row
/// magnitude makes the verdict independent of how the model happens to be
/// written.
///
/// Uses the **accepting** direction (`is_negligible`), which is never stricter
/// than the plain absolute `tol`. A pure relative test was tried first and
/// rejected genuine solutions: the solver converges to *absolute* residuals, so
/// on a row of magnitude `1e-3` a residual of `1e-8` is converged, while a
/// relative test at `tol = 1e-6` would demand `1e-9`.
///
/// The non-finite case is handled here rather than inside the primitive, which
/// reports an unjudgeable value as not-negligible-and-not-significant. A point
/// carrying `NaN` or `±inf` is not a point at all and must be rejected — which
/// is what `box_violation` returning infinity encodes.
pub fn row_is_violated(viol: Number, magnitude: Number, feas_tol: Number) -> bool {
    if !viol.is_finite() {
        return true;
    }
    !is_negligible(viol, magnitude, feas_tol)
}

/// A row's declared name, or a positional `x[i]` / `c[i]` stand-in.
pub fn name_at(names: &[String], i: usize, kind: char) -> String {
    match names.get(i) {
        Some(s) if !s.is_empty() => s.clone(),
        _ => format!("{kind}[{i}]"),
    }
}
