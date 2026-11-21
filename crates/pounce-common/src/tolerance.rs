//! Scale-aware significance tests for feasibility and optimality decisions.
//!
//! # Why this exists
//!
//! Multiplying a constraint row by a positive constant leaves the feasible set
//! *exactly* unchanged — it is the same problem written differently. So a
//! solver's verdict must not depend on it. Comparing a scale-*dependent*
//! quantity (a constraint residual) against an *absolute* threshold breaks that
//! invariant, and has produced defects in the restoration gates, presolve
//! certification, and the solution verifier.
//!
//! The sharpest example: `x >= 2` over `x ∈ [0, 1]` reports
//! `Infeasible_Problem_Detected` as written, and `Solve_Succeeded` when every
//! row is multiplied by `1e-12` — because at that scale the residual falls
//! under an absolute tolerance. Same empty feasible set, opposite verdicts.
//!
//! # The rule
//!
//! Compare a residual against `tol * scale`, where `scale` is the quantity's
//! own natural magnitude. Both sides then move together under row scaling, so
//! the test is invariant.
//!
//! A clamped form — `tol * max(scale, 1)` — looks safer and is **wrong**: the
//! clamp reinstates the absolute floor for `scale < 1`, which is exactly the
//! down-scaled direction that fails. This was measured, not assumed:
//!
//! ```text
//!   k    residual   scale     tol*max(s,1)  fires?  |  tol*s     fires?
//! -12    1.00e-12   1.00e-12  1.00e-08      false   |  1.00e-20  true
//!  -8    1.00e-08   1.00e-08  1.00e-08      false   |  1.00e-16  true
//!   0    1.00e+00   1.00e+00  1.00e-08      true    |  1.00e-08  true
//!  12    1.00e+12   1.00e+12  1.00e+04      true    |  1.00e+04  true
//! ```
//!
//! # Direction of failure
//!
//! Both functions **fail closed**: a residual that cannot be judged (`NaN`, or
//! non-finite) is reported *not* significant. For an infeasibility test that is
//! the safe direction — it withholds a verdict rather than fabricating one. A
//! caller that needs the opposite polarity must handle non-finite values itself
//! rather than inverting the result.

use crate::types::Number;

/// Whether `value` is large enough, relative to its own natural magnitude, to
/// be treated as a real quantity rather than numerical noise.
///
/// The threshold is `tol * |scale|`. When `scale` is zero or non-finite the
/// relative test is undefined, so it degrades to the absolute `tol` — that case
/// is a degenerate row with no magnitude, where there is nothing to be relative
/// to.
///
/// Returns `false` for a non-finite `value` (see the module note on failing
/// closed).
///
/// ```
/// use pounce_common::tolerance::is_significant;
/// // Same model at three row scalings: the verdict must not move.
/// assert!(is_significant(1.0e-12, 1.0e-12, 1e-8));
/// assert!(is_significant(1.0, 1.0, 1e-8));
/// assert!(is_significant(1.0e12, 1.0e12, 1e-8));
/// // Noise at any scale is still noise.
/// assert!(!is_significant(1.0e-20, 1.0, 1e-8));
/// ```
pub fn is_significant(value: Number, scale: Number, tol: Number) -> bool {
    if !value.is_finite() {
        return false;
    }
    let s = scale.abs();
    let threshold = if s.is_finite() && s > 0.0 {
        tol * s
    } else {
        tol
    };
    value.abs() > threshold
}

/// [`is_significant`] with the boundary included: `value` *at* `tol * |scale|`
/// counts.
///
/// For a question asked with `constr_viol_tol` — "is this constraint
/// violation real?" — the user's declared tolerance is the line between
/// feasible and violated, and a violation sitting exactly on it is a
/// violation (gh#508: the status mapping already reads it that way, and
/// gh#1217 found the restoration certificate reading it the other way).
/// Same scaling rule and same fail-closed handling of a non-finite `value`.
///
/// ```
/// use pounce_common::tolerance::{is_significant, reaches_tolerance};
/// assert!(reaches_tolerance(1e-4, 1.0, 1e-4));
/// assert!(!is_significant(1e-4, 1.0, 1e-4));
/// assert!(!reaches_tolerance(0.99e-4, 1.0, 1e-4));
/// assert!(!reaches_tolerance(f64::NAN, 1.0, 1e-4));
/// ```
pub fn reaches_tolerance(value: Number, scale: Number, tol: Number) -> bool {
    if !value.is_finite() {
        return false;
    }
    let s = scale.abs();
    let threshold = if s.is_finite() && s > 0.0 {
        tol * s
    } else {
        tol
    };
    value.abs() >= threshold
}

/// Whether `value` is small enough, relative to its own natural magnitude, to
/// be treated as satisfied — the **accepting** direction.
///
/// Threshold is `tol * max(|scale|, 1)`: never *stricter* than the plain
/// absolute `tol`, but scaled up for a large row.
///
/// # Why this is not the negation of [`is_significant`]
///
/// The two answer different questions and are conservative in opposite
/// directions, so they need different thresholds:
///
/// * *"Is this residual small enough to call the point feasible?"* — must never
///   demand more precision than the solver promised. A solver converges to
///   **absolute** residuals, so on a row of magnitude `1e-3` a residual of
///   `1e-8` is a converged solution; a pure relative test with `tol = 1e-6`
///   would reject it at a `1e-9` threshold. Hence the clamp.
/// * *"Is this residual large enough to prove no point is feasible?"* — must not
///   depend on how the model is written, so it uses [`is_significant`]'s pure
///   `tol * scale`. A clamp there would reinstate the absolute floor and miss
///   the down-scaled direction entirely.
///
/// Using one form for both was measured and fails: pure relative in the
/// accepting direction rejects genuine solutions on small-magnitude rows.
///
/// ```
/// use pounce_common::tolerance::is_negligible;
/// // A converged absolute residual stays acceptable on a small row.
/// assert!(is_negligible(1e-8, 1e-3, 1e-6));
/// // On a row near 5e13, a residual of 3.15e3 is eleven relative digits.
/// assert!(is_negligible(3.15e3, 5e13, 1e-6));
/// // Genuine violations are still caught.
/// assert!(!is_negligible(0.5, 1.0, 1e-6));
/// ```
pub fn is_negligible(value: Number, scale: Number, tol: Number) -> bool {
    if !value.is_finite() {
        return false;
    }
    let s = scale.abs();
    let widened = if s.is_finite() { s.max(1.0) } else { 1.0 };
    value.abs() <= tol * widened
}

/// The natural magnitude of a row, from the NLP scaling factor applied to it.
///
/// POUNCE's scaling picks `dc_i` so that `dc_i * c_i` is O(1); the row's own
/// magnitude is therefore `1 / dc_i`. Using the factor the solver already
/// computed avoids inventing a second, possibly disagreeing, notion of scale —
/// `c_scale_vec` / `d_scale_vec` are the authority.
///
/// A missing, zero, or non-finite factor means "no scaling applied", which is
/// magnitude `1.0`.
///
/// ```
/// use pounce_common::tolerance::row_scale_from_factor;
/// assert_eq!(row_scale_from_factor(1.0), 1.0);
/// assert_eq!(row_scale_from_factor(1e-6), 1e6);   // row shrunk by 1e-6 => magnitude 1e6
/// assert_eq!(row_scale_from_factor(0.0), 1.0);    // degenerate => unscaled
/// ```
pub fn row_scale_from_factor(factor: Number) -> Number {
    if factor.is_finite() && factor > 0.0 {
        1.0 / factor
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole module exists for: scaling a residual and its
    /// magnitude together must not change the answer.
    #[test]
    fn verdict_is_invariant_under_row_scaling() {
        let tol = 1e-8;
        for k in -12..=12 {
            let s = 10f64.powi(k);
            assert!(
                is_significant(1.0 * s, s, tol),
                "a unit violation at scale 10^{k} must stay significant"
            );
            assert!(
                !is_significant(1e-12 * s, s, tol),
                "noise at scale 10^{k} must stay insignificant"
            );
        }
    }

    /// Pins the bug in the clamped form, so nobody reintroduces it.
    #[test]
    fn clamped_form_would_lose_the_down_scaled_direction() {
        let tol = 1e-8;
        let (value, scale) = (1e-12, 1e-12); // a full unit violation at 1e-12 scale
        assert!(is_significant(value, scale, tol));
        // What `tol * max(scale, 1)` would have concluded:
        assert!(
            !(value.abs() > tol * scale.abs().max(1.0)),
            "the clamped form misses this — that is why it is not used"
        );
    }

    #[test]
    fn degenerate_scale_falls_back_to_absolute() {
        let tol = 1e-8;
        assert!(is_significant(1e-6, 0.0, tol));
        assert!(!is_significant(1e-10, 0.0, tol));
        assert!(is_significant(1e-6, f64::INFINITY, tol));
        assert!(is_significant(1e-6, f64::NAN, tol));
    }

    #[test]
    fn non_finite_value_is_not_evidence() {
        let tol = 1e-8;
        assert!(!is_significant(f64::NAN, 1.0, tol));
        assert!(!is_significant(f64::INFINITY, 1.0, tol));
        assert!(!is_significant(f64::NEG_INFINITY, 1.0, tol));
    }

    #[test]
    fn exactly_at_threshold_is_not_significant() {
        // Strict `>` keeps the boundary on the conservative side.
        assert!(!is_significant(1e-8, 1.0, 1e-8));
        assert!(is_significant(1.0000001e-8, 1.0, 1e-8));
    }

    #[test]
    fn accepting_direction_never_stricter_than_absolute() {
        let tol = 1e-6;
        // The case that made a pure relative test reject genuine solutions:
        // a converged absolute residual on a small-magnitude row.
        assert!(is_negligible(1e-8, 1e-3, tol));
        // Pure relative would have rejected it:
        assert!(
            1e-8f64 > tol * 1e-3,
            "pure relative flags it — that is the bug"
        );
    }

    #[test]
    fn accepting_direction_scales_up_for_large_rows() {
        let tol = 1e-6;
        // Eleven relative digits on a 5e13 row: acceptable.
        assert!(is_negligible(3.15e3, 5e13, tol));
        // A real violation at the same scale is not.
        assert!(!is_negligible(1e10, 5e13, tol));
    }

    #[test]
    fn the_two_directions_are_not_negations() {
        let tol = 1e-6;
        let (value, scale) = (1e-8, 1e-3);
        // Both can be true at once: too small to accept as a violation, and
        // too small to prove infeasibility from. They are different questions.
        assert!(is_negligible(value, scale, tol));
        assert!(is_significant(value, scale, tol));
    }

    #[test]
    fn non_finite_is_never_negligible() {
        let tol = 1e-6;
        assert!(!is_negligible(f64::NAN, 1.0, tol));
        assert!(!is_negligible(f64::INFINITY, 1.0, tol));
    }

    #[test]
    fn row_scale_inverts_the_factor() {
        assert_eq!(row_scale_from_factor(1.0), 1.0);
        assert_eq!(row_scale_from_factor(1e-6), 1e6);
        assert_eq!(row_scale_from_factor(1e6), 1e-6);
        // Degenerate factors mean "unscaled".
        assert_eq!(row_scale_from_factor(0.0), 1.0);
        assert_eq!(row_scale_from_factor(-1.0), 1.0);
        assert_eq!(row_scale_from_factor(f64::NAN), 1.0);
        assert_eq!(row_scale_from_factor(f64::INFINITY), 1.0);
    }

    /// End-to-end: a factor from `c_scale_vec` feeding the significance test.
    #[test]
    fn factor_and_significance_compose() {
        let tol = 1e-8;
        // Solver shrank this row by 1e-6, so its natural magnitude is 1e6.
        let scale = row_scale_from_factor(1e-6);
        assert_eq!(scale, 1e6);
        // Threshold is tol * scale = 1e-8 * 1e6 = 1e-2.
        assert!(
            !is_significant(1e-3, scale, tol),
            "1e-3 is below the 1e-2 threshold"
        );
        assert!(is_significant(1e-1, scale, tol), "1e-1 is above it");
    }
}
