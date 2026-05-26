//! Warm-start helpers — building a [`pounce_qp::WorkingSet`] from
//! a converged IPM (or any) iterate so the next SQP solve can pick
//! up where the IPM left off (Phase 5c §7.5 + sensitivity
//! corrector handoff).
//!
//! The classifier is the **multiplier-sign + primal-distance**
//! heuristic standard in mixed IPM/SQP warm-start pipelines
//! (Wächter-Biegler 2006 §6; Forsgren-Gill-Wright 2002 §5). It is
//! intentionally lossy at degenerate active sets — the QP solver
//! will detect and correct any misclassification in the first
//! step of the next QP, so correctness is preserved.

use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_common::Number;
use pounce_qp::{BoundStatus, ConsStatus, WorkingSet};

/// Classify the active set at iterate `(x, λ_x, λ_g)` against the
/// supplied bounds and constraint-bound vectors.
///
/// Inputs:
/// - `lambda_x`: packed signed bound multipliers (`z_l − z_u`) of
///   length `n`. Positive ⇒ lower bound active; negative ⇒ upper.
/// - `lambda_g`: stacked constraint multipliers `[y_c ; y_d]` of
///   length `m = m_eq + m_ineq`. Same sign convention.
/// - `m_eq`: number of equality rows at the start of `lambda_g`.
///   Used to flag rows as [`ConsStatus::Equality`] without
///   consulting `g_l`/`g_u`.
/// - `x`, `x_l`, `x_u`: primal iterate and variable bounds, length
///   `n`. The bound-classifier double-checks the primal is close
///   to the bound (within `primal_tol`) — guards against the case
///   where a multiplier is large but the primal hasn't actually
///   reached the bound (e.g. near-degenerate KKT or a bad
///   multiplier estimate).
/// - `g`, `g_l`, `g_u`: constraint values and bounds, length `m`.
///   Used identically for constraint rows.
/// - `mult_tol`: multiplier-magnitude threshold; a row whose
///   `|λ|` falls below this is classified as `Inactive`
///   regardless of primal distance.
/// - `primal_tol`: distance threshold between `x[i]` and `x_l[i]`
///   / `x_u[i]` (resp. `g[i]` vs `g_l[i]` / `g_u[i]`) below which
///   a row is treated as "at the bound".
///
/// Variable bounds with `x_l[i] == x_u[i]` are classified
/// [`BoundStatus::Fixed`]; constraint rows in the first `m_eq`
/// slots are [`ConsStatus::Equality`]. Both are unconditionally
/// active.
#[allow(clippy::too_many_arguments)]
pub fn classify_working_set(
    lambda_x: &[Number],
    lambda_g: &[Number],
    m_eq: usize,
    x: &[Number],
    x_l: &[Number],
    x_u: &[Number],
    g: &[Number],
    g_l: &[Number],
    g_u: &[Number],
    mult_tol: Number,
    primal_tol: Number,
) -> WorkingSet {
    let n = lambda_x.len();
    let m = lambda_g.len();
    debug_assert_eq!(x.len(), n);
    debug_assert_eq!(x_l.len(), n);
    debug_assert_eq!(x_u.len(), n);
    debug_assert_eq!(g.len(), m);
    debug_assert_eq!(g_l.len(), m);
    debug_assert_eq!(g_u.len(), m);
    debug_assert!(m_eq <= m);

    // Bound-finiteness uses the same `NLP_*_BOUND_INF` sentinels
    // pounce uses everywhere else (default ±1e19). Naive
    // `.is_finite()` would falsely include `−1e19` as a real lower
    // bound and tag any unbounded variable at that value as
    // `AtLower` (PR #50 review A4).
    let mut bounds = Vec::with_capacity(n);
    for i in 0..n {
        let lo_fin = x_l[i] > NLP_LOWER_BOUND_INF;
        let up_fin = x_u[i] < NLP_UPPER_BOUND_INF;
        if lo_fin && up_fin && (x_u[i] - x_l[i]).abs() < primal_tol {
            bounds.push(BoundStatus::Fixed);
            continue;
        }
        let mu = lambda_x[i];
        let at_lo = lo_fin && (x[i] - x_l[i]).abs() < primal_tol;
        let at_up = up_fin && (x_u[i] - x[i]).abs() < primal_tol;
        let status = if mu > mult_tol && at_lo {
            BoundStatus::AtLower
        } else if mu < -mult_tol && at_up {
            BoundStatus::AtUpper
        } else if at_lo && mu >= 0.0 {
            BoundStatus::AtLower
        } else if at_up && mu <= 0.0 {
            BoundStatus::AtUpper
        } else {
            BoundStatus::Inactive
        };
        bounds.push(status);
    }

    let mut constraints = Vec::with_capacity(m);
    for i in 0..m {
        if i < m_eq {
            constraints.push(ConsStatus::Equality);
            continue;
        }
        let lo_fin = g_l[i] > NLP_LOWER_BOUND_INF;
        let up_fin = g_u[i] < NLP_UPPER_BOUND_INF;
        if lo_fin && up_fin && (g_u[i] - g_l[i]).abs() < primal_tol {
            constraints.push(ConsStatus::Equality);
            continue;
        }
        let mu = lambda_g[i];
        let at_lo = lo_fin && (g[i] - g_l[i]).abs() < primal_tol;
        let at_up = up_fin && (g_u[i] - g[i]).abs() < primal_tol;
        let status = if mu > mult_tol && at_lo {
            ConsStatus::AtLower
        } else if mu < -mult_tol && at_up {
            ConsStatus::AtUpper
        } else if at_lo && mu >= 0.0 {
            ConsStatus::AtLower
        } else if at_up && mu <= 0.0 {
            ConsStatus::AtUpper
        } else {
            ConsStatus::Inactive
        };
        constraints.push(status);
    }

    WorkingSet {
        bounds,
        constraints,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_treats_nlp_bound_inf_sentinel_as_unbounded() {
        // PR #50 review A4 regression. Variables with `x_l =
        // NLP_LOWER_BOUND_INF` (the −1e19 sentinel) are unbounded
        // below; even a primal value at exactly that sentinel must
        // be tagged `Inactive`, not `AtLower`. Prior to the fix
        // `is_finite()` would treat `−1e19` as a real bound.
        let ws = classify_working_set(
            &[0.0],
            &[],
            0,
            &[-1.0e19],
            &[NLP_LOWER_BOUND_INF],
            &[NLP_UPPER_BOUND_INF],
            &[],
            &[],
            &[],
            1e-8,
            1e-6,
        );
        assert_eq!(ws.bounds[0], BoundStatus::Inactive);
    }

    #[test]
    fn classify_all_inactive_when_strictly_interior() {
        // 1-D unconstrained, x* in the interior, no multipliers.
        let ws = classify_working_set(
            &[0.0],
            &[],
            0,
            &[0.5],
            &[-1.0],
            &[1.0],
            &[],
            &[],
            &[],
            1e-8,
            1e-8,
        );
        assert_eq!(ws.bounds[0], BoundStatus::Inactive);
        assert!(ws.constraints.is_empty());
    }

    #[test]
    fn classify_lower_bound_active_when_primal_at_bound_and_mult_positive() {
        let ws = classify_working_set(
            &[2.0],
            &[],
            0,
            &[0.0],
            &[0.0],
            &[1.0],
            &[],
            &[],
            &[],
            1e-8,
            1e-8,
        );
        assert_eq!(ws.bounds[0], BoundStatus::AtLower);
    }

    #[test]
    fn classify_upper_bound_active_when_primal_at_bound_and_mult_negative() {
        let ws = classify_working_set(
            &[-2.0],
            &[],
            0,
            &[1.0],
            &[0.0],
            &[1.0],
            &[],
            &[],
            &[],
            1e-8,
            1e-8,
        );
        assert_eq!(ws.bounds[0], BoundStatus::AtUpper);
    }

    #[test]
    fn classify_fixed_when_bounds_equal() {
        let ws = classify_working_set(
            &[0.0],
            &[],
            0,
            &[2.0],
            &[2.0],
            &[2.0],
            &[],
            &[],
            &[],
            1e-8,
            1e-8,
        );
        assert_eq!(ws.bounds[0], BoundStatus::Fixed);
    }

    #[test]
    fn classify_equality_constraint_always_active() {
        // 1 eq constraint at row 0, no ineqs.
        let ws = classify_working_set(
            &[],
            &[1.0],
            1,
            &[],
            &[],
            &[],
            &[5.0],
            &[5.0],
            &[5.0],
            1e-8,
            1e-8,
        );
        assert_eq!(ws.constraints[0], ConsStatus::Equality);
    }

    #[test]
    fn classify_inequality_at_lower_bound() {
        let ws = classify_working_set(
            &[],
            &[3.0],
            0,
            &[],
            &[],
            &[],
            &[1.0],
            &[1.0],
            &[10.0],
            1e-8,
            1e-8,
        );
        assert_eq!(ws.constraints[0], ConsStatus::AtLower);
    }

    #[test]
    fn classify_inequality_at_upper_bound() {
        let ws = classify_working_set(
            &[],
            &[-3.0],
            0,
            &[],
            &[],
            &[],
            &[10.0],
            &[0.0],
            &[10.0],
            1e-8,
            1e-8,
        );
        assert_eq!(ws.constraints[0], ConsStatus::AtUpper);
    }

    #[test]
    fn classify_inactive_when_primal_off_bound_despite_large_multiplier() {
        // Bound multiplier is large but primal is mid-range —
        // tag as Inactive, not AtLower. This guards against
        // stale-multiplier carry from a slightly mis-aligned
        // perturbation.
        let ws = classify_working_set(
            &[2.0],
            &[],
            0,
            &[0.5],
            &[0.0],
            &[1.0],
            &[],
            &[],
            &[],
            1e-8,
            1e-8,
        );
        assert_eq!(ws.bounds[0], BoundStatus::Inactive);
    }
}
