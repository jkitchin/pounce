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

use pounce_common::Number;
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_qp::{BoundStatus, ConsStatus, WorkingSet};

/// Classify the active set at iterate `(x, λ_x, λ_g)` against the
/// supplied bounds and constraint-bound vectors.
///
/// Inputs:
/// - `lambda_x`: packed signed bound multipliers (`z_l − z_u`) of
///   length `n`. Positive ⇒ lower bound active; negative ⇒ upper.
/// - `lambda_g`: stacked constraint multipliers `[y_c ; y_d]` of
///   length `m = m_eq + m_ineq`. The **opposite** sign convention to
///   `lambda_x`: negative ⇒ lower bound active, positive ⇒ upper. Both
///   follow from the one stationarity form `∇f + Jᵀλ_g − λ_x = 0` that
///   pounce-qp, the SQP driver's `check_kkt`, and IPOPT's `(y, z_l, z_u)`
///   all share; the bound block enters it negated, so its signs flip.
/// - `m_eq`: number of leading rows to flag [`ConsStatus::Equality`]
///   without consulting `g_l`/`g_u` — the SQP's internal
///   equalities-first order. For rows in a **model's** own `g` order
///   (what `finalize_solution` reports and
///   `IpoptApplication::set_sqp_warm_start` takes), pass `0`: a row
///   with `g_l == g_u` is classified `Equality` from its bounds anyway,
///   wherever it sits, and the result is then in that same row order.
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
///   regardless of primal distance. `0` takes every at-bound row
///   with a multiplier of the right sign (zero included) as active.
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
        // Both halves of the test must hold: at the bound *and* a multiplier
        // of the right sign at least `mult_tol` in magnitude. Two fallback
        // arms used to take any at-bound row with a multiplier of the right
        // sign, however small, which left `mult_tol` with no effect on
        // anything within `primal_tol` — the opposite of what it documents
        // (gh#1206 B3). `mult_tol = 0` keeps that old reading: a zero
        // multiplier at the bound is then still active.
        let status = if at_lo && mu >= mult_tol.max(0.0) {
            BoundStatus::AtLower
        } else if at_up && mu <= -mult_tol.max(0.0) {
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
        // Constraint-row multipliers carry the OPPOSITE sign to bound
        // multipliers, because they enter stationarity with the opposite
        // sign: `Hx + g + Aᵀλ_g − λ_x = 0`. So `λ_g ≤ 0` at an active lower
        // bound and `λ_g ≥ 0` at an active upper bound — the reverse of the
        // bound rules above, and pinned by
        // `classify_matches_pounce_qp_row_sign_convention` below.
        //
        // This block read the bound signs until gh#612. That is lossy rather
        // than wrong — a row the estimate calls `Inactive` is simply one the
        // QP has to re-add on its first pivot, and the returned solution is
        // unaffected — which is why nothing caught it: no test asserts a
        // working-set *estimate*, only the solutions it warm-starts. It
        // matters here because crossover's whole purpose is to hand the
        // active-set path an estimate that is already right (KNITRO §7 step
        // 2), and the old rules classified every active inequality row as
        // inactive.
        let mu = lambda_g[i];
        let at_lo = lo_fin && (g[i] - g_l[i]).abs() < primal_tol;
        let at_up = up_fin && (g_u[i] - g[i]).abs() < primal_tol;
        // Same two-sided test as the bounds, with the row signs (gh#1206 B3).
        let status = if at_lo && mu <= -mult_tol.max(0.0) {
            ConsStatus::AtLower
        } else if at_up && mu >= mult_tol.max(0.0) {
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
        // λ_g ≤ 0 at an active lower bound — the row convention, opposite to
        // the bound convention two tests up. See
        // `classify_matches_pounce_qp_row_sign_convention`.
        let ws = classify_working_set(
            &[],
            &[-3.0],
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
            &[3.0],
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

    /// Anchor the classifier's row-sign convention to the engine that
    /// consumes its output, rather than to a hand-written expectation.
    ///
    /// The classifier exists to hand `pounce-qp` a working set built from
    /// someone else's multipliers, so "which sign means AtLower" is not ours
    /// to choose — it is whatever `pounce-qp` returns. Asserting a literal
    /// (`-3.0 ⇒ AtLower`) restates a belief; solving the QP and feeding its
    /// own multipliers back through the classifier tests the agreement, and
    /// fails if either side's convention moves. gh#612: the two had silently
    /// disagreed on rows since the classifier was written.
    #[test]
    fn classify_matches_pounce_qp_row_sign_convention() {
        use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
        use pounce_qp::{HessianInertia, QpOptions, QpProblem, QpSolver};

        // min ½‖x‖²  s.t.  x₀ + x₁ ≥ 2.  Solution x = (1,1) with the row
        // active at its lower bound.
        let n = 2usize;
        let mut h = SymTMatrix::new(SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]));
        h.set_values(&[1.0, 1.0]);
        let mut a = GenTMatrix::new(GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]));
        a.set_values(&[1.0, 1.0]);
        let g = [0.0, 0.0];
        let bl = [2.0];
        let bu = [NLP_UPPER_BOUND_INF];
        let xl = [NLP_LOWER_BOUND_INF; 2];
        let xu = [NLP_UPPER_BOUND_INF; 2];
        let qp = QpProblem {
            n,
            m: 1,
            h: &h,
            g: &g,
            a: &a,
            bl: &bl,
            bu: &bu,
            xl: &xl,
            xu: &xu,
            hessian_inertia: HessianInertia::Psd,
        };
        let mut solver = pounce_qp::ParametricActiveSetSolver::new(Box::new(
            pounce_feral::FeralSolverInterface::new(),
        ));
        let sol = solver
            .solve(&qp, None, &QpOptions::default())
            .expect("QP solve");
        assert_eq!(sol.working.constraints[0], ConsStatus::AtLower);

        // Now the actual claim: re-deriving the working set from the
        // solution the engine returned reproduces the engine's own labels.
        let g_vals = [sol.x[0] + sol.x[1]];
        let ws = classify_working_set(
            &sol.lambda_x,
            &sol.lambda_g,
            0,
            &sol.x,
            &xl,
            &xu,
            &g_vals,
            &bl,
            &bu,
            1e-8,
            1e-6,
        );
        assert_eq!(
            ws.constraints, sol.working.constraints,
            "classifier disagrees with pounce-qp on row activity \
             (λ_g = {:?})",
            sol.lambda_g
        );
        assert_eq!(ws.bounds, sol.working.bounds);
    }

    /// gh#1206 B3: a row at its bound whose multiplier is below `mult_tol`
    /// is `Inactive`, as documented — bounds and constraint rows alike. Two
    /// fallback arms used to activate it anyway, so raising
    /// `crossover_mult_tol` changed nothing within `primal_tol`.
    #[test]
    fn classify_respects_mult_tol_at_the_bound() {
        // Lower bound, upper bound, and a row at each side; all primal-tight,
        // all with multipliers of the right sign but below `mult_tol`.
        let ws = classify_working_set(
            &[1e-4, -1e-4],
            &[-1e-4, 1e-4],
            0,
            &[0.0, 1.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 10.0],
            &[1.0, 0.0],
            &[10.0, 10.0],
            1e-3,
            1e-6,
        );
        assert_eq!(ws.bounds, vec![BoundStatus::Inactive; 2]);
        assert_eq!(ws.constraints, vec![ConsStatus::Inactive; 2]);

        // The same point with `mult_tol` below the multipliers: all active.
        let ws = classify_working_set(
            &[1e-4, -1e-4],
            &[-1e-4, 1e-4],
            0,
            &[0.0, 1.0],
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 10.0],
            &[1.0, 0.0],
            &[10.0, 10.0],
            1e-5,
            1e-6,
        );
        assert_eq!(ws.bounds, vec![BoundStatus::AtLower, BoundStatus::AtUpper]);
        assert_eq!(
            ws.constraints,
            vec![ConsStatus::AtLower, ConsStatus::AtUpper]
        );
    }

    /// `mult_tol = 0` keeps the old reading: a zero multiplier at the bound
    /// is active, a wrong-signed one is not.
    #[test]
    fn classify_zero_mult_tol_takes_zero_multiplier_at_bound_as_active() {
        let ws = classify_working_set(
            &[0.0, 0.0, -1.0],
            &[],
            0,
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            &[],
            &[],
            &[],
            0.0,
            1e-6,
        );
        assert_eq!(
            ws.bounds,
            vec![
                BoundStatus::AtLower,
                BoundStatus::AtUpper,
                BoundStatus::Inactive
            ]
        );
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
