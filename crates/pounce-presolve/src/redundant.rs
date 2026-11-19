//! Phase 2 — redundant linear-constraint detection.
//!
//! For a linear row `lo ≤ Σ a_j x_j ≤ hi`, the activity bounds
//! `[act_lo, act_hi]` computed from the (possibly already
//! Phase-1-tightened) box `[x_l, x_u]` give the row's full range. If
//! `[act_lo, act_hi] ⊆ [lo, hi]` then the row imposes no additional
//! constraint and can be dropped — Andersen §3 calls these
//! *forcing/dominated* constraints in their stronger form, but the
//! pure-redundancy case is what we ship here.
//!
//! The containment test is **scale-relative** (gh #1197). `tol` is applied
//! to the row's own magnitude — the largest of its finite activity terms
//! `|a_j|·|x_j|` over the box and its finite bounds — never as an absolute
//! slack on the activity. Multiplying a row by `s > 0` scales its activity
//! interval and its bounds alike, so the verdict must not depend on `s`. An
//! absolute `1e-9` did: at `s = 1e-12` the row `s·x == 0.2·s` over
//! `x ∈ [0, 1]` has activity `[0, 1e-12]`, both ends within `1e-9` of
//! `0.2e-12`, and was dropped as redundant although no point of the box is
//! within 60% of satisfying it — turning an infeasible model `Optimal`.

use crate::bound_tighten::{INF_BOUND, LinearRow, row_activity_pub};
use pounce_common::types::Number;

/// Compute a boolean mask, `true` at rows that are redundant given
/// the current variable box. `tol` is relative to each row's magnitude
/// (see the module docs), so it absorbs the float error of summing the
/// activity and nothing more.
pub fn find_redundant_rows(
    rows: &[LinearRow],
    x_l: &[Number],
    x_u: &[Number],
    tol: Number,
) -> Vec<bool> {
    rows.iter()
        .map(|row| is_redundant(row, x_l, x_u, tol))
        .collect()
}

fn is_redundant(row: &LinearRow, x_l: &[Number], x_u: &[Number], tol: Number) -> bool {
    let act = row_activity_pub(row, x_l, x_u);
    let tol = tol * row_scale(row, x_l, x_u);
    let lo_satisfied =
        row.lo <= -INF_BOUND || (act.lo_neg_inf == 0 && act.lo_finite >= row.lo - tol);
    let hi_satisfied =
        row.hi >= INF_BOUND || (act.hi_pos_inf == 0 && act.hi_finite <= row.hi + tol);
    lo_satisfied && hi_satisfied
}

/// Magnitude of a row: the largest finite `|a_j|·|bound of x_j|` term of its
/// activity over the box, or finite row bound. Scales linearly with the row,
/// so `tol * row_scale` is a relative test. Absent bounds contribute nothing;
/// a row with an unbounded activity side is decided by the infinity counts,
/// not by this tolerance.
fn row_scale(row: &LinearRow, x_l: &[Number], x_u: &[Number]) -> Number {
    let finite = |v: Number| if v.abs() < INF_BOUND { v.abs() } else { 0.0 };
    let mut scale = finite(row.lo).max(finite(row.hi));
    for &(j, a) in &row.entries {
        let j = j as usize;
        let a = a.abs();
        if a.is_finite() {
            scale = scale.max(a * finite(x_l[j])).max(a * finite(x_u[j]));
        }
    }
    scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_common::types::Index;

    fn row(entries: &[(Index, Number)], lo: Number, hi: Number) -> LinearRow {
        LinearRow {
            entries: entries.to_vec(),
            lo,
            hi,
        }
    }

    #[test]
    fn double_sided_implied_by_box() {
        // x ∈ [0,1], y ∈ [0,1], 0 ≤ x + y ≤ 2.  Activity is [0,2] ⊆ [0,2].
        let r = vec![row(&[(0, 1.0), (1, 1.0)], 0.0, 2.0)];
        let mask = find_redundant_rows(&r, &[0.0, 0.0], &[1.0, 1.0], 1e-12);
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn equality_row_never_redundant_unless_pinned() {
        // x + y = 1 with x,y ∈ [0,1]: act [0,2] not ⊆ [1,1].
        let r = vec![row(&[(0, 1.0), (1, 1.0)], 1.0, 1.0)];
        let mask = find_redundant_rows(&r, &[0.0, 0.0], &[1.0, 1.0], 1e-12);
        assert_eq!(mask, vec![false]);
    }

    #[test]
    fn open_upper_redundancy() {
        // x ∈ [0,1], x ≤ 5  — upper unsatisfied by act_hi=1 ≤ 5. lo open.
        let r = vec![row(&[(0, 1.0)], -INF_BOUND, 5.0)];
        let mask = find_redundant_rows(&r, &[0.0], &[1.0], 1e-12);
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn unbounded_var_blocks_redundancy() {
        // x ∈ [-inf, +inf], x ≤ 5 — act_hi unbounded above ⇒ not redundant.
        let r = vec![row(&[(0, 1.0)], -INF_BOUND, 5.0)];
        let mask = find_redundant_rows(&r, &[-INF_BOUND], &[INF_BOUND], 1e-12);
        assert_eq!(mask, vec![false]);
    }

    /// gh #1197: `s·x == 0.2·s` over `x ∈ [0, 1]` is violated everywhere in
    /// the box except `x = 0.2`; it must never be redundant, at any row scale.
    #[test]
    fn violated_row_is_not_redundant_at_any_scale() {
        for k in -15..=15 {
            let s = 10.0_f64.powi(k);
            let r = vec![row(&[(0, s)], 0.2 * s, 0.2 * s)];
            let mask = find_redundant_rows(&r, &[0.0], &[1.0], 1e-9);
            assert_eq!(mask, vec![false], "row scale 1e{k}");
        }
    }

    /// The other branch: a row the box genuinely implies stays redundant at
    /// every row scale, including when the box is pinned and the activity
    /// carries float error from the propagated bound.
    #[test]
    fn implied_row_is_redundant_at_any_scale() {
        for k in -15..=15 {
            let s = 10.0_f64.powi(k);
            let r = vec![
                row(&[(0, s), (1, s)], 0.0, 2.0 * s),
                row(&[(0, s)], 0.1 * s, 0.1 * s),
            ];
            // x0 pinned at 0.1 with a one-ulp wobble, as Phase 1 leaves it.
            let x0 = 0.1_f64;
            let mask = find_redundant_rows(
                &r,
                &[x0, 0.0],
                &[f64::from_bits(x0.to_bits() + 1), 1.0],
                1e-9,
            );
            assert_eq!(mask, vec![true, true], "row scale 1e{k}");
        }
    }
}
