//! Phase 2 — redundant linear-constraint detection.
//!
//! For a linear row `lo ≤ Σ a_j x_j ≤ hi`, the activity bounds
//! `[act_lo, act_hi]` computed from the (possibly already
//! Phase-1-tightened) box `[x_l, x_u]` give the row's full range. If
//! `[act_lo, act_hi] ⊆ [lo, hi]` then the row imposes no additional
//! constraint and can be dropped — Andersen §3 calls these
//! *forcing/dominated* constraints in their stronger form, but the
//! pure-redundancy case is what we ship here.

use crate::bound_tighten::{INF_BOUND, LinearRow, row_activity_pub};
use pounce_common::types::Number;

/// Compute a boolean mask, `true` at rows that are redundant given
/// the current variable box.
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
    let lo_satisfied =
        row.lo <= -INF_BOUND || (act.lo_neg_inf == 0 && act.lo_finite >= row.lo - tol);
    let hi_satisfied =
        row.hi >= INF_BOUND || (act.hi_pos_inf == 0 && act.hi_finite <= row.hi + tol);
    lo_satisfied && hi_satisfied
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
}
