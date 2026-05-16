//! Phase 1 — bound tightening via constraint propagation.
//!
//! Implements Andersen & Andersen, *Presolving in Linear Programming*,
//! Math. Prog. 71 (1995) §2, restricted to LINEAR constraint rows of
//! the NLP. Nonlinear rows contribute nothing here (a future Phase
//! could add McCormick / interval propagation; see issue #20's
//! "Out of scope (explicitly)").
//!
//! For each linear row `lo ≤ Σ a_j x_j ≤ hi` and each variable
//! `j` with `a_j ≠ 0`, the implied bounds are:
//!
//! ```text
//!   if a_j > 0:
//!     x_j ≥ (lo - others_max) / a_j
//!     x_j ≤ (hi - others_min) / a_j
//!   if a_j < 0:
//!     x_j ≥ (hi - others_min) / a_j
//!     x_j ≤ (lo - others_max) / a_j
//! ```
//!
//! where `others_min`/`others_max` are the row's activity bounds with
//! `j`'s contribution removed.
//!
//! Inputs are CSR-ish: for each linear row, a slice of `(j, a_{i,j})`
//! pairs. This decouples the algorithm from the TNLP/Jacobian access
//! pattern so it is straightforward to unit-test on hand-built
//! fixtures.

use pounce_common::types::{Index, Number};

/// Anything outside `(-INF_BOUND, +INF_BOUND)` is treated as
/// unbounded — matches `nlp_lower_bound_inf` / `nlp_upper_bound_inf`.
pub const INF_BOUND: Number = 1.0e19;

/// One linear constraint as `(coefficient, variable_index)` pairs and
/// the row's two-sided bounds.
#[derive(Debug, Clone)]
pub struct LinearRow {
    pub entries: Vec<(Index, Number)>,
    pub lo: Number,
    pub hi: Number,
}

/// Summary of one pass of [`tighten_bounds`].
#[derive(Debug, Clone, Default)]
pub struct TightenReport {
    /// Number of (j, side) bound updates that were actually tighter
    /// than the incoming bound by more than `tol`.
    pub n_tightened: Index,
    /// Number of bounds whose finite/-INF or finite/+INF status
    /// flipped on this pass (those count once each in `n_tightened`).
    pub n_new_finite: Index,
    /// True if the algorithm detected an empty feasible region
    /// (`x_l[j] > x_u[j]` after propagation, beyond `tol`).
    pub infeasible: bool,
}

/// Run bound tightening to a fixed point. Mutates `x_l` / `x_u`
/// in-place. Stops after `max_passes` rounds or when a full pass
/// changes no bound by more than `tol`.
///
/// Returns the aggregated report.
pub fn tighten_bounds(
    rows: &[LinearRow],
    x_l: &mut [Number],
    x_u: &mut [Number],
    max_passes: Index,
    tol: Number,
) -> TightenReport {
    let mut total = TightenReport::default();
    for _ in 0..max_passes.max(1) {
        let pass = tighten_pass(rows, x_l, x_u, tol);
        total.n_tightened += pass.n_tightened;
        total.n_new_finite += pass.n_new_finite;
        if pass.infeasible {
            total.infeasible = true;
            return total;
        }
        if pass.n_tightened == 0 {
            break;
        }
    }
    total
}

fn tighten_pass(
    rows: &[LinearRow],
    x_l: &mut [Number],
    x_u: &mut [Number],
    tol: Number,
) -> TightenReport {
    let mut report = TightenReport::default();
    for row in rows {
        // Row activity, tracked as (finite_sum, count_of_infinite_terms).
        // When the count is 0, the "others" activity is fully known;
        // when it is exactly 1 *and* var j is the one infinite term,
        // its removal still yields the finite sum.
        let act = row_activity(row, x_l, x_u);
        for &(j, a) in &row.entries {
            if a == 0.0 {
                continue;
            }
            let j = j as usize;
            let (others_lo, others_hi) = act.others_for(a, x_l[j], x_u[j]);

            let (new_lo, new_hi) =
                implied_bounds_for_var(row.lo, row.hi, a, others_lo, others_hi);

            if let Some(nl) = new_lo {
                if nl > x_l[j] + tol {
                    let was_inf = x_l[j] <= -INF_BOUND;
                    x_l[j] = nl;
                    report.n_tightened += 1;
                    if was_inf && nl > -INF_BOUND {
                        report.n_new_finite += 1;
                    }
                }
            }
            if let Some(nh) = new_hi {
                if nh < x_u[j] - tol {
                    let was_inf = x_u[j] >= INF_BOUND;
                    x_u[j] = nh;
                    report.n_tightened += 1;
                    if was_inf && nh < INF_BOUND {
                        report.n_new_finite += 1;
                    }
                }
            }
            if x_l[j] > x_u[j] + tol {
                report.infeasible = true;
                return report;
            }
        }
    }
    report
}

/// Activity of a linear row, split into finite-sum and a count of
/// ±∞ contributors. Allows precise removal of a single variable's
/// contribution even when other variables are unbounded.
///
/// `pub(crate)` so the `redundant` module can read these fields when
/// deciding whether a row is implied by the current variable box.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowActivity {
    /// Sum of finite contributions to the *minimum* activity.
    pub(crate) lo_finite: Number,
    /// Count of contributions equal to -∞.
    pub(crate) lo_neg_inf: u32,
    /// Sum of finite contributions to the *maximum* activity.
    pub(crate) hi_finite: Number,
    /// Count of contributions equal to +∞.
    pub(crate) hi_pos_inf: u32,
}

impl RowActivity {
    /// Activity of the row across *all* its variables.
    fn others_for(
        &self,
        a: Number,
        xl: Number,
        xu: Number,
    ) -> (Option<Number>, Option<Number>) {
        let (cj_lo, cj_hi) = contribution(a, xl, xu);

        // others_lo: subtract cj_lo (which may be -∞) from (lo_finite, lo_neg_inf).
        let others_lo = if cj_lo == Number::NEG_INFINITY {
            // Removing one -∞ contributor.
            if self.lo_neg_inf == 1 {
                Some(self.lo_finite)
            } else {
                None
            }
        } else if self.lo_neg_inf > 0 {
            None
        } else {
            Some(self.lo_finite - cj_lo)
        };

        let others_hi = if cj_hi == Number::INFINITY {
            if self.hi_pos_inf == 1 {
                Some(self.hi_finite)
            } else {
                None
            }
        } else if self.hi_pos_inf > 0 {
            None
        } else {
            Some(self.hi_finite - cj_hi)
        };

        (others_lo, others_hi)
    }
}

/// `pub(crate)` re-export of the row-activity helper, for the
/// redundant-row detector. Computes both endpoints of the activity
/// interval treating `|x| ≥ INF_BOUND` as ±∞.
pub(crate) fn row_activity_pub(row: &LinearRow, x_l: &[Number], x_u: &[Number]) -> RowActivity {
    row_activity(row, x_l, x_u)
}

fn row_activity(row: &LinearRow, x_l: &[Number], x_u: &[Number]) -> RowActivity {
    let mut a = RowActivity {
        lo_finite: 0.0,
        lo_neg_inf: 0,
        hi_finite: 0.0,
        hi_pos_inf: 0,
    };
    for &(j, coef) in &row.entries {
        let j = j as usize;
        let (cj_lo, cj_hi) = contribution(coef, x_l[j], x_u[j]);
        if cj_lo == Number::NEG_INFINITY {
            a.lo_neg_inf += 1;
        } else {
            a.lo_finite += cj_lo;
        }
        if cj_hi == Number::INFINITY {
            a.hi_pos_inf += 1;
        } else {
            a.hi_finite += cj_hi;
        }
    }
    a
}

/// Contribution of `a * x_j` to (min activity, max activity), with
/// `|x| >= INF_BOUND` propagating to ±∞ in the result.
fn contribution(a: Number, xl: Number, xu: Number) -> (Number, Number) {
    if a > 0.0 {
        (mul_bound(a, xl), mul_bound(a, xu))
    } else {
        (mul_bound(a, xu), mul_bound(a, xl))
    }
}

fn mul_bound(a: Number, x: Number) -> Number {
    if x >= INF_BOUND {
        if a > 0.0 {
            Number::INFINITY
        } else {
            Number::NEG_INFINITY
        }
    } else if x <= -INF_BOUND {
        if a > 0.0 {
            Number::NEG_INFINITY
        } else {
            Number::INFINITY
        }
    } else {
        a * x
    }
}

/// Row-implied bounds on `x_j` given row bounds, a_j, and the
/// activity of the other variables. `None` on a side means no
/// implied bound from this row (the other variables' contribution
/// is unbounded on the relevant side).
fn implied_bounds_for_var(
    lo: Number,
    hi: Number,
    a: Number,
    others_lo: Option<Number>,
    others_hi: Option<Number>,
) -> (Option<Number>, Option<Number>) {
    let row_lo_finite = lo > -INF_BOUND;
    let row_hi_finite = hi < INF_BOUND;

    let mut new_lo = None;
    let mut new_hi = None;

    // From a_j * x_j ≥ lo - others_hi.
    if row_lo_finite {
        if let Some(oh) = others_hi {
            let rhs = (lo - oh) / a;
            if a > 0.0 {
                new_lo = Some(rhs);
            } else {
                new_hi = Some(rhs);
            }
        }
    }
    // From a_j * x_j ≤ hi - others_lo.
    if row_hi_finite {
        if let Some(ol) = others_lo {
            let rhs = (hi - ol) / a;
            if a > 0.0 {
                new_hi = Some(rhs);
            } else {
                new_lo = Some(rhs);
            }
        }
    }
    (new_lo, new_hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    type RowSpec<'a> = (&'a [(Index, Number)], Number, Number);
    fn rows(specs: &[RowSpec<'_>]) -> Vec<LinearRow> {
        specs
            .iter()
            .map(|(es, lo, hi)| LinearRow {
                entries: es.to_vec(),
                lo: *lo,
                hi: *hi,
            })
            .collect()
    }

    #[test]
    fn no_propagation_when_already_tight() {
        // x in [0,1], y in [0,1], x + y = 1.5 (infeasible-ish but doesn't matter)
        let r = rows(&[(&[(0, 1.0), (1, 1.0)], 1.5, 1.5)]);
        let mut xl = vec![0.0, 0.0];
        let mut xu = vec![1.0, 1.0];
        let rep = tighten_bounds(&r, &mut xl, &mut xu, 3, 1e-12);
        // Each var's implied lower is 0.5; implied upper is 1.5 (clamped by current 1.0).
        assert_eq!(xl, vec![0.5, 0.5]);
        assert_eq!(xu, vec![1.0, 1.0]);
        assert!(rep.n_tightened >= 2);
    }

    #[test]
    fn two_round_propagation_on_chain() {
        // x in [0,10], y in [0,10], z in [0,10]
        // x + y = 1   ⇒ x ≤ 1, y ≤ 1
        // y + z = 1   ⇒ z ≤ 1 (after y ≤ 1 propagates)
        let r = rows(&[
            (&[(0, 1.0), (1, 1.0)], 1.0, 1.0),
            (&[(1, 1.0), (2, 1.0)], 1.0, 1.0),
        ]);
        let mut xl = vec![0.0; 3];
        let mut xu = vec![10.0; 3];
        let rep = tighten_bounds(&r, &mut xl, &mut xu, 3, 1e-12);
        for (j, &xuj) in xu.iter().enumerate() {
            assert!(xuj <= 1.0 + 1e-12, "var {j} upper {} > 1", xuj);
        }
        assert!(rep.n_tightened >= 3);
        assert!(!rep.infeasible);
    }

    #[test]
    fn unbounded_other_var_blocks_propagation() {
        // x ∈ [0,1], y ∈ [-inf, +inf], x + y ≤ 5
        // For x: others (y) has unbounded max ⇒ no upper tightening of x.
        // For y: others (x) is bounded — y ≤ 5 - 0 = 5.
        let r = rows(&[(&[(0, 1.0), (1, 1.0)], -INF_BOUND, 5.0)]);
        let mut xl = vec![0.0, -INF_BOUND];
        let mut xu = vec![1.0, INF_BOUND];
        let rep = tighten_bounds(&r, &mut xl, &mut xu, 3, 1e-12);
        assert_eq!(xu[0], 1.0, "x upper should not have moved");
        assert!(xu[1] <= 5.0 + 1e-12, "y upper should be ≤ 5, got {}", xu[1]);
        assert!(rep.n_new_finite >= 1);
    }

    #[test]
    fn negative_coefficient_flips_sides() {
        // -x + y = 0  with y ∈ [2, 3]  ⇒  x ∈ [2, 3]
        let r = rows(&[(&[(0, -1.0), (1, 1.0)], 0.0, 0.0)]);
        let mut xl = vec![-INF_BOUND, 2.0];
        let mut xu = vec![INF_BOUND, 3.0];
        let _ = tighten_bounds(&r, &mut xl, &mut xu, 3, 1e-12);
        assert!((xl[0] - 2.0).abs() < 1e-12, "x_l = {}", xl[0]);
        assert!((xu[0] - 3.0).abs() < 1e-12, "x_u = {}", xu[0]);
    }

    #[test]
    fn infeasibility_detected() {
        // x ∈ [0,1], x ≥ 2.
        let r = rows(&[(&[(0, 1.0)], 2.0, INF_BOUND)]);
        let mut xl = vec![0.0];
        let mut xu = vec![1.0];
        let rep = tighten_bounds(&r, &mut xl, &mut xu, 3, 1e-12);
        assert!(rep.infeasible);
    }

    #[test]
    fn max_passes_caps_work() {
        // A trivially convergent fixture, but force max_passes=1 and
        // confirm we still run one pass.
        let r = rows(&[(&[(0, 1.0), (1, 1.0)], 1.0, 1.0)]);
        let mut xl = vec![0.0, 0.0];
        let mut xu = vec![10.0, 10.0];
        let rep = tighten_bounds(&r, &mut xl, &mut xu, 1, 1e-12);
        assert!(rep.n_tightened > 0);
    }
}
