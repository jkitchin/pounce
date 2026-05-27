//! Trivial-elimination pre-pass for the auxiliary-equality
//! preprocessing pipeline.
//!
//! PR 13 of the auxiliary-presolve port (issue #53). Detects three
//! classes of "presolved-by-inspection" structure that the full
//! incidence → matching → DM → BTF pipeline shouldn't have to
//! reason about:
//!
//! - **Fixed variables** (`lb_i == ub_i` within `eq_tol`, both
//!   finite) — already pinned; including them in the incidence
//!   makes them look like elimination candidates when they aren't.
//! - **Free rows** (`g_l[i] <= -big_bound && g_u[i] >= big_bound`)
//!   — carry no constraint, just noise in the graph.
//! - **Trivially-slack inequalities** — linear inequality rows
//!   whose activity range computed from the variable box is
//!   already strictly inside `[g_l, g_u]`. They can't constrain
//!   anything in this pass and shouldn't trip coupling detection.
//!
//! All three are reported to the orchestrator as masks; the
//! incidence builder respects the masks so the downstream graph is
//! exactly what gets shrunk.

use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::Linearity;

/// Lists of indices flagged by the pre-pass.
#[derive(Debug, Clone, Default)]
pub struct TrivialEliminationReport {
    /// Variable indices where `lb == ub` (within `eq_tol`, both
    /// finite).
    pub fixed_vars: Vec<usize>,
    /// Row indices where both `g_l` and `g_u` exceed `big_bound`
    /// (in magnitude, opposite signs) — i.e. unconstrained rows.
    pub free_rows: Vec<usize>,
    /// Linear inequality rows whose activity range from the
    /// variable box is already strictly inside `[g_l, g_u]`.
    pub trivially_slack_rows: Vec<usize>,
}

/// Run the pre-pass. Returns indices in ascending order.
///
/// `big_bound` is the threshold above which a bound counts as
/// "infinite". `1e19` matches AMPL/Ipopt convention.
#[allow(clippy::too_many_arguments)]
pub fn find_trivial_eliminations(
    n_vars: usize,
    n_rows: usize,
    x_l: &[Number],
    x_u: &[Number],
    g_l: &[Number],
    g_u: &[Number],
    jac_irow: &[Index],
    jac_jcol: &[Index],
    jac_values: &[Number],
    linearity: &[Linearity],
    one_based: bool,
    eq_tol: Number,
    big_bound: Number,
) -> TrivialEliminationReport {
    let mut fixed_vars: Vec<usize> = Vec::new();
    for i in 0..n_vars {
        if x_l[i].is_finite() && x_u[i].is_finite() && (x_u[i] - x_l[i]).abs() <= eq_tol {
            fixed_vars.push(i);
        }
    }

    let mut free_rows: Vec<usize> = Vec::new();
    for i in 0..n_rows {
        if g_l[i] <= -big_bound && g_u[i] >= big_bound {
            free_rows.push(i);
        }
    }

    // Bucket Jacobian entries by row for the activity computation.
    let nnz = jac_irow.len();
    let mut by_row: Vec<Vec<(usize, Number)>> = vec![Vec::new(); n_rows];
    for k in 0..nnz {
        let i = if one_based {
            (jac_irow[k] as isize - 1) as usize
        } else {
            jac_irow[k] as usize
        };
        if i >= n_rows {
            continue;
        }
        let j = if one_based {
            (jac_jcol[k] as isize - 1) as usize
        } else {
            jac_jcol[k] as usize
        };
        if j >= n_vars {
            continue;
        }
        by_row[i].push((j, jac_values[k]));
    }

    let mut trivially_slack_rows: Vec<usize> = Vec::new();
    for i in 0..n_rows {
        // Skip equalities — they go through the equality pipeline.
        if (g_u[i] - g_l[i]).abs() <= eq_tol {
            continue;
        }
        // Skip rows we already counted as free.
        if g_l[i] <= -big_bound && g_u[i] >= big_bound {
            continue;
        }
        // Slack detection only meaningful for linear rows (constant
        // Jacobian); skip nonlinear conservatively.
        if !matches!(linearity[i], Linearity::Linear) {
            continue;
        }

        // Compute activity bounds [lo, hi] = Σ J[i][j] * x[j] over
        // the variable box. If any variable's bound is infinite the
        // activity is also unbounded → don't flag.
        let mut lo: Number = 0.0;
        let mut hi: Number = 0.0;
        let mut bounded = true;
        for &(j, v) in &by_row[i] {
            if v == 0.0 {
                continue;
            }
            let xl = x_l[j];
            let xu = x_u[j];
            if !xl.is_finite() || !xu.is_finite() {
                bounded = false;
                break;
            }
            if v > 0.0 {
                lo += v * xl;
                hi += v * xu;
            } else {
                lo += v * xu;
                hi += v * xl;
            }
        }
        if !bounded {
            continue;
        }

        // Tolerance buffer: a row whose activity exactly equals
        // `g_l` or `g_u` is not slack in any useful sense (the
        // bound is potentially active). Require strict interior.
        let buffer = eq_tol.max(1e-12);
        if g_l[i] + buffer <= lo && hi + buffer <= g_u[i] {
            trivially_slack_rows.push(i);
        }
    }

    TrivialEliminationReport {
        fixed_vars,
        free_rows,
        trivially_slack_rows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        n_vars: usize,
        n_rows: usize,
        x_l: &[Number],
        x_u: &[Number],
        g_l: &[Number],
        g_u: &[Number],
        jac_irow: &[Index],
        jac_jcol: &[Index],
        jac_values: &[Number],
        linearity: &[Linearity],
    ) -> TrivialEliminationReport {
        find_trivial_eliminations(
            n_vars, n_rows, x_l, x_u, g_l, g_u, jac_irow, jac_jcol, jac_values, linearity, false,
            1e-12, 1e19,
        )
    }

    #[test]
    fn detects_fixed_var() {
        let r = run(
            3,
            0,
            &[1.0, -1e19, 5.0],
            &[1.0, 1e19, 7.0],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        assert_eq!(r.fixed_vars, vec![0]);
        assert!(r.free_rows.is_empty());
        assert!(r.trivially_slack_rows.is_empty());
    }

    #[test]
    fn detects_free_row() {
        let r = run(
            1,
            2,
            &[-1e19],
            &[1e19],
            &[-1e20, 0.0],
            &[1e20, 5.0],
            &[],
            &[],
            &[],
            &[Linearity::Linear, Linearity::Linear],
        );
        assert_eq!(r.free_rows, vec![0]);
    }

    #[test]
    fn slack_inequality_with_bounded_vars() {
        // Single row: g(x) = x, x ∈ [0, 1], g ∈ [-10, 10]. Activity
        // [0, 1] is strictly inside [-10, 10] → slack.
        let r = run(
            1,
            1,
            &[0.0],
            &[1.0],
            &[-10.0],
            &[10.0],
            &[0],
            &[0],
            &[1.0],
            &[Linearity::Linear],
        );
        assert_eq!(r.trivially_slack_rows, vec![0]);
    }

    #[test]
    fn tight_inequality_not_flagged_slack() {
        // Activity touches the upper bound → not slack.
        let r = run(
            1,
            1,
            &[0.0],
            &[10.0],
            &[-10.0],
            &[10.0],
            &[0],
            &[0],
            &[1.0],
            &[Linearity::Linear],
        );
        assert!(r.trivially_slack_rows.is_empty());
    }

    #[test]
    fn equality_row_never_flagged_slack() {
        // g_l == g_u → equality, never counted as slack regardless
        // of activity.
        let r = run(
            1,
            1,
            &[0.0],
            &[1.0],
            &[5.0],
            &[5.0],
            &[0],
            &[0],
            &[1.0],
            &[Linearity::Linear],
        );
        assert!(r.trivially_slack_rows.is_empty());
    }

    #[test]
    fn nonlinear_row_never_flagged_slack() {
        // Same shape as `slack_inequality_with_bounded_vars` but
        // tagged NonLinear — activity bound is not valid.
        let r = run(
            1,
            1,
            &[0.0],
            &[1.0],
            &[-10.0],
            &[10.0],
            &[0],
            &[0],
            &[1.0],
            &[Linearity::NonLinear],
        );
        assert!(r.trivially_slack_rows.is_empty());
    }

    #[test]
    fn unbounded_var_blocks_slack_detection() {
        // x ∈ [-∞, 1] makes the activity unbounded below → no slack.
        let r = run(
            1,
            1,
            &[-1e19],
            &[1.0],
            &[-10.0],
            &[10.0],
            &[0],
            &[0],
            &[1.0],
            &[Linearity::Linear],
        );
        assert!(r.trivially_slack_rows.is_empty());
    }
}
