//! Inequality projection for `InequalityCoupled` candidate blocks.
//!
//! PR 14 of the auxiliary-presolve port (issue #53). Today the
//! orchestrator rejects any block whose variables also appear in an
//! inequality row — fixing the block could violate the inequality.
//! This module widens the accepted class: when both the block's
//! equality system AND the coupled inequality rows are linear, we
//! substitute the block solution into the inequalities and check
//! whether the result is implied by the variable box on the
//! surviving variables. If yes, the block is safe to eliminate
//! without further IPM machinery.
//!
//! Algorithm: given the block's linear system
//!
//! ```text
//!   J_block · b + J_other · y = const_block
//! ```
//!
//! solve for `b = M y + p` (one dense LU factorisation, used as a
//! right-hand-side solve once per surviving column plus once for
//! the constant). Then for each coupled inequality
//! `A_b · b + A_y · y ∈ [g_l, g_u]`:
//!
//! ```text
//!   coef = A_b · M + A_y
//!   rhs_l = g_l - A_b · p
//!   rhs_u = g_u - A_b · p
//! ```
//!
//! Compute the activity range `[lo, hi]` of `coef · y` over `y` in
//! `[x_l, x_u]`. If `rhs_l ≤ lo` AND `hi ≤ rhs_u`, this inequality
//! is implied. If every coupled inequality is implied, the block
//! is safe.
//!
//! The master plan refers to this as "Fourier-Motzkin projection."
//! Since BTF blocks are square (the equalities uniquely determine
//! `b`), substitution and FM elimination produce equivalent
//! projected constraints. General FM (handling underdetermined
//! sub-blocks, range inequalities producing new constraints to add
//! back to the IPM problem) is deferred; it requires extending
//! `PresolveTnlp` to synthesize rows.

use pounce_common::types::{Index, Number};

use crate::block_solve::{lu_factor_partial_pivot, lu_solve};

/// Outcome of projecting all coupled inequalities of one candidate
/// block.
#[derive(Debug, Clone)]
pub struct ProjectionResult {
    /// True iff every coupled inequality is implied by the
    /// variable box after the block substitution.
    pub all_implied: bool,
    /// Per-inequality details, in the order of `coupled_ineq_rows`
    /// passed in.
    pub per_row: Vec<ProjectedRow>,
}

#[derive(Debug, Clone)]
pub struct ProjectedRow {
    /// Inner-row index this projection came from.
    pub inner_row: usize,
    /// Coefficient vector over surviving (non-block) variables;
    /// length `n_vars`, zero at block columns.
    pub coef: Vec<Number>,
    /// Projected lower bound `g_l - A_b · p`.
    pub rhs_l: Number,
    /// Projected upper bound `g_u - A_b · p`.
    pub rhs_u: Number,
    /// Activity-range lower bound of `coef · y` over `[x_l, x_u]`.
    pub activity_lo: Number,
    /// Activity-range upper bound of `coef · y` over `[x_l, x_u]`.
    pub activity_hi: Number,
    /// True iff `rhs_l ≤ activity_lo ≤ activity_hi ≤ rhs_u`.
    pub implied: bool,
}

/// Project the coupled inequalities by substituting the block
/// solution `b = M y + p`. Returns `None` when `J_block` is
/// singular (block solve would have failed anyway) or when the
/// system can't be built coherently.
#[allow(clippy::too_many_arguments)]
pub fn project_inequalities(
    block_rows: &[usize],
    block_cols: &[usize],
    coupled_ineq_rows: &[usize],
    n_vars: usize,
    x_l: &[Number],
    x_u: &[Number],
    g_l: &[Number],
    g_u: &[Number],
    jac_irow: &[Index],
    jac_jcol: &[Index],
    jac_values: &[Number],
    g_at_probe: &[Number],
    x_probe: &[Number],
    one_based: bool,
) -> Option<ProjectionResult> {
    let k = block_rows.len();
    if k == 0 || k != block_cols.len() {
        return None;
    }

    // Bucket Jacobian entries by row → (col, value).
    let nnz = jac_irow.len();
    let mut by_row: std::collections::HashMap<usize, Vec<(usize, Number)>> =
        std::collections::HashMap::new();
    for kk in 0..nnz {
        let i = if one_based {
            (jac_irow[kk] as isize - 1) as usize
        } else {
            jac_irow[kk] as usize
        };
        let j = if one_based {
            (jac_jcol[kk] as isize - 1) as usize
        } else {
            jac_jcol[kk] as usize
        };
        if j >= n_vars {
            continue;
        }
        by_row.entry(i).or_default().push((j, jac_values[kk]));
    }

    // Index lookup: which position in `block_cols` does an inner
    // column belong to? -1 (None) if it's a surviving variable.
    let mut col_to_block_pos: Vec<Option<usize>> = vec![None; n_vars];
    for (pos, &c) in block_cols.iter().enumerate() {
        col_to_block_pos[c] = Some(pos);
    }

    // Build `J_block` (k×k, row-major) and `const_block` (k,
    // = g_l[r] - linear constant of row r).
    // Linear constant: for a linear row, g(x) = J · x + c, so c =
    // g(x_probe) - J · x_probe.
    let mut j_block = vec![0.0; k * k];
    let mut const_block = vec![0.0; k];
    for (i_block, &r) in block_rows.iter().enumerate() {
        let entries = match by_row.get(&r) {
            Some(e) => e,
            None => return None, // empty equality row — block ill-posed
        };
        let mut sum_jx = 0.0;
        for &(c, v) in entries {
            sum_jx += v * x_probe[c];
            if let Some(j_pos) = col_to_block_pos[c] {
                // Accumulate: duplicate (row, col) triplets sum, matching the
                // `sum_jx` convention used for the linear constant just above
                // (L43). Assignment would drop all but the last duplicate.
                j_block[i_block * k + j_pos] += v;
            }
        }
        let c_r = g_at_probe[r] - sum_jx;
        const_block[i_block] = g_l[r] - c_r;
    }

    // LU-factor J_block.
    let mut j_block_lu = j_block.clone();
    let piv = lu_factor_partial_pivot(&mut j_block_lu, k).ok()?;

    // Solve J_block · p = const_block.
    let mut p = const_block.clone();
    lu_solve(&j_block_lu, &piv, &mut p, k);

    // Solve J_block · M_col = J_other_col for each surviving col,
    // then negate to get the column of M. Store M as
    // `m_by_y[(surviving_col_index_in_block_var_space)][(block_var_idx)]`.
    // Cheaper: per surviving var v that appears in any block row,
    // build its J_other column, LU-solve, store. Vars that don't
    // appear in any block row have a zero column in M.
    let mut m_columns: std::collections::HashMap<usize, Vec<Number>> =
        std::collections::HashMap::new();
    for (i_block, &r) in block_rows.iter().enumerate() {
        let entries = by_row.get(&r).expect("checked above");
        for &(c, v) in entries {
            if col_to_block_pos[c].is_some() {
                continue;
            }
            // Accumulate J_other column for surviving var c.
            let col = m_columns.entry(c).or_insert_with(|| vec![0.0; k]);
            col[i_block] += v;
            let _ = v;
            let _ = i_block;
        }
    }
    let mut m: std::collections::HashMap<usize, Vec<Number>> =
        std::collections::HashMap::with_capacity(m_columns.len());
    for (c, mut col) in m_columns {
        lu_solve(&j_block_lu, &piv, &mut col, k);
        // Negate.
        for v in col.iter_mut() {
            *v = -*v;
        }
        m.insert(c, col);
    }

    // Now project each coupled inequality.
    let mut per_row: Vec<ProjectedRow> = Vec::with_capacity(coupled_ineq_rows.len());
    let mut all_implied = true;
    for &r in coupled_ineq_rows {
        let entries = by_row.get(&r).cloned().unwrap_or_default();
        // Compute the linear constant for this row.
        let mut sum_jx = 0.0;
        for &(c, v) in &entries {
            sum_jx += v * x_probe[c];
        }
        let const_r = g_at_probe[r] - sum_jx;
        let gl = g_l[r] - const_r;
        let gu = g_u[r] - const_r;
        // a_b · p: contribution from block columns hitting `p`.
        // Build A_b (in block-column order) and A_y (per-surviving-col).
        let mut a_b = vec![0.0; k];
        let mut a_y: std::collections::HashMap<usize, Number> = std::collections::HashMap::new();
        for &(c, v) in &entries {
            if let Some(pos) = col_to_block_pos[c] {
                // Accumulate duplicates, consistent with the `a_y` surviving-
                // column branch below and with `sum_jx` (L43).
                a_b[pos] += v;
            } else {
                *a_y.entry(c).or_insert(0.0) += v;
            }
        }
        let mut a_b_dot_p: Number = 0.0;
        for (i_block, &val) in a_b.iter().enumerate() {
            a_b_dot_p += val * p[i_block];
        }
        let rhs_l = gl - a_b_dot_p;
        let rhs_u = gu - a_b_dot_p;
        // coef[c] = a_y[c] + Σ_i a_b[i] * M[c][i]   (for surviving c)
        let mut coef = vec![0.0; n_vars];
        for (&c, &ay_val) in &a_y {
            coef[c] += ay_val;
        }
        for (&c, m_col) in &m {
            let mut s = 0.0;
            for i_block in 0..k {
                s += a_b[i_block] * m_col[i_block];
            }
            coef[c] += s;
        }
        // Activity range of `coef · y` over `y ∈ [x_l, x_u]`.
        // Treat block columns as their fixed values (b = M y + p
        // means the activity over block cols is zero in `coef`
        // because we substituted them out — coef is zero there).
        let mut lo: Number = 0.0;
        let mut hi: Number = 0.0;
        let mut bounded = true;
        for c in 0..n_vars {
            let v = coef[c];
            if v == 0.0 {
                continue;
            }
            // For block columns, x_l/x_u may already be clamped or
            // wide; but `coef[block_col]` should be zero by
            // construction.
            let xl = x_l[c];
            let xu = x_u[c];
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
        let (activity_lo, activity_hi) = if bounded {
            (lo, hi)
        } else {
            (Number::NEG_INFINITY, Number::INFINITY)
        };
        let implied = rhs_l <= activity_lo && activity_hi <= rhs_u;
        if !implied {
            all_implied = false;
        }
        per_row.push(ProjectedRow {
            inner_row: r,
            coef,
            rhs_l,
            rhs_u,
            activity_lo,
            activity_hi,
            implied,
        });
    }

    Some(ProjectionResult {
        all_implied,
        per_row,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_implied_singleton() {
        // Block: row 0, var 0. Equality: 1·b = 3, so b = 3.
        // Coupled ineq: row 1: 1·b + 0·y in [-10, 10]. After
        // substituting b=3: 3 in [-10, 10] → implied.
        // n_vars = 2 (b is x[0], y is x[1]).
        let result = project_inequalities(
            &[0], // block_rows
            &[0], // block_cols
            &[1], // coupled_ineq_rows
            2,
            &[-1e19, 0.0],
            &[1e19, 1.0],
            &[3.0, -10.0],
            &[3.0, 10.0],
            &[0, 1],
            &[0, 0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            false,
        )
        .expect("non-singular");
        assert!(result.all_implied);
        let row = &result.per_row[0];
        // No `y` coefficient → coef[1] = 0; activity range [0,0].
        assert_eq!(row.coef[0], 0.0); // block col
        assert!(row.coef[1].abs() < 1e-12);
        assert!((row.rhs_l - (-13.0)).abs() < 1e-12); // -10 - 3
        assert!((row.rhs_u - 7.0).abs() < 1e-12); // 10 - 3
        assert!(row.implied);
    }

    #[test]
    fn projection_not_implied_tight() {
        // Block: row 0, var 0. Equality: 1·b = 5, so b = 5.
        // Coupled ineq: row 1: 1·b in [-1, 1]. After substituting:
        // 5 in [-1, 1] → NOT implied. activity_lo = activity_hi = 0
        // (no y contribution), rhs_l = -6, rhs_u = -4. Check
        // [-6, -4] is required to contain [0, 0]? rhs_u = -4 < 0 →
        // not implied.
        let result = project_inequalities(
            &[0],
            &[0],
            &[1],
            2,
            &[-1e19, 0.0],
            &[1e19, 1.0],
            &[5.0, -1.0],
            &[5.0, 1.0],
            &[0, 1],
            &[0, 0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            false,
        )
        .expect("non-singular");
        assert!(!result.all_implied);
        assert!(!result.per_row[0].implied);
    }

    #[test]
    fn projection_2x2_block_implied() {
        // 2-var block. Equalities:
        //   r0: b0 + b1 = 3
        //   r1: b0 - b1 = 1
        // → b0 = 2, b1 = 1.
        // Coupled ineq r2: b0 + b1 + 0·y in [0, 100]. After
        // substitution: 3 ∈ [0, 100] → implied.
        let result = project_inequalities(
            &[0, 1],
            &[0, 1],
            &[2],
            3,
            &[-1e19, -1e19, 0.0],
            &[1e19, 1e19, 1.0],
            &[3.0, 1.0, 0.0],
            &[3.0, 1.0, 100.0],
            &[0, 0, 1, 1, 2, 2],
            &[0, 1, 0, 1, 0, 1],
            &[1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            false,
        )
        .expect("non-singular");
        assert!(result.all_implied);
    }

    #[test]
    fn projection_unbounded_var_in_y_blocks_admit() {
        // Block: row 0, var 0. r0: 1·b = 3 → b = 3.
        // Coupled ineq: row 1: 1·b + 1·y ≤ 100, with y unbounded
        // above. Activity hi = +∞ → not implied.
        let result = project_inequalities(
            &[0],
            &[0],
            &[1],
            2,
            &[-1e19, 0.0],
            &[1e19, 1e19], // y unbounded
            &[3.0, -1e19],
            &[3.0, 100.0],
            &[0, 1, 1],
            &[0, 0, 1],
            &[1.0, 1.0, 1.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            false,
        )
        .expect("non-singular");
        assert!(!result.all_implied);
    }

    #[test]
    fn projection_singular_block() {
        // 2-var block with rank-1 equality system → LU fails →
        // returns None.
        let r = project_inequalities(
            &[0, 1],
            &[0, 1],
            &[],
            2,
            &[-1e19, -1e19],
            &[1e19, 1e19],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 2.0, 2.0, 4.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            false,
        );
        assert!(r.is_none());
    }

    #[test]
    fn projection_sums_duplicate_jacobian_entries_in_block() {
        // L43: duplicate Jacobian triplets for the same (row, col) must be
        // *summed* (the convention the linear constant `sum_jx` already uses),
        // not overwritten. Here the equality row 0 carries TWO entries for the
        // block column 0 — `2.0` and `-1.0` — whose sum is the true coefficient
        // `1.0`. With assignment-instead-of-accumulation the last write (`-1.0`)
        // wins, so the block solves `-1·b = 3 → b = -3` instead of `1·b = 3 →
        // b = 3`.
        //
        // The coupled inequality (row 1, coefficient 1 on b) is `b ∈ [-5, -1]`.
        // With the correct `b = 3` it is NOT implied (3 ∉ [-5,-1]) — a real
        // constraint that must be kept. With the buggy `b = -3` it looks
        // implied (-3 ∈ [-5,-1]) and would be silently dropped → unsafe. So
        // the correct verdict is `all_implied == false`.
        let result = project_inequalities(
            &[0],              // block_rows
            &[0],              // block_cols
            &[1],              // coupled_ineq_rows
            1,                 // n_vars (only b)
            &[-1e19],          // x_l
            &[1e19],           // x_u
            &[3.0, -5.0],      // g_l: row0 equality = 3, row1 ineq lower
            &[3.0, -1.0],      // g_u: row0 equality = 3, row1 ineq upper
            &[0, 0, 1],        // jac_irow: row 0 twice (duplicate), row 1 once
            &[0, 0, 0],        // jac_jcol: all column 0
            &[2.0, -1.0, 1.0], // jac_values: 2 + (-1) = 1 for row 0
            &[0.0, 0.0],       // g_at_probe
            &[0.0],            // x_probe
            false,
        )
        .expect("non-singular: summed coefficient 1.0 is invertible");
        // Fail-first: with assignment, b = -3 makes the row look implied and
        // this is `true`.
        assert!(
            !result.all_implied,
            "row b ∈ [-5,-1] with b=3 is a real constraint, not implied",
        );
        assert!(!result.per_row[0].implied);
        // The block solution drives rhs: rhs_l = -5 - b, rhs_u = -1 - b with
        // b = 3 ⇒ [-8, -4], which does NOT contain the activity 0 ⇒ not implied.
        assert!(
            (result.per_row[0].rhs_u - (-4.0)).abs() < 1e-9,
            "rhs_u = {} (expected -1 - 3 = -4, i.e. b solved to 3 not -3)",
            result.per_row[0].rhs_u,
        );
    }
}
