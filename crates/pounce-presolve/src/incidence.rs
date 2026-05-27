//! Bipartite incidence graph between equality rows and variables.
//!
//! PR 2 of the auxiliary-presolve port (issue #53). Driven by the
//! inner TNLP's Jacobian sparsity plus the equality filter on
//! `(g_l, g_u)`. ripopt anchor:
//! `src/auxiliary_preprocessing.rs:2282-2318`.

use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::Linearity;

/// View into the data we need to build an [`EqualityIncidence`].
/// Decoupled from `TNLP` itself so this module stays unit-testable
/// without spinning up a full inner problem.
#[derive(Debug, Clone, Copy)]
pub struct ProbeView<'a> {
    /// Number of variables.
    pub n_vars: usize,
    /// Number of inner constraint rows.
    pub m_rows: usize,
    /// Inner Jacobian sparsity (one entry per structural nonzero).
    pub jac_irow: &'a [Index],
    pub jac_jcol: &'a [Index],
    /// Optional Jacobian values at a probe point. When provided,
    /// entries that evaluate to exactly `0.0` are dropped — this
    /// removes structural zeros that linearity hasn't already
    /// excluded.
    pub jac_values: Option<&'a [Number]>,
    pub g_l: &'a [Number],
    pub g_u: &'a [Number],
    /// Optional per-row linearity tags; unused by PR 2 but plumbed
    /// for PR 5's coupling classifier.
    pub linearity: Option<&'a [Linearity]>,
    /// `true` when the inner TNLP uses Fortran (1-based) indexing.
    pub one_based: bool,
    /// Tolerance for `g_l[i] == g_u[i]`. Rows tighter than this are
    /// treated as equalities.
    pub eq_tol: Number,
    /// PR 13: per-variable mask. `true` entries are excluded from
    /// the incidence graph. Used to drop trivially-fixed variables
    /// (`x_l[i] == x_u[i]`) before matching.
    pub excluded_vars: Option<&'a [bool]>,
    /// PR 13: per-row mask. `true` entries are excluded from the
    /// incidence graph. Used to drop free rows and trivially-slack
    /// inequalities before matching.
    pub excluded_rows: Option<&'a [bool]>,
}

/// CSR-style bipartite adjacency: equality rows ↔ variables.
///
/// # Example
///
/// ```
/// use pounce_presolve::incidence::{EqualityIncidence, ProbeView};
///
/// // 2 rows × 2 vars, one equality (row 0), one inequality (row 1).
/// // Jacobian touches (0,0), (0,1), (1,0).
/// let irow = [0, 0, 1];
/// let jcol = [0, 1, 0];
/// let g_l = [1.0, 0.0];
/// let g_u = [1.0, 5.0]; // row 1 is g(x) ∈ [0, 5], not an equality.
/// let p = ProbeView {
///     n_vars: 2,
///     m_rows: 2,
///     jac_irow: &irow,
///     jac_jcol: &jcol,
///     jac_values: None,
///     g_l: &g_l,
///     g_u: &g_u,
///     linearity: None,
///     one_based: false,
///     eq_tol: 1e-12,
///     excluded_vars: None,
///     excluded_rows: None,
/// };
/// let inc = EqualityIncidence::from_probe(&p);
/// assert_eq!(inc.n_eq_rows(), 1);
/// assert_eq!(inc.neighbors(0), &[0, 1]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct EqualityIncidence {
    /// Number of variables (columns) in the original problem.
    pub n_vars: usize,
    /// Inner-row indices of the equality rows, in ascending order.
    /// Length = `self.n_eq_rows()`.
    pub eq_row_inner_idx: Vec<usize>,
    /// CSR row pointers (length `n_eq_rows + 1`).
    pub adj_ptr: Vec<usize>,
    /// Sorted, deduped column indices per row.
    pub vars: Vec<usize>,
}

impl EqualityIncidence {
    /// Build an incidence graph from a probe.
    pub fn from_probe(p: &ProbeView<'_>) -> Self {
        // 1. Identify equality rows in inner-row index order.
        //    Excludes any row marked in `excluded_rows` (PR 13).
        let mut eq_row_inner_idx: Vec<usize> = Vec::new();
        let mut inner_to_eq: Vec<Option<usize>> = vec![None; p.m_rows];
        for (i, slot) in inner_to_eq.iter_mut().enumerate() {
            if let Some(mask) = p.excluded_rows {
                if mask[i] {
                    continue;
                }
            }
            if (p.g_u[i] - p.g_l[i]).abs() <= p.eq_tol {
                *slot = Some(eq_row_inner_idx.len());
                eq_row_inner_idx.push(i);
            }
        }
        let n_eq = eq_row_inner_idx.len();

        // 2. Bucket Jacobian entries by equality-row index.
        let mut per_row: Vec<Vec<usize>> = vec![Vec::new(); n_eq];
        let nnz = p.jac_irow.len();
        for k in 0..nnz {
            // Skip exact structural zeros when values are available.
            if let Some(vals) = p.jac_values {
                if vals[k] == 0.0 {
                    continue;
                }
            }
            let i = if p.one_based {
                (p.jac_irow[k] as isize - 1) as usize
            } else {
                p.jac_irow[k] as usize
            };
            if i >= p.m_rows {
                continue;
            }
            let Some(eq_k) = inner_to_eq[i] else { continue };
            let j = if p.one_based {
                (p.jac_jcol[k] as isize - 1) as usize
            } else {
                p.jac_jcol[k] as usize
            };
            if j >= p.n_vars {
                continue;
            }
            // PR 13: drop entries touching trivially-fixed vars.
            if let Some(mask) = p.excluded_vars {
                if mask[j] {
                    continue;
                }
            }
            per_row[eq_k].push(j);
        }

        // 3. Sort + dedupe each row; pack into CSR.
        let mut adj_ptr: Vec<usize> = Vec::with_capacity(n_eq + 1);
        let mut vars: Vec<usize> = Vec::new();
        adj_ptr.push(0);
        for row in per_row.iter_mut() {
            row.sort_unstable();
            row.dedup();
            vars.extend_from_slice(row);
            adj_ptr.push(vars.len());
        }

        Self {
            n_vars: p.n_vars,
            eq_row_inner_idx,
            adj_ptr,
            vars,
        }
    }

    /// Number of equality rows in the incidence graph.
    pub fn n_eq_rows(&self) -> usize {
        self.eq_row_inner_idx.len()
    }

    /// Sorted column neighbours of equality row `k` (0-based into
    /// `eq_row_inner_idx`). Panics if `k >= self.n_eq_rows()`.
    pub fn neighbors(&self, k: usize) -> &[usize] {
        let lo = self.adj_ptr[k];
        let hi = self.adj_ptr[k + 1];
        &self.vars[lo..hi]
    }
}

/// CSR-style bipartite adjacency: **inequality** rows ↔ variables.
///
/// Mirror of [`EqualityIncidence`] but for rows where
/// `|g_u - g_l| > eq_tol` (any range or one-sided bound). Used by
/// PR 5's coupling classifier: a candidate block is unsafe to
/// eliminate if its variables also appear in any inequality row.
#[derive(Debug, Clone, Default)]
pub struct InequalityIncidence {
    /// Number of variables (columns).
    pub n_vars: usize,
    /// Inner-row indices of the inequality rows, in ascending order.
    pub ineq_row_inner_idx: Vec<usize>,
    /// CSR row pointers.
    pub adj_ptr: Vec<usize>,
    /// Sorted, deduped column indices per row.
    pub vars: Vec<usize>,
    /// Reverse adjacency: for each variable, the list of inequality
    /// rows that touch it (sorted ascending).
    pub col_to_rows_ptr: Vec<usize>,
    pub col_to_rows: Vec<usize>,
}

impl InequalityIncidence {
    /// Build the inequality-incidence graph from the same probe
    /// [`EqualityIncidence`] uses; we just invert the equality
    /// filter.
    pub fn from_probe(p: &ProbeView<'_>) -> Self {
        // 1. Identify inequality rows. Excludes any row marked in
        //    `excluded_rows` (PR 13: free rows + trivially-slack
        //    inequalities).
        let mut ineq_row_inner_idx: Vec<usize> = Vec::new();
        let mut inner_to_ineq: Vec<Option<usize>> = vec![None; p.m_rows];
        for (i, slot) in inner_to_ineq.iter_mut().enumerate() {
            if let Some(mask) = p.excluded_rows {
                if mask[i] {
                    continue;
                }
            }
            if (p.g_u[i] - p.g_l[i]).abs() > p.eq_tol {
                *slot = Some(ineq_row_inner_idx.len());
                ineq_row_inner_idx.push(i);
            }
        }
        let n_ineq = ineq_row_inner_idx.len();

        // 2. Bucket Jacobian entries by inequality-row index.
        let mut per_row: Vec<Vec<usize>> = vec![Vec::new(); n_ineq];
        let nnz = p.jac_irow.len();
        for k in 0..nnz {
            if let Some(vals) = p.jac_values {
                if vals[k] == 0.0 {
                    continue;
                }
            }
            let i = if p.one_based {
                (p.jac_irow[k] as isize - 1) as usize
            } else {
                p.jac_irow[k] as usize
            };
            if i >= p.m_rows {
                continue;
            }
            let Some(ineq_k) = inner_to_ineq[i] else {
                continue;
            };
            let j = if p.one_based {
                (p.jac_jcol[k] as isize - 1) as usize
            } else {
                p.jac_jcol[k] as usize
            };
            if j >= p.n_vars {
                continue;
            }
            // PR 13: drop entries touching trivially-fixed vars.
            if let Some(mask) = p.excluded_vars {
                if mask[j] {
                    continue;
                }
            }
            per_row[ineq_k].push(j);
        }

        // 3. CSR pack.
        let mut adj_ptr: Vec<usize> = Vec::with_capacity(n_ineq + 1);
        let mut vars: Vec<usize> = Vec::new();
        adj_ptr.push(0);
        for row in per_row.iter_mut() {
            row.sort_unstable();
            row.dedup();
            vars.extend_from_slice(row);
            adj_ptr.push(vars.len());
        }

        // 4. Reverse adjacency for cheap "does var j touch an
        // inequality?" queries.
        let mut col_to_rows_ptr = vec![0usize; p.n_vars + 1];
        for &v in &vars {
            col_to_rows_ptr[v + 1] += 1;
        }
        for i in 1..=p.n_vars {
            col_to_rows_ptr[i] += col_to_rows_ptr[i - 1];
        }
        let mut col_to_rows = vec![0usize; col_to_rows_ptr[p.n_vars]];
        let mut cursor = col_to_rows_ptr[..p.n_vars].to_vec();
        for (ineq_k, row) in per_row.iter().enumerate() {
            for &v in row {
                col_to_rows[cursor[v]] = ineq_k;
                cursor[v] += 1;
            }
        }

        Self {
            n_vars: p.n_vars,
            ineq_row_inner_idx,
            adj_ptr,
            vars,
            col_to_rows_ptr,
            col_to_rows,
        }
    }

    /// Number of inequality rows.
    pub fn n_ineq_rows(&self) -> usize {
        self.ineq_row_inner_idx.len()
    }

    /// Sorted variable neighbours of inequality row `k`.
    pub fn neighbors(&self, k: usize) -> &[usize] {
        let lo = self.adj_ptr[k];
        let hi = self.adj_ptr[k + 1];
        &self.vars[lo..hi]
    }

    /// Sorted inequality-row neighbours of variable `j` (0-based into
    /// `0..n_vars`). Returns an empty slice if no inequality touches
    /// `j`.
    pub fn rows_for_var(&self, j: usize) -> &[usize] {
        let lo = self.col_to_rows_ptr[j];
        let hi = self.col_to_rows_ptr[j + 1];
        &self.col_to_rows[lo..hi]
    }

    /// True iff variable `j` appears in any inequality row.
    pub fn var_in_inequality(&self, j: usize) -> bool {
        !self.rows_for_var(j).is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe<'a>(
        n_vars: usize,
        m_rows: usize,
        irow: &'a [Index],
        jcol: &'a [Index],
        g_l: &'a [Number],
        g_u: &'a [Number],
    ) -> ProbeView<'a> {
        ProbeView {
            n_vars,
            m_rows,
            jac_irow: irow,
            jac_jcol: jcol,
            jac_values: None,
            g_l,
            g_u,
            linearity: None,
            one_based: false,
            eq_tol: 1e-12,
            excluded_vars: None,
            excluded_rows: None,
        }
    }

    #[test]
    fn incidence_empty_problem_is_empty() {
        let p = probe(0, 0, &[], &[], &[], &[]);
        let inc = EqualityIncidence::from_probe(&p);
        assert_eq!(inc.n_eq_rows(), 0);
        assert_eq!(inc.vars.len(), 0);
        assert_eq!(inc.adj_ptr, vec![0]);
    }

    #[test]
    fn incidence_filters_inequalities() {
        // Row 0 is g = 1 (equality). Row 1 is g ∈ [0, 5] (range).
        let p = probe(2, 2, &[0, 0, 1], &[0, 1, 0], &[1.0, 0.0], &[1.0, 5.0]);
        let inc = EqualityIncidence::from_probe(&p);
        assert_eq!(inc.n_eq_rows(), 1);
        assert_eq!(inc.eq_row_inner_idx, vec![0]);
        assert_eq!(inc.neighbors(0), &[0, 1]);
    }

    #[test]
    fn incidence_dedupes_and_sorts_columns() {
        // Same equality row mentions column 1 twice and column 0
        // after column 1 — output must be sorted [0, 1] without dupes.
        let p = probe(2, 1, &[0, 0, 0, 0], &[1, 1, 0, 1], &[2.5], &[2.5]);
        let inc = EqualityIncidence::from_probe(&p);
        assert_eq!(inc.neighbors(0), &[0, 1]);
    }

    #[test]
    fn incidence_respects_fortran_indexing() {
        let mut p = probe(
            2,
            1,
            &[1, 1], // Fortran rows 1..=1
            &[1, 2], // Fortran cols 1..=2
            &[0.0],
            &[0.0],
        );
        p.one_based = true;
        let inc = EqualityIncidence::from_probe(&p);
        assert_eq!(inc.n_eq_rows(), 1);
        assert_eq!(inc.neighbors(0), &[0, 1]);
    }

    #[test]
    fn incidence_drops_structural_zeros_when_values_provided() {
        // Row 0 touches columns 0 and 1, but the (0, 1) entry has
        // value 0.0 at the probe point.
        let vals = [3.5, 0.0];
        let p = ProbeView {
            n_vars: 2,
            m_rows: 1,
            jac_irow: &[0, 0],
            jac_jcol: &[0, 1],
            jac_values: Some(&vals),
            g_l: &[1.0],
            g_u: &[1.0],
            linearity: None,
            one_based: false,
            eq_tol: 1e-12,
            excluded_vars: None,
            excluded_rows: None,
        };
        let inc = EqualityIncidence::from_probe(&p);
        assert_eq!(inc.neighbors(0), &[0]);
    }

    #[test]
    fn inequality_incidence_filters_equalities() {
        // Row 0 is g = 1 (equality); row 1 is g ∈ [0, 5] (range).
        // Only row 1 should appear in InequalityIncidence.
        let p = probe(2, 2, &[0, 0, 1], &[0, 1, 0], &[1.0, 0.0], &[1.0, 5.0]);
        let ineq = InequalityIncidence::from_probe(&p);
        assert_eq!(ineq.n_ineq_rows(), 1);
        assert_eq!(ineq.ineq_row_inner_idx, vec![1]);
        assert_eq!(ineq.neighbors(0), &[0]);
        assert!(ineq.var_in_inequality(0));
        assert!(!ineq.var_in_inequality(1));
    }

    #[test]
    fn inequality_incidence_range_row() {
        // Single row, g ∈ [-2, 5] over both variables.
        let p = probe(2, 1, &[0, 0], &[0, 1], &[-2.0], &[5.0]);
        let ineq = InequalityIncidence::from_probe(&p);
        assert_eq!(ineq.n_ineq_rows(), 1);
        assert_eq!(ineq.neighbors(0), &[0, 1]);
        // Reverse adjacency points each variable back to row 0.
        assert_eq!(ineq.rows_for_var(0), &[0]);
        assert_eq!(ineq.rows_for_var(1), &[0]);
    }

    #[test]
    fn inequality_incidence_one_sided() {
        // g(x) ≤ 5 (lower = -∞ in Ipopt land). Encoded with a very
        // negative lower bound; abs gap >> eq_tol.
        let p = probe(1, 1, &[0], &[0], &[-1e19], &[5.0]);
        let ineq = InequalityIncidence::from_probe(&p);
        assert_eq!(ineq.n_ineq_rows(), 1);
        assert_eq!(ineq.neighbors(0), &[0]);
    }
}
