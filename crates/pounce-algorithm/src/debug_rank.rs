//! Numerical rank diagnosis of the active-constraint Jacobian.
//!
//! When the interior-point solver applies dual regularization δ_c (or
//! reports wrong KKT inertia), the usual root cause is a (near)
//! rank-deficient constraint Jacobian — linearly dependent or redundant
//! equality constraints, i.e. an LICQ failure. The debugger already
//! *detects* this as a scalar (`δ_c > 0`) and names *structurally*
//! dependent rows via a Dulmage–Mendelsohn pass on the sparsity pattern
//! (`pounce-presolve`). This module closes the remaining gap: a
//! *numerical* rank-revealing SVD of the equality Jacobian at the current
//! iterate, which localizes the dependency to the specific equations that
//! participate in it — including dependencies that are numerical only
//! (cancelling values over a full-rank pattern), which the structural
//! pass cannot see.
//!
//! The math. For the equality Jacobian `A` (m_c × n) with SVD `A = U Σ Vᵀ`,
//! the left singular vectors `u_k` whose singular value `σ_k ≈ 0` span the
//! left null space — the row combinations `u_kᵀ A ≈ 0` that vanish. Row
//! `i`'s participation in that null space,
//! `w_i = Σ_{k : σ_k ≤ τ} u[i,k]² ∈ [0, 1]`, localizes the dependency to
//! specific equations (`w_i = 1` ⇒ row `i` lies entirely in the null
//! space). The REPL resolves the implicated rows to model names via the
//! same `.row` path as `print equation`.

use crate::debug::ResidKind;
use faer::Mat;
use pounce_common::types::Number;

/// Rows whose null-space participation is below this are treated as not
/// implicated (numerical dust). A genuine dependency among `k` rows
/// spreads weight ~`1/k` across them, well above this floor.
const CULPRIT_WEIGHT_TOL: Number = 1e-3;

/// Identity of one active row in a [`RankReport`], so the REPL can resolve
/// it to a model name (the equality name pool for [`ResidKind::Eq`], the
/// inequality pool for [`ResidKind::Ineq`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankRow {
    pub kind: ResidKind,
    pub index: usize,
}

/// One row implicated in the Jacobian's near-null space, with its
/// participation weight `Σ_{k∈null} u[i,k]² ∈ [0, 1]`.
#[derive(Clone, Copy, Debug)]
pub struct RankCulprit {
    /// Index into [`RankReport::rows`].
    pub row: usize,
    /// Null-space participation weight in `[0, 1]`.
    pub weight: Number,
}

/// Numerical rank diagnosis of the active-constraint Jacobian at one
/// iterate. See the module docs for the underlying SVD analysis.
#[derive(Clone, Debug)]
pub struct RankReport {
    /// The active rows analyzed, in block order (currently the equality
    /// block `J_c`). `rows[i]` is the identity of dense row `i`.
    pub rows: Vec<RankRow>,
    /// Number of variables (columns of the analyzed block).
    pub n_cols: usize,
    /// Singular values, nonincreasing.
    pub singular_values: Vec<Number>,
    /// Numerical-rank threshold `τ = σ_max · max(m, n) · ε`.
    pub tol: Number,
    /// Numerical rank (count of `σ > τ`).
    pub rank: usize,
    /// Condition number `σ_max / σ_min` (`∞` if `σ_min == 0`).
    pub cond: Number,
    /// Rows participating in the near-null space, sorted by weight
    /// descending. Empty when the block has full row rank.
    pub culprits: Vec<RankCulprit>,
}

impl RankReport {
    /// Number of active rows analyzed (`m`).
    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }

    /// Row-rank deficiency `m − rank` — the number of dependent rows.
    pub fn deficiency(&self) -> usize {
        self.rows.len().saturating_sub(self.rank)
    }

    /// Whether the analyzed block is (numerically) row-rank-deficient.
    pub fn is_rank_deficient(&self) -> bool {
        self.rank < self.rows.len()
    }

    /// Smallest singular value (0 if there are none).
    pub fn sigma_min(&self) -> Number {
        self.singular_values.last().copied().unwrap_or(0.0)
    }

    /// Largest singular value (0 if there are none).
    pub fn sigma_max(&self) -> Number {
        self.singular_values.first().copied().unwrap_or(0.0)
    }
}

/// Run the rank-revealing SVD on a dense **row-major** `m × n` matrix and
/// build a [`RankReport`]. `rows` carries the identity of each of the `m`
/// rows (for downstream naming) and must have length `m`. Returns `None`
/// if the block is empty or the SVD fails to converge.
pub(crate) fn svd_rank(
    m: usize,
    n: usize,
    dense_row_major: &[Number],
    rows: Vec<RankRow>,
) -> Option<RankReport> {
    if m == 0 || n == 0 {
        return None;
    }
    debug_assert_eq!(dense_row_major.len(), m * n, "dense buffer is not m*n");
    debug_assert_eq!(rows.len(), m, "row-identity count must equal m");

    let a = Mat::from_fn(m, n, |i, j| dense_row_major[i * n + j]);
    let svd = a.svd().ok()?;

    let s: Vec<Number> = svd.S().column_vector().iter().copied().collect();
    let smax = s.first().copied().unwrap_or(0.0);
    // The standard LAPACK/NumPy numerical-rank threshold.
    let tol = smax * (m.max(n) as Number) * Number::EPSILON;
    let rank = s.iter().filter(|&&sv| sv > tol).count();
    let smin = s.last().copied().unwrap_or(0.0);
    let cond = if smin > 0.0 {
        smax / smin
    } else {
        Number::INFINITY
    };

    // Per-row participation in the left null space: the columns of U whose
    // singular value is ≤ τ. For a "tall" block (m > n) the columns in
    // `[n, m)` carry no singular value and span guaranteed null space — the
    // `rank..m` range below covers both cases (U is m × m).
    let u = svd.U();
    let mut weights = vec![0.0_f64; m];
    for k in rank..m {
        let col = u.col(k);
        for (i, &val) in col.iter().enumerate() {
            weights[i] += val * val;
        }
    }

    let mut culprits: Vec<RankCulprit> = weights
        .iter()
        .enumerate()
        .filter(|&(_, &w)| w > CULPRIT_WEIGHT_TOL)
        .map(|(row, &w)| RankCulprit { row, weight: w })
        .collect();
    culprits.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Some(RankReport {
        rows,
        n_cols: n,
        singular_values: s,
        tol,
        rank,
        cond,
        culprits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eq_rows(m: usize) -> Vec<RankRow> {
        (0..m)
            .map(|index| RankRow {
                kind: ResidKind::Eq,
                index,
            })
            .collect()
    }

    #[test]
    fn full_rank_block_has_no_culprits() {
        // 2×3 with independent rows.
        let dense = vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0,
        ];
        let r = svd_rank(2, 3, &dense, eq_rows(2)).expect("svd");
        assert_eq!(r.rank, 2);
        assert!(!r.is_rank_deficient());
        assert_eq!(r.deficiency(), 0);
        assert!(r.culprits.is_empty());
        assert!(r.cond.is_finite());
    }

    #[test]
    fn duplicate_rows_are_flagged_as_culprits() {
        // Rows 0 and 2 are identical ⇒ rank 2, one dependency shared
        // between rows 0 and 2 (each ~0.5 participation), row 1 clean.
        let dense = vec![
            1.0, 2.0, 3.0, //
            0.0, 1.0, 0.0, //
            1.0, 2.0, 3.0,
        ];
        let r = svd_rank(3, 3, &dense, eq_rows(3)).expect("svd");
        assert_eq!(r.rank, 2, "one redundant row");
        assert!(r.is_rank_deficient());
        assert_eq!(r.deficiency(), 1);
        let flagged: Vec<usize> = r.culprits.iter().map(|c| c.row).collect();
        assert!(
            flagged.contains(&0),
            "row 0 should be implicated: {flagged:?}"
        );
        assert!(
            flagged.contains(&2),
            "row 2 should be implicated: {flagged:?}"
        );
        assert!(!flagged.contains(&1), "row 1 is independent: {flagged:?}");
        // The two duplicates split the null direction roughly evenly.
        for c in &r.culprits {
            assert!((c.weight - 0.5).abs() < 0.1, "weight {} ~ 0.5", c.weight);
        }
    }

    #[test]
    fn value_dependence_is_caught_numerically() {
        // Row 2 = row 0 + row 1 exactly: the sparsity pattern is full
        // (every entry nonzero), so a structural Dulmage–Mendelsohn pass
        // sees full rank — but the *values* make the rows dependent. This
        // is precisely the gap the numerical SVD closes.
        let dense = vec![
            1.0, 0.0, 1.0, //
            0.0, 1.0, 1.0, //
            1.0, 1.0, 2.0,
        ];
        let r = svd_rank(3, 3, &dense, eq_rows(3)).expect("svd");
        assert_eq!(r.rank, 2, "value-dependent third row");
        assert!(r.is_rank_deficient());
        assert_eq!(r.deficiency(), 1);
        assert!(
            r.cond > 1e14 || r.cond.is_infinite(),
            "ill-conditioned: cond={:.2e}",
            r.cond
        );
        assert!(!r.culprits.is_empty(), "the dependency must be localized");
    }

    #[test]
    fn rank_threshold_keeps_small_but_resolvable_singular_values() {
        // A perturbation (1e-9) well above τ ≈ σ_max·3·ε must NOT be
        // mistaken for a dependency — the block is genuinely full rank.
        let dense = vec![
            1.0,
            0.0,
            1.0, //
            0.0,
            1.0,
            1.0, //
            1.0,
            1.0,
            2.0 + 1e-9,
        ];
        let r = svd_rank(3, 3, &dense, eq_rows(3)).expect("svd");
        assert_eq!(r.rank, 3, "1e-9 perturbation is above the rank tol");
        assert!(!r.is_rank_deficient());
        assert!(r.culprits.is_empty());
    }

    #[test]
    fn empty_block_returns_none() {
        assert!(svd_rank(0, 3, &[], vec![]).is_none());
        assert!(svd_rank(3, 0, &[], eq_rows(3)).is_none());
    }
}
