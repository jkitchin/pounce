//! Symmetric Ruiz ∞-norm equilibration — port of the algorithm in
//! Ruiz, D., *A scaling algorithm to equilibrate both rows and columns
//! norms in matrices*, CERFACS TR/PA/01/14 (2001).
//!
//! For a symmetric matrix `K`, this computes a diagonal `D = diag(d)`
//! such that `D K D` has every row (and hence column, by symmetry) of
//! ∞-norm approximately 1. Single-diagonal form is mandatory for the
//! IPM augmented-system use case: we need the scaled matrix to remain
//! symmetric so the downstream symmetric factorization (MA57, MUMPS,
//! FERAL/SSIDS) doesn't lose its pivoting properties (issue #61).
//!
//! # Algorithm
//!
//! Start with `d_i = 1`. Then iterate:
//!
//! ```text
//!   for i in 0..n:
//!       r_i = max_j |K_ij| * d_i * d_j           // row ∞-norm of D K D
//!   for i in 0..n:
//!       d_i /= sqrt(r_i)                          // damp by sqrt of imbalance
//! ```
//!
//! repeating until `max_i |r_i - 1| < tol` or `iter == max_iter`. The
//! `sqrt` is the symmetric balancing trick: applying `d_i ← d_i /
//! sqrt(r_i)` halves the action of `d_i` on each row *and* column, so
//! the symmetric structure is preserved iteration by iteration.
//!
//! # Zero / near-zero rows
//!
//! A row that is structurally all-zero (or smaller than
//! [`Self::min_row_value`]) has `r_i == 0`. Dividing by `sqrt(0)` would
//! produce `Inf`; we instead leave such a `d_i` at its previous value
//! (effectively treating that row as already balanced).
//!
//! # Differences from upstream Ipopt
//!
//! Upstream Ipopt's only symmetric scaler is `IpMc19TSymScalingMethod`
//! (Curtis-Reid via HSL `mc19ad`). Ruiz lives in pounce because it is
//! simpler, has no Fortran dependency, and converges on the same
//! `O(log)` iterations on the matrix classes the IPM produces. The two
//! options coexist under `linear_system_scaling` (`mc19` vs `ruiz`).
//!
//! # References
//!
//! * Ruiz, D. CERFACS TR/PA/01/14.
//!   <https://cerfacs.fr/wp-content/uploads/2017/06/14_DanielRuiz.pdf>
//! * pounce issue #61.

use crate::scaling::TSymScalingMethod;
use pounce_common::types::{Index, Number};

/// `linear_system_scaling=ruiz` — iterative symmetric ∞-norm
/// equilibration.
#[derive(Debug, Clone, Copy)]
pub struct RuizTSymScalingMethod {
    /// Iteration cap. Defaults to 10 (issue #61's `scaling_max_iter`).
    /// Ruiz is geometrically convergent on the imbalance, so 10
    /// iterations is comfortable headroom for any practically
    /// reasonable matrix; the early-exit on `tol` usually fires
    /// well before this.
    pub max_iter: usize,
    /// Stop early when `max_i |r_i - 1| < tol`.
    pub tol: Number,
    /// Treat row ∞-norms below this as "already zero" — leaves the
    /// corresponding `d_i` unchanged for that iteration instead of
    /// dividing by `sqrt(≈0)`.
    pub min_row_value: Number,
}

impl Default for RuizTSymScalingMethod {
    fn default() -> Self {
        Self {
            max_iter: 10,
            tol: 1e-2,
            min_row_value: Number::MIN_POSITIVE,
        }
    }
}

impl RuizTSymScalingMethod {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_tol(mut self, tol: Number) -> Self {
        self.tol = tol;
        self
    }
}

impl TSymScalingMethod for RuizTSymScalingMethod {
    fn compute_sym_t_scaling_factors(
        &mut self,
        n: Index,
        nnz: Index,
        airn: &[Index],
        ajcn: &[Index],
        a: &[Number],
        scaling_factors: &mut [Number],
    ) -> bool {
        let n_us = n as usize;
        let nnz_us = nnz as usize;
        debug_assert_eq!(scaling_factors.len(), n_us);
        debug_assert!(airn.len() >= nnz_us);
        debug_assert!(ajcn.len() >= nnz_us);
        debug_assert!(a.len() >= nnz_us);

        if n_us == 0 {
            return true;
        }

        // Detect Fortran-style 1-based triplets. Upstream's
        // `TSymLinearSolver` hands triplets through unchanged; both
        // index styles appear in the test bed. The base cannot always be
        // read from the indices alone, but for an n×n matrix two signals
        // are decisive: a 1-based triplet never references index 0, and a
        // 0-based triplet never references index n (its valid range is
        // [0, n-1]). The previous min-only rule (`min_idx >= 1 ⇒
        // 1-based`) misclassified a 0-based triplet whose row 0 is
        // structurally empty (so `min_idx >= 1`) as 1-based, shifting
        // every scaling factor onto the wrong row. Use *both* extremes,
        // and resolve the residual ambiguity (neither 0 nor n present)
        // toward 0-based when the indices already cover the last row
        // (`max_idx == n - 1`, the hallmark of a full 0-based n×n
        // system); otherwise keep the historical 1-based assumption of
        // the in-tree caller.
        let mut min_idx = airn[0];
        let mut max_idx = airn[0];
        for k in 0..nnz_us {
            min_idx = min_idx.min(airn[k]).min(ajcn[k]);
            max_idx = max_idx.max(airn[k]).max(ajcn[k]);
        }
        let offset: Index = if min_idx == 0 {
            0
        } else if max_idx >= n {
            1
        } else if max_idx == n - 1 {
            0
        } else {
            1
        };

        // Initialize d = 1.
        for s in scaling_factors.iter_mut() {
            *s = 1.0;
        }

        // Workspace: per-row maxima of |D K D|.
        let mut row_max = vec![0.0 as Number; n_us];

        for _iter in 0..self.max_iter {
            for v in row_max.iter_mut() {
                *v = 0.0;
            }
            // Walk the triplet. For each entry, contribute to both i and
            // j row-maxes (symmetric storage convention: upstream
            // triplets list each off-diagonal once, so the (j, i)
            // mirror contribution must be added by us).
            for k in 0..nnz_us {
                let i = (airn[k] - offset) as usize;
                let j = (ajcn[k] - offset) as usize;
                if i >= n_us || j >= n_us {
                    return false;
                }
                let v = a[k].abs() * scaling_factors[i] * scaling_factors[j];
                if v > row_max[i] {
                    row_max[i] = v;
                }
                if i != j && v > row_max[j] {
                    row_max[j] = v;
                }
            }

            // Check convergence: max imbalance.
            let mut max_imbalance: Number = 0.0;
            for &r in row_max.iter() {
                if r < self.min_row_value {
                    continue;
                }
                let imb = (r - 1.0).abs();
                if imb > max_imbalance {
                    max_imbalance = imb;
                }
            }
            if max_imbalance < self.tol {
                break;
            }

            // Update d: d_i /= sqrt(r_i) where r_i is non-trivial.
            for i in 0..n_us {
                let r = row_max[i];
                if r >= self.min_row_value {
                    scaling_factors[i] /= r.sqrt();
                }
            }
        }

        // Guard: any NaN / non-finite means the matrix had pathological
        // structure (e.g. row of pure NaNs). Signal failure so the
        // caller can fall back to identity scaling (mirrors MC19's
        // `factors > 1e40` defensive guard).
        for s in scaling_factors.iter() {
            if !s.is_finite() {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the symmetric triplet for `diag([1e6, 1e-6])`. After
    /// equilibration the scaled diagonal should be approximately
    /// `[1, 1]`.
    #[test]
    fn equilibrates_diagonal_extremes() {
        let mut method = RuizTSymScalingMethod::new();
        let irn = [0, 1];
        let jcn = [0, 1];
        let vals = [1.0e6, 1.0e-6];
        let mut s = vec![0.0; 2];
        assert!(method.compute_sym_t_scaling_factors(2, 2, &irn, &jcn, &vals, &mut s));

        // s_i = 1 / sqrt(K_ii) → K_ii scaled to 1.
        let scaled_00 = vals[0] * s[0] * s[0];
        let scaled_11 = vals[1] * s[1] * s[1];
        assert!(
            (scaled_00 - 1.0).abs() < 1e-3,
            "diag(0)→{}, want ≈1; s={:?}",
            scaled_00,
            s
        );
        assert!(
            (scaled_11 - 1.0).abs() < 1e-3,
            "diag(1)→{}, want ≈1; s={:?}",
            scaled_11,
            s
        );
    }

    /// Symmetric off-diagonal: `K = [[100, 10], [10, 1]]`. After
    /// equilibration max(row/col ∞-norm) - 1 < tol.
    #[test]
    fn equilibrates_off_diagonal_block() {
        let mut method = RuizTSymScalingMethod::new();
        // Triplet stores upper triangle (or lower, equivalently for
        // symmetric storage). Use entries (0,0)=100, (1,0)=10,
        // (1,1)=1 — i.e. one off-diagonal listed once.
        let irn = [0, 1, 1];
        let jcn = [0, 0, 1];
        let vals = [100.0, 10.0, 1.0];
        let mut s = vec![0.0; 2];
        assert!(method.compute_sym_t_scaling_factors(2, 3, &irn, &jcn, &vals, &mut s));

        // Compute scaled row ∞-norms.
        let row0 = (vals[0] * s[0] * s[0])
            .abs()
            .max((vals[1] * s[1] * s[0]).abs());
        let row1 = (vals[1] * s[1] * s[0])
            .abs()
            .max((vals[2] * s[1] * s[1]).abs());
        assert!(
            (row0 - 1.0).abs() < method.tol + 1e-9,
            "row0={}, want ≈1",
            row0
        );
        assert!(
            (row1 - 1.0).abs() < method.tol + 1e-9,
            "row1={}, want ≈1",
            row1
        );
    }

    /// Zero row → corresponding scale stays at 1 (no NaN propagation).
    #[test]
    fn zero_row_keeps_unit_scale() {
        let mut method = RuizTSymScalingMethod::new();
        // K = diag(4, 0).
        let irn = [0];
        let jcn = [0];
        let vals = [4.0];
        let mut s = vec![0.0; 2];
        assert!(method.compute_sym_t_scaling_factors(2, 1, &irn, &jcn, &vals, &mut s));
        assert!((s[0] - 0.5).abs() < 1e-6, "K_00=4 → s_0≈0.5, got {}", s[0]);
        assert!(
            (s[1] - 1.0).abs() < 1e-12,
            "zero row keeps s=1, got {}",
            s[1]
        );
    }

    /// Fortran 1-based triplets must be handled the same as 0-based.
    #[test]
    fn fortran_index_style() {
        let mut method = RuizTSymScalingMethod::new();
        let irn = [1, 2];
        let jcn = [1, 2];
        let vals = [1.0e6, 1.0e-6];
        let mut s = vec![0.0; 2];
        assert!(method.compute_sym_t_scaling_factors(2, 2, &irn, &jcn, &vals, &mut s));
        let scaled_00 = vals[0] * s[0] * s[0];
        let scaled_11 = vals[1] * s[1] * s[1];
        assert!((scaled_00 - 1.0).abs() < 1e-3);
        assert!((scaled_11 - 1.0).abs() < 1e-3);
    }

    /// A 0-based triplet whose row 0 is structurally empty has
    /// `min_idx >= 1`, so the old min-only base detection misread it as
    /// Fortran 1-based and shifted every factor down one row (issue L8).
    /// Here `K = diag([0, 4, 9])` (0-based; row 0 empty): the factors
    /// must land on rows 1 and 2, and the empty row 0 must keep `d=1`.
    #[test]
    fn zero_based_with_empty_first_row_is_not_misread_as_fortran() {
        let mut method = RuizTSymScalingMethod::new();
        // 0-based indices: entries only on rows/cols 1 and 2. min_idx==1,
        // max_idx==2==n-1 ⇒ must be detected as 0-based (offset 0).
        let irn = [1, 2];
        let jcn = [1, 2];
        let vals = [4.0, 9.0];
        let mut s = vec![0.0; 3];
        assert!(method.compute_sym_t_scaling_factors(3, 2, &irn, &jcn, &vals, &mut s));

        // Empty row 0 → untouched unit scale. Pre-fix this slot received
        // the factor meant for row 1 (≈0.5) because offset was 1.
        assert!(
            (s[0] - 1.0).abs() < 1e-12,
            "empty row 0 must keep d=1, got {} (factor leaked from a \
             misdetected 1-based offset)",
            s[0]
        );
        // The actual entries (rows 1,2) must be equilibrated to ≈1.
        let scaled_11 = vals[0] * s[1] * s[1];
        let scaled_22 = vals[1] * s[2] * s[2];
        assert!(
            (scaled_11 - 1.0).abs() < 1e-3,
            "K_11=4 → scaled {}, want ≈1; s={:?}",
            scaled_11,
            s
        );
        assert!(
            (scaled_22 - 1.0).abs() < 1e-3,
            "K_22=9 → scaled {}, want ≈1; s={:?}",
            scaled_22,
            s
        );
    }

    /// Symmetric balance: after equilibration the max row/col ∞-norm
    /// ratio is within tol of 1 (issue #61's fuzz acceptance).
    #[test]
    fn fuzz_reduces_imbalance() {
        // Build a 5×5 symmetric matrix with entries spanning 1e-4..1e4.
        let n = 5usize;
        let mut irn: Vec<Index> = Vec::new();
        let mut jcn: Vec<Index> = Vec::new();
        let mut vals: Vec<Number> = Vec::new();
        // Deterministic "random" entries.
        let raw = [
            (0, 0, 1.0e4),
            (1, 0, 1.0e2),
            (1, 1, 1.0),
            (2, 0, 1.0e-2),
            (2, 2, 1.0e-4),
            (3, 1, 5.0),
            (3, 3, 50.0),
            (4, 2, 0.1),
            (4, 4, 25.0),
        ];
        for (i, j, v) in raw.iter() {
            irn.push(*i as Index);
            jcn.push(*j as Index);
            vals.push(*v);
        }
        let nnz = irn.len() as Index;

        // Unscaled row ∞-norms.
        let mut pre = vec![0.0 as Number; n];
        for k in 0..vals.len() {
            let i = irn[k] as usize;
            let j = jcn[k] as usize;
            let v = vals[k].abs();
            if v > pre[i] {
                pre[i] = v;
            }
            if i != j && v > pre[j] {
                pre[j] = v;
            }
        }
        let pre_max = pre.iter().cloned().fold(0.0_f64, f64::max);
        let pre_min = pre.iter().cloned().fold(f64::INFINITY, f64::min);

        // Scaled row ∞-norms.
        let mut method = RuizTSymScalingMethod::new();
        let mut s = vec![0.0; n];
        assert!(method.compute_sym_t_scaling_factors(n as Index, nnz, &irn, &jcn, &vals, &mut s));
        let mut post = vec![0.0 as Number; n];
        for k in 0..vals.len() {
            let i = irn[k] as usize;
            let j = jcn[k] as usize;
            let v = (vals[k] * s[i] * s[j]).abs();
            if v > post[i] {
                post[i] = v;
            }
            if i != j && v > post[j] {
                post[j] = v;
            }
        }
        let post_max = post.iter().cloned().fold(0.0_f64, f64::max);
        let post_min = post.iter().cloned().fold(f64::INFINITY, f64::min);

        let pre_ratio = pre_max / pre_min;
        let post_ratio = post_max / post_min;
        assert!(
            post_ratio < pre_ratio,
            "Ruiz must reduce row-∞-norm ratio: pre={pre_ratio}, post={post_ratio}"
        );
        // Issue #61 fuzz acceptance: post_max/post_min within ε of 1.
        // Allow the configured tolerance plus a small slack.
        assert!(
            (post_ratio - 1.0).abs() < method.tol + 5e-2,
            "post ratio {} should be ≈1",
            post_ratio
        );
    }
}
