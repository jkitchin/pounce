//! Symmetric equilibration for the RSLAB adapter.
//!
//! `rslab::LdltSolver` equilibrates `A_hat = D A D` before it factors, but the
//! low-level entry points the adapter uses (`analyze_with` +
//! `factor_numeric`) do not, and RSLAB's `equilibrate_with` /
//! `scaling::compute_scaling` are private. Running unscaled would be a
//! silent divergence from RSLAB's own default and would put the adapter's
//! reported pivot magnitudes in a different space from feral's (whose
//! `min_pivot_magnitude` is documented as living in the scaled space of
//! `S·A·S`), so the one-pass step is reproduced here instead.
//!
//! This is the whole of what the adapter reimplements from RSLAB, and it is
//! nine lines of arithmetic. `tests/inertia.rs::the_adapters_equilibration_matches_rslabs`
//! pins it against `rslab::LdltSolver`'s own factorization.

use pounce_common::types::Number;
use rslab::CscMatrix;

/// Which symmetric equilibration to apply before factoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Equilibration {
    /// `s_i = 1 / sqrt(max_j |A_ij|)` over the symmetric row — one
    /// Knight-Ruiz step. Reproduces `rslab::ScalingStrategy::OnePassInfNorm`,
    /// which is what `rslab::SolverSettings::default()` selects and therefore
    /// what `rslab::LdltSolver` applies.
    #[default]
    OnePassInfNorm,
    /// No scaling (`s ≡ 1`). For regression work and for callers that scale
    /// the KKT themselves — POUNCE's own `TSymLinearSolver` can apply Ruiz
    /// equilibration one layer up (`linear_system_scaling = ruiz`), and
    /// stacking the two is measurable but not obviously desirable.
    Identity,
}

/// The equilibration diagonal `s` for `a` under `strategy`.
///
/// `a` holds the **lower triangle only**, so the sweep folds each
/// off-diagonal entry into both of its rows — the transpose entry is not
/// stored. A row whose largest magnitude is zero, non-finite, or would produce
/// a non-finite reciprocal square root holds `s_i = 1`; that guard is RSLAB's
/// `inv_sqrt_scale_guarded` and it is what lets the one-pass step tolerate a
/// structurally zero diagonal, which every saddle-point KKT has.
pub fn compute(strategy: Equilibration, a: &CscMatrix<f64>) -> Vec<Number> {
    match strategy {
        Equilibration::Identity => vec![1.0; a.n],
        Equilibration::OnePassInfNorm => {
            let mut row_max = vec![0.0_f64; a.n];
            for j in 0..a.n {
                for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                    let i = a.row_idx[k];
                    let m = a.values[k].abs();
                    if m > row_max[i] {
                        row_max[i] = m;
                    }
                    if i != j && m > row_max[j] {
                        row_max[j] = m;
                    }
                }
            }
            row_max.iter().map(|&r| inv_sqrt_guarded(r)).collect()
        }
    }
}

/// `1/sqrt(m)`, held at `1.0` whenever that is not a finite positive number.
#[inline]
fn inv_sqrt_guarded(m: f64) -> f64 {
    if m > 0.0 {
        let cand = 1.0 / m.sqrt();
        if cand.is_finite() && cand > 0.0 {
            return cand;
        }
    }
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri(n: usize, rows: &[usize], cols: &[usize], vals: &[f64]) -> CscMatrix<f64> {
        CscMatrix::<f64>::from_triplets(n, rows, cols, vals).unwrap()
    }

    /// The off-diagonal entry is stored once but bounds two rows.
    #[test]
    fn an_offdiagonal_entry_scales_both_of_its_rows() {
        // [[1, 100], [100, 4]] — row maxima are both 100.
        let a = tri(2, &[0, 1, 1], &[0, 0, 1], &[1.0, 100.0, 4.0]);
        let s = compute(Equilibration::OnePassInfNorm, &a);
        let want = 1.0 / 100.0_f64.sqrt();
        assert!((s[0] - want).abs() < 1e-15, "s0 = {}", s[0]);
        assert!((s[1] - want).abs() < 1e-15, "s1 = {}", s[1]);
    }

    /// A structurally zero diagonal — the shape of every saddle-point (2,2)
    /// block — must not produce a zero or non-finite scale.
    #[test]
    fn a_zero_diagonal_row_is_scaled_by_its_offdiagonals() {
        // [[0, 2], [2, 0]]
        let a = tri(2, &[0, 1, 1], &[0, 0, 1], &[0.0, 2.0, 0.0]);
        let s = compute(Equilibration::OnePassInfNorm, &a);
        let want = 1.0 / 2.0_f64.sqrt();
        assert!(s.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!((s[0] - want).abs() < 1e-15 && (s[1] - want).abs() < 1e-15);
    }

    /// A structurally empty row (no stored entry touches it at all) holds at
    /// 1.0 rather than dividing by zero.
    #[test]
    fn an_empty_row_holds_at_unit_scale() {
        let a = tri(3, &[0, 2], &[0, 2], &[4.0, 9.0]);
        let s = compute(Equilibration::OnePassInfNorm, &a);
        assert_eq!(s[1], 1.0);
        assert!((s[0] - 0.5).abs() < 1e-15);
        assert!((s[2] - 1.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn identity_is_all_ones() {
        let a = tri(3, &[0, 1, 2], &[0, 1, 2], &[4.0, 9.0, 1e30]);
        assert_eq!(compute(Equilibration::Identity, &a), vec![1.0; 3]);
    }

    /// A non-finite entry holds at unit scale instead of propagating.
    ///
    /// `1/sqrt(m)` cannot overflow for any finite positive `f64` — the
    /// smallest subnormal, `5e-324`, gives `4.5e161` — so the guard's reachable
    /// cases are `m = 0` (covered above), `m = +inf` (`1/inf = 0`, not
    /// positive) and `m = NaN` (fails `m > 0`). Both of the latter would
    /// otherwise put a zero or a NaN on the equilibration diagonal, which
    /// silently destroys the factorization rather than failing it.
    #[test]
    fn a_non_finite_row_max_holds_at_unit_scale() {
        for bad in [f64::INFINITY, f64::NAN] {
            let a = tri(2, &[0, 1], &[0, 1], &[bad, 4.0]);
            let s = compute(Equilibration::OnePassInfNorm, &a);
            assert_eq!(s[0], 1.0, "row max {bad} must hold at unit scale");
            assert!((s[1] - 0.5).abs() < 1e-15, "the healthy row is unaffected");
        }
    }

    /// The smallest subnormal is still a finite positive scale, so it is
    /// scaled rather than held — the guard is not a magnitude cutoff.
    #[test]
    fn the_smallest_subnormal_row_max_is_still_scaled() {
        let tiny = f64::MIN_POSITIVE * f64::EPSILON;
        let a = tri(1, &[0], &[0], &[tiny]);
        let s = compute(Equilibration::OnePassInfNorm, &a);
        assert!(s[0].is_finite() && s[0] > 1e160, "s0 = {}", s[0]);
    }
}
