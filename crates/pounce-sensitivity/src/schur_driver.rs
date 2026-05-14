//! `SchurDriver` trait + `DenseGenSchurDriver` implementation.
//!
//! The Schur driver sits on top of the [`PCalculator`](crate::PCalculator)
//! and provides the **second** linear-algebra step in sIPOPT's sensitivity
//! update: factor the dense Schur complement `S = -B K⁻¹ A` once, then
//! apply `S⁻¹` to RHS vectors during the per-perturbation step calc.
//!
//! Mirrors upstream
//! [`SensSchurDriver.hpp`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSchurDriver.hpp)
//! (abstract trait) and
//! [`SensDenseGenSchurDriver.{hpp,cpp}`](../../../ref/Ipopt/contrib/sIPOPT/src/SensDenseGenSchurDriver.cpp)
//! (general-dense LU concrete implementation).
//!
//! # Pipeline
//!
//! ```text
//! [PCalculator] ──→ S = -B K⁻¹ A  ──→ [SchurDriver] ──→ S⁻¹ · rhs
//! ```
//!
//! Phase B.1 ships the trait + a dense-LU implementation reusing the
//! same `DenseLuBacksolver` machinery as `crate::backsolver`. The
//! sparse / parallel Schur-driver flavors lacking in upstream stay
//! out of scope.

use crate::backsolver::{DenseLuBacksolver, SensBacksolver};
use crate::p_calculator::PCalculator;
use pounce_common::types::Number;

/// Factor the Schur complement once, then apply `S⁻¹` to RHS vectors.
///
/// Mirrors `Ipopt::SchurDriver`
/// ([`SensSchurDriver.hpp:17-118`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSchurDriver.hpp)).
///
/// # Lifecycle
///
/// 1. Construct with a `PCalculator` that owns the parameter rows
///    (`A` and the converged backsolver).
/// 2. Call [`Self::schur_build_and_factor`] — populates the dense
///    Schur matrix internally and factors it.
/// 3. Call [`Self::schur_solve`] one or more times with different
///    RHS vectors.
pub trait SchurDriver {
    /// Build the Schur complement and factor it. Returns `false` on
    /// any backend failure (PCalculator solve, factor singular, …).
    fn schur_build_and_factor(&mut self, b: &dyn crate::SchurData) -> bool;

    /// Solve `S x = rhs` against the factored Schur complement.
    /// `rhs` and `x` are both of length `n_b` (the row count of the
    /// `B` matrix used at build time). Returns `false` if the driver
    /// hasn't been factored yet, or on a downstream solve failure.
    fn schur_solve(&self, rhs: &[Number], x: &mut [Number]) -> bool;

    /// Dimension of the factored Schur complement. Returns `None`
    /// when the driver hasn't been built / factored yet.
    fn schur_dim(&self) -> Option<usize>;
}

/// Dense general (non-symmetric) Schur driver.
///
/// LU-factors the `n_b × n_b` Schur complement `S = -B K⁻¹ A` via
/// pivoted Gaussian elimination, then services `schur_solve` calls
/// against the factor. Mirrors upstream
/// `DenseGenSchurDriver`'s flow at
/// [`SensDenseGenSchurDriver.cpp:23-106`](../../../ref/Ipopt/contrib/sIPOPT/src/SensDenseGenSchurDriver.cpp).
///
/// Pounce reuses `DenseLuBacksolver` as the LU primitive — the small
/// `n_b` (typically the parameter count, often < 100) makes a dense
/// LU the natural choice and avoids pulling in another linear-algebra
/// dependency.
pub struct DenseGenSchurDriver<P: PCalculator, B: SensBacksolver> {
    pcalc: P,
    /// Factored `S`, lazily populated by `schur_build_and_factor`.
    schur_lu: Option<DenseLuBacksolver>,
    /// Marker so the trait `SensBacksolver` type parameter is
    /// captured at construction time — pcalc.compute_p() needs the
    /// concrete backsolver type internally.
    _b: std::marker::PhantomData<B>,
}

impl<P: PCalculator, B: SensBacksolver> DenseGenSchurDriver<P, B> {
    /// Wrap a `PCalculator` and produce an un-built driver. Call
    /// [`Self::schur_build_and_factor`] before `schur_solve`.
    pub fn new(pcalc: P) -> Self {
        Self {
            pcalc,
            schur_lu: None,
            _b: std::marker::PhantomData,
        }
    }

    /// Borrow the underlying P-calculator (for inspection in tests
    /// and for chaining additional sensitivity computations that
    /// reuse the same converged `K`).
    pub fn pcalc(&self) -> &P {
        &self.pcalc
    }
}

impl<P: PCalculator, B: SensBacksolver> SchurDriver for DenseGenSchurDriver<P, B> {
    fn schur_build_and_factor(&mut self, b: &dyn crate::SchurData) -> bool {
        let n_b = b.nrows() as usize;
        let n_a = self.pcalc.data_a().nrows() as usize;
        if n_b != n_a {
            // DenseGen wants a square Schur complement. The
            // rectangular case (B ≠ A in dimensions) corresponds to
            // upstream's non-square Schur which always produces a
            // square output by construction (`B` and `A` have the
            // same number of rows in the parametric / reduced-Hess
            // settings); reject the mismatch here rather than later.
            return false;
        }
        let mut s_col_major = vec![0.0; n_b * n_a];
        if !self.pcalc.schur_matrix(b, &mut s_col_major) {
            return false;
        }
        // `DenseLuBacksolver::from_dense` expects row-major; transpose
        // S in place to feed it.
        let mut s_row_major = vec![0.0; n_b * n_a];
        for i in 0..n_b {
            for j in 0..n_a {
                s_row_major[i * n_a + j] = s_col_major[j * n_b + i];
            }
        }
        match DenseLuBacksolver::from_dense(n_b, &s_row_major) {
            Ok(lu) => {
                self.schur_lu = Some(lu);
                true
            }
            Err(_) => false,
        }
    }

    fn schur_solve(&self, rhs: &[Number], x: &mut [Number]) -> bool {
        match self.schur_lu.as_ref() {
            Some(lu) => lu.solve(rhs, x),
            None => false,
        }
    }

    fn schur_dim(&self) -> Option<usize> {
        self.schur_lu.as_ref().map(|lu| lu.dim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backsolver::DenseLuBacksolver;
    use crate::p_calculator::IndexPCalculator;
    use crate::schur_data::IndexSchurData;

    /// End-to-end Schur factor + solve.
    ///
    /// Same `K` (3×3 SPD tridiag) and `A` (e_0 || e_2) as the
    /// `p_calculator` tests. With `B = A`, `S = -B K⁻¹ A` is
    /// 2×2 with closed-form values from `K⁻¹` columns:
    ///   S = -[[3/4, 1/4], [1/4, 3/4]].
    /// Determinant is -(9/16 - 1/16) = -1/2.
    /// `S x = (1, 0)ᵀ` has solution `x = -2 · (3/4, -1/4)ᵀ = (-3/2, 1/2)ᵀ`
    /// (verified via `Sᵀ` × x check).
    #[test]
    fn dense_gen_schur_driver_factors_and_solves() {
        #[rustfmt::skip]
        let k = vec![
             2.0, -1.0,  0.0,
            -1.0,  2.0, -1.0,
             0.0, -1.0,  2.0,
        ];
        let backsolver = DenseLuBacksolver::from_dense(3, &k).unwrap();
        let a = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).unwrap();
        let pc = IndexPCalculator::new(backsolver, a);
        let mut driver = DenseGenSchurDriver::<_, DenseLuBacksolver>::new(pc);

        let b = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).unwrap();
        assert!(driver.schur_build_and_factor(&b));
        assert_eq!(driver.schur_dim(), Some(2));

        // Solve S · x = (1, 0). Closed-form x = (-3/2, 1/2).
        let rhs = [1.0, 0.0];
        let mut x = [0.0; 2];
        assert!(driver.schur_solve(&rhs, &mut x));
        assert!((x[0] - (-1.5)).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((x[1] - 0.5).abs() < 1e-10, "x[1] = {}", x[1]);

        // Roundtrip: S · x should equal rhs to 1e-10. Reconstruct S
        // from p_columns to verify.
        // S[0,0] = -K⁻¹[0,0] = -3/4
        // S[0,1] = -K⁻¹[0,2] = -1/4
        // S[1,0] = -K⁻¹[2,0] = -1/4
        // S[1,1] = -K⁻¹[2,2] = -3/4
        let s00 = -0.75; let s01 = -0.25;
        let s10 = -0.25; let s11 = -0.75;
        let recon0 = s00 * x[0] + s01 * x[1];
        let recon1 = s10 * x[0] + s11 * x[1];
        assert!((recon0 - 1.0).abs() < 1e-10);
        assert!((recon1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn schur_solve_before_factor_fails() {
        #[rustfmt::skip]
        let k = vec![ 1.0, 0.0, 0.0, 1.0 ];
        let backsolver = DenseLuBacksolver::from_dense(2, &k).unwrap();
        let a = IndexSchurData::from_parts(vec![0], vec![1]).unwrap();
        let pc = IndexPCalculator::new(backsolver, a);
        let driver = DenseGenSchurDriver::<_, DenseLuBacksolver>::new(pc);
        let rhs = [1.0];
        let mut x = [0.0; 1];
        assert!(!driver.schur_solve(&rhs, &mut x));
        assert_eq!(driver.schur_dim(), None);
    }

    #[test]
    fn schur_build_rejects_b_a_dim_mismatch() {
        #[rustfmt::skip]
        let k = vec![
             2.0, 0.0, 0.0,
             0.0, 3.0, 0.0,
             0.0, 0.0, 4.0,
        ];
        let backsolver = DenseLuBacksolver::from_dense(3, &k).unwrap();
        let a = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).unwrap();
        let pc = IndexPCalculator::new(backsolver, a);
        let mut driver = DenseGenSchurDriver::<_, DenseLuBacksolver>::new(pc);
        // B has only one row; A has two; DenseGen wants square.
        let b = IndexSchurData::from_parts(vec![1], vec![1]).unwrap();
        assert!(!driver.schur_build_and_factor(&b));
    }
}
