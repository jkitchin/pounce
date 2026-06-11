//! `PCalculator` trait surface.
//!
//! Mirrors upstream
//! [`SensPCalculator.hpp`](../../../ref/Ipopt/contrib/sIPOPT/src/SensPCalculator.hpp)
//! (lines 17–133). Pounce ships only the trait + a wiring shim in
//! Phase A; the concrete `IndexPCalculator` numerical driver lands in
//! Phase B once the sensitivity-side backsolver is plumbed in.
//!
//! # What `PCalculator` computes
//!
//! Given the augmented system
//!
//! ```text
//! ⎡ K   A ⎤
//! ⎣ B   0 ⎦
//! ```
//!
//! with `K` the converged KKT matrix, `A`/`B` parameter-row data
//! supplied via [`crate::SchurData`], a `PCalculator` is responsible
//! for computing the dense `P = K⁻¹ A` (call `compute_p`) and the
//! Schur-complement matrix `S = B K⁻¹ A` (call `schur_matrix`).
//!
//! Reference: Pirnay, López-Negrete & Biegler 2012, §3 (DOI:
//! [10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2)).

use crate::backsolver::SensBacksolver;
use crate::schur_data::{IndexSchurData, SchurData};
use pounce_common::types::{Index, Number};
use std::collections::HashMap;

/// Algorithmic-strategy surface for computing the sensitivity matrix
/// `P = K⁻¹ A` and the Schur complement `S = B K⁻¹ A`.
///
/// Mirror of `Ipopt::PCalculator`
/// ([`SensPCalculator.hpp:26-133`](../../../ref/Ipopt/contrib/sIPOPT/src/SensPCalculator.hpp)).
///
/// # Why a trait rather than a struct
///
/// Upstream takes the same factoring choice
/// ([`SensPCalculator.hpp:50-60`](../../../ref/Ipopt/contrib/sIPOPT/src/SensPCalculator.hpp))
/// — `PCalculator` is abstract so the index-only flavor
/// (`IndexPCalculator`) and the dense-Schur-driver flavor
/// (`DenseGenSchurDriver` via the higher-level `SchurDriver`)
/// can share a consumer interface. Pounce's Phase-A surface omits
/// the journalist printing and the `AlgorithmStrategyObject`
/// inheritance — both are convenience APIs we'll add when the
/// numerical drivers land in Phase B.
pub trait PCalculator {
    /// The `A`-side schur data this calculator owns. Multiple inner
    /// drivers may share the same `A` matrix, so it's exposed by
    /// reference. Upstream `data_A()`
    /// ([`SensPCalculator.hpp:112-115`](../../../ref/Ipopt/contrib/sIPOPT/src/SensPCalculator.hpp)).
    fn data_a(&self) -> &dyn SchurData;

    /// Compute `P = K⁻¹ A` and stash it inside the implementation.
    /// Returns `false` on backsolver failure (mirrors upstream's
    /// `bool ComputeP()` —
    /// [`SensPCalculator.hpp:51`](../../../ref/Ipopt/contrib/sIPOPT/src/SensPCalculator.hpp)).
    ///
    /// Phase A: default returns `false` (no backsolver wired). Phase B
    /// overrides this with the real implementation.
    fn compute_p(&mut self) -> bool {
        false
    }

    /// Fill `dense_schur` (size `b.nrows() × a_nrows`, row-major) with
    /// `S = B · P` where `P = K⁻¹ A` was computed by `compute_p`.
    /// Upstream `GetSchurMatrix(B, S)`
    /// ([`SensPCalculator.hpp:57-60`](../../../ref/Ipopt/contrib/sIPOPT/src/SensPCalculator.hpp)).
    ///
    /// Phase A: default returns `false`. Phase B implements the real
    /// row-by-row construction over `B`'s `multiplying_row` surface.
    fn schur_matrix(&mut self, _b: &dyn SchurData, _dense_schur: &mut [Number]) -> bool {
        false
    }
}

/// Concrete `PCalculator` for ±1-flagged parameter matrices.
///
/// Stores `P = K⁻¹ A` as a hash map keyed by column index. Builds
/// each column lazily on first access via `compute_p`, then reuses
/// it for any `schur_matrix(B, …)` call. Mirrors upstream
/// [`IndexPCalculator`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexPCalculator.cpp)
/// (lines 20–173).
///
/// # Schur sign convention
///
/// Upstream writes `S[i, j] = -P[B_col_idx[i], A_col_idx[j]]`
/// ([`SensIndexPCalculator.cpp:225`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexPCalculator.cpp)).
/// The leading minus follows from sIPOPT's augmented-system
/// reduction sign convention (the bottom-right block of the
/// augmented system is `0`, but the reduction adds `−S` to the
/// trailing block). Pounce keeps the same sign so downstream
/// `SchurDriver` consumers don't have to track which convention
/// they're under.
pub struct IndexPCalculator<B: SensBacksolver> {
    backsolver: B,
    data_a: IndexSchurData,
    n_full: usize,
    /// Column of `P = K⁻¹ A` keyed by the corresponding column index
    /// in `A_data`. Built lazily by `compute_p`.
    p_cols: HashMap<Index, Vec<Number>>,
}

impl<B: SensBacksolver> IndexPCalculator<B> {
    /// Build an `IndexPCalculator` from a converged backsolver and
    /// the `A` parameter-row data. `n_full` is the backsolver's KKT
    /// dimension; it must match what `data_a`'s rows reference.
    pub fn new(backsolver: B, data_a: IndexSchurData) -> Self {
        let n_full = backsolver.dim();
        Self {
            backsolver,
            data_a,
            n_full,
            p_cols: HashMap::new(),
        }
    }

    /// Number of full-state entries in the backsolver — `n_full` per
    /// upstream's `nrows_` field
    /// ([`SensIndexPCalculator.hpp`](../../../ref/Ipopt/contrib/sIPOPT/src/SensIndexPCalculator.hpp)).
    pub fn n_full(&self) -> usize {
        self.n_full
    }

    /// Read-only access to the cached `P = K⁻¹ A` columns. Used by
    /// the test suite + Phase-B.2 SchurDriver to verify rows.
    pub fn p_columns(&self) -> &HashMap<Index, Vec<Number>> {
        &self.p_cols
    }

    /// Borrow the underlying backsolver. Exposed so the
    /// [`crate::WithBacksolver`] bridge can hand it to
    /// [`crate::StdStepCalc`] for the final `K⁻¹` step.
    pub fn backsolver(&self) -> &B {
        &self.backsolver
    }
}

impl<B: SensBacksolver> PCalculator for IndexPCalculator<B> {
    fn data_a(&self) -> &dyn SchurData {
        &self.data_a
    }

    fn compute_p(&mut self) -> bool {
        // For each (col, sign) pair in A, backsolve K · p_col = sign · e_col,
        // store the result in p_cols keyed by the column index.
        let cols = self.data_a.col_indices().to_vec();
        let signs = self.data_a.signs().to_vec();
        for (i, &col) in cols.iter().enumerate() {
            if self.p_cols.contains_key(&col) {
                // Already cached — A_data may have duplicate column
                // indices across runs, no work needed.
                continue;
            }
            let mut rhs = vec![0.0; self.n_full];
            let c_us = col as usize;
            if c_us >= self.n_full {
                return false;
            }
            rhs[c_us] = signs[i] as Number;
            let mut p_col = vec![0.0; self.n_full];
            if !self.backsolver.solve(&rhs, &mut p_col) {
                return false;
            }
            self.p_cols.insert(col, p_col);
        }
        true
    }

    fn schur_matrix(&mut self, b: &dyn SchurData, dense_schur: &mut [Number]) -> bool {
        let n_b = b.nrows() as usize;
        let n_a = self.data_a.nrows() as usize;
        if dense_schur.len() != n_b * n_a {
            return false;
        }
        // Need P populated for every A-column before reading from p_cols.
        if !self.compute_p() {
            return false;
        }
        let a_cols = self.data_a.col_indices().to_vec();
        // Column-major layout: S[i, j] = dense_schur[j * n_b + i].
        for (j, &a_col) in a_cols.iter().enumerate() {
            let p_col = match self.p_cols.get(&a_col) {
                Some(v) => v,
                None => return false,
            };
            // For each row `i` of B, pick the single non-zero column
            // index that row points to (Index_SchurData contract).
            for i in 0..n_b {
                let (b_idx_vec, _facs) = match b.multiplying_row(i as Index) {
                    Ok(t) => t,
                    Err(_) => return false,
                };
                let b_col = b_idx_vec[0] as usize;
                if b_col >= p_col.len() {
                    return false;
                }
                dense_schur[j * n_b + i] = -p_col[b_col];
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schur_data::IndexSchurData;

    /// Phase-A placeholder calculator — wraps an `IndexSchurData` for
    /// `A` and returns the trait defaults for `compute_p` / `schur_matrix`.
    /// Exists so the trait surface compiles + the default behavior is
    /// exercised. Replaced by the real numerical driver in Phase B.
    struct StubPCalculator {
        a: IndexSchurData,
    }

    impl PCalculator for StubPCalculator {
        fn data_a(&self) -> &dyn SchurData {
            &self.a
        }
    }

    #[test]
    fn trait_default_compute_p_returns_false() {
        let a = IndexSchurData::from_parts(vec![0, 2], vec![1, -1]).unwrap();
        let mut pc = StubPCalculator { a };
        assert!(
            !pc.compute_p(),
            "default compute_p must return false until Phase B"
        );
    }

    #[test]
    fn trait_default_schur_matrix_returns_false() {
        let a = IndexSchurData::from_parts(vec![0, 2], vec![1, -1]).unwrap();
        let b = IndexSchurData::from_parts(vec![1], vec![1]).unwrap();
        let mut pc = StubPCalculator { a };
        let mut out = vec![0.0; 1 * 2];
        assert!(!pc.schur_matrix(&b, &mut out));
    }

    #[test]
    fn data_a_round_trips_to_concrete_schur_data() {
        let a = IndexSchurData::from_parts(vec![3, 5], vec![1, 1]).unwrap();
        let pc = StubPCalculator { a };
        assert_eq!(pc.data_a().nrows(), 2);
    }

    // ---- IndexPCalculator numeric tests (Phase B.1) ----
    //
    // Strategy: build a known-good `K` matrix, factor it via
    // `DenseLuBacksolver`, pick a small `A` selecting two columns,
    // run `compute_p` + `schur_matrix`, and compare against the
    // closed-form result `S = -B K⁻¹ A`.

    use crate::backsolver::DenseLuBacksolver;

    #[test]
    fn compute_p_solves_each_a_column_against_k_matrix() {
        // K is the 3×3 SPD example from the backsolver test.
        //   2 -1  0
        //  -1  2 -1
        //   0 -1  2
        // Inverse (verified by hand):
        //   1/4 * [[3 2 1], [2 4 2], [1 2 3]] ... let me just check against
        //   the unit columns through the backsolver.
        #[rustfmt::skip]
        let k = vec![
             2.0, -1.0,  0.0,
            -1.0,  2.0, -1.0,
             0.0, -1.0,  2.0,
        ];
        let backsolver = DenseLuBacksolver::from_dense(3, &k).expect("factor");
        // A picks columns 0 and 2 with +1 signs.
        let a = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).unwrap();
        let mut pc = IndexPCalculator::new(backsolver, a);
        assert!(pc.compute_p());

        // K⁻¹ e_0 = (3/4, 1/2, 1/4)   (from the prior solver test).
        let p0 = pc.p_columns().get(&0).expect("col 0 cached");
        assert!((p0[0] - 0.75).abs() < 1e-12);
        assert!((p0[1] - 0.50).abs() < 1e-12);
        assert!((p0[2] - 0.25).abs() < 1e-12);

        // K⁻¹ e_2 by symmetry = (1/4, 1/2, 3/4).
        let p2 = pc.p_columns().get(&2).expect("col 2 cached");
        assert!((p2[0] - 0.25).abs() < 1e-12);
        assert!((p2[1] - 0.50).abs() < 1e-12);
        assert!((p2[2] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn compute_p_uses_sign_from_a_data() {
        // Pick column 1 with sign -1. K⁻¹ (-e_1) = -(K⁻¹ e_1)
        // = -(1/2, 1, 1/2) = (-1/2, -1, -1/2).
        #[rustfmt::skip]
        let k = vec![
             2.0, -1.0,  0.0,
            -1.0,  2.0, -1.0,
             0.0, -1.0,  2.0,
        ];
        let backsolver = DenseLuBacksolver::from_dense(3, &k).unwrap();
        let a = IndexSchurData::from_parts(vec![1], vec![-1]).unwrap();
        let mut pc = IndexPCalculator::new(backsolver, a);
        assert!(pc.compute_p());
        let p1 = pc.p_columns().get(&1).expect("col 1 cached");
        assert!((p1[0] - (-0.5)).abs() < 1e-12);
        assert!((p1[1] - (-1.0)).abs() < 1e-12);
        assert!((p1[2] - (-0.5)).abs() < 1e-12);
    }

    /// End-to-end Schur complement check: `S = -B · (K⁻¹ A)`.
    ///
    /// Same K as above. A selects columns {0, 2} with +1.
    /// B selects column {1} with +1.
    /// K⁻¹ has been verified analytically as
    ///   1/4 * [[3 2 1], [2 4 2], [1 2 3]].
    /// A is e_0 || e_2 (3×2), so
    ///   K⁻¹ A = column 0 of K⁻¹  ||  column 2 of K⁻¹
    ///        = [[3/4, 1/4], [1/2, 1/2], [1/4, 3/4]].
    /// B = [[0, 1, 0]], so
    ///   B K⁻¹ A = [[1/2, 1/2]]
    /// and the upstream-convention output is
    ///   S = -B K⁻¹ A = [[-1/2, -1/2]].
    #[test]
    fn schur_matrix_matches_closed_form_minus_b_kinv_a() {
        #[rustfmt::skip]
        let k = vec![
             2.0, -1.0,  0.0,
            -1.0,  2.0, -1.0,
             0.0, -1.0,  2.0,
        ];
        let backsolver = DenseLuBacksolver::from_dense(3, &k).unwrap();
        let a = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).unwrap();
        let b = IndexSchurData::from_parts(vec![1], vec![1]).unwrap();
        let mut pc = IndexPCalculator::new(backsolver, a);
        // Column-major: 1 row × 2 cols → length-2 buffer.
        let mut s = vec![0.0; 1 * 2];
        assert!(pc.schur_matrix(&b, &mut s));
        // S[0, 0] should be -1/2; S[0, 1] should be -1/2.
        assert!((s[0] - (-0.5)).abs() < 1e-12, "S[0,0] = {}", s[0]);
        assert!((s[1] - (-0.5)).abs() < 1e-12, "S[0,1] = {}", s[1]);
    }

    /// Cross-check via independent matrix arithmetic — build `K⁻¹ A`
    /// as a 3×2 dense matrix, multiply by `B` column-by-column, and
    /// confirm the calculator agrees with that reference.
    #[test]
    fn schur_matrix_reproduces_independent_computation() {
        #[rustfmt::skip]
        let k = vec![
             4.0,  1.0,  0.0,  0.0,
             1.0,  4.0,  1.0,  0.0,
             0.0,  1.0,  4.0,  1.0,
             0.0,  0.0,  1.0,  4.0,
        ];
        let backsolver = DenseLuBacksolver::from_dense(4, &k).unwrap();

        // A picks columns {1, 3} with signs {+1, -1}: A is 4×2 with
        // A[:,0] = e_1 and A[:,1] = -e_3.
        let a_data = IndexSchurData::from_parts(vec![1, 3], vec![1, -1]).unwrap();
        let mut pc = IndexPCalculator::new(backsolver, a_data);
        assert!(pc.compute_p());

        // Independently compute K⁻¹ A by solving twice.
        let bs2 = DenseLuBacksolver::from_dense(4, &k).unwrap();
        let mut kinv_e1 = vec![0.0; 4];
        bs2.solve(&[0.0, 1.0, 0.0, 0.0], &mut kinv_e1);
        let mut kinv_minus_e3 = vec![0.0; 4];
        bs2.solve(&[0.0, 0.0, 0.0, -1.0], &mut kinv_minus_e3);
        // Construct dense K⁻¹ A in row-major (4 × 2).
        let mut kinv_a = vec![0.0; 4 * 2];
        for r in 0..4 {
            kinv_a[r * 2 + 0] = kinv_e1[r];
            kinv_a[r * 2 + 1] = kinv_minus_e3[r];
        }

        // B selects rows {0, 2} with signs {+1, +1}: B is 2×4 with
        // B[0,:] = e_0^T and B[1,:] = e_2^T. Then B (K⁻¹ A) is 2 × 2.
        let b_data = IndexSchurData::from_parts(vec![0, 2], vec![1, 1]).unwrap();
        let mut s_actual = vec![0.0; 2 * 2];
        assert!(pc.schur_matrix(&b_data, &mut s_actual));

        // Reference: B (K⁻¹ A)[i, j] = (K⁻¹ A)[idx_B[i], j], then negate.
        let mut s_expected = vec![0.0; 2 * 2];
        let b_idx = [0usize, 2];
        for (i, &row) in b_idx.iter().enumerate() {
            for j in 0..2 {
                // Column-major output: [j * n_b + i]
                s_expected[j * 2 + i] = -kinv_a[row * 2 + j];
            }
        }
        // Compare entrywise.
        for k in 0..4 {
            assert!(
                (s_actual[k] - s_expected[k]).abs() < 1e-10,
                "S[{}] actual={}, expected={}",
                k,
                s_actual[k],
                s_expected[k],
            );
        }
    }
}
