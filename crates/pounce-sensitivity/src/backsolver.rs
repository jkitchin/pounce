//! `SensBacksolver` trait — abstract backsolver against a converged KKT factor.
//!
//! Mirrors upstream
//! [`SensBacksolver.hpp`](../../../ref/Ipopt/contrib/sIPOPT/src/SensBacksolver.hpp).
//!
//! # What this is
//!
//! After pounce's IPM converges, the KKT factor `K` is stored on the
//! algorithm side (`pounce-algorithm::kkt::PdFullSpaceSolver`).
//! sIPOPT runs a small number of backsolves against that factor to
//! build the sensitivity matrix `P = K⁻¹ A` (see
//! [`crate::p_calculator::PCalculator`]). The trait surface is just
//! "solve `K · lhs = rhs`"; concrete impls plug in either the real
//! `AugSystemSolver` (Phase B.2) or a synthetic dense-LU backsolver
//! (Phase B.1, this file, for unit testing the sensitivity math).
//!
//! Upstream `SensSimpleBacksolver` is the analogous wrapper around
//! Ipopt's `AugSystemSolver`
//! ([`SensSimpleBacksolver.{hpp,cpp}`](../../../ref/Ipopt/contrib/sIPOPT/src/SensSimpleBacksolver.hpp)).

use pounce_common::types::Number;

/// Solve `K · lhs = rhs` against the converged KKT factor. Returns
/// `false` on failure (e.g. backend reports `Singular`).
///
/// Mirrors `Ipopt::SensBacksolver::Solve`
/// ([`SensBacksolver.hpp:28-31`](../../../ref/Ipopt/contrib/sIPOPT/src/SensBacksolver.hpp)).
/// Pounce takes flat `&[Number]` / `&mut [Number]` rather than
/// upstream's block-structured `IteratesVector` because the
/// sensitivity-side data is naturally flat; if the algorithm-side
/// wrapper in Phase B.2 needs the block layout it converts before
/// calling.
pub trait SensBacksolver {
    /// Size of the linear system in entries (length of `lhs` and
    /// `rhs`). The backsolver's notion of "full state" — pounce's
    /// IPM uses the compound `(x, s, λ_c, λ_d, z_l, z_u, v_l, v_u)`
    /// concatenation here.
    fn dim(&self) -> usize;

    /// Solve `K · lhs = rhs`. The implementation may use `rhs` as
    /// scratch; callers should treat it as moved-from on return.
    /// `lhs` must have length `self.dim()`.
    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool;
}

/// Synthetic dense-LU backsolver. Used in this crate's tests to
/// validate the sensitivity math against known-good linear-algebra
/// answers without standing up the full pounce IPM. Phase B.2 ships
/// a real `pounce-algorithm`-backed implementation; this stays for
/// regression tests and as a reference for the trait contract.
///
/// Stores a row-major `n × n` matrix and reuses an in-place
/// Gaussian-elimination LU factor with partial pivoting. Numerical
/// stability is fine for the small problem sizes the unit tests
/// exercise (n ≤ 16).
#[derive(Debug, Clone)]
pub struct DenseLuBacksolver {
    n: usize,
    /// Factored `K` in row-major order: contains `L` (unit lower)
    /// and `U` (upper) packed in-place per Doolittle.
    lu: Vec<Number>,
    /// Row-permutation order: row `piv[i]` of the original `K`
    /// ended up in row `i` of the factor.
    piv: Vec<usize>,
}

impl DenseLuBacksolver {
    /// Build from a row-major `n × n` matrix. Returns `Err(())` if
    /// the matrix is exactly singular at LU time (zero pivot after
    /// pivoting).
    pub fn from_dense(n: usize, a_row_major: &[Number]) -> Result<Self, ()> {
        if a_row_major.len() != n * n {
            return Err(());
        }
        let mut lu = a_row_major.to_vec();
        let mut piv: Vec<usize> = (0..n).collect();
        for k in 0..n {
            // Partial pivot: find row with largest |lu[r,k]| for r >= k.
            let mut best = k;
            let mut best_mag = lu[k * n + k].abs();
            for r in (k + 1)..n {
                let mag = lu[r * n + k].abs();
                if mag > best_mag {
                    best = r;
                    best_mag = mag;
                }
            }
            if best_mag == 0.0 {
                return Err(());
            }
            if best != k {
                piv.swap(k, best);
                for j in 0..n {
                    let tmp = lu[k * n + j];
                    lu[k * n + j] = lu[best * n + j];
                    lu[best * n + j] = tmp;
                }
            }
            // Eliminate below pivot.
            let inv_p = 1.0 / lu[k * n + k];
            for r in (k + 1)..n {
                let m = lu[r * n + k] * inv_p;
                lu[r * n + k] = m;
                for j in (k + 1)..n {
                    let upd = lu[k * n + j];
                    lu[r * n + j] -= m * upd;
                }
            }
        }
        Ok(Self { n, lu, piv })
    }
}

impl SensBacksolver for DenseLuBacksolver {
    fn dim(&self) -> usize {
        self.n
    }

    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        if rhs.len() != self.n || lhs.len() != self.n {
            return false;
        }
        // Apply row permutation first.
        for i in 0..self.n {
            lhs[i] = rhs[self.piv[i]];
        }
        // Forward solve: L · y = P · rhs (L unit lower in `lu`).
        for i in 0..self.n {
            let mut s = lhs[i];
            for j in 0..i {
                s -= self.lu[i * self.n + j] * lhs[j];
            }
            lhs[i] = s;
        }
        // Back solve: U · x = y.
        for i in (0..self.n).rev() {
            let mut s = lhs[i];
            for j in (i + 1)..self.n {
                s -= self.lu[i * self.n + j] * lhs[j];
            }
            let p = self.lu[i * self.n + i];
            if p == 0.0 {
                return false;
            }
            lhs[i] = s / p;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Solve the small symmetric system `A x = b`:
    ///   2 -1  0 | 1
    ///  -1  2 -1 | 0
    ///   0 -1  2 | 0
    /// Closed-form: x = (3/4, 1/2, 1/4).
    #[test]
    fn dense_lu_solves_3x3_symmetric() {
        #[rustfmt::skip]
        let a = vec![
             2.0, -1.0,  0.0,
            -1.0,  2.0, -1.0,
             0.0, -1.0,  2.0,
        ];
        let solver = DenseLuBacksolver::from_dense(3, &a).expect("factor");
        let b = [1.0, 0.0, 0.0];
        let mut x = [0.0; 3];
        assert!(solver.solve(&b, &mut x));
        assert!((x[0] - 0.75).abs() < 1e-12, "x[0] = {}", x[0]);
        assert!((x[1] - 0.50).abs() < 1e-12, "x[1] = {}", x[1]);
        assert!((x[2] - 0.25).abs() < 1e-12, "x[2] = {}", x[2]);
    }

    /// Pivoting check: a matrix whose first pivot is zero (must swap
    /// rows). Direct closed-form: A x = b for
    ///   0  1  | 2     →  x = (2, 1) after swap
    ///   1  0  | 1
    #[test]
    fn dense_lu_handles_zero_first_pivot() {
        let a = vec![0.0, 1.0, 1.0, 0.0];
        let solver = DenseLuBacksolver::from_dense(2, &a).expect("factor");
        let b = [2.0, 1.0];
        let mut x = [0.0; 2];
        assert!(solver.solve(&b, &mut x));
        assert!((x[0] - 1.0).abs() < 1e-12, "x[0] = {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-12, "x[1] = {}", x[1]);
    }

    #[test]
    fn dense_lu_rejects_singular_matrix() {
        // Rank-deficient: rows are linearly dependent.
        let a = vec![1.0, 2.0, 2.0, 4.0];
        assert!(DenseLuBacksolver::from_dense(2, &a).is_err());
    }

    #[test]
    fn solve_rejects_wrong_dim() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let s = DenseLuBacksolver::from_dense(2, &a).expect("ok");
        let b = [1.0];
        let mut x = [0.0; 2];
        assert!(!s.solve(&b, &mut x));
    }
}
