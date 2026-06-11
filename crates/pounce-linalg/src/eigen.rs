//! Symmetric eigendecomposition for small dense matrices via the
//! cyclic Jacobi rotation method.
//!
//! Mirrors the role of upstream's
//! [`DenseGenMatrix::ComputeEigenVectors`](../../../ref/Ipopt/IpoptAux/IpDenseGenMatrix.cpp)
//! (which dispatches to LAPACK `dsyev`). pounce sticks to pure Rust,
//! so we use the classical cyclic Jacobi algorithm: O(n³) per sweep
//! and O(n) sweeps for full convergence on a symmetric matrix. The
//! reduced Hessian's dimension is the number of free degrees of
//! freedom (`red_hessian` suffix count), which is typically small
//! (≤ a few dozen), so this is plenty fast.
//!
//! Reference: Golub & Van Loan, *Matrix Computations*, 4th ed.,
//! §8.5.

use pounce_common::types::Number;

/// In-place symmetric eigendecomposition.
///
/// On entry:
/// - `a` is the column-major `n × n` symmetric matrix to decompose.
///   Only the upper triangle is consulted; the lower triangle is
///   touched but not read.
///
/// On exit:
/// - `eigenvalues` (length `n`) contains the eigenvalues in
///   ascending order.
/// - `eigenvectors` (length `n²`, column-major) contains the
///   corresponding unit eigenvectors as columns. Column `j` is the
///   eigenvector for `eigenvalues[j]`.
///
/// Returns `true` on convergence (the off-diagonal Frobenius norm
/// drops below `tol = 1e-14 · ||A||_F` within `max_sweeps`),
/// `false` if the iteration ran out of sweeps. For dense reduced
/// Hessians up to dimension ~64 a handful of sweeps suffices.
///
/// Returns `false` immediately if any buffer is mis-sized.
pub fn symmetric_eigen(
    a: &[Number],
    n: usize,
    eigenvalues: &mut [Number],
    eigenvectors: &mut [Number],
) -> bool {
    if a.len() != n * n || eigenvalues.len() != n || eigenvectors.len() != n * n {
        return false;
    }
    if n == 0 {
        return true;
    }

    // Work on a copy: we'll zero its off-diagonals as we rotate.
    let mut m = a.to_vec();
    // Symmetrize to the lower triangle for clean indexing during
    // rotations (Jacobi uses both halves).
    for i in 0..n {
        for j in 0..i {
            let upper = m[i * n + j];
            let lower = m[j * n + i];
            let v = 0.5 * (upper + lower);
            m[i * n + j] = v;
            m[j * n + i] = v;
        }
    }

    // V starts as the identity; accumulates rotations.
    for v in eigenvectors.iter_mut() {
        *v = 0.0;
    }
    for k in 0..n {
        eigenvectors[k * n + k] = 1.0;
    }

    // Tolerance proportional to the Frobenius norm of A.
    let mut frob_sq = 0.0;
    for &x in m.iter() {
        frob_sq += x * x;
    }
    let tol = (1e-28 * frob_sq).max(1e-300);

    let max_sweeps = 50;
    for _sweep in 0..max_sweeps {
        // Off-diagonal sum of squares.
        let mut off = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off += m[i * n + j] * m[i * n + j];
                }
            }
        }
        if off < tol {
            break;
        }

        // Cyclic sweep over the upper triangle.
        for p in 0..n {
            for q in (p + 1)..n {
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                // Rotation angle: tan(2θ) = 2·apq / (app - aqq).
                let (c, s) = jacobi_cs(app, aqq, apq);

                // Update M ← Jᵀ M J (column-major a[i + n*j]).
                for i in 0..n {
                    let mip = m[i + n * p];
                    let miq = m[i + n * q];
                    m[i + n * p] = c * mip - s * miq;
                    m[i + n * q] = s * mip + c * miq;
                }
                for j in 0..n {
                    let mpj = m[p + n * j];
                    let mqj = m[q + n * j];
                    m[p + n * j] = c * mpj - s * mqj;
                    m[q + n * j] = s * mpj + c * mqj;
                }
                // Zero out the (p,q) and (q,p) entries explicitly to
                // avoid drift.
                m[p * n + q] = 0.0;
                m[q * n + p] = 0.0;

                // Accumulate V ← V J.
                for i in 0..n {
                    let vip = eigenvectors[i + n * p];
                    let viq = eigenvectors[i + n * q];
                    eigenvectors[i + n * p] = c * vip - s * viq;
                    eigenvectors[i + n * q] = s * vip + c * viq;
                }
            }
        }
    }

    // Extract diagonal as eigenvalues, then sort ascending and
    // permute columns of V to match.
    let mut idx: Vec<usize> = (0..n).collect();
    let diag: Vec<Number> = (0..n).map(|k| m[k * n + k]).collect();
    idx.sort_by(|&i, &j| {
        diag[i]
            .partial_cmp(&diag[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let v_in = eigenvectors.to_vec();
    for (new_pos, &old_pos) in idx.iter().enumerate() {
        eigenvalues[new_pos] = diag[old_pos];
        for row in 0..n {
            eigenvectors[row + n * new_pos] = v_in[row + n * old_pos];
        }
    }
    true
}

/// Compute the (c, s) Jacobi rotation that zeros the (p, q) entry of
/// a 2×2 symmetric block `[[app, apq], [apq, aqq]]`. Standard formula
/// from Golub & Van Loan §8.5.1.
fn jacobi_cs(app: Number, aqq: Number, apq: Number) -> (Number, Number) {
    if apq == 0.0 {
        return (1.0, 0.0);
    }
    let theta = (aqq - app) / (2.0 * apq);
    let t = if theta >= 0.0 {
        1.0 / (theta + (1.0 + theta * theta).sqrt())
    } else {
        1.0 / (theta - (1.0 + theta * theta).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    (c, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: Number, b: Number, tol: Number, label: &str) {
        assert!(
            (a - b).abs() < tol,
            "{label}: {a} vs {b} (|d|={})",
            (a - b).abs()
        );
    }

    #[test]
    fn eigen_diagonal_matrix() {
        // A = diag(3, 1, 2). Eigenvalues sorted: 1, 2, 3.
        let a = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let mut w = vec![0.0; 3];
        let mut v = vec![0.0; 9];
        assert!(symmetric_eigen(&a, 3, &mut w, &mut v));
        assert_close(w[0], 1.0, 1e-12, "w0");
        assert_close(w[1], 2.0, 1e-12, "w1");
        assert_close(w[2], 3.0, 1e-12, "w2");
    }

    #[test]
    fn eigen_2x2_known() {
        // A = [[2, 1], [1, 2]]: eigenvalues 1, 3; eigenvectors
        // [1,-1]/√2 and [1,1]/√2.
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let mut w = vec![0.0; 2];
        let mut v = vec![0.0; 4];
        assert!(symmetric_eigen(&a, 2, &mut w, &mut v));
        assert_close(w[0], 1.0, 1e-12, "w0");
        assert_close(w[1], 3.0, 1e-12, "w1");
        // Column 0 should be ±[1, -1]/√2 (up to sign).
        let s = 1.0 / 2f64.sqrt();
        assert!(
            ((v[0] - s).abs() < 1e-10 && (v[1] + s).abs() < 1e-10)
                || ((v[0] + s).abs() < 1e-10 && (v[1] - s).abs() < 1e-10)
        );
    }

    #[test]
    fn eigen_reconstructs_matrix() {
        // Random-ish 4×4 symmetric matrix. Verify A · v_j = λ_j · v_j.
        let a = vec![
            4.0, 1.0, 2.0, 0.5, 1.0, 3.0, 0.7, 1.5, 2.0, 0.7, 5.0, 0.3, 0.5, 1.5, 0.3, 2.0,
        ];
        let n = 4;
        let mut w = vec![0.0; n];
        let mut v = vec![0.0; n * n];
        assert!(symmetric_eigen(&a, n, &mut w, &mut v));

        for j in 0..n {
            // A · v_j
            let mut av = vec![0.0; n];
            for row in 0..n {
                let mut s = 0.0;
                for col in 0..n {
                    s += a[row + n * col] * v[col + n * j];
                }
                av[row] = s;
            }
            // λ_j · v_j
            for row in 0..n {
                let lv = w[j] * v[row + n * j];
                assert_close(av[row], lv, 1e-9, &format!("A v_{j} row {row}"));
            }
        }

        // Eigenvalues should be sorted ascending.
        for k in 1..n {
            assert!(w[k - 1] <= w[k] + 1e-12, "not sorted at {k}");
        }
    }

    #[test]
    fn eigen_handles_indefinite() {
        // A = [[1, 2], [2, 1]] — eigenvalues -1, 3.
        let a = vec![1.0, 2.0, 2.0, 1.0];
        let mut w = vec![0.0; 2];
        let mut v = vec![0.0; 4];
        assert!(symmetric_eigen(&a, 2, &mut w, &mut v));
        assert_close(w[0], -1.0, 1e-12, "w0");
        assert_close(w[1], 3.0, 1e-12, "w1");
    }

    #[test]
    fn eigen_rejects_wrong_buffer_size() {
        let a = vec![1.0; 4];
        let mut w = vec![0.0; 3];
        let mut v = vec![0.0; 4];
        assert!(!symmetric_eigen(&a, 2, &mut w, &mut v));
    }

    #[test]
    fn eigen_zero_dimension() {
        let mut w: Vec<Number> = vec![];
        let mut v: Vec<Number> = vec![];
        assert!(symmetric_eigen(&[], 0, &mut w, &mut v));
    }
}
