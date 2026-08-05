//! Exact rational `LDLᵀ` factorization of the objective Hessian.
//!
//! The PSD witness in the certificate is a unit-lower-triangular `L` and a
//! nonnegative diagonal `D` with `Q = L·diag(D)·Lᵀ`. Unlike Cholesky, `LDLᵀ`
//! uses **no square roots**, so for a rational `Q` the factors stay in ℚ and the
//! identity holds exactly — which is exactly what the Lean side rechecks by
//! `ring`/`norm_num`.
//!
//! We do **not** pivot. A convex QP's Hessian is PSD; if the no-pivot recursion
//! hits a negative pivot the matrix is indefinite, and if it hits a zero pivot
//! with a nonzero off-diagonal numerator the matrix is PSD-but-singular in a way
//! this witness form can't express. Either way we error out rather than emit a
//! factorization that won't certify.

use num_rational::BigRational;
use num_traits::Zero;

/// `Q = L·diag(D)·Lᵀ` with `L` unit-lower-triangular and `D ≥ 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ldl {
    /// Strictly-below-diagonal entries of `L` as `(i, j, value)` with `i > j`.
    /// The unit diagonal is implied and omitted.
    pub l_below: Vec<(usize, usize, BigRational)>,
    /// The diagonal `D`, length `n`, each entry `≥ 0`.
    pub d: Vec<BigRational>,
}

/// Why a factorization could not be produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LdlError {
    /// Input was not square.
    NotSquare,
    /// A negative pivot at column `col`: `Q` is indefinite, not PSD.
    Indefinite { col: usize },
    /// A zero pivot at `col` with a nonzero numerator at row `row`: PSD-singular,
    /// not expressible as unit-lower `LDLᵀ` without pivoting.
    SingularNeedsPivot { col: usize, row: usize },
}

/// Factor a symmetric matrix `q` (given densely, `n×n`; only the lower triangle
/// is read) into exact rational `LDLᵀ` with `D ≥ 0`.
#[allow(clippy::needless_range_loop)] // index loops cross-reference q, l, d
pub fn ldlt(q: &[Vec<BigRational>]) -> Result<Ldl, LdlError> {
    let n = q.len();
    if q.iter().any(|row| row.len() != n) {
        return Err(LdlError::NotSquare);
    }

    // Dense unit-lower work matrix (below-diagonal entries filled as we go).
    let mut l = vec![vec![BigRational::zero(); n]; n];
    let mut d = vec![BigRational::zero(); n];

    for j in 0..n {
        // d[j] = Q[j][j] - Σ_{k<j} L[j][k]² · d[k]
        let mut dj = q[j][j].clone();
        for k in 0..j {
            dj -= &l[j][k] * &l[j][k] * &d[k];
        }
        if dj < BigRational::zero() {
            return Err(LdlError::Indefinite { col: j });
        }

        for i in (j + 1)..n {
            // numerator = Q[i][j] - Σ_{k<j} L[i][k]·L[j][k]·d[k]
            let mut num = q[i][j].clone();
            for k in 0..j {
                num -= &l[i][k] * &l[j][k] * &d[k];
            }
            if dj.is_zero() {
                if !num.is_zero() {
                    return Err(LdlError::SingularNeedsPivot { col: j, row: i });
                }
                // L[i][j] stays zero.
            } else {
                l[i][j] = &num / &dj;
            }
        }
        d[j] = dj;
    }

    let mut l_below = Vec::new();
    for i in 0..n {
        for (j, lij) in l[i].iter().enumerate().take(i) {
            if !lij.is_zero() {
                l_below.push((i, j, lij.clone()));
            }
        }
    }
    Ok(Ldl { l_below, d })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> BigRational {
        BigRational::new(n.into(), d.into())
    }

    /// Reconstruct `L·diag(D)·Lᵀ` densely from an [`Ldl`] for checking.
    #[allow(clippy::needless_range_loop)]
    fn reconstruct(ldl: &Ldl, n: usize) -> Vec<Vec<BigRational>> {
        let mut l = vec![vec![BigRational::zero(); n]; n];
        for i in 0..n {
            l[i][i] = r(1, 1); // unit diagonal
        }
        for (i, j, v) in &ldl.l_below {
            l[*i][*j] = v.clone();
        }
        // M = L · diag(D) · Lᵀ  ⇒  M[i][k] = Σ_j L[i][j]·D[j]·L[k][j]
        let mut m = vec![vec![BigRational::zero(); n]; n];
        for i in 0..n {
            for k in 0..n {
                let mut acc = BigRational::zero();
                for j in 0..n {
                    acc += &l[i][j] * &ldl.d[j] * &l[k][j];
                }
                m[i][k] = acc;
            }
        }
        m
    }

    #[test]
    fn diagonal_q_is_identity_l() {
        // The reference QP's Q = diag(2,2): L = I, D = (2,2).
        let q = vec![vec![r(2, 1), r(0, 1)], vec![r(0, 1), r(2, 1)]];
        let ldl = ldlt(&q).unwrap();
        assert!(
            ldl.l_below.is_empty(),
            "diagonal Q needs no below-diagonal L"
        );
        assert_eq!(ldl.d, vec![r(2, 1), r(2, 1)]);
        assert_eq!(reconstruct(&ldl, 2), q);
    }

    #[test]
    fn dense_psd_factorizes_exactly() {
        // Q = [[2,1],[1,2]]: L[1][0] = 1/2, D = (2, 3/2).
        let q = vec![vec![r(2, 1), r(1, 1)], vec![r(1, 1), r(2, 1)]];
        let ldl = ldlt(&q).unwrap();
        assert_eq!(ldl.l_below, vec![(1, 0, r(1, 2))]);
        assert_eq!(ldl.d, vec![r(2, 1), r(3, 2)]);
        assert_eq!(reconstruct(&ldl, 2), q, "L·diag(D)·Lᵀ must equal Q exactly");
    }

    #[test]
    fn indefinite_is_rejected() {
        // Q = [[1,2],[2,1]] has a negative second pivot (1 - 4 = -3).
        let q = vec![vec![r(1, 1), r(2, 1)], vec![r(2, 1), r(1, 1)]];
        assert_eq!(ldlt(&q), Err(LdlError::Indefinite { col: 1 }));
    }

    #[test]
    fn psd_singular_with_consistent_column_is_ok() {
        // Q = [[1,1],[1,1]] is PSD rank-1: D = (1, 0), L[1][0] = 1.
        let q = vec![vec![r(1, 1), r(1, 1)], vec![r(1, 1), r(1, 1)]];
        let ldl = ldlt(&q).unwrap();
        assert_eq!(ldl.d, vec![r(1, 1), r(0, 1)]);
        assert_eq!(reconstruct(&ldl, 2), q);
    }
}
