//! Exact reduced Hessian `ZᵀHZ` and its strictly-positive `LDLᵀ` witness.
//!
//! This is the curvature half of a Tier 2 (`local-min-strict`) certificate. The
//! null-space half lives in [`crate::nullspace`]; here we take the `Z` it
//! produced and form the reduced Hessian exactly over ℚ, then factor it.
//!
//! Two things distinguish this from the ordinary `hessian_psd` witness.
//!
//! **The factorization must be strictly positive.** Second-order *sufficiency*
//! needs `ZᵀHZ ≻ 0`, not `⪰ 0`. [`crate::ldlt::ldlt`] enforces only `D ≥ 0`,
//! since that is what convexity requires, so a zero pivot passes it happily. A
//! zero pivot here means the reduced Hessian is singular — the objective is flat
//! along some feasible direction — and flatness is precisely the case where
//! strict minimality *fails*. We therefore re-check `D > 0` entrywise and refuse
//! otherwise. Nothing else in the crate needs this distinction, which is why it
//! is enforced here rather than in `ldlt`.
//!
//! **`H` itself may be indefinite.** That is the entire point of the tier: a
//! saddle in the full space can be a strict minimum once restricted to the
//! feasible directions. So a negative eigenvalue of `H` is not an error, and the
//! emitter must not check `H` for definiteness — only `ZᵀHZ`. Conversely, a
//! reduced Hessian that comes out indefinite is a genuine refusal: the point is
//! a constrained saddle and no Tier 2 certificate exists for it.

use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::ldlt::{Ldl, LdlError, ldlt};

/// The reduced Hessian together with its strict factorization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReducedHessian {
    /// `ZᵀHZ`, dense `k × k`, exact.
    pub matrix: Vec<Vec<BigRational>>,
    /// `L`, `D` with `matrix = L·diag(D)·Lᵀ` and every `D i > 0`.
    pub ldl: Ldl,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReducedError {
    /// `H` is not square, or `Z`'s row count disagrees with it.
    Shape,
    /// `H` is not symmetric. A Hessian must be; an asymmetric input means the
    /// caller assembled it wrongly, and `ZᵀHZ` would silently lose information.
    NotSymmetric { row: usize, col: usize },
    /// The reduced Hessian has a negative pivot: `x*` is a constrained saddle,
    /// not a strict minimum. A correct refusal, not a defect.
    ReducedIndefinite { col: usize },
    /// The reduced Hessian is singular — flat along some feasible direction, so
    /// strict minimality does not hold.
    ReducedSingular { col: usize },
    /// The self-check `L·diag(D)·Lᵀ = ZᵀHZ` failed. Should be unreachable;
    /// present so a factorization bug cannot escape into a certificate.
    SelfCheck { row: usize, col: usize },
}

/// Form `ZᵀHZ` exactly. `h` is `n × n` dense, `z` is `n × k` dense.
#[allow(clippy::needless_range_loop)] // index loops cross-reference h and z
pub fn reduced_hessian_matrix(
    h: &[Vec<BigRational>],
    z: &[Vec<BigRational>],
) -> Result<Vec<Vec<BigRational>>, ReducedError> {
    let n = h.len();
    if h.iter().any(|row| row.len() != n) || z.len() != n {
        return Err(ReducedError::Shape);
    }
    let k = if n == 0 { 0 } else { z[0].len() };
    if z.iter().any(|row| row.len() != k) {
        return Err(ReducedError::Shape);
    }
    for i in 0..n {
        for j in 0..i {
            if h[i][j] != h[j][i] {
                return Err(ReducedError::NotSymmetric { row: i, col: j });
            }
        }
    }

    // (HZ)[i][j] = Σ_t H[i][t]·Z[t][j]
    let hz: Vec<Vec<BigRational>> = (0..n)
        .map(|i| {
            (0..k)
                .map(|j| {
                    let mut acc = BigRational::zero();
                    for t in 0..n {
                        acc += &h[i][t] * &z[t][j];
                    }
                    acc
                })
                .collect()
        })
        .collect();

    // (ZᵀHZ)[a][b] = Σ_i Z[i][a]·(HZ)[i][b]
    Ok((0..k)
        .map(|a| {
            (0..k)
                .map(|b| {
                    let mut acc = BigRational::zero();
                    for i in 0..n {
                        acc += &z[i][a] * &hz[i][b];
                    }
                    acc
                })
                .collect()
        })
        .collect())
}

/// Form `ZᵀHZ` and factor it with a **strictly** positive diagonal, verifying
/// the identity before returning.
#[allow(clippy::needless_range_loop)] // index loops cross-reference L, D and the matrix
pub fn reduced_hessian(
    h: &[Vec<BigRational>],
    z: &[Vec<BigRational>],
) -> Result<ReducedHessian, ReducedError> {
    let matrix = reduced_hessian_matrix(h, z)?;
    let k = matrix.len();

    let ldl = ldlt(&matrix).map_err(|e| match e {
        LdlError::NotSquare => ReducedError::Shape,
        LdlError::Indefinite { col } => ReducedError::ReducedIndefinite { col },
        // A zero pivot that needs pivoting is singular for our purposes too:
        // either way the strict-positivity requirement below is unmet.
        LdlError::SingularNeedsPivot { col, .. } => ReducedError::ReducedSingular { col },
    })?;

    // The strictness gate. `ldlt` accepts `D ≥ 0`; positive-definiteness needs
    // `> 0`, and a zero here means a flat feasible direction.
    let zero = BigRational::zero();
    for (i, d) in ldl.d.iter().enumerate() {
        if *d > zero {
            continue;
        }
        // `ldlt` rejects negative pivots already, so in practice this only fires
        // on `d == 0`. Distinguish anyway rather than mislabel a sign error as
        // singularity if that ever changes.
        return Err(if d.is_zero() {
            ReducedError::ReducedSingular { col: i }
        } else {
            ReducedError::ReducedIndefinite { col: i }
        });
    }

    // Self-check `L·diag(D)·Lᵀ = ZᵀHZ` exactly, so a factorization bug cannot
    // reach a certificate. `L` is unit lower triangular with the diagonal implied.
    let l = |i: usize, j: usize| -> BigRational {
        if i == j {
            BigRational::one()
        } else {
            ldl.l_below
                .iter()
                .find(|(a, b, _)| *a == i && *b == j)
                .map_or_else(BigRational::zero, |(_, _, v)| v.clone())
        }
    };
    for i in 0..k {
        for j in 0..k {
            let mut acc = BigRational::zero();
            for t in 0..k {
                acc += l(i, t) * &ldl.d[t] * l(j, t);
            }
            if acc != matrix[i][j] {
                return Err(ReducedError::SelfCheck { row: i, col: j });
            }
        }
    }

    Ok(ReducedHessian { matrix, ldl })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64) -> BigRational {
        BigRational::from_integer(n.into())
    }

    fn mat(rows: &[&[i64]]) -> Vec<Vec<BigRational>> {
        rows.iter()
            .map(|r0| r0.iter().map(|&v| r(v)).collect())
            .collect()
    }

    /// **The case the tier exists for**, and the same instance the Lean example
    /// `Examples/SecondOrderIndefinite` certifies: `H = diag(1, −1)` is
    /// indefinite, but restricted to the `x₀` axis it is the `1×1` matrix `(1)`,
    /// which is positive definite. An emitter that screened `H` for definiteness
    /// would wrongly refuse this problem.
    #[test]
    fn an_indefinite_h_can_have_a_positive_definite_reduction() {
        let h = mat(&[&[1, 0], &[0, -1]]);
        let z = mat(&[&[1], &[0]]);
        let out = reduced_hessian(&h, &z).unwrap();
        assert_eq!(out.matrix, vec![vec![r(1)]]);
        assert_eq!(out.ldl.d, vec![r(1)]);
    }

    /// The mirror image: keep the *negative* direction instead. Same `H`, same
    /// null-space machinery, but now `x*` is a constrained saddle and Tier 2
    /// must refuse rather than certify.
    #[test]
    fn selecting_the_negative_direction_is_refused_as_a_saddle() {
        let h = mat(&[&[1, 0], &[0, -1]]);
        let z = mat(&[&[0], &[1]]);
        assert_eq!(
            reduced_hessian(&h, &z),
            Err(ReducedError::ReducedIndefinite { col: 0 })
        );
    }

    /// A flat feasible direction is singular, not definite: strict minimality
    /// genuinely fails (every point along it ties), so this must be refused even
    /// though `ldlt` alone would accept `D = 0` as PSD.
    #[test]
    fn a_flat_direction_is_refused_though_ldlt_alone_would_accept_it() {
        let h = mat(&[&[1, 0], &[0, 0]]);
        let z = mat(&[&[0], &[1]]);
        // ldlt is happy: D = [0] is nonnegative.
        assert_eq!(ldlt(&mat(&[&[0]])).unwrap().d, vec![r(0)]);
        // The strict gate is not.
        assert_eq!(
            reduced_hessian(&h, &z),
            Err(ReducedError::ReducedSingular { col: 0 })
        );
    }

    /// A genuine 2-dimensional reduction with off-diagonal coupling, so the
    /// `LDLᵀ` recursion does real work rather than reading off a diagonal.
    #[test]
    fn two_dimensional_reduction_with_coupling() {
        // H = [[2,1,0],[1,2,0],[0,0,-5]]; Z picks the first two coordinates.
        let h = mat(&[&[2, 1, 0], &[1, 2, 0], &[0, 0, -5]]);
        let z = mat(&[&[1, 0], &[0, 1], &[0, 0]]);
        let out = reduced_hessian(&h, &z).unwrap();
        assert_eq!(out.matrix, mat(&[&[2, 1], &[1, 2]]));
        // D = [2, 3/2] — both strictly positive.
        assert_eq!(out.ldl.d[0], r(2));
        assert_eq!(out.ldl.d[1], BigRational::new(3.into(), 2.into()));
    }

    /// A `Z` that mixes coordinates, so `ZᵀHZ` is not a submatrix of `H`.
    #[test]
    fn a_rotated_basis_reduces_exactly() {
        // H = diag(1, -1), Z = (2, 1)ᵀ: ZᵀHZ = 4 − 1 = 3.
        let h = mat(&[&[1, 0], &[0, -1]]);
        let z = mat(&[&[2], &[1]]);
        let out = reduced_hessian(&h, &z).unwrap();
        assert_eq!(out.matrix, vec![vec![r(3)]]);
    }

    /// Fractional entries stay exact — no rounding in the reduction.
    #[test]
    fn fractional_entries_stay_exact() {
        let third = BigRational::new(1.into(), 3.into());
        let h = vec![vec![third.clone(), r(0)], vec![r(0), r(1)]];
        let z = vec![vec![r(1)], vec![r(0)]];
        let out = reduced_hessian(&h, &z).unwrap();
        assert_eq!(out.matrix, vec![vec![third]]);
    }

    /// An asymmetric `H` is a caller error, not something to silently reduce.
    #[test]
    fn asymmetric_h_is_rejected() {
        let h = mat(&[&[1, 2], &[3, 1]]);
        let z = mat(&[&[1], &[0]]);
        assert!(matches!(
            reduced_hessian(&h, &z),
            Err(ReducedError::NotSymmetric { .. })
        ));
    }

    /// Mismatched shapes are refused rather than indexed out of bounds.
    #[test]
    fn mismatched_shapes_are_refused() {
        let h = mat(&[&[1, 0], &[0, 1]]);
        let z = mat(&[&[1]]); // 1 row, but H is 2×2
        assert_eq!(reduced_hessian(&h, &z), Err(ReducedError::Shape));
    }

    /// An empty null space (`k = 0`) reduces to an empty matrix, vacuously
    /// definite. This is the fully-determined case: no feasible directions
    /// remain, so `x*` is the only feasible point.
    #[test]
    fn empty_nullspace_reduces_to_an_empty_matrix() {
        let h = mat(&[&[1, 0], &[0, -1]]);
        let z: Vec<Vec<BigRational>> = vec![Vec::new(), Vec::new()];
        let out = reduced_hessian(&h, &z).unwrap();
        assert!(out.matrix.is_empty());
        assert!(out.ldl.d.is_empty());
    }
}
