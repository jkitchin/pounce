//! Null-space basis `Z` of the active-constraint Jacobian, as a witness the
//! consumer can *verify* rather than trust.
//!
//! Second-order sufficiency reads `Zᵀ H Z ≻ 0`, where `Z` spans the null space
//! of the active constraint gradients. Witnesses are untrusted, so shipping `Z`
//! is not enough: a forged `Z` spanning some *smaller* subspace would make a
//! saddle point look like a strict local minimum. The consumer must confirm two
//! things —
//!
//! 1. `A_active · Z = 0` — the columns really lie in the null space;
//! 2. `Z` has full column rank — they really span a space of the claimed
//!    dimension, i.e. no column is redundant.
//!
//! (1) is an exact rational matrix product. (2) is the interesting one, because
//! the obvious route — a determinant, or a rank computation — is exactly the
//! kind of `O(n³)` rational-matrix decision procedure that has already proven
//! too slow in the kernel for dense Hessians.
//!
//! The way out is to make full rank **structural instead of computational**. The
//! RREF null-space basis assigns each basis vector to a distinct *free* column,
//! setting that entry to 1 and every other free entry to 0. So the rows of `Z`
//! indexed by the free columns are exactly the identity matrix. Given that, if
//! `Z v = 0` then reading off those rows gives `v = 0` directly — full column
//! rank falls out in `O(k)` index lookups with no arithmetic at all.
//!
//! We therefore ship the free-column indices alongside `Z` as `identity_rows`.
//! They are not extra trust: the consumer checks `Z[identity_rows, :] = I`
//! itself, and a forged list simply fails that check. It is a *hint* that turns
//! an expensive proof obligation into a cheap one, which is the same trick the
//! `LDLᵀ` witness plays for positive-semidefiniteness.

use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::linalg::nullspace_exact;

/// A null-space basis together with the data that makes its rank checkable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NullspaceBasis {
    /// `n × k`, column `j` a basis vector. `k = n − rank(A_active)`.
    pub z: Vec<Vec<BigRational>>,
    /// The `k` row indices on which `Z` restricts to the identity.
    pub identity_rows: Vec<usize>,
}

impl NullspaceBasis {
    /// Number of variables (rows of `Z`).
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.z.len()
    }

    /// Dimension of the null space (columns of `Z`).
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.identity_rows.len()
    }
}

/// What can go wrong. Each variant is a self-check the emitter runs *before*
/// writing, so that a certificate which would not verify is never emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NullspaceError {
    /// `a_active` is ragged or its row length disagrees with `n`.
    Shape,
    /// `A · Z ≠ 0` — the computed basis does not lie in the null space.
    NotInNullspace { row: usize, col: usize },
    /// `Z[identity_rows, :] ≠ I`, so full column rank is not witnessed.
    NotIdentity { row: usize, col: usize },
}

/// Compute the null-space basis of `a_active` (an `m × n` matrix) in the form
/// the certificate ships, self-checking both consumer obligations exactly.
///
/// An empty `a_active` is a legitimate input — no active constraints means the
/// null space is all of `ℝⁿ` and `Z = I`.
pub fn nullspace_basis(
    a_active: &[Vec<BigRational>],
    n: usize,
) -> Result<NullspaceBasis, NullspaceError> {
    if a_active.iter().any(|row| row.len() != n) {
        return Err(NullspaceError::Shape);
    }

    // `nullspace_exact` returns one vector per free column, each of length `n`.
    let vectors = nullspace_exact(a_active, n);

    // Recover the free columns from the basis itself rather than re-deriving
    // the RREF: vector `j` is the unique one whose distinguished entry is 1,
    // and that entry is its free column. Reading it back this way means the
    // identity claim is checked against the data actually shipped, not against
    // an internal variable that might have drifted from it.
    let identity_rows: Vec<usize> = vectors
        .iter()
        .enumerate()
        .map(|(j, v)| {
            // The free column of vector `j` is the one where it is 1 and every
            // *other* basis vector is 0. Locate it by that property.
            (0..n)
                .find(|&c| {
                    v[c].is_one()
                        && vectors
                            .iter()
                            .enumerate()
                            .all(|(i, w)| i == j || w[c].is_zero())
                })
                .unwrap_or(usize::MAX)
        })
        .collect();

    let basis = NullspaceBasis {
        // Transpose into `n × k` so `z[i][j]` is the natural row-major shape the
        // schema and the consumer both index.
        z: (0..n)
            .map(|i| vectors.iter().map(|v| v[i].clone()).collect())
            .collect(),
        identity_rows,
    };

    check(a_active, &basis)?;
    Ok(basis)
}

/// Run exactly the two checks the consumer will run. Public so the negative
/// tests can drive it against deliberately corrupted witnesses — the emitter's
/// guarantee is only as good as the checks being the *same* ones.
pub fn check(a_active: &[Vec<BigRational>], basis: &NullspaceBasis) -> Result<(), NullspaceError> {
    let k = basis.n_cols();
    let n = basis.n_rows();

    if basis.z.iter().any(|row| row.len() != k) {
        return Err(NullspaceError::Shape);
    }
    if basis.identity_rows.iter().any(|&r| r >= n) {
        return Err(NullspaceError::Shape);
    }

    // (1) A · Z = 0, exactly.
    for (i, arow) in a_active.iter().enumerate() {
        if arow.len() != n {
            return Err(NullspaceError::Shape);
        }
        for j in 0..k {
            let mut acc = BigRational::zero();
            for (t, a) in arow.iter().enumerate() {
                acc += a * &basis.z[t][j];
            }
            if !acc.is_zero() {
                return Err(NullspaceError::NotInNullspace { row: i, col: j });
            }
        }
    }

    // (2) Z[identity_rows, :] = I. This is what buys full column rank.
    for (j, &r) in basis.identity_rows.iter().enumerate() {
        for c in 0..k {
            let want_one = c == j;
            let entry = &basis.z[r][c];
            if want_one != entry.is_one() || (!want_one && !entry.is_zero()) {
                return Err(NullspaceError::NotIdentity { row: r, col: c });
            }
        }
    }

    // Distinct rows: a repeated index would let one row masquerade as two
    // identity rows and inflate the apparent rank.
    let mut seen = basis.identity_rows.clone();
    seen.sort_unstable();
    seen.dedup();
    if seen.len() != k {
        return Err(NullspaceError::Shape);
    }

    Ok(())
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

    /// One constraint in 3 variables leaves a 2-dimensional null space.
    #[test]
    fn single_row_gives_codimension_one() {
        let a = mat(&[&[1, 1, 1]]);
        let b = nullspace_basis(&a, 3).unwrap();
        assert_eq!(b.n_cols(), 2);
        assert_eq!(b.n_rows(), 3);
        check(&a, &b).unwrap();
    }

    /// No active constraints: the null space is everything, `Z = I`.
    #[test]
    fn empty_active_set_gives_the_identity() {
        let b = nullspace_basis(&[], 3).unwrap();
        assert_eq!(b.n_cols(), 3);
        assert_eq!(b.identity_rows, vec![0, 1, 2]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(b.z[i][j], if i == j { r(1) } else { r(0) });
            }
        }
    }

    /// A *degenerate* active set — more rows than variables, and dependent.
    /// The rank is what matters, not the row count; this is the case real
    /// active-set solves produce and the one a naive `n − m` would get wrong.
    #[test]
    fn dependent_rows_are_counted_by_rank_not_by_row_count() {
        // Row 3 = row 1 + row 2, so rank is 2 and the null space is 1-dimensional.
        let a = mat(&[&[1, 0, 1], &[0, 1, 1], &[1, 1, 2]]);
        let b = nullspace_basis(&a, 3).unwrap();
        assert_eq!(b.n_cols(), 1, "rank 2 in 3 variables leaves dimension 1");
        check(&a, &b).unwrap();
    }

    /// Full rank: the null space is trivial and `Z` has no columns.
    #[test]
    fn full_rank_gives_an_empty_basis() {
        let a = mat(&[&[1, 0], &[0, 1]]);
        let b = nullspace_basis(&a, 2).unwrap();
        assert_eq!(b.n_cols(), 0);
        check(&a, &b).unwrap();
    }

    /// Rational, non-integer entries survive exactly — no rounding anywhere.
    #[test]
    fn fractional_entries_stay_exact() {
        let a = vec![vec![
            BigRational::new(1.into(), 3.into()),
            BigRational::new(1.into(), 7.into()),
        ]];
        let b = nullspace_basis(&a, 2).unwrap();
        assert_eq!(b.n_cols(), 1);
        // 1/3·z₀ + 1/7·z₁ = 0 exactly; check() already asserted it, but pin the
        // actual value so a change in basis convention is visible.
        let dot = BigRational::new(1.into(), 3.into()) * &b.z[0][0]
            + BigRational::new(1.into(), 7.into()) * &b.z[1][0];
        assert!(dot.is_zero());
    }

    // --- negative tests: the checks must reject what they are there to reject ---

    /// A `Z` that does not lie in the null space is caught.
    #[test]
    fn a_forged_column_outside_the_nullspace_is_rejected() {
        let a = mat(&[&[1, 1, 1]]);
        let mut b = nullspace_basis(&a, 3).unwrap();
        b.z[0][0] += r(1); // perturb one entry
        assert!(matches!(
            check(&a, &b),
            Err(NullspaceError::NotInNullspace { .. })
        ));
    }

    /// The rank-inflation attack this design exists to stop: duplicate a column
    /// so `Z` claims a bigger null space than it spans. `A · Z = 0` still holds
    /// — the duplicate *is* in the null space — so only the identity check can
    /// catch it. This is the test that justifies shipping `identity_rows`.
    #[test]
    fn a_duplicated_column_inflates_rank_and_is_caught_only_by_the_identity_check() {
        let a = mat(&[&[1, 1, 1]]);
        let b = nullspace_basis(&a, 3).unwrap();

        let dup = NullspaceBasis {
            z: b.z
                .iter()
                .map(|row| vec![row[0].clone(), row[0].clone()])
                .collect(),
            identity_rows: b.identity_rows.clone(),
        };

        // The forged basis passes the null-space product...
        for arow in &a {
            for j in 0..2 {
                let acc: BigRational = arow
                    .iter()
                    .enumerate()
                    .map(|(t, av)| av * &dup.z[t][j])
                    .sum();
                assert!(acc.is_zero(), "duplicate column is genuinely in ker A");
            }
        }
        // ...and is still rejected, because it is not the identity on those rows.
        assert!(matches!(
            check(&a, &dup),
            Err(NullspaceError::NotIdentity { .. })
        ));
    }

    /// A repeated `identity_rows` entry must not be able to double-count a row.
    #[test]
    fn repeated_identity_rows_are_rejected() {
        let a = mat(&[&[1, 1, 1]]);
        let mut b = nullspace_basis(&a, 3).unwrap();
        let first = b.identity_rows[0];
        b.identity_rows[1] = first;
        assert!(check(&a, &b).is_err());
    }

    /// Out-of-range indices are a shape error, not a panic.
    #[test]
    fn out_of_range_identity_row_is_a_shape_error() {
        let a = mat(&[&[1, 1, 1]]);
        let mut b = nullspace_basis(&a, 3).unwrap();
        b.identity_rows[0] = 99;
        assert_eq!(check(&a, &b), Err(NullspaceError::Shape));
    }

    /// Ragged input is refused rather than silently mis-shaped.
    #[test]
    fn ragged_input_is_a_shape_error() {
        let a = vec![vec![r(1), r(1)], vec![r(1)]];
        assert_eq!(nullspace_basis(&a, 2), Err(NullspaceError::Shape));
    }
}
