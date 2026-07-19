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
//!
//! ## (3) `Z` must span the *whole* null space
//!
//! Checks (1) and (2) are necessary and read naturally as sufficient. They are
//! not, and the first version of this module shipped without noticing — the gap
//! surfaced only when the consumer-side Lean proof of second-order sufficiency
//! would not close without a spanning hypothesis.
//!
//! Together they give `range(Z) ⊆ ker(A_active)` with `dim range(Z) = k`: they
//! bound the spanned dimension from *below*. Soundness needs it bounded from
//! *above*. A `Z` spanning a strict subspace of the null space satisfies both
//! checks, and if the subspace it omits is a direction of negative curvature,
//! then `ZᵀHZ ≻ 0` holds at a genuine saddle point and the `local-min-strict`
//! verdict is wrong. Under-reporting the null space is the dangerous direction;
//! over-reporting merely fails check (1).
//!
//! The fix follows the same principle as `identity_rows`: convert a decision
//! into a check. We ship `n − k` row indices, `n − k` column indices, and the
//! exact inverse `M` of the square submatrix they select. The consumer verifies
//! `A[rows, cols] · M = I` — one matrix product, no determinant — which gives
//! `rank(A_active) ≥ n − k`, hence `dim ker(A_active) ≤ k`, hence
//! `range(Z) = ker(A_active)` exactly. The inverse is a byproduct the emitter's
//! elimination already computed.

use num_rational::BigRational;
use num_traits::{One, Zero};

use crate::linalg::{invert_exact, nullspace_exact, pivot_columns, select_independent_rows};

/// Witness that `A_active` has rank at least `n − k`, which caps the null space
/// at dimension `k` and so forces `Z` to span *all* of it.
///
/// `rows` and `cols` select an `(n−k) × (n−k)` submatrix of `A_active`;
/// `inverse` is its exact inverse. The consumer checks the single product
/// `A[rows, cols] · inverse = I`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankWitness {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub inverse: Vec<Vec<BigRational>>,
}

/// A null-space basis together with the data that makes its rank checkable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NullspaceBasis {
    /// `n × k`, column `j` a basis vector. `k = n − rank(A_active)`.
    pub z: Vec<Vec<BigRational>>,
    /// The `k` row indices on which `Z` restricts to the identity.
    pub identity_rows: Vec<usize>,
    /// Witness that `Z` spans the *whole* null space, not merely a subspace.
    ///
    /// A required field, not an option. The first version of this struct
    /// shipped without it and looked complete: `A·Z = 0` and full column rank
    /// are both necessary, read naturally as sufficient, and are not. Making
    /// the field mandatory means no construction path can omit the check
    /// again.
    pub spanning: RankWitness,
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
    /// `A[rows, cols] · inverse ≠ I`, so the rank lower bound is not witnessed
    /// and `Z` might span only part of the null space.
    NotSpanning { row: usize, col: usize },
    /// The rank witness is not `(n−k) × (n−k)`, or indexes out of range.
    BadRankWitness,
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
        spanning: rank_witness(a_active, n, vectors.len())?,
        identity_rows,
    };

    check(a_active, &basis)?;
    Ok(basis)
}

/// Build the rank witness: an invertible `(n−k) × (n−k)` submatrix of
/// `A_active`, together with its exact inverse.
///
/// The pivot columns of the RREF index a maximal independent set of columns, so
/// there are exactly `rank(A) = n − k` of them and `A[:, cols]` has full column
/// rank. Picking `n − k` independent *rows* of that tall matrix yields a square
/// nonsingular block.
pub fn rank_witness(
    a_active: &[Vec<BigRational>],
    n: usize,
    k: usize,
) -> Result<RankWitness, NullspaceError> {
    let r = n.checked_sub(k).ok_or(NullspaceError::BadRankWitness)?;
    if r == 0 {
        // Rank 0: `A_active` constrains nothing, the null space is all of ℝⁿ,
        // and `Z = I` already spans it. The empty product check is vacuous.
        return Ok(RankWitness {
            rows: Vec::new(),
            cols: Vec::new(),
            inverse: Vec::new(),
        });
    }

    let cols = pivot_columns(a_active, n);
    if cols.len() != r {
        // rank(A) disagrees with n − dim(null space) — impossible by
        // rank-nullity unless a caller passed mismatched data.
        return Err(NullspaceError::BadRankWitness);
    }

    // Restrict to the pivot columns, then choose independent rows of that block.
    let narrowed: Vec<Vec<BigRational>> = a_active
        .iter()
        .map(|row| cols.iter().map(|&c| row[c].clone()).collect())
        .collect();
    let rows = select_independent_rows(&narrowed);
    if rows.len() != r {
        return Err(NullspaceError::BadRankWitness);
    }

    let square: Vec<Vec<BigRational>> = rows
        .iter()
        .map(|&i| cols.iter().map(|&c| a_active[i][c].clone()).collect())
        .collect();
    let inverse = invert_exact(&square).ok_or(NullspaceError::BadRankWitness)?;

    Ok(RankWitness {
        rows,
        cols,
        inverse,
    })
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

    // (3) A[rows, cols] · inverse = I, which caps dim ker(A) at k and so forces
    // range(Z) to be the *whole* null space rather than a subspace of it.
    check_spanning(a_active, &basis.spanning, n, k)?;

    Ok(())
}

/// The spanning check, separated so the negative tests can target it directly.
///
/// Establishes `rank(A_active) ≥ n − k`. Combined with `A·Z = 0` (so
/// `range(Z) ⊆ ker A`) and full column rank of `Z` (so `dim range(Z) = k`),
/// rank-nullity gives `dim ker(A) ≤ k` and hence `range(Z) = ker(A)` exactly.
///
/// Without this, checks (1) and (2) bound the spanned dimension only from
/// *below*. Soundness needs it bounded from above: a `Z` spanning a strict
/// subspace of the null space passes both while omitting a direction the
/// second-order test must see.
pub fn check_spanning(
    a_active: &[Vec<BigRational>],
    w: &RankWitness,
    n: usize,
    k: usize,
) -> Result<(), NullspaceError> {
    let r = n.checked_sub(k).ok_or(NullspaceError::BadRankWitness)?;
    if w.rows.len() != r || w.cols.len() != r || w.inverse.len() != r {
        return Err(NullspaceError::BadRankWitness);
    }
    if w.inverse.iter().any(|row| row.len() != r) {
        return Err(NullspaceError::BadRankWitness);
    }
    if w.rows.iter().any(|&i| i >= a_active.len()) || w.cols.iter().any(|&c| c >= n) {
        return Err(NullspaceError::BadRankWitness);
    }
    // Repeated indices would let a single row or column stand in for several,
    // faking a rank the matrix does not have.
    let distinct = |v: &[usize]| {
        let mut t = v.to_vec();
        t.sort_unstable();
        t.dedup();
        t.len() == v.len()
    };
    if !distinct(&w.rows) || !distinct(&w.cols) {
        return Err(NullspaceError::BadRankWitness);
    }

    for i in 0..r {
        for j in 0..r {
            let mut acc = BigRational::zero();
            for t in 0..r {
                acc += &a_active[w.rows[i]][w.cols[t]] * &w.inverse[t][j];
            }
            let want_one = i == j;
            if want_one != acc.is_one() || (!want_one && !acc.is_zero()) {
                return Err(NullspaceError::NotSpanning { row: i, col: j });
            }
        }
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
            spanning: b.spanning.clone(),
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

    /// **The regression this whole field exists for.**
    ///
    /// One constraint in 3 variables leaves a 2-dimensional null space. Hand a
    /// `Z` with only *one* of those two columns: it satisfies `A·Z = 0`, it has
    /// full column rank, and its `identity_rows` are genuinely the identity —
    /// every check the module originally shipped passes. But it spans a line
    /// inside a plane, so a reduced Hessian computed against it is blind to the
    /// missing direction. Only the spanning witness rejects it.
    #[test]
    fn a_z_spanning_a_strict_subspace_passes_the_old_checks_and_fails_the_new_one() {
        let a = mat(&[&[1, 1, 1]]);
        let full = nullspace_basis(&a, 3).unwrap();
        assert_eq!(full.n_cols(), 2, "the true null space is 2-dimensional");

        // Keep only column 0.
        let truncated_z: Vec<Vec<BigRational>> =
            full.z.iter().map(|row| vec![row[0].clone()]).collect();
        let id_row = full.identity_rows[0];

        // It genuinely lies in the null space...
        for arow in &a {
            let acc: BigRational = arow
                .iter()
                .enumerate()
                .map(|(t, av)| av * &truncated_z[t][0])
                .sum();
            assert!(acc.is_zero(), "the retained column is in ker A");
        }
        // ...and is genuinely the identity on its row, so full column rank holds.
        assert!(truncated_z[id_row][0].is_one());

        // The rank witness for the *claimed* k = 1 would need rank(A) >= 2,
        // but A has rank 1, so no such witness can be built.
        assert_eq!(
            rank_witness(&a, 3, 1),
            Err(NullspaceError::BadRankWitness),
            "a 1-column Z in a 2-dimensional null space must not be witnessable"
        );

        // And splicing in the real (k = 2) witness does not rescue it either:
        // the shapes no longer agree with the claimed dimension.
        let forged = NullspaceBasis {
            z: truncated_z,
            identity_rows: vec![id_row],
            spanning: full.spanning.clone(),
        };
        assert!(matches!(
            check(&a, &forged),
            Err(NullspaceError::BadRankWitness)
        ));
    }

    /// The spanning witness must reject a forged inverse.
    #[test]
    fn a_forged_inverse_is_rejected() {
        let a = mat(&[&[1, 1, 1]]);
        let mut b = nullspace_basis(&a, 3).unwrap();
        b.spanning.inverse[0][0] += r(1);
        assert!(matches!(
            check(&a, &b),
            Err(NullspaceError::NotSpanning { .. })
        ));
    }

    /// Repeated indices must not let one row stand in for several and fake rank.
    #[test]
    fn repeated_rank_witness_indices_are_rejected() {
        // Rank 2 in 3 variables, so the witness is 2x2 and has room to repeat.
        let a = mat(&[&[1, 0, 1], &[0, 1, 1]]);
        let b = nullspace_basis(&a, 3).unwrap();
        assert_eq!(b.spanning.rows.len(), 2);

        let mut forged = b.clone();
        forged.spanning.cols[1] = forged.spanning.cols[0];
        assert_eq!(
            check_spanning(&a, &forged.spanning, 3, 1),
            Err(NullspaceError::BadRankWitness)
        );
    }

    /// The honest witness verifies, at every size exercised here.
    #[test]
    fn the_emitted_spanning_witness_verifies() {
        for a in [
            mat(&[&[1, 1, 1]]),
            mat(&[&[1, 0, 1], &[0, 1, 1]]),
            mat(&[&[1, 0, 1], &[0, 1, 1], &[1, 1, 2]]),
            mat(&[&[2, -3, 5]]),
        ] {
            let b = nullspace_basis(&a, 3).unwrap();
            check_spanning(&a, &b.spanning, 3, b.n_cols()).unwrap();
        }
    }

    /// Ragged input is refused rather than silently mis-shaped.
    #[test]
    fn ragged_input_is_a_shape_error() {
        let a = vec![vec![r(1), r(1)], vec![r(1)]];
        assert_eq!(nullspace_basis(&a, 2), Err(NullspaceError::Shape));
    }
}
