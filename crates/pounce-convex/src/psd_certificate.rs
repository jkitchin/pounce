//! Conservative positive-semidefiniteness certificates for sparse matrices.

use std::collections::{BTreeMap, HashMap};

use pounce_common::types::Index;
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};

use crate::Triplet;

/// An input error reported by [`certify_psd_lower_triangle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsdCertificateError {
    /// `tolerance` was negative, NaN, or infinite.
    InvalidTolerance,
    /// A triplet coordinate is outside the declared `n × n` matrix.
    OutOfBounds {
        /// Zero-based row coordinate.
        row: usize,
        /// Zero-based column coordinate.
        col: usize,
        /// Declared matrix dimension.
        dimension: usize,
    },
    /// The caller supplied an upper-triangle triplet to a lower-triangle API.
    UpperTriangle { row: usize, col: usize },
    /// A triplet value is not finite.
    NonFiniteValue { row: usize, col: usize },
    /// Summing duplicate triplets overflowed to a non-finite value.
    NonFiniteSum { row: usize, col: usize },
    /// The matrix pattern cannot be represented by the linear-solver index
    /// type on this target.
    DimensionTooLarge,
}

/// Certify that a sparse symmetric matrix is positive semidefinite within an
/// absolute tolerance.
///
/// `triplets` supplies the matrix's lower triangle in zero-based
/// coordinates (`row >= col`). Duplicate coordinates are summed. The
/// diagonal fast path accepts an eigenvalue down to `-tolerance`, while coupled
/// matrices are tested by factoring `A + tolerance * I` and inspecting its
/// inertia. These branches differ at the exact boundary: the diagonal test is
/// inclusive (`lambda_min(A) >= -tolerance`), but the coupled route must first
/// accept the shifted matrix as a nonsingular factorization and is therefore
/// effectively strict (`lambda_min(A) > -tolerance`) with the current FERAL
/// backend.
///
/// At `lambda_min(A) = -tolerance`, the shifted matrix has a zero eigenvalue.
/// FERAL reports that zero pivot as `Singular`; `Factorization::new` converts
/// that status to an error, so the certificate returns `Ok(false)` before
/// `number_of_neg_evals` is consulted. This is separate from the later
/// `provides_inertia()` guard: a backend that reports no inertia, or any other
/// factorization failure, likewise produces `Ok(false)` rather than a
/// certificate.
///
/// For a symmetric matrix, shifting every eigenvalue by `tolerance` gives
/// `lambda_min(A + tolerance I) = lambda_min(A) + tolerance`. Therefore a
/// successful factorization with zero negative eigenvalues certifies exactly
/// `lambda_min(A) >= -tolerance`; it does not require the original matrix to
/// be nonsingular. The strict lower-triangle convention is important here:
/// each off-diagonal is supplied once and the factorization layer mirrors it
/// according to its symmetric-matrix contract.
///
/// More explicitly, the factorization is an `L D L^T` factorization of the
/// shifted symmetric matrix, up to the backend's permutation. The factor is a
/// congruence, so Sylvester's law of inertia makes the reported negative-pivot
/// count equal to the shifted matrix's number of negative eigenvalues. Since
/// every eigenvalue is shifted by the same `tolerance`, `negative == 0` is
/// equivalent to `lambda_min(A) + tolerance >= 0`. A singular PSD matrix has
/// `lambda_min(A) = 0`, so the shift moves its zero eigenvalue to the strictly
/// positive value `tolerance` and it is certified just like an SPD matrix
/// (subject to the backend's finite-precision singularity policy).
///
/// The diagonal branch is intentionally separate: once duplicate entries have
/// been summed, a diagonal matrix's eigenvalues are its diagonal entries, so a
/// single sign scan settles it in `O(nnz)` without constructing or factoring a
/// backend. That cost matters for the large separable and least-squares QP
/// shapes, whose Hessians can contain tens of thousands of diagonal entries
/// even though their convexity is trivial.
///
/// Coupled Hessians use the sparse factorization instead of a dense Jacobi or
/// eigensolve. The latter allocates a dense `k × k` matrix and costs
/// `O(k^3)` for the `k` active variables; sparse inertia follows the nonzero
/// pattern and its fill. This is what makes the coupled, sparse Hessians in
/// the CVXQP family (typically around `k = 1000`) practical to classify
/// without paying a dense cubic eigenvalue computation before the solve.
///
/// Empty and diagonal matrices are settled without constructing `make_backend`.
/// A coupled matrix is compressed to the variables present in its nonzero
/// pattern before it is factored, so structurally zero rows do not inflate the
/// factorization.
///
/// The shift is deliberately absolute: it preserves the classifier's
/// `lambda_min(A) >= -tolerance` contract. At very large matrix scales an
/// absolute shift can be below floating-point resolution, and an exact
/// rank-deficient PSD matrix may consequently be rejected as singular by a
/// backend. That is a conservative dispatch fallback, not evidence of a
/// negative eigenvalue; changing to a norm-relative tolerance would be a
/// separate numerical-policy change.
pub fn certify_psd_lower_triangle<F>(
    n: usize,
    triplets: &[Triplet],
    tolerance: f64,
    make_backend: F,
) -> Result<bool, PsdCertificateError>
where
    F: FnOnce() -> Box<dyn SparseSymLinearSolverInterface>,
{
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(PsdCertificateError::InvalidTolerance);
    }
    let entries = sum_lower_triangle(n, triplets)?;
    certify_summed(entries, tolerance, make_backend)
}

/// The tolerance in the frontends' convexity screen: an eigenvalue below
/// `-PSD_TOL` is treated as a genuine negative direction (nonconvex); within
/// `±PSD_TOL` it is treated as zero. Scaled tolerances would be better once
/// we have problem scaling in this path; a fixed absolute tolerance is
/// adequate here and errs toward the safe (more general) class. Used through
/// [`psd_band`], which scales it down for a small `‖H‖∞`.
///
/// Moved here from `pounce-cli/src/dispatch.rs` (gh#1147 PR review) so the
/// CLI's dispatch and the C API's `PounceQp*` handle read one definition.
pub const PSD_TOL: f64 = 1e-9;

/// The band around zero within which a Hessian eigenvalue counts as zero,
/// for a Hessian whose (summed) entries are `values`.
///
/// [`PSD_TOL`] scaled by `‖H‖∞`, and deliberately only in the *lowering*
/// direction (`.min(1.0)`): tightening the band on a small `H` fixes gh#872,
/// while widening it on a large `H` would hand the convex engine Hessians it
/// rejects today — a wrong answer, not a slower one. The asymmetry is the
/// point, so this is not a plain `PSD_TOL * h_scale`.
///
/// Moved here from `pounce-cli/src/dispatch.rs` with [`PSD_TOL`]; there it
/// took the CLI's `QuadHessian` map, here the map's values.
pub fn psd_band<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let h_scale = values.into_iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    PSD_TOL * h_scale.min(1.0)
}

/// [`certify_psd_lower_triangle`] at [`psd_band`] of the matrix itself: the
/// band is read off the summed entries the certificate builds anyway, so a
/// caller holding raw triplets (duplicates allowed) does not sum them a
/// second time just to find `‖H‖∞`. Same errors, except that the tolerance
/// is always valid.
pub fn certify_psd_lower_triangle_in_band<F>(
    n: usize,
    triplets: &[Triplet],
    make_backend: F,
) -> Result<bool, PsdCertificateError>
where
    F: FnOnce() -> Box<dyn SparseSymLinearSolverInterface>,
{
    let entries = sum_lower_triangle(n, triplets)?;
    let tolerance = psd_band(entries.values().copied());
    certify_summed(entries, tolerance, make_backend)
}

/// Validate `triplets` (lower triangle of an `n × n` matrix, finite) and sum
/// duplicates, refusing a sum that overflows.
fn sum_lower_triangle(
    n: usize,
    triplets: &[Triplet],
) -> Result<HashMap<(usize, usize), f64>, PsdCertificateError> {
    let mut entries = HashMap::<(usize, usize), f64>::with_capacity(triplets.len());
    for &Triplet { row, col, val } in triplets {
        if row >= n || col >= n {
            return Err(PsdCertificateError::OutOfBounds {
                row,
                col,
                dimension: n,
            });
        }
        if row < col {
            return Err(PsdCertificateError::UpperTriangle { row, col });
        }
        if !val.is_finite() {
            return Err(PsdCertificateError::NonFiniteValue { row, col });
        }
        let sum = entries.entry((row, col)).or_default();
        *sum += val;
        if !sum.is_finite() {
            return Err(PsdCertificateError::NonFiniteSum { row, col });
        }
    }
    Ok(entries)
}

/// The certificate proper, on summed entries and a validated tolerance.
fn certify_summed<F>(
    mut entries: HashMap<(usize, usize), f64>,
    tolerance: f64,
    make_backend: F,
) -> Result<bool, PsdCertificateError>
where
    F: FnOnce() -> Box<dyn SparseSymLinearSolverInterface>,
{
    entries.retain(|_, value| *value != 0.0);

    if entries.is_empty() {
        return Ok(true);
    }

    if entries.keys().all(|(row, col)| row == col) {
        return Ok(entries.values().all(|value| *value >= -tolerance));
    }

    let mut active = Vec::with_capacity(2 * entries.len());
    for &(row, col) in entries.keys() {
        active.push(row);
        active.push(col);
    }
    active.sort_unstable();
    active.dedup();

    let Ok(dim) = Index::try_from(active.len()) else {
        return Err(PsdCertificateError::DimensionTooLarge);
    };
    let active_index: HashMap<usize, usize> = active
        .iter()
        .copied()
        .enumerate()
        .map(|(compressed, original)| (original, compressed))
        .collect();
    let mut shifted = BTreeMap::<(usize, usize), f64>::new();
    for ((row, col), value) in entries {
        // Both coordinates were used to construct `active`, so these lookups
        // are total for every entry in the validated map.
        let compressed_row = active_index[&row];
        let compressed_col = active_index[&col];
        shifted.insert((compressed_row, compressed_col), value);
    }

    for (diagonal, &original) in active.iter().enumerate() {
        let value = shifted.entry((diagonal, diagonal)).or_default();
        *value += tolerance;
        if !value.is_finite() {
            return Err(PsdCertificateError::NonFiniteSum {
                row: original,
                col: original,
            });
        }
    }
    if Index::try_from(shifted.len()).is_err() {
        return Err(PsdCertificateError::DimensionTooLarge);
    }

    let mut rows = Vec::with_capacity(shifted.len());
    let mut cols = Vec::with_capacity(shifted.len());
    let mut values = Vec::with_capacity(shifted.len());
    for ((row, col), value) in shifted {
        rows.push(row as Index + 1);
        cols.push(col as Index + 1);
        values.push(value);
    }

    let backend = make_backend();
    if !backend.provides_inertia() {
        return Ok(false);
    }
    let Ok(factor) = Factorization::new(dim, rows, cols, values, backend) else {
        return Ok(false);
    };
    Ok(factor.number_of_neg_evals() == Some(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_linsol::{EMatrixFormat, ESymSolverStatus};
    use std::cell::RefCell;
    use std::rc::Rc;

    fn feral_backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(pounce_feral::FeralSolverInterface::serial())
    }

    /// `psd_band` (moved from the CLI's dispatch, gh#1147 PR review) clamps
    /// at `PSD_TOL` above unit scale and scales down below it.
    #[test]
    fn psd_band_scales_down_only() {
        assert_eq!(psd_band([1e6, -3.0]), PSD_TOL);
        assert_eq!(psd_band([1.0]), PSD_TOL);
        assert_eq!(psd_band([-1e-4, 5e-5]), PSD_TOL * 1e-4);
        assert_eq!(psd_band(std::iter::empty()), 0.0);
    }

    /// `certify_psd_lower_triangle_in_band` is `certify_psd_lower_triangle`
    /// at `psd_band` of the *summed* entries, on each branch: diagonal and
    /// coupled, inside and outside the band, with a duplicate whose sum sets
    /// the scale (`0.25 + 0.25`, so a per-triplet max would read half of it).
    #[test]
    fn the_in_band_certificate_is_the_certificate_at_the_band_of_the_sum() {
        let t = |r, c, v| Triplet::new(r, c, v);
        let cases: Vec<(Vec<Triplet>, bool)> = vec![
            // Diagonal, a negative eigenvalue inside / outside the band.
            (vec![t(0, 0, 0.25), t(0, 0, 0.25), t(1, 1, -1e-10)], true),
            (vec![t(0, 0, 0.25), t(0, 0, 0.25), t(1, 1, -1e-9)], false),
            // Coupled PSD and indefinite.
            (vec![t(0, 0, 2.0), t(1, 0, 1.0), t(1, 1, 2.0)], true),
            (vec![t(0, 0, 1.0), t(1, 0, 2.0), t(1, 1, 1.0)], false),
        ];
        for (trip, want) in cases {
            let summed = sum_lower_triangle(2, &trip).unwrap();
            let tol = psd_band(summed.values().copied());
            assert_eq!(
                certify_psd_lower_triangle_in_band(2, &trip, feral_backend),
                Ok(want),
                "{trip:?}"
            );
            assert_eq!(
                certify_psd_lower_triangle(2, &trip, tol, feral_backend),
                Ok(want),
                "{trip:?}"
            );
        }
        assert_eq!(
            sum_lower_triangle(1, &[t(0, 0, 0.25), t(0, 0, 0.25)]).unwrap()[&(0, 0)],
            0.5
        );
        // Same refusals as the certificate.
        assert_eq!(
            certify_psd_lower_triangle_in_band(2, &[t(0, 1, 1.0)], feral_backend),
            Err(PsdCertificateError::UpperTriangle { row: 0, col: 1 })
        );
        assert_eq!(
            certify_psd_lower_triangle_in_band(
                1,
                &[t(0, 0, f64::MAX), t(0, 0, f64::MAX)],
                feral_backend
            ),
            Err(PsdCertificateError::NonFiniteSum { row: 0, col: 0 })
        );
    }

    #[test]
    fn empty_and_diagonal_matrices_do_not_construct_a_backend() {
        let no_backend = || -> Box<dyn SparseSymLinearSolverInterface> {
            panic!("the diagonal fast path must not construct a backend")
        };
        assert!(certify_psd_lower_triangle(4, &[], 1e-9, no_backend).unwrap());

        let diagonal = [
            Triplet::new(0, 0, 3.0),
            Triplet::new(1, 1, -2.0),
            Triplet::new(1, 1, 2.0 - 5e-10),
        ];
        assert!(
            certify_psd_lower_triangle(2, &diagonal, 1e-9, || {
                panic!("the diagonal fast path must not construct a backend")
            })
            .unwrap()
        );
        assert!(
            !certify_psd_lower_triangle(2, &diagonal, 1e-11, || {
                panic!("the diagonal fast path must not construct a backend")
            })
            .unwrap()
        );
    }

    #[test]
    fn coupled_psd_and_indefinite_matrices_are_distinguished() {
        let singular_psd = [
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 1.0),
        ];
        assert!(certify_psd_lower_triangle(2, &singular_psd, 1e-9, feral_backend).unwrap());

        let indefinite = [
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 0, 0.25),
            Triplet::new(1, 1, -1e-4),
        ];
        assert!(!certify_psd_lower_triangle(2, &indefinite, 1e-9, feral_backend).unwrap());
    }

    #[test]
    fn coupled_tolerance_boundary_is_applied_to_the_spectrum() {
        const TOLERANCE: f64 = 1e-6;

        // [[1, 1 + delta], [1 + delta, 1]] has eigenvalues
        // -delta and 2 + delta.
        let matrix = |delta| {
            [
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 0, 1.0 + delta),
                Triplet::new(1, 1, 1.0),
            ]
        };
        assert!(
            certify_psd_lower_triangle(2, &matrix(0.5 * TOLERANCE), TOLERANCE, feral_backend)
                .unwrap()
        );
        assert!(
            !certify_psd_lower_triangle(2, &matrix(2.0 * TOLERANCE), TOLERANCE, feral_backend)
                .unwrap()
        );
    }

    #[test]
    fn duplicate_off_diagonals_are_summed_before_factorization() {
        // The off-diagonal duplicates cancel, leaving diag(1, 1).
        let matrix = [
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 0, 4.0),
            Triplet::new(1, 0, -4.0),
            Triplet::new(1, 1, 1.0),
        ];
        assert!(
            certify_psd_lower_triangle(2, &matrix, 0.0, || {
                panic!("cancelled coupling should use the diagonal fast path")
            })
            .unwrap()
        );
    }

    #[test]
    fn malformed_inputs_are_not_certified() {
        let cases = [
            (vec![Triplet::new(0, 1, 1.0)], 1e-9),
            (vec![Triplet::new(2, 0, 1.0)], 1e-9),
            (vec![Triplet::new(0, 0, f64::NAN)], 1e-9),
            (vec![Triplet::new(0, 0, 1.0)], -1.0),
            (vec![Triplet::new(0, 0, 1.0)], f64::INFINITY),
        ];
        for (matrix, tolerance) in cases {
            assert!(
                certify_psd_lower_triangle(2, &matrix, tolerance, || {
                    panic!("invalid input must not construct a backend")
                })
                .is_err()
            );
        }
        assert!(matches!(
            certify_psd_lower_triangle(2, &[Triplet::new(0, 1, 1.0)], 1e-9, || {
                panic!("upper-triangle input must not construct a backend")
            }),
            Err(PsdCertificateError::UpperTriangle { row: 0, col: 1 })
        ));
    }

    struct MockBackend {
        values: Vec<f64>,
        solve_status: ESymSolverStatus,
        inertia: Option<Index>,
        panic_on_initialize: bool,
    }

    impl SparseSymLinearSolverInterface for MockBackend {
        fn initialize_structure(
            &mut self,
            _dim: Index,
            nonzeros: Index,
            _ia: &[Index],
            _ja: &[Index],
        ) -> ESymSolverStatus {
            assert!(
                !self.panic_on_initialize,
                "a backend without inertia must not be initialized"
            );
            self.values.resize(nonzeros as usize, 0.0);
            ESymSolverStatus::Success
        }

        fn values_array_mut(&mut self) -> &mut [f64] {
            &mut self.values
        }

        fn multi_solve(
            &mut self,
            _new_matrix: bool,
            _ia: &[Index],
            _ja: &[Index],
            _nrhs: Index,
            _rhs_vals: &mut [f64],
            _check_neg_evals: bool,
            _number_of_neg_evals: Index,
        ) -> ESymSolverStatus {
            self.solve_status
        }

        fn number_of_neg_evals(&self) -> Index {
            self.inertia.unwrap_or_default()
        }

        fn increase_quality(&mut self) -> bool {
            false
        }

        fn provides_inertia(&self) -> bool {
            self.inertia.is_some()
        }

        fn matrix_format(&self) -> EMatrixFormat {
            EMatrixFormat::TripletFormat
        }
    }

    #[derive(Debug, Default)]
    struct BackendRecord {
        dim: Option<Index>,
        rows: Vec<Index>,
        cols: Vec<Index>,
        values: Vec<f64>,
    }

    struct RecordingBackend {
        values: Vec<f64>,
        record: Rc<RefCell<BackendRecord>>,
    }

    impl SparseSymLinearSolverInterface for RecordingBackend {
        fn initialize_structure(
            &mut self,
            dim: Index,
            nonzeros: Index,
            ia: &[Index],
            ja: &[Index],
        ) -> ESymSolverStatus {
            self.values.resize(nonzeros as usize, 0.0);
            let mut record = self.record.borrow_mut();
            record.dim = Some(dim);
            record.rows = ia.to_vec();
            record.cols = ja.to_vec();
            ESymSolverStatus::Success
        }

        fn values_array_mut(&mut self) -> &mut [f64] {
            &mut self.values
        }

        fn multi_solve(
            &mut self,
            _new_matrix: bool,
            _ia: &[Index],
            _ja: &[Index],
            _nrhs: Index,
            _rhs_vals: &mut [f64],
            _check_neg_evals: bool,
            _number_of_neg_evals: Index,
        ) -> ESymSolverStatus {
            self.record.borrow_mut().values = self.values.clone();
            ESymSolverStatus::Success
        }

        fn number_of_neg_evals(&self) -> Index {
            0
        }

        fn increase_quality(&mut self) -> bool {
            false
        }

        fn provides_inertia(&self) -> bool {
            true
        }

        fn matrix_format(&self) -> EMatrixFormat {
            EMatrixFormat::TripletFormat
        }
    }

    #[test]
    fn coupled_matrix_is_normalized_compressed_shifted_and_one_based() {
        let matrix = [
            Triplet::new(4, 4, 3.0),
            Triplet::new(4, 1, 0.75),
            Triplet::new(7, 7, 4.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(4, 1, 0.25),
        ];
        let record = Rc::new(RefCell::new(BackendRecord::default()));
        let backend_record = Rc::clone(&record);

        assert!(
            certify_psd_lower_triangle(8, &matrix, 0.5, move || {
                Box::new(RecordingBackend {
                    values: Vec::new(),
                    record: backend_record,
                })
            })
            .unwrap()
        );

        let record = record.borrow();
        assert_eq!(record.dim, Some(3));
        assert_eq!(record.rows, [1, 2, 2, 3]);
        assert_eq!(record.cols, [1, 1, 2, 3]);
        assert_eq!(record.values, [2.5, 1.0, 3.5, 4.5]);
    }

    fn coupled_matrix() -> [Triplet; 3] {
        [
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 0, 0.1),
            Triplet::new(1, 1, 2.0),
        ]
    }

    #[test]
    fn missing_inertia_and_factorization_failures_are_inconclusive() {
        let make_mock = |solve_status, inertia| {
            move || -> Box<dyn SparseSymLinearSolverInterface> {
                Box::new(MockBackend {
                    values: vec![],
                    solve_status,
                    inertia,
                    panic_on_initialize: inertia.is_none(),
                })
            }
        };
        assert!(
            !certify_psd_lower_triangle(
                2,
                &coupled_matrix(),
                1e-9,
                make_mock(ESymSolverStatus::Success, None),
            )
            .unwrap()
        );
        assert!(
            !certify_psd_lower_triangle(
                2,
                &coupled_matrix(),
                1e-9,
                make_mock(ESymSolverStatus::Singular, Some(0)),
            )
            .unwrap()
        );
    }
}
