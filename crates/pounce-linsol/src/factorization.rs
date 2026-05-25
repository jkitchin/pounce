//! Public factor-once / solve-many handle.
//!
//! [`Factorization`] is the user-facing value type for "I have a sparse
//! symmetric matrix, factor it once, then solve against the factor
//! repeatedly." It wraps an arbitrary [`SparseSymLinearSolverInterface`]
//! backend (feral, MA57, etc.) and the [`TSymLinearSolver`] driver that
//! handles triplet→CSR conversion and the `CallAgain` retry loop for
//! backends that may grow their factor arrays.
//!
//! Two operations preserve the cached factor:
//!
//! * [`Factorization::solve`] — back-substitute against the current
//!   factor. Cheap; reuses the LDLᵀ factors.
//! * [`Factorization::refactor`] — supply new numeric values for the
//!   same sparsity pattern; backend reuses its symbolic factor (AMD
//!   ordering, elimination tree, pattern cache) and only redoes the
//!   numeric work.
//!
//! See the `examples/shift_invert.rs` example for a worked use of
//! factor-once / many-RHS for a shift-invert eigenvalue probe.
//!
//! # Example
//!
//! ```
//! use pounce_linsol::{Factorization, FactorizationError};
//! # // The example uses a dummy in-tree backend; real callers supply
//! # // feral or MA57.
//! # struct Dummy; // placeholder; full example needs a backend
//! ```

use crate::error::FactorizationError;
use crate::sparse_sym_iface::SparseSymLinearSolverInterface;
use crate::t_sym_solver::TSymLinearSolver;
use pounce_common::types::{Index, Number};

/// Value-typed handle holding a sparse symmetric factorization.
///
/// Construction (via [`Factorization::new`]) performs the symbolic +
/// numeric factor; subsequent [`Factorization::solve`] calls are pure
/// back-substitution. [`Factorization::refactor`] replaces the numeric
/// values without redoing the symbolic work.
///
/// The matrix is supplied in **triplet (COO) format with 1-based
/// indices over the lower triangle** — the universal denominator the
/// trait expects. Backends that prefer CSR are fed via the standard
/// `TripletToCsrConverter` inside the wrapper.
pub struct Factorization {
    inner: TSymLinearSolver,
    dim: Index,
    nnz: Index,
    values: Vec<Number>,
    inertia_known: bool,
}

impl std::fmt::Debug for Factorization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Factorization")
            .field("dim", &self.dim)
            .field("nnz", &self.nnz)
            .field("inertia_known", &self.inertia_known)
            .finish_non_exhaustive()
    }
}

impl Factorization {
    /// Factor a new matrix. Pattern (`airn`, `ajcn`) and `values` are
    /// the lower-triangle triplet of `A`, 1-based indices, length
    /// `nnz` each. `backend` is any implementor (feral, MA57, …).
    ///
    /// Performs both the symbolic and numeric factorization. Subsequent
    /// [`Self::solve`] calls are back-substitution only;
    /// [`Self::refactor`] redoes the numeric work but reuses the
    /// symbolic factor.
    ///
    /// # Errors
    ///
    /// * [`FactorizationError::Singular`] — the supplied matrix is
    ///   numerically singular.
    /// * [`FactorizationError::FatalError`] — unrecoverable backend
    ///   error.
    ///
    /// # Panics
    ///
    /// Panics if `airn.len() != ajcn.len()` or if `values.len() !=
    /// airn.len()`.
    pub fn new(
        dim: Index,
        airn: Vec<Index>,
        ajcn: Vec<Index>,
        values: Vec<Number>,
        backend: Box<dyn SparseSymLinearSolverInterface>,
    ) -> Result<Self, FactorizationError> {
        assert_eq!(airn.len(), ajcn.len(), "airn and ajcn must have same length");
        assert_eq!(values.len(), airn.len(), "values must match nnz");
        let nnz = airn.len() as Index;
        let mut inner = TSymLinearSolver::new(backend, None, false);
        FactorizationError::from_status(inner.initialize_structure(dim, &airn, &ajcn))?;

        let mut me = Self {
            inner,
            dim,
            nnz,
            values,
            inertia_known: false,
        };

        // Initial factor. We issue a no-op back-solve (nrhs=1 with a
        // zero RHS we then discard) because the trait does not expose
        // a factor-only entry point — every multi_solve also runs back
        // substitution. The cost is one triangular solve per
        // construction, which is negligible relative to the factor.
        me.do_factor()?;
        Ok(me)
    }

    /// Back-substitute against the cached factor. `rhs` packs `nrhs`
    /// columns (each length `dim`) in column-major layout; solutions
    /// overwrite `rhs` in place.
    ///
    /// # Errors
    ///
    /// [`FactorizationError::FatalError`] — backend solve failed.
    ///
    /// # Panics
    ///
    /// Panics if `rhs.len() != dim * nrhs`.
    pub fn solve(&mut self, rhs: &mut [Number], nrhs: usize) -> Result<(), FactorizationError> {
        assert_eq!(
            rhs.len(),
            self.dim as usize * nrhs,
            "rhs length must equal dim * nrhs"
        );
        let status = self.inner.multi_solve(
            &self.values,
            false, // new_matrix = false: pure back-substitution
            nrhs as Index,
            rhs,
            false,
            0,
        );
        FactorizationError::from_status(status)
    }

    /// Convenience for the common `nrhs=1` case. Identical to
    /// `solve(rhs, 1)`.
    pub fn solve_one(&mut self, rhs: &mut [Number]) -> Result<(), FactorizationError> {
        self.solve(rhs, 1)
    }

    /// Replace the numeric values and refactor. Pattern is unchanged;
    /// the backend reuses its symbolic factor / AMD ordering.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`].
    ///
    /// # Panics
    ///
    /// Panics if `new_values.len() != nnz`.
    pub fn refactor(&mut self, new_values: &[Number]) -> Result<(), FactorizationError> {
        assert_eq!(
            new_values.len(),
            self.nnz as usize,
            "new_values length must equal nnz",
        );
        self.values.copy_from_slice(new_values);
        self.inertia_known = false;
        self.do_factor()
    }

    /// Number of negative eigenvalues from the most recent factor, if
    /// the backend reports inertia. `None` otherwise.
    pub fn number_of_neg_evals(&self) -> Option<Index> {
        use crate::sym_solver::SymLinearSolver;
        if self.inertia_known && self.inner.provides_inertia() {
            Some(self.inner.number_of_neg_evals())
        } else {
            None
        }
    }

    /// Dimension `n` of the factored `n × n` matrix.
    pub fn dim(&self) -> Index {
        self.dim
    }

    /// Number of nonzeros in the triplet pattern.
    pub fn nnz(&self) -> Index {
        self.nnz
    }

    /// Internal helper: issue a factor (and discard the back-solve).
    fn do_factor(&mut self) -> Result<(), FactorizationError> {
        let mut dummy_rhs = vec![0.0; self.dim as usize];
        let status = self.inner.multi_solve(
            &self.values,
            true, // new_matrix = true: factor now
            1,
            &mut dummy_rhs,
            false,
            0,
        );
        FactorizationError::from_status(status)?;
        self.inertia_known = true;
        Ok(())
    }
}

// Compile-time assertion that a Factorization is Send — backends are
// `Box<dyn SparseSymLinearSolverInterface>` which doesn't currently
// require Send, so this is intentionally not asserted at the type
// level. Users threading Factorizations across threads must ensure
// their backend is Send (feral and MA57 both are in practice).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_sym_iface::EMatrixFormat;
    use crate::status::ESymSolverStatus;

    /// Minimal in-test backend that solves a dense symmetric system via
    /// LU. Lets us exercise the Factorization API without depending on
    /// feral or MA57 in this crate's test suite.
    struct DenseLuBackend {
        dim: usize,
        nnz: usize,
        rows: Vec<Index>, // 1-based
        cols: Vec<Index>, // 1-based
        values: Vec<Number>,
        // Cached LU factor of the dense matrix; rebuilt on each factor().
        factor: Option<DenseLu>,
    }

    struct DenseLu {
        a: Vec<Vec<f64>>, // L\U combined (Doolittle)
        perm: Vec<usize>,
        neg_evals: Index,
    }

    impl DenseLuBackend {
        fn new() -> Self {
            Self {
                dim: 0,
                nnz: 0,
                rows: Vec::new(),
                cols: Vec::new(),
                values: Vec::new(),
                factor: None,
            }
        }

        fn assemble_dense(&self) -> Vec<Vec<f64>> {
            let n = self.dim;
            let mut a = vec![vec![0.0; n]; n];
            for k in 0..self.nnz {
                let i = (self.rows[k] - 1) as usize;
                let j = (self.cols[k] - 1) as usize;
                a[i][j] += self.values[k];
                if i != j {
                    a[j][i] += self.values[k];
                }
            }
            a
        }

        fn factor_dense(&mut self) -> ESymSolverStatus {
            let n = self.dim;
            let mut a = self.assemble_dense();
            let mut perm: Vec<usize> = (0..n).collect();
            // Partial-pivoted LU.
            for k in 0..n {
                // Pivot.
                let mut p = k;
                let mut maxv = a[perm[k]][k].abs();
                for i in (k + 1)..n {
                    let v = a[perm[i]][k].abs();
                    if v > maxv {
                        maxv = v;
                        p = i;
                    }
                }
                if maxv < 1e-300 {
                    return ESymSolverStatus::Singular;
                }
                perm.swap(k, p);
                let pk = perm[k];
                for &pi in &perm[(k + 1)..n] {
                    let factor = a[pi][k] / a[pk][k];
                    a[pi][k] = factor;
                    #[allow(clippy::needless_range_loop)]
                    for j in (k + 1)..n {
                        a[pi][j] -= factor * a[pk][j];
                    }
                }
            }
            // Count negative diagonal entries of U as a stand-in for
            // inertia (correct for symmetric matrices with no pivoting,
            // which this backend does pivot — so this is only an
            // approximation, used to exercise the inertia code path in
            // tests).
            let mut neg = 0;
            for k in 0..n {
                if a[perm[k]][k] < 0.0 {
                    neg += 1;
                }
            }
            self.factor = Some(DenseLu {
                a,
                perm,
                neg_evals: neg as Index,
            });
            ESymSolverStatus::Success
        }

        fn solve_one(&self, b: &mut [f64]) {
            let factor = self.factor.as_ref().unwrap();
            let n = self.dim;
            // Permute.
            let mut x: Vec<f64> = factor.perm.iter().map(|&p| b[p]).collect();
            // Forward substitution (unit-lower).
            for i in 0..n {
                let pi = factor.perm[i];
                for j in 0..i {
                    x[i] -= factor.a[pi][j] * x[j];
                }
            }
            // Back substitution (upper).
            for i in (0..n).rev() {
                let pi = factor.perm[i];
                for j in (i + 1)..n {
                    x[i] -= factor.a[pi][j] * x[j];
                }
                x[i] /= factor.a[pi][i];
            }
            b.copy_from_slice(&x);
        }
    }

    impl SparseSymLinearSolverInterface for DenseLuBackend {
        fn initialize_structure(
            &mut self,
            dim: Index,
            nonzeros: Index,
            ia: &[Index],
            ja: &[Index],
        ) -> ESymSolverStatus {
            self.dim = dim as usize;
            self.nnz = nonzeros as usize;
            self.rows = ia.to_vec();
            self.cols = ja.to_vec();
            self.values = vec![0.0; self.nnz];
            ESymSolverStatus::Success
        }

        fn values_array_mut(&mut self) -> &mut [Number] {
            &mut self.values
        }

        fn multi_solve(
            &mut self,
            new_matrix: bool,
            _ia: &[Index],
            _ja: &[Index],
            nrhs: Index,
            rhs_vals: &mut [Number],
            check_neg_evals: bool,
            number_of_neg_evals: Index,
        ) -> ESymSolverStatus {
            if new_matrix {
                let s = self.factor_dense();
                if s != ESymSolverStatus::Success {
                    return s;
                }
                if check_neg_evals {
                    let actual = self.factor.as_ref().unwrap().neg_evals;
                    if actual != number_of_neg_evals {
                        return ESymSolverStatus::WrongInertia;
                    }
                }
            }
            let n = self.dim;
            for k in 0..nrhs as usize {
                let base = k * n;
                self.solve_one(&mut rhs_vals[base..base + n]);
            }
            ESymSolverStatus::Success
        }

        fn number_of_neg_evals(&self) -> Index {
            self.factor.as_ref().map(|f| f.neg_evals).unwrap_or(0)
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

    /// SPD 2x2: `[[2,1],[1,3]]`. Lower-triangle 1-based triplets.
    /// Solving against (3, 4) gives (1, 1).
    #[test]
    fn factors_spd_2x2_and_solves_one_rhs() {
        let airn = vec![1, 2, 2];
        let ajcn = vec![1, 1, 2];
        let values = vec![2.0, 1.0, 3.0];
        let mut f =
            Factorization::new(2, airn, ajcn, values, Box::new(DenseLuBackend::new())).unwrap();
        let mut rhs = vec![3.0, 4.0];
        f.solve_one(&mut rhs).unwrap();
        assert!((rhs[0] - 1.0).abs() < 1e-12);
        assert!((rhs[1] - 1.0).abs() < 1e-12);
    }

    /// Same SPD 2x2, multiple RHS packed; results match single-RHS
    /// solves done individually.
    #[test]
    fn packed_multi_rhs_matches_one_at_a_time() {
        let airn = vec![1, 2, 2];
        let ajcn = vec![1, 1, 2];
        let values = vec![2.0, 1.0, 3.0];
        let backend1 = Box::new(DenseLuBackend::new());
        let backend2 = Box::new(DenseLuBackend::new());
        let mut f1 = Factorization::new(2, airn.clone(), ajcn.clone(), values.clone(), backend1)
            .unwrap();
        let mut f2 =
            Factorization::new(2, airn, ajcn, values, backend2).unwrap();

        // Packed 3-RHS solve.
        let mut packed = vec![
            3.0, 4.0, // col 0 → expect (1, 1)
            5.0, 5.0, // col 1
            2.0, 6.0, // col 2
        ];
        f1.solve(&mut packed, 3).unwrap();

        // One-at-a-time for the same RHS columns.
        let mut col0 = vec![3.0, 4.0];
        let mut col1 = vec![5.0, 5.0];
        let mut col2 = vec![2.0, 6.0];
        f2.solve_one(&mut col0).unwrap();
        f2.solve_one(&mut col1).unwrap();
        f2.solve_one(&mut col2).unwrap();

        for (i, &v) in col0.iter().enumerate() {
            assert!((packed[i] - v).abs() < 1e-12, "col0 mismatch at {i}");
        }
        for (i, &v) in col1.iter().enumerate() {
            assert!((packed[2 + i] - v).abs() < 1e-12, "col1 mismatch at {i}");
        }
        for (i, &v) in col2.iter().enumerate() {
            assert!((packed[4 + i] - v).abs() < 1e-12, "col2 mismatch at {i}");
        }
    }

    /// Refactor with perturbed values; residual against the perturbed
    /// system is small.
    #[test]
    fn refactor_yields_correct_solution_for_new_values() {
        let airn = vec![1, 2, 2];
        let ajcn = vec![1, 1, 2];
        let mut f = Factorization::new(
            2,
            airn,
            ajcn,
            vec![2.0, 1.0, 3.0],
            Box::new(DenseLuBackend::new()),
        )
        .unwrap();

        // Perturb to `[[4, 1], [1, 5]]`.
        f.refactor(&[4.0, 1.0, 5.0]).unwrap();
        let mut rhs = vec![5.0, 6.0]; // expect ~ (19/19, 19/19) = (1, 1)
        f.solve_one(&mut rhs).unwrap();
        // Check residual: A x - b where A = [[4,1],[1,5]], b = (5, 6).
        let r0 = 4.0 * rhs[0] + rhs[1] - 5.0;
        let r1 = rhs[0] + 5.0 * rhs[1] - 6.0;
        assert!(r0.abs() < 1e-10);
        assert!(r1.abs() < 1e-10);
    }

    /// Singular matrix → Singular error.
    #[test]
    fn singular_matrix_returns_singular_error() {
        // `[[0, 1], [1, 0]]` is symmetric indefinite but the LU with
        // partial pivoting on it succeeds (it pivots the off-diagonal
        // up). Use a genuinely singular matrix instead: `[[1,1],[1,1]]`.
        let airn = vec![1, 2, 2];
        let ajcn = vec![1, 1, 2];
        let err = Factorization::new(
            2,
            airn,
            ajcn,
            vec![1.0, 1.0, 1.0],
            Box::new(DenseLuBackend::new()),
        )
        .unwrap_err();
        assert_eq!(err, FactorizationError::Singular);
    }

    /// `solve_one` and `solve(.., 1)` produce identical results.
    #[test]
    fn solve_one_matches_solve_with_nrhs_one() {
        let airn = vec![1, 2, 2];
        let ajcn = vec![1, 1, 2];
        let values = vec![2.0, 1.0, 3.0];
        let mut f1 = Factorization::new(
            2,
            airn.clone(),
            ajcn.clone(),
            values.clone(),
            Box::new(DenseLuBackend::new()),
        )
        .unwrap();
        let mut f2 =
            Factorization::new(2, airn, ajcn, values, Box::new(DenseLuBackend::new())).unwrap();

        let mut rhs1 = vec![3.0, 4.0];
        let mut rhs2 = vec![3.0, 4.0];
        f1.solve_one(&mut rhs1).unwrap();
        f2.solve(&mut rhs2, 1).unwrap();
        assert_eq!(rhs1, rhs2);
    }

    /// Inertia is reported after construction; backend says it
    /// provides inertia.
    #[test]
    fn inertia_is_reported_when_backend_provides_it() {
        let airn = vec![1, 2, 2];
        let ajcn = vec![1, 1, 2];
        let f = Factorization::new(
            2,
            airn,
            ajcn,
            vec![2.0, 1.0, 3.0], // SPD, so 0 negative eigenvalues
            Box::new(DenseLuBackend::new()),
        )
        .unwrap();
        assert_eq!(f.number_of_neg_evals(), Some(0));
        assert_eq!(f.dim(), 2);
        assert_eq!(f.nnz(), 3);
    }
}
