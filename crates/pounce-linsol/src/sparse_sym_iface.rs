//! Low-level sparse-symmetric backend interface — port of
//! `IpSparseSymLinearSolverInterface.hpp`.
//!
//! Concrete implementors:
//! * `pounce_hsl::Ma57SolverInterface` (v1.0).
//! * Future: MUMPS, FERAL.

use crate::status::ESymSolverStatus;
use pounce_common::types::{Index, Number};

/// Snapshot of the most recent LDLᵀ factor's sparsity pattern (and
/// optionally values) plus the fill-reducing permutation. Backends
/// produce this on demand from [`SparseSymLinearSolverInterface::factor_pattern`]
/// — it is purely diagnostic and is not part of the solve / refine
/// hot path.
///
/// All `irn` / `jcn` indices are **1-based** in *permuted* coordinates
/// (i.e. they reference the matrix `Pᵀ K P` that the backend actually
/// factored, not the original-variable ordering). The `perm` array
/// closes the loop: `perm[k] = original_row` for the k-th permuted
/// row, so a consumer can render the L pattern in either coordinate
/// system. `perm` is **0-based** to keep the array directly indexable.
///
/// Only the **strict lower triangle** of L is populated — the unit
/// diagonal is implicit (`L_ii = 1`).
#[derive(Debug, Clone)]
pub struct FactorPattern {
    /// Matrix dimension (rows = cols).
    pub n: usize,
    /// Fill-reducing permutation, 0-based, length `n`. `perm[k]` is
    /// the original-variable row that landed at permuted-row `k`.
    pub perm: Vec<usize>,
    /// Row indices of L's strict-lower nonzeros, 1-based, permuted
    /// coordinates.
    pub l_irn: Vec<Index>,
    /// Column indices of L's strict-lower nonzeros, 1-based, permuted
    /// coordinates. Same length as `l_irn`.
    pub l_jcn: Vec<Index>,
    /// Optional numerical values aligned with `l_irn` / `l_jcn`. `None`
    /// when only the pattern was requested.
    pub l_vals: Option<Vec<Number>>,
}

/// Sparse matrix format that a backend wants its triplet/CSR data in.
/// Mirrors `SparseSymLinearSolverInterface::EMatrixFormat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EMatrixFormat {
    /// Triplet (COO) of the lower triangle, 1-based indices
    /// (MA27 / MA57 / MUMPS convention).
    TripletFormat,
    /// CSR of the upper triangle, 0-based indices.
    CsrFormat0Offset,
    /// CSR of the upper triangle, 1-based indices.
    CsrFormat1Offset,
    /// Full CSR (lower + upper), 0-based indices.
    CsrFullFormat0Offset,
    /// Full CSR (lower + upper), 1-based indices.
    CsrFullFormat1Offset,
}

/// Backend-side trait. The lifecycle mirrors upstream's narrative
/// comment in `IpSparseSymLinearSolverInterface.hpp`:
///
/// 1. caller asks [`Self::matrix_format`].
/// 2. caller calls [`Self::initialize_structure`] once with `(ia, ja)`.
/// 3. caller takes the values pointer from
///    [`Self::values_array_mut`], fills it.
/// 4. caller calls [`Self::multi_solve`] with `new_matrix=true` for
///    each new value pattern.
/// 5. caller may query [`Self::number_of_neg_evals`] /
///    [`Self::increase_quality`] between solves.
///
/// `new_matrix=false` requests a back-substitution against the
/// existing factorization.
pub trait SparseSymLinearSolverInterface {
    /// Initialize backend internal structures for a matrix of given
    /// dimension and pattern.
    fn initialize_structure(
        &mut self,
        dim: Index,
        nonzeros: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> ESymSolverStatus;

    /// Slice into which the caller writes the matrix nonzeros (in the
    /// same order as `ja` from [`Self::initialize_structure`]).
    fn values_array_mut(&mut self) -> &mut [Number];

    /// Factor (if `new_matrix`) and back-substitute against `nrhs`
    /// right-hand sides packed in `rhs_vals` (length `nrhs * dim`).
    /// Solutions overwrite `rhs_vals`.
    #[allow(clippy::too_many_arguments)]
    fn multi_solve(
        &mut self,
        new_matrix: bool,
        ia: &[Index],
        ja: &[Index],
        nrhs: Index,
        rhs_vals: &mut [Number],
        check_neg_evals: bool,
        number_of_neg_evals: Index,
    ) -> ESymSolverStatus;

    /// Number of negative eigenvalues found in the most recent
    /// factorization. Caller must check [`Self::provides_inertia`]
    /// first.
    fn number_of_neg_evals(&self) -> Index;

    /// Ask the backend to use a more accurate (but slower) pivot
    /// strategy on the next solve. Returns `false` if the maximum
    /// quality is already reached.
    fn increase_quality(&mut self) -> bool;

    /// Whether this backend reports the number of negative
    /// eigenvalues post-factor.
    fn provides_inertia(&self) -> bool;

    /// Required matrix layout. Caller marshals data into this format.
    fn matrix_format(&self) -> EMatrixFormat;

    /// Whether [`Self::determine_dependent_rows`] is supported.
    fn provides_degeneracy_detection(&self) -> bool {
        false
    }

    /// Find linearly dependent rows — used by Ipopt's degeneracy
    /// probe. Default is `FatalError` matching upstream's default
    /// implementation.
    fn determine_dependent_rows(
        &mut self,
        _ia: &[Index],
        _ja: &[Index],
        _c_deps: &mut Vec<Index>,
    ) -> ESymSolverStatus {
        ESymSolverStatus::FatalError
    }

    /// Snapshot of the most recent factor's L pattern and permutation.
    /// Backends that expose their factor data structures (e.g. feral)
    /// return `Some(_)`; backends that don't (e.g. MA57, which keeps
    /// its factors inside opaque Fortran work arrays) return `None`.
    /// Diagnostic-only — consumed by the `--dump kkt:*+L` path. Set
    /// `want_values=true` to populate [`FactorPattern::l_vals`].
    fn factor_pattern(&self, _want_values: bool) -> Option<FactorPattern> {
        None
    }
}
