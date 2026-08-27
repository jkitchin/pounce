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

    /// Whether a blocked `multi_solve` of `nrhs` columns returns
    /// **bit-identical** results to `nrhs` separate `nrhs = 1` calls
    /// against the same factor.
    ///
    /// Backends that block the triangular substitution across columns
    /// reassociate the floating-point sums, so their batched answer is
    /// tolerance-equal but not bit-equal to the per-column one. That is
    /// fine for a caller that only wants *a* solution, and not fine for
    /// a caller batching purely to save time inside an iteration whose
    /// trajectory must not move: on a nonconvex problem the perturbation
    /// can select a different local optimum. `pooling_rt2stp` under MA57
    /// does exactly that (gh#729), landing on an objective 25% worse
    /// while still reporting `Optimal Solution Found`.
    ///
    /// Defaults to `false` — the conservative answer, so a new backend
    /// has to opt in deliberately rather than inherit a trajectory
    /// change by omission. This gates only opportunistic batching;
    /// callers that batch for their own reasons (`pounce-sensitivity`'s
    /// `jacrev` backward, where each cotangent is an independent
    /// question) do not consult it.
    /// The answer is allowed to depend on `nrhs`: a backend may run a
    /// bit-identical rank-1 cascade for narrow blocks and switch to a
    /// reassociating BLAS-3 panel kernel once the block is wide enough
    /// to pay for it. feral does exactly that.
    ///
    /// It is not guaranteed that such a narrow window exists. MA57 has
    /// none — measured, `ma57cd` reassociates from the second column on,
    /// below its own `ICNTL(13)` level-3 BLAS threshold — so
    /// `pounce-hsl` returns `false` at every width and spells that out
    /// rather than inheriting this default silently (gh#810). A backend
    /// author reaching for an `nrhs`-capped override should measure
    /// first; the guards to copy are
    /// `Ma57SolverInterface::multi_solve_reassociates_from_two_columns_up`
    /// and feral's
    /// `multi_solve_bitwise_matches_single_solve_at_the_documented_ceiling`.
    fn multi_solve_matches_single_solve(&self, _nrhs: usize) -> bool {
        false
    }

    /// Required matrix layout. Caller marshals data into this format.
    fn matrix_format(&self) -> EMatrixFormat;

    /// Whether [`Self::determine_dependent_rows`] is supported.
    fn provides_degeneracy_detection(&self) -> bool {
        false
    }

    /// Find the linearly dependent rows of a constraint Jacobian `J`
    /// (the Ipopt-style degeneracy probe). `J` is `n_rows × n_cols`,
    /// supplied as a **1-based triplet** `(irn, jcn, vals)`; on
    /// success `c_deps` is filled with the **0-based** indices of a
    /// set of rows whose removal leaves `J` full row rank (each
    /// dropped row is a linear combination of the retained ones).
    ///
    /// Callers must check [`Self::provides_degeneracy_detection`]
    /// first. The default returns `FatalError`, matching upstream's
    /// "not supported" default; backends that implement this set
    /// `provides_degeneracy_detection() -> true`.
    fn determine_dependent_rows(
        &mut self,
        _n_rows: Index,
        _n_cols: Index,
        _irn: &[Index],
        _jcn: &[Index],
        _vals: &[Number],
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
