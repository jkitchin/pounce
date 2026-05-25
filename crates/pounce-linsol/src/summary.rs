//! Linear-solver post-mortem summary — shared shape that concrete
//! backends populate and downstream report builders consume.
//!
//! Kept dep-light on purpose: no serde derives here. The CLI's solve
//! report crate owns the serializable mirror.

/// Aggregate stats accumulated over the lifetime of one linear-solver
/// instance. All fields default to zero / `None` so a backend that
/// declines to populate them still produces a valid summary.
#[derive(Debug, Clone, Default)]
pub struct LinearSolverSummary {
    /// Short identifier of the backend that produced this summary:
    /// `"feral"`, `"ma57"`, etc. Empty for the `Default` value.
    pub solver_name: String,
    /// Number of `factor()` calls completed (including those that
    /// reused the cached symbolic factorisation).
    pub n_factors: u64,
    /// Of `n_factors`, how many reused the previous symbolic
    /// factorisation (sparsity pattern unchanged). Healthy IPM workloads
    /// expect this to dominate after the first iter.
    pub n_pattern_reuse: u64,
    /// Of `n_factors`, how many required a fresh symbolic factorisation
    /// (sparsity pattern changed). Inverse of `n_pattern_reuse` modulo
    /// the very first factor.
    pub n_pattern_changes: u64,
    /// Maximum `nnz(L) / nnz(A)` observed across factors. Values much
    /// greater than ~10 on KKT-style systems indicate ordering trouble.
    pub max_fill_ratio: Option<f64>,
    /// Minimum `|pivot|` observed across factors. Approaches the
    /// working-precision floor when the matrix is near-singular.
    pub min_abs_pivot: Option<f64>,
    /// Maximum `|pivot|` observed across factors.
    pub max_abs_pivot: Option<f64>,
    /// Inertia of the final factorisation as `(positive, negative, zero)`.
    pub last_inertia: Option<(usize, usize, usize)>,
    /// `nnz(A)` of the final factorisation's matrix.
    pub last_nnz_a: Option<usize>,
    /// `nnz(L)` of the final factorisation.
    pub last_nnz_l: Option<usize>,
}

impl LinearSolverSummary {
    /// Returns `true` if the summary carries no signal beyond the
    /// solver name — useful for "did we collect anything?" checks.
    pub fn is_empty(&self) -> bool {
        self.n_factors == 0
    }
}
