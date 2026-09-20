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

    // ---- Work and pivoting totals (structured-KKT Phase 0a) --------
    //
    // Summed over every recorded factorization, so a caller divides by
    // the iteration count for a per-iteration figure. Regularization
    // retries are factorizations too and are included on purpose: they
    // are part of what an iteration costs.
    /// Wall-clock seconds spent inside the backend's numeric factor
    /// call, summed. Measured with a monotonic clock around every
    /// factor, independent of the `timing_statistics` option. On the
    /// Schur path this is the whole block factorization: the eliminated
    /// block plus forming and factoring `S`.
    pub total_factor_secs: f64,
    /// Sum of the backend's a-priori work proxy (FERAL: `Σ ncol·nrow²`
    /// over supernodes). A proxy for comparing orderings and sizes, not
    /// a time; see `feral::WorkEstimate`.
    pub total_factor_flops: f64,
    /// Delayed-column entries, summed. A column delayed up `k` levels
    /// of the elimination tree counts `k` times, which is what makes
    /// this a measure of the extra work delayed pivoting causes.
    pub total_delayed_cols: u64,
    /// 2×2 pivot blocks, summed.
    pub total_two_by_two: u64,
    /// Pivots statically perturbed to the pivot floor, summed (MUMPS
    /// `INFO(25)` analogue).
    pub total_n_tiny: u64,

    // ---- The final factorization, in more detail --------------------
    /// Work proxy of the final factorization.
    pub last_factor_flops: Option<f64>,
    /// Predicted peak bytes (factor plus transient contribution
    /// blocks) of the final factorization's symbolic analysis.
    pub last_peak_bytes: Option<usize>,
    /// Supernodes in the final factorization's elimination tree.
    pub last_n_supernodes: Option<usize>,
    /// Rows in the final factorization's largest frontal matrix.
    pub last_max_front_rows: Option<usize>,
    /// Delayed-column entries in the final factorization.
    pub last_delayed_cols: Option<usize>,
    /// 2×2 pivot blocks in the final factorization.
    pub last_two_by_two: Option<usize>,
    /// Statically perturbed pivots in the final factorization.
    pub last_n_tiny: Option<usize>,
    /// Concrete fill-reducing ordering the final factorization used
    /// (never `auto`: the method the dispatcher resolved to).
    pub last_ordering: Option<String>,
    /// Ordering-stage preprocessing the final factorization applied
    /// (`none`, or `ldlt_compress`: the matching-compressed ordering that
    /// keeps each 2x2 KKT pivot pair together).
    pub last_ordering_preprocess: Option<String>,

    /// Present when the KKT solve went through the block-parallel path
    /// (a declared block-diagonal-plus-border structure) and it actually
    /// factored; absent when no structure was given or the path fell back.
    pub blocks: Option<BlockSummary>,

    /// Present when the KKT solve went through the block-triangular /
    /// Schur path (`set_kkt_schur_block`) and it actually factored.
    /// Absent when no Schur block was set **or** the path fell back to
    /// the standard solver, so its presence is the signal of which path
    /// ran. The sparse eliminated block's factorizations are recorded in
    /// the fields above like any other factorization.
    pub schur: Option<SchurSummary>,

    /// Factorizations inside the restoration phase's sub-solves, kept
    /// apart so the fields above describe the main solve alone. `None`
    /// when restoration never factored. Only the outermost summary
    /// carries one.
    pub restoration: Option<Box<LinearSolverSummary>>,
}

/// Block-parallel KKT path. See [`LinearSolverSummary::blocks`].
#[derive(Debug, Clone, Default)]
pub struct BlockSummary {
    /// Blocks the KKT was partitioned into.
    pub n_blocks: usize,
    /// Order of the shared border.
    pub border_dim: usize,
    /// Order of the largest block.
    pub largest_block: usize,
    /// Completed block factorizations (blocks + border count as one).
    pub n_factors: u64,
    /// Seconds in those factorizations, summed (wall time, so the blocks'
    /// parallelism is already in it).
    pub factor_secs: f64,
}

/// Block-triangular / Schur KKT path breakdown. See
/// [`LinearSolverSummary::schur`].
#[derive(Debug, Clone, Default)]
pub struct SchurSummary {
    /// Order of the eliminated (sparse) block `A_FF`.
    pub n_eliminated: usize,
    /// Order of the Schur block `S`.
    pub n_schur: usize,
    /// Completed Schur factorizations (eliminated block, `S` formed,
    /// `S` factored).
    pub n_factors: u64,
    /// Seconds factoring the eliminated block, summed.
    pub eliminated_factor_secs: f64,
    /// Seconds forming `S = A_SS − A_SFᵀ A_FF⁻¹ A_FS`, summed.
    pub form_schur_secs: f64,
    /// Seconds factoring `S`, summed.
    pub schur_factor_secs: f64,
}

/// One factorization, as a backend reports it to
/// [`LinearSolverSummary::record`]. Backend-neutral: a backend fills
/// what it can measure and leaves the rest `None`.
#[derive(Debug, Clone, Default)]
pub struct FactorRecord {
    /// Whether this factor reused the previous symbolic analysis.
    pub pattern_reused: bool,
    /// `nnz(L) / nnz(A)`.
    pub fill_ratio: f64,
    /// Smallest `|pivot|`.
    pub min_abs_pivot: f64,
    /// Largest `|pivot|`.
    pub max_abs_pivot: f64,
    /// `(positive, negative, zero)`.
    pub inertia: (usize, usize, usize),
    /// `nnz(A)`.
    pub nnz_a: usize,
    /// `nnz(L)`.
    pub nnz_l: usize,
    /// Wall-clock seconds in the numeric factor call.
    pub factor_secs: f64,
    /// Work proxy, when the backend has a symbolic analysis to read it from.
    pub factor_flops: Option<f64>,
    /// Predicted peak bytes.
    pub peak_bytes: Option<usize>,
    /// Supernodes.
    pub n_supernodes: Option<usize>,
    /// Largest front's row count.
    pub max_front_rows: Option<usize>,
    /// Delayed-column entries.
    pub delayed_cols: Option<usize>,
    /// 2×2 pivot blocks.
    pub two_by_two: Option<usize>,
    /// Statically perturbed pivots.
    pub n_tiny: Option<usize>,
    /// Concrete ordering used.
    pub ordering: Option<String>,
    /// Ordering preprocessing actually applied (`none` / `ldlt_compress`).
    pub ordering_preprocess: Option<String>,
}

impl LinearSolverSummary {
    /// Returns `true` if the summary carries no signal beyond the
    /// solver name — useful for "did we collect anything?" checks.
    pub fn is_empty(&self) -> bool {
        self.n_factors == 0
    }

    /// Fold one factorization into the running aggregate: counters and
    /// totals accumulate, extremals widen, `last_*` fields take this
    /// record's values.
    ///
    /// Folding (rather than replacing the whole summary) is what lets
    /// several backend instances share one sink: under L-BFGS the
    /// algorithm builds two backends from one factory (the low-rank
    /// solver's and its bypass), and each used to overwrite the sink
    /// with its own running summary, so every total reflected whichever
    /// instance factored last.
    pub fn record(&mut self, r: &FactorRecord) {
        self.n_factors += 1;
        if r.pattern_reused {
            self.n_pattern_reuse += 1;
        } else {
            self.n_pattern_changes += 1;
        }
        self.max_fill_ratio = Some(
            self.max_fill_ratio
                .map_or(r.fill_ratio, |p| p.max(r.fill_ratio)),
        );
        self.min_abs_pivot = Some(
            self.min_abs_pivot
                .map_or(r.min_abs_pivot, |p| p.min(r.min_abs_pivot)),
        );
        self.max_abs_pivot = Some(
            self.max_abs_pivot
                .map_or(r.max_abs_pivot, |p| p.max(r.max_abs_pivot)),
        );
        self.last_inertia = Some(r.inertia);
        self.last_nnz_a = Some(r.nnz_a);
        self.last_nnz_l = Some(r.nnz_l);

        self.total_factor_secs += r.factor_secs;
        self.total_factor_flops += r.factor_flops.unwrap_or(0.0);
        self.total_delayed_cols += r.delayed_cols.unwrap_or(0) as u64;
        self.total_two_by_two += r.two_by_two.unwrap_or(0) as u64;
        self.total_n_tiny += r.n_tiny.unwrap_or(0) as u64;

        self.last_factor_flops = r.factor_flops;
        self.last_peak_bytes = r.peak_bytes;
        self.last_n_supernodes = r.n_supernodes;
        self.last_max_front_rows = r.max_front_rows;
        self.last_delayed_cols = r.delayed_cols;
        self.last_two_by_two = r.two_by_two;
        self.last_n_tiny = r.n_tiny;
        self.last_ordering = r.ordering.clone();
        self.last_ordering_preprocess = r.ordering_preprocess.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(fill: f64, secs: f64, delayed: usize) -> FactorRecord {
        FactorRecord {
            pattern_reused: true,
            fill_ratio: fill,
            min_abs_pivot: fill,
            max_abs_pivot: fill,
            inertia: (3, 2, 0),
            nnz_a: 10,
            nnz_l: 20,
            factor_secs: secs,
            factor_flops: Some(100.0),
            delayed_cols: Some(delayed),
            two_by_two: Some(1),
            n_tiny: Some(0),
            ordering: Some("amd".into()),
            ..Default::default()
        }
    }

    #[test]
    fn record_accumulates_totals_and_keeps_last() {
        let mut s = LinearSolverSummary::default();
        s.record(&rec(2.0, 0.5, 3));
        s.record(&rec(1.5, 0.25, 4));
        assert_eq!(s.n_factors, 2);
        assert_eq!(s.n_pattern_reuse, 2);
        assert_eq!(s.max_fill_ratio, Some(2.0));
        assert_eq!(s.min_abs_pivot, Some(1.5));
        assert_eq!(s.total_factor_secs, 0.75);
        assert_eq!(s.total_factor_flops, 200.0);
        assert_eq!(s.total_delayed_cols, 7);
        assert_eq!(s.total_two_by_two, 2);
        assert_eq!(s.last_delayed_cols, Some(4));
        assert_eq!(s.last_ordering.as_deref(), Some("amd"));
    }

    /// Two backends folding into one summary produce the sum of both,
    /// not whichever wrote last (the L-BFGS two-backend case).
    #[test]
    fn two_writers_fold_rather_than_overwrite() {
        let mut shared = LinearSolverSummary::default();
        let mut a = LinearSolverSummary::default();
        let mut b = LinearSolverSummary::default();
        for (mine, r) in [(&mut a, rec(1.0, 1.0, 1)), (&mut b, rec(1.0, 1.0, 1))] {
            mine.record(&r);
            shared.record(&r);
        }
        a.record(&rec(1.0, 1.0, 1));
        shared.record(&rec(1.0, 1.0, 1));
        assert_eq!(a.n_factors, 2);
        assert_eq!(b.n_factors, 1);
        assert_eq!(shared.n_factors, 3);
        assert_eq!(shared.total_delayed_cols, 3);
    }
}
