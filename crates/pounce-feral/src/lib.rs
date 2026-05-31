//! FERAL backend — pure-Rust sparse symmetric LDL^T factor.
//!
//! Implements [`SparseSymLinearSolverInterface`] over [`feral::Solver`].
//! The lifecycle mirrors `pounce_hsl::Ma57SolverInterface`:
//!
//! * `matrix_format()` returns [`EMatrixFormat::TripletFormat`] (1-based,
//!   lower-triangle COO) so the IPM `TSymLinearSolver` wrapper requires
//!   no changes versus the MA57 path.
//! * `initialize_structure` caches the 0-based row/col arrays needed by
//!   FERAL's [`CscMatrix::from_triplets`] and allocates the values
//!   buffer.
//! * `multi_solve` rebuilds `CscMatrix` from the cached pattern + caller-
//!   filled values and dispatches to [`feral::Solver::factor`] /
//!   [`feral::Solver::solve_many`]. FERAL's pattern-fingerprint cache
//!   reuses the symbolic factorization across iterates with identical
//!   structure (the IPM common case).
//! * `increase_quality` delegates to [`feral::Solver::increase_quality`]
//!   and uses MA57's `pivtol_changed` / `CallAgain` protocol so the
//!   upper-layer reload-and-retry semantics line up.

use std::sync::{Arc, Mutex};

use feral::symbolic::SupernodeParams;
use feral::{CscMatrix, FactorStats, FactorStatus, NumericParams, Solver};

/// Re-export so option-aware callers can construct a
/// [`FeralConfig`] without taking a direct dependency on `feral`.
pub use feral::symbolic::OrderingMethod;
use pounce_common::types::{Index, Number};
use pounce_linsol::summary::LinearSolverSummary;
use pounce_linsol::{
    EMatrixFormat, ESymSolverStatus, FactorPattern, SparseSymLinearSolverInterface,
};

/// EXPERIMENT helper (gpu-batched-layers §9, step 1): is f32-precision
/// inner-solve emulation requested? Env-gated, default off, read per call
/// so it can be toggled per process without touching any production path.
fn emulate_f32() -> bool {
    std::env::var_os("POUNCE_FERAL_EMULATE_F32").is_some()
}

/// FERAL solver implementing the IPM-side sparse symmetric backend
/// contract.
pub struct FeralSolverInterface {
    solver: Solver,

    initialized: bool,
    pivtol_changed: bool,
    refactorize: bool,
    refine: bool,

    dim: Index,
    nonzeros: Index,

    /// 0-based row indices, fixed by `initialize_structure`.
    rows_0: Vec<usize>,
    /// 0-based column indices, fixed by `initialize_structure`.
    cols_0: Vec<usize>,
    /// Caller-filled numerical values, in the same order as
    /// `(rows_0, cols_0)`.
    values: Vec<Number>,

    /// Last factored matrix, retained so `backsolve` can run iterative
    /// refinement against it (feral's `solve_*_refined` requires `A`).
    matrix: Option<CscMatrix>,

    negevals: Index,

    /// Fill-reducing ordering configured at construction; surfaced on
    /// the `linear_solve` tracing span after each factorization
    /// (pounce#71).
    ordering: OrderingMethod,

    /// Absolute near-singularity floor; see
    /// [`FeralConfig::singular_pivot_floor`].
    singular_pivot_floor: f64,

    /// Running aggregate updated after every successful `factor()`.
    /// Exposed via [`Self::summary`] and (mirrored to) the optional
    /// `sink` so an out-of-band consumer can read it post-solve
    /// without plumbing through the algorithm's wrapper layers.
    summary: LinearSolverSummary,

    /// Optional shared sink updated alongside `summary`. Decouples
    /// the algorithm-internal solver lifecycle from CLI / report
    /// consumers — pounce-cli installs an `Arc<Mutex<...>>` via
    /// [`Self::with_summary_sink`] and reads it after
    /// `optimize_tnlp` returns.
    sink: Option<Arc<Mutex<LinearSolverSummary>>>,
}

/// Construction-time configuration for [`FeralSolverInterface`].
///
/// Mirrors the pounce-extension options registered in
/// `pounce-algorithm`'s `upstream_options::register_all_options`
/// (`feral_cascade_break`, `feral_fma`, `feral_refine`,
/// `feral_singular_pivot_floor`). The IPM
/// caller reads those off its `OptionsList`, builds a `FeralConfig`,
/// and passes it to [`FeralSolverInterface::with_config`]. For
/// non-option callers (tests, standalone use, the env-only legacy
/// path), [`FeralSolverInterface::new`] keeps reading the
/// `POUNCE_FERAL_*` env vars to preserve the historic defaults.
#[derive(Debug, Clone, Copy)]
pub struct FeralConfig {
    /// Tri-state. `None` (the pounce default) inherits whatever FERAL's
    /// `NumericParams::default()` ships with — as of FERAL Phase B
    /// (issue #55, commit 7554a78) that is CB armed with
    /// `ratio = 0.5, eps = 1e-10` and a symbolic-time delay budget.
    /// `Some(true)` explicitly arms with the same constants (matches
    /// the FERAL default; useful when a caller wants the intent
    /// recorded). `Some(false)` explicitly disarms by setting both
    /// `cascade_break_ratio` and `cascade_break_eps` to `None`; this
    /// is what enables FERAL's `DelayBudgetExceeded` path for non-root
    /// cascade victims and should only be used to reproduce the
    /// pre-Phase-B behaviour.
    pub cascade_break: Option<bool>,
    pub fma: bool,
    pub refine: bool,
    /// Near-singularity trigger: if the smallest accepted D-block pivot
    /// magnitude `min|λ(D)|` (scaled space) falls below this absolute
    /// floor, `factor()` returns [`ESymSolverStatus::Singular`] even
    /// though feral force-accepted the pivot and reported `Success`.
    /// This is pounce's analog of MA57's `CNTL(2)` small-pivot
    /// threshold — an absolute magnitude on the *scaled* pivot, not a
    /// ratio: a genuinely rank-deficient pivot sits at the working-
    /// precision floor regardless of the rest of the spectrum, whereas
    /// `min/max` ≈ 1/κ(D) collapses on any healthy interior-point KKT
    /// as `μ→0`. Routes into the IPM's `PerturbForSingularity` branch
    /// so `δ_w` is bumped. `0` disables the trigger. See
    /// `dev/research/near-singularity-signal.md` (feral) §4.
    pub singular_pivot_floor: f64,
    /// Relative Bunch-Kaufman partial-pivoting threshold `u`: a
    /// candidate diagonal pivot is rejected when `|d| < u * col_max`.
    /// Direct analog of Ipopt's `ma27_pivtol` / `ma57_pivtol`. Smaller
    /// `u` preserves the AMD ordering and keeps `L` sparse; larger
    /// `u` rejects more candidates, delaying pivots / forcing 2x2
    /// blocks for accuracy. LAPACK's textbook maximum-stability value
    /// is `0.5`. Default `1e-8` matches feral's `NumericParams`.
    pub pivtol: f64,
    /// Fill-reducing ordering method passed to
    /// [`feral::Solver::with_ordering`]. Default
    /// [`OrderingMethod::Auto`]: the adaptive dispatcher picks a
    /// concrete method per matrix from cheap pattern features (very-
    /// large-and-sparse → AMD; `n ≤ 10 000` → AMF; otherwise →
    /// MetisND). Override via the `feral_ordering` OptionsList option
    /// or the `POUNCE_FERAL_ORDERING` env var when a specific
    /// concrete method (`amd`, `amf`, `metis`, `scotch`, `kahip`) or
    /// the symbolic-time race (`auto_race`) is wanted. See
    /// `feral/src/symbolic/mod.rs::OrderingMethod` for the
    /// per-variant rationale.
    pub ordering: OrderingMethod,
}

impl Default for FeralConfig {
    fn default() -> Self {
        Self {
            cascade_break: None,
            fma: false,
            refine: true,
            // MA57 `CNTL(2)` default — an absolute small-pivot
            // magnitude on the scaled matrix. Only pivots essentially
            // at the working-precision floor are flagged singular.
            singular_pivot_floor: 1e-20,
            pivtol: 1e-8,
            ordering: OrderingMethod::Auto,
        }
    }
}

impl FeralConfig {
    /// Read the knobs from `POUNCE_FERAL_CASCADE_BREAK`,
    /// `POUNCE_FERAL_FMA`, `POUNCE_FERAL_REFINE`,
    /// `POUNCE_FERAL_SINGULAR_PIVOT_FLOOR`, `POUNCE_FERAL_ORDERING`
    /// environment variables. Used as a fallback when the IPM has no
    /// `OptionsList` to consult (tests, legacy callers).
    pub fn from_env() -> Self {
        Self {
            cascade_break: match std::env::var("POUNCE_FERAL_CASCADE_BREAK").as_deref() {
                Ok("1") | Ok("on") | Ok("true") | Ok("yes") => Some(true),
                Ok("0") | Ok("off") | Ok("false") | Ok("no") => Some(false),
                _ => None,
            },
            fma: matches!(
                std::env::var("POUNCE_FERAL_FMA").as_deref(),
                Ok("1") | Ok("on") | Ok("true") | Ok("yes"),
            ),
            refine: !matches!(
                std::env::var("POUNCE_FERAL_REFINE").as_deref(),
                Ok("0") | Ok("false") | Ok("off") | Ok("no"),
            ),
            singular_pivot_floor: std::env::var("POUNCE_FERAL_SINGULAR_PIVOT_FLOOR")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(1e-20),
            pivtol: std::env::var("FERAL_PIVTOL")
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(1e-8),
            ordering: std::env::var("POUNCE_FERAL_ORDERING")
                .ok()
                .as_deref()
                .and_then(parse_ordering_method)
                .unwrap_or(OrderingMethod::Auto),
        }
    }
}

/// Parse a case-insensitive ordering tag (the values accepted by the
/// `feral_ordering` OptionsList option and the `POUNCE_FERAL_ORDERING`
/// env var) into the corresponding [`OrderingMethod`]. Returns `None`
/// for unrecognized tags so the caller can fall back to the default.
pub fn parse_ordering_method(s: &str) -> Option<OrderingMethod> {
    match s.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(OrderingMethod::Auto),
        "auto_race" | "autorace" | "race" => Some(OrderingMethod::AutoRace),
        "amd" => Some(OrderingMethod::Amd),
        "amf" => Some(OrderingMethod::Amf),
        "metis" | "metis_nd" | "metisnd" => Some(OrderingMethod::MetisND),
        "scotch" | "scotch_nd" | "scotchnd" => Some(OrderingMethod::ScotchND),
        "kahip" | "kahip_nd" | "kahipnd" => Some(OrderingMethod::KahipND),
        _ => None,
    }
}

impl FeralSolverInterface {
    /// Construct with config read from environment variables. Retained
    /// for legacy callers (tests, anything without an IPM options
    /// list). Prefer [`Self::with_config`] from option-aware sites so
    /// the `.opt`-file knobs take effect.
    pub fn new() -> Self {
        Self::with_config(FeralConfig::from_env())
    }

    /// Construct with explicit configuration. Cascade-break
    /// (`ratio=0.5, eps=1e-10`) was off by default in pounce for a
    /// period after the issue-17/issue-18 inertia investigations,
    /// when the FERAL Bunch-Kaufman heuristic could not bound the
    /// per-supernode delayed-pivot catchment and spurious
    /// `WrongInertia` returns on borderline iterates (robot_1600
    /// iter-3, NARX_CFy iters 1+, ~250 spurious records — feral
    /// journal 2026-05-16 21:30) cost more than CB's per-factor
    /// speedup (pinene_3200_0009: 33 ms cb-on vs 94 s cb-off).
    /// FERAL issue #55 Phase B (commit 7554a78) bounds the catchment
    /// at symbolic-analysis time and arms CB out of the box, so
    /// pounce now inherits the FERAL default (CB on) when the
    /// `feral_cascade_break` option is left unset. See
    /// [`FeralConfig::cascade_break`] for the tri-state semantics.
    pub fn with_config(cfg: FeralConfig) -> Self {
        // `POUNCE_FERAL_MIN_PAR_FLOPS=<u64>` overrides feral's parallel-
        // dispatch flop gate (feral#19, default 10^8). `0` fires the
        // gate on every multi-child tree ≥ N_PAR_MIN supernodes (pre-
        // feral#19 behavior). `u64::MAX` rejects all parallel dispatch
        // at the tree level. Useful when the per-hardware break-even
        // is far from feral's default.
        let mut np = NumericParams::default();
        if let Ok(s) = std::env::var("POUNCE_FERAL_MIN_PAR_FLOPS") {
            if let Ok(v) = s.parse::<u64>() {
                np.min_parallel_flops = Some(v);
            }
        }
        // Relative Bunch-Kaufman partial-pivoting threshold — the
        // analog of Ipopt's `ma27_pivtol` / `ma57_pivtol`. Surfaced as
        // the `feral_pivtol` OptionsList option (registered in
        // pounce-algorithm's `upstream_options::register_all_options`),
        // with `FERAL_PIVTOL` env var as a fallback read in
        // `FeralConfig::from_env`.
        np.bk.pivot_threshold = cfg.pivtol;
        // Cascade-break (FERAL issue #55 Phase B, commit 7554a78):
        // `NumericParams::default()` now arms CB with `ratio = 0.5,
        // eps = 1e-10` and bounds delayed-pivot catchment via a
        // symbolic-time delay budget. Pounce no longer disables CB by
        // default — the tri-state `cfg.cascade_break` only intervenes
        // when the caller explicitly set the option:
        //   None        — leave `np` alone (inherit FERAL default = on)
        //   Some(true)  — explicit re-arm (matches the default; no-op
        //                 in behaviour, but records caller intent)
        //   Some(false) — explicit disarm (re-enables FERAL's
        //                 `DelayBudgetExceeded` path on non-root
        //                 cascade victims; only meaningful for
        //                 reproducing pre-Phase-B behaviour)
        match cfg.cascade_break {
            None => {}
            Some(true) => {
                np.cascade_break_ratio = Some(0.5);
                np.cascade_break_eps = Some(1e-10);
            }
            Some(false) => {
                np.cascade_break_ratio = None;
                np.cascade_break_eps = None;
            }
        }
        let mut solver = Solver::with_params(np, SupernodeParams::default());
        if matches!(
            std::env::var("FERAL_PARALLEL").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        ) {
            solver = solver.with_parallel(false);
        }
        if cfg.fma {
            solver = solver.with_fma(true);
        }
        // Fill-reducing ordering. `OrderingMethod::Auto` is pounce's
        // default — it picks a concrete method per-matrix from cheap
        // pattern features. Override via the `feral_ordering`
        // OptionsList option or `POUNCE_FERAL_ORDERING` env var.
        solver = solver.with_ordering(cfg.ordering);
        Self {
            solver,
            initialized: false,
            pivtol_changed: false,
            refactorize: false,
            refine: cfg.refine,
            dim: 0,
            nonzeros: 0,
            rows_0: Vec::new(),
            cols_0: Vec::new(),
            values: Vec::new(),
            matrix: None,
            negevals: 0,
            ordering: cfg.ordering,
            singular_pivot_floor: cfg.singular_pivot_floor,
            summary: LinearSolverSummary {
                solver_name: "feral".to_string(),
                ..Default::default()
            },
            sink: None,
        }
    }

    /// Install a shared summary sink. The interface updates the sink
    /// (and the internal `summary`) after every successful
    /// `factor()`. Default is no sink — calls then go only to the
    /// internal `summary`.
    pub fn with_summary_sink(mut self, sink: Arc<Mutex<LinearSolverSummary>>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Snapshot of the post-solve aggregate. Always populated (no
    /// opt-in needed for the always-on Phase A stats from feral 0.7).
    pub fn summary(&self) -> LinearSolverSummary {
        self.summary.clone()
    }

    /// Fold a single feral `FactorStats` into the running summary,
    /// then mirror the snapshot into the sink if one is installed.
    fn record_factor_stats(&mut self, stats: FactorStats) {
        let s = &mut self.summary;
        s.n_factors += 1;
        if stats.pattern_reused {
            s.n_pattern_reuse += 1;
        } else {
            s.n_pattern_changes += 1;
        }
        s.max_fill_ratio = Some(match s.max_fill_ratio {
            Some(prev) => prev.max(stats.fill_ratio),
            None => stats.fill_ratio,
        });
        s.min_abs_pivot = Some(match s.min_abs_pivot {
            Some(prev) => prev.min(stats.min_abs_pivot),
            None => stats.min_abs_pivot,
        });
        s.max_abs_pivot = Some(match s.max_abs_pivot {
            Some(prev) => prev.max(stats.max_abs_pivot),
            None => stats.max_abs_pivot,
        });
        s.last_inertia = Some((
            stats.inertia.positive,
            stats.inertia.negative,
            stats.inertia.zero,
        ));
        s.last_nnz_a = Some(stats.nnz_a);
        s.last_nnz_l = Some(stats.nnz_l);

        if let Some(sink) = self.sink.as_ref() {
            if let Ok(mut guard) = sink.lock() {
                *guard = s.clone();
            }
        }

        // Surface the linear-solve characteristics on the enclosing
        // `linear_solve` tracing span (pounce#71). A no-op unless that
        // span is active and declared these fields, so non-IPM callers
        // and the no-subscriber case pay nothing. Re-factorizations
        // (regularization retries) overwrite with last-wins, so the
        // span reflects the accepted factorization.
        let span = tracing::Span::current();
        span.record("n", self.dim);
        span.record("matrix_nnz", stats.nnz_a);
        span.record("factor_nnz", stats.nnz_l);
        span.record("inertia_neg", stats.inertia.negative);
        span.record("fill_ratio", stats.fill_ratio);
        span.record("ordering", tracing::field::debug(self.ordering));
    }

    /// Build the lower-triangle CSC view, factor it, and stash the
    /// strict-negative-eigenvalue count (IPOPT / MA57 `INFO(24)`
    /// convention). Rank deficiency (zero pivots) is reported as
    /// `Singular` so the outer loop routes to `perturb_for_singular`.
    fn factor(&mut self, check_neg_evals: bool, number_of_neg_evals: Index) -> ESymSolverStatus {
        let n = self.dim as usize;
        // EXPERIMENT (gpu-batched-layers §9, step 1): emulate an f32-precision
        // inner solve through pounce's real IPM. Round the KKT values to f32
        // before factoring so FERAL factors the f32-representable matrix and
        // reads inertia from f32 pivots — exactly what a GPU f32 batched solve
        // would see. Env-gated (`POUNCE_FERAL_EMULATE_F32`), default off.
        if emulate_f32() {
            for v in self.values.iter_mut() {
                *v = (*v as f32) as Number;
            }
        }
        // Hand the KKT to feral with its structure intact. Where a
        // constraint multiplier is zero (e.g. the initial point) the
        // (2,2) diagonal lands as structurally-present exact `0.0`
        // values; feral handles those explicit zeros correctly and
        // without a delayed-pivot penalty, so pounce must NOT strip
        // them — dropping them leaves the constraint columns with no
        // diagonal, which is the structurally-absent-(2,2) cascade
        // (feral#46) the strip was meant to avoid.
        let matrix = match CscMatrix::from_triplets(n, &self.rows_0, &self.cols_0, &self.values) {
            Ok(m) => m,
            Err(_) => return ESymSolverStatus::FatalError,
        };

        let status = self.solver.factor(&matrix, None);
        // Keep the matrix for refinement in backsolve, regardless of
        // factor outcome — the caller may still issue solves against
        // a stale factor in some restart paths.
        self.matrix = Some(matrix);
        match status {
            FactorStatus::Success => {
                if let Some(stats) = self.solver.last_factor_stats() {
                    self.record_factor_stats(stats);
                }
                // IPOPT / MA57 convention: `number_of_neg_evals` is the
                // count of strict negative pivots (MA57's INFO(24)). Zero
                // pivots are reported separately by signalling `Singular`,
                // which routes the outer loop to `perturb_for_singular`
                // (bumping δ_c on rank-deficient constraint rows) instead
                // of `perturb_for_wrong_inertia` (bumping δ_x). Folding
                // zero into negevals — the SSIDS bookkeeping convention —
                // is correct for spectral accounting but breaks IPOPT's
                // singularity branch on LP-shaped KKTs whose (3,3) block
                // is structurally zero. See pounce gh#52 / feral gh#54.
                let (neg, zero) = match self.solver.inertia() {
                    Some(i) => (i.negative, i.zero),
                    None => (self.solver.num_negative_eigenvalues(), 0),
                };
                self.negevals = neg as Index;
                if zero > 0 {
                    tracing::debug!(
                        target: "pounce::linsol",
                        neg, zero, expected = number_of_neg_evals, dim = self.dim,
                        "inertia singular"
                    );
                    return ESymSolverStatus::Singular;
                }
                if check_neg_evals && self.negevals != number_of_neg_evals {
                    tracing::debug!(
                        target: "pounce::linsol",
                        got_neg = self.negevals, expected = number_of_neg_evals, dim = self.dim,
                        "inertia mismatch"
                    );
                    return ESymSolverStatus::WrongInertia;
                }
                // Near-singularity (MA57 CNTL(2) analog). feral's default
                // `ZeroPivotAction::ForceAccept` completes the factorization
                // and reports `Success` even on a pivot at the working-
                // precision floor. We flag `Singular` only when the smallest
                // accepted D-block pivot magnitude drops below an absolute
                // floor — the literal `CNTL(2)` quantity. A ratio test
                // `min/max` ≈ 1/κ(D) is wrong here: an interior-point KKT
                // is *designed* to become ill-conditioned as `μ→0`, so the
                // ratio collapses on healthy full-rank systems near the
                // solution. The absolute floor moves with neither `μ` nor
                // the spectral spread. See
                // `dev/research/near-singularity-signal.md` (feral) §4.
                if self.singular_pivot_floor > 0.0 {
                    if let Some(min_piv) = self.solver.min_pivot_magnitude() {
                        if min_piv < self.singular_pivot_floor {
                            return ESymSolverStatus::Singular;
                        }
                    }
                }
                ESymSolverStatus::Success
            }
            FactorStatus::Singular => ESymSolverStatus::Singular,
            FactorStatus::WrongInertia { .. } => {
                // Should not occur — we passed `None` for check_inertia.
                ESymSolverStatus::FatalError
            }
            FactorStatus::FatalError(_) => ESymSolverStatus::FatalError,
        }
    }

    fn backsolve(&self, nrhs: Index, rhs_vals: &mut [Number]) -> ESymSolverStatus {
        let n = self.dim as usize;
        let nrhs = nrhs as usize;
        debug_assert_eq!(rhs_vals.len(), n * nrhs);

        // EXPERIMENT (gpu-batched-layers §9, step 1): under f32 emulation,
        // round the RHS to f32 and skip f64 iterative refinement so nothing
        // recovers accuracy the f32 GPU solve would not have. The solution is
        // rounded back to f32 below.
        let emulate = emulate_f32();
        if emulate {
            for r in rhs_vals.iter_mut() {
                *r = (*r as f32) as Number;
            }
        }
        let use_refine = self.refine && !emulate;
        let solved = match (use_refine, self.matrix.as_ref(), nrhs == 1) {
            (true, Some(m), true) => self.solver.solve_refined(m, rhs_vals),
            (true, Some(m), false) => self.solver.solve_many_refined(m, rhs_vals, nrhs),
            (_, _, true) => self.solver.solve(rhs_vals),
            (_, _, false) => self.solver.solve_many(rhs_vals, nrhs),
        };
        match solved {
            Ok(mut x) => {
                if emulate {
                    for v in x.iter_mut() {
                        *v = (*v as f32) as Number;
                    }
                }
                rhs_vals.copy_from_slice(&x);
                ESymSolverStatus::Success
            }
            Err(_) => ESymSolverStatus::FatalError,
        }
    }
}

impl Default for FeralSolverInterface {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for FeralSolverInterface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeralSolverInterface")
            .field("dim", &self.dim)
            .field("nonzeros", &self.nonzeros)
            .field("initialized", &self.initialized)
            .field("negevals", &self.negevals)
            .finish_non_exhaustive()
    }
}

impl SparseSymLinearSolverInterface for FeralSolverInterface {
    fn initialize_structure(
        &mut self,
        dim: Index,
        nonzeros: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> ESymSolverStatus {
        assert_eq!(ia.len(), nonzeros as usize);
        assert_eq!(ja.len(), nonzeros as usize);

        self.dim = dim;
        self.nonzeros = nonzeros;
        self.values = vec![0.0; nonzeros as usize];

        // Convert 1-based MA57-style indices to 0-based for FERAL, and
        // canonicalize each entry to the lower triangle. MA57 accepts
        // either triangle of a symmetric COO; pounce's KKT assembly
        // takes advantage of that and emits a mix of lower- and
        // upper-triangle entries. FERAL's `CscMatrix::from_triplets`
        // documents "Entries must be lower-triangle (row >= col)" but
        // does NOT check it — upper-triangle entries get stored in the
        // CSC structure where the LDL^T factorization ignores them,
        // silently dropping them from the factored matrix.
        self.rows_0 = Vec::with_capacity(nonzeros as usize);
        self.cols_0 = Vec::with_capacity(nonzeros as usize);
        for k in 0..nonzeros as usize {
            let i = (ia[k] - 1) as usize;
            let j = (ja[k] - 1) as usize;
            if i >= j {
                self.rows_0.push(i);
                self.cols_0.push(j);
            } else {
                self.rows_0.push(j);
                self.cols_0.push(i);
            }
        }

        self.initialized = true;
        ESymSolverStatus::Success
    }

    fn values_array_mut(&mut self) -> &mut [Number] {
        debug_assert!(self.initialized);
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
        // Quality was bumped since the last factor → caller must refill
        // values and we'll re-factor. Mirrors MA57's protocol.
        if self.pivtol_changed {
            self.pivtol_changed = false;
            if !new_matrix {
                self.refactorize = true;
                return ESymSolverStatus::CallAgain;
            }
        }

        if new_matrix || self.refactorize {
            let status = self.factor(check_neg_evals, number_of_neg_evals);
            if status != ESymSolverStatus::Success {
                return status;
            }
            self.refactorize = false;
        }

        self.backsolve(nrhs, rhs_vals)
    }

    fn number_of_neg_evals(&self) -> Index {
        debug_assert!(self.initialized);
        self.negevals
    }

    fn increase_quality(&mut self) -> bool {
        // Mirror ipopt-feral (IpFeralSolverInterface.cpp:134): no pivtol
        // escalation here. Returning false hands recovery to
        // PDPerturbationHandler so matrix-side regularization (`lg(rg)`)
        // is the single escalator, matching ipopt-feral's trajectory.
        false
    }

    fn provides_inertia(&self) -> bool {
        true
    }

    fn matrix_format(&self) -> EMatrixFormat {
        EMatrixFormat::TripletFormat
    }

    /// Walk feral's per-supernode `NodeFactors` to assemble the LDLᵀ
    /// factor's strict-lower nonzero pattern in *permuted*
    /// coordinates. `perm` is feral's global fill-reducing
    /// permutation (new-to-old). When `want_values` is true the
    /// per-supernode `l` block is also gathered into `l_vals` —
    /// indexed by the post-BK pivot perm so the order matches the
    /// (irn, jcn) arrays.
    ///
    /// Returns `None` before the first successful factor (`factors()`
    /// returns `None`).
    fn factor_pattern(&self, want_values: bool) -> Option<FactorPattern> {
        let factors = self.solver.factors()?;

        // Conservative upper bound on nnz_L (strict-lower): per-supernode
        //   nelim*(nelim-1)/2 + (nrow - nelim) * nelim
        // (the diagonal is excluded). Doubles as a single allocation
        // for both irn and jcn, plus l_vals when requested.
        let mut nnz_upper: usize = 0;
        for nf in &factors.node_factors {
            let ff = &nf.frontal_factors;
            let nelim = ff.nelim;
            let nrow = ff.nrow;
            let trailing = nrow.saturating_sub(nelim) * nelim;
            nnz_upper += nelim * nelim.saturating_sub(1) / 2 + trailing;
        }

        let mut l_irn: Vec<Index> = Vec::with_capacity(nnz_upper);
        let mut l_jcn: Vec<Index> = Vec::with_capacity(nnz_upper);
        let mut l_vals: Option<Vec<Number>> = if want_values {
            Some(Vec::with_capacity(nnz_upper))
        } else {
            None
        };

        for nf in &factors.node_factors {
            let ff = &nf.frontal_factors;
            let nelim = ff.nelim;
            let nrow = ff.nrow;
            // perm[i] = pre-BK supernode row that landed at post-BK
            // pivot position i. Indices [nelim, nrow) are identity.
            let perm = &ff.perm;
            let l = &ff.l;
            for j in 0..nelim {
                // Column j of L: global col index in permuted coords.
                let col_local = perm[j];
                let col_global = nf.row_indices[col_local];
                let col1 = (col_global as Index) + 1;
                // Strict-lower entries: rows i in (j, nrow).
                for i in (j + 1)..nrow {
                    let row_local = if i < nelim { perm[i] } else { i };
                    let row_global = nf.row_indices[row_local];
                    l_irn.push((row_global as Index) + 1);
                    l_jcn.push(col1);
                    if let Some(vals) = l_vals.as_mut() {
                        vals.push(l[j * nrow + i]);
                    }
                }
            }
        }

        Some(FactorPattern {
            n: factors.n,
            perm: factors.perm.clone(),
            l_irn,
            l_jcn,
            l_vals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2x2 SPD matrix `[[2,1],[1,3]]`. Lower-triangle 1-based triplets.
    /// Solving against (3, 4) gives x = (1, 1).
    #[test]
    fn factor_and_solve_spd_2x2() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];

        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);

        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        assert!((rhs[0] - 1.0).abs() < 1e-12, "x0 = {}", rhs[0]);
        assert!((rhs[1] - 1.0).abs() < 1e-12, "x1 = {}", rhs[1]);
        assert_eq!(s.number_of_neg_evals(), 0);
        assert!(s.provides_inertia());
        assert_eq!(s.matrix_format(), EMatrixFormat::TripletFormat);
    }

    /// 2x2 indefinite `[[1,2],[2,1]]` — eigenvalues 3, -1.
    #[test]
    fn detects_one_negative_eigenvalue() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];

        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[1.0, 2.0, 1.0]);

        let mut rhs = [3.0, 3.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 1),
            ESymSolverStatus::Success
        );
        assert_eq!(s.number_of_neg_evals(), 1);
        assert!((rhs[0] - 1.0).abs() < 1e-12);
        assert!((rhs[1] - 1.0).abs() < 1e-12);
    }

    /// Wrong expected inertia → `WrongInertia` (and no solve).
    #[test]
    fn check_neg_evals_mismatch_returns_wrong_inertia() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]); // SPD
        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 1),
            ESymSolverStatus::WrongInertia
        );
    }

    /// `increase_quality` then resolve with `new_matrix=false`
    /// returns `CallAgain`; refilling values and retrying succeeds.
    #[test]
    fn increase_quality_then_resolve_triggers_call_again() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );

        if s.increase_quality() {
            // Quality bumped; new_matrix=false → CallAgain.
            let mut rhs = [3.0, 4.0];
            assert_eq!(
                s.multi_solve(false, &irn, &jcn, 1, &mut rhs, false, 0),
                ESymSolverStatus::CallAgain
            );
            // Refill values and retry.
            s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
            let mut rhs = [3.0, 4.0];
            assert_eq!(
                s.multi_solve(false, &irn, &jcn, 1, &mut rhs, false, 0),
                ESymSolverStatus::Success
            );
            assert!((rhs[0] - 1.0).abs() < 1e-12);
            assert!((rhs[1] - 1.0).abs() < 1e-12);
        }
    }

    /// Pounce emits some symmetric entries as upper-triangle
    /// `(i, j)` with `i < j` because MA57 accepts either half. The
    /// FERAL wrapper must canonicalize to lower triangle (row >= col)
    /// before handing entries to `CscMatrix::from_triplets`, which
    /// silently drops upper-triangle entries during LDL^T. A regression
    /// in this canonicalization would corrupt residuals and inertia
    /// (see jkitchin/feral#6).
    #[test]
    fn upper_triangle_entries_are_canonicalized() {
        let mut s = FeralSolverInterface::new();
        // Same matrix as `factor_and_solve_spd_2x2`, but the (2,1)
        // off-diagonal is given as upper-triangle (1,2).
        let irn: [Index; 3] = [1, 1, 2];
        let jcn: [Index; 3] = [1, 2, 2];
        s.initialize_structure(2, 3, &irn, &jcn);
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);

        let mut rhs = [3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        assert!((rhs[0] - 1.0).abs() < 1e-12, "x0 = {}", rhs[0]);
        assert!((rhs[1] - 1.0).abs() < 1e-12, "x1 = {}", rhs[1]);
    }

    /// `factor_pattern` returns the L sparsity (strict-lower) after a
    /// successful factor. For the SPD 2x2 above, L has exactly one
    /// strict-lower entry (the single off-diagonal), and `perm` is a
    /// permutation of `0..n`.
    #[test]
    fn factor_pattern_returns_l_after_factor() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        s.initialize_structure(2, 3, &irn, &jcn);
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
        let mut rhs = [3.0, 4.0];
        s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0);

        // Pattern-only.
        let pat = s.factor_pattern(false).expect("factors present");
        assert_eq!(pat.n, 2);
        assert_eq!(pat.perm.len(), 2);
        assert!(pat.perm.contains(&0) && pat.perm.contains(&1));
        assert_eq!(pat.l_irn.len(), 1, "L strict-lower nnz = 1 for SPD 2x2");
        assert_eq!(pat.l_jcn.len(), 1);
        assert!(pat.l_vals.is_none(), "values not requested");

        // With values.
        let pat = s.factor_pattern(true).expect("factors present");
        let vals = pat.l_vals.as_ref().expect("values requested");
        assert_eq!(vals.len(), pat.l_irn.len());
        // The single strict-lower L entry should be finite.
        assert!(vals[0].is_finite());
    }

    /// Before any factor, `factor_pattern` returns `None`.
    #[test]
    fn factor_pattern_none_before_factor() {
        let s = FeralSolverInterface::new();
        assert!(s.factor_pattern(false).is_none());
        assert!(s.factor_pattern(true).is_none());
    }

    /// Two-RHS solve via `solve_many`.
    #[test]
    fn multi_rhs_solve() {
        let mut s = FeralSolverInterface::new();
        let irn: [Index; 3] = [1, 2, 2];
        let jcn: [Index; 3] = [1, 1, 2];
        assert_eq!(
            s.initialize_structure(2, 3, &irn, &jcn),
            ESymSolverStatus::Success
        );
        s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);

        // Column 1: A x = (3, 4) → x = (1, 1)
        // Column 2: A x = (4, 5) → x = (7/5, 6/5)
        let mut rhs = [3.0, 4.0, 4.0, 5.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 2, &mut rhs, false, 0),
            ESymSolverStatus::Success
        );
        assert!((rhs[0] - 1.0).abs() < 1e-10);
        assert!((rhs[1] - 1.0).abs() < 1e-10);
        assert!((rhs[2] - 7.0 / 5.0).abs() < 1e-10);
        assert!((rhs[3] - 6.0 / 5.0).abs() < 1e-10);
    }

    /// `parse_ordering_method` accepts every documented tag (in either
    /// case) and rejects unknown ones.
    #[test]
    fn parse_ordering_method_accepts_documented_tags() {
        use OrderingMethod::*;
        let cases: &[(&str, OrderingMethod)] = &[
            ("auto", Auto),
            ("AUTO", Auto),
            ("auto_race", AutoRace),
            ("autorace", AutoRace),
            ("race", AutoRace),
            ("amd", Amd),
            ("AMD", Amd),
            ("amf", Amf),
            ("metis", MetisND),
            ("metis_nd", MetisND),
            ("MetisND", MetisND),
            ("scotch", ScotchND),
            ("kahip", KahipND),
        ];
        for (tag, expected) in cases {
            assert_eq!(
                parse_ordering_method(tag),
                Some(*expected),
                "tag {tag:?} should parse"
            );
        }
        assert_eq!(parse_ordering_method("not_a_method"), None);
        assert_eq!(parse_ordering_method(""), None);
    }

    /// Each `OrderingMethod` variant constructs a usable solver and
    /// can factor a tiny SPD system.
    #[test]
    fn every_ordering_constructs_and_factors() {
        use OrderingMethod::*;
        for method in [Auto, AutoRace, Amd, Amf, MetisND, ScotchND, KahipND] {
            let cfg = FeralConfig {
                ordering: method,
                ..FeralConfig::default()
            };
            let mut s = FeralSolverInterface::with_config(cfg);
            let irn: [Index; 3] = [1, 2, 2];
            let jcn: [Index; 3] = [1, 1, 2];
            assert_eq!(
                s.initialize_structure(2, 3, &irn, &jcn),
                ESymSolverStatus::Success,
                "structure init for {method:?}"
            );
            s.values_array_mut().copy_from_slice(&[2.0, 1.0, 3.0]);
            let mut rhs = [3.0, 4.0];
            assert_eq!(
                s.multi_solve(true, &irn, &jcn, 1, &mut rhs, false, 0),
                ESymSolverStatus::Success,
                "solve for {method:?}"
            );
            assert!((rhs[0] - 1.0).abs() < 1e-10, "x0 for {method:?}");
            assert!((rhs[1] - 1.0).abs() < 1e-10, "x1 for {method:?}");
        }
    }
}
