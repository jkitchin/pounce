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
use pounce_common::types::{Index, Number};
use pounce_linsol::summary::LinearSolverSummary;
use pounce_linsol::{EMatrixFormat, ESymSolverStatus, SparseSymLinearSolverInterface};

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
    pub cascade_break: bool,
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
}

impl Default for FeralConfig {
    fn default() -> Self {
        Self {
            cascade_break: false,
            fma: false,
            refine: true,
            // MA57 `CNTL(2)` default — an absolute small-pivot
            // magnitude on the scaled matrix. Only pivots essentially
            // at the working-precision floor are flagged singular.
            singular_pivot_floor: 1e-20,
            pivtol: 1e-8,
        }
    }
}

impl FeralConfig {
    /// Read the knobs from `POUNCE_FERAL_CASCADE_BREAK`,
    /// `POUNCE_FERAL_FMA`, `POUNCE_FERAL_REFINE`,
    /// `POUNCE_FERAL_SINGULAR_PIVOT_FLOOR` environment variables.
    /// Used as a fallback when the IPM has no `OptionsList` to
    /// consult (tests, legacy callers).
    pub fn from_env() -> Self {
        Self {
            cascade_break: matches!(
                std::env::var("POUNCE_FERAL_CASCADE_BREAK").as_deref(),
                Ok("1") | Ok("on") | Ok("true") | Ok("yes"),
            ),
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
        }
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
    /// (`ratio=0.5, eps=1e-10`) was on by default until the
    /// issue-17/issue-18 inertia investigations: it accelerates IPM
    /// KKT matrices with cascade-overloaded supernodes
    /// (pinene_3200_0009: 33ms with cb on vs 94s with cb off, feral
    /// 585d739) but introduces WrongInertia loops on others
    /// (robot_1600 iter-3; NARX_CFy iters 1+, ~250 spurious
    /// WrongInertia status records — see feral journal 2026-05-16
    /// 21:30). The C-API now defaults cb off (da23d13) for the same
    /// reason. Default here matches that; per-problem `.opt` files
    /// (e.g. `benchmarks/mittelmann/profiles/pinene_3200.opt`) flip
    /// `feral_cascade_break yes` for problems where cb is a net win.
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
        let mut solver = if cfg.cascade_break {
            Solver::with_params(np, SupernodeParams::default())
                .with_cascade_break(0.5)
                .with_cascade_break_eps(1e-10)
        } else {
            Solver::with_params(np, SupernodeParams::default())
        };
        if matches!(
            std::env::var("FERAL_PARALLEL").as_deref(),
            Ok("0") | Ok("false") | Ok("off")
        ) {
            solver = solver.with_parallel(false);
        }
        if cfg.fma {
            solver = solver.with_fma(true);
        }
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
    }

    /// Build the lower-triangle CSC view, factor it, and stash the
    /// strict-negative-eigenvalue count (IPOPT / MA57 `INFO(24)`
    /// convention). Rank deficiency (zero pivots) is reported as
    /// `Singular` so the outer loop routes to `perturb_for_singular`.
    fn factor(&mut self, check_neg_evals: bool, number_of_neg_evals: Index) -> ESymSolverStatus {
        let n = self.dim as usize;
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
                    if std::env::var_os("POUNCE_DBG_INERTIA").is_some() {
                        eprintln!(
                            "[INERTIA] singular: neg={} zero={} expected={} dim={}",
                            neg, zero, number_of_neg_evals, self.dim
                        );
                    }
                    return ESymSolverStatus::Singular;
                }
                if check_neg_evals && self.negevals != number_of_neg_evals {
                    if std::env::var_os("POUNCE_DBG_INERTIA").is_some() {
                        eprintln!(
                            "[INERTIA] mismatch: got_neg={} expected={} dim={}",
                            self.negevals, number_of_neg_evals, self.dim
                        );
                    }
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

        let solved = match (self.refine, self.matrix.as_ref(), nrhs == 1) {
            (true, Some(m), true) => self.solver.solve_refined(m, rhs_vals),
            (true, Some(m), false) => self.solver.solve_many_refined(m, rhs_vals, nrhs),
            (_, _, true) => self.solver.solve(rhs_vals),
            (_, _, false) => self.solver.solve_many(rhs_vals, nrhs),
        };
        match solved {
            Ok(x) => {
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
}
