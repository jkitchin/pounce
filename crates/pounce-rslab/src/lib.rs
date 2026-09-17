//! RSLAB backend — experimental pure-Rust sparse symmetric LDLᵀ factor.
//!
//! Implements [`SparseSymLinearSolverInterface`] over RSLAB
//! (<https://github.com/milanofthe/rslab>), a fork of FERAL that adds an
//! unsymmetric LU path, complex-symmetric support and solver-in-the-loop
//! diagnostics. This crate exists to answer one question — *is RSLAB a
//! numerically credible KKT backend for POUNCE?* — and is deliberately not
//! wired into the optimizer: nothing in `pounce-algorithm` knows it exists.
//! See `README.md` for why, and `dev-notes/rslab-backend-assessment.md` for
//! the measurement it was built to support.
//!
//! The lifecycle mirrors [`pounce_feral::FeralSolverInterface`] so the two are
//! interchangeable behind the trait object:
//!
//! * `matrix_format()` returns [`EMatrixFormat::TripletFormat`] (1-based,
//!   lower-triangle COO), so `TSymLinearSolver` needs no changes;
//! * `initialize_structure` caches the 0-based row/col arrays and allocates
//!   the values buffer;
//! * the first `multi_solve` with `new_matrix = true` builds the
//!   [`rslab::CscMatrix`], records the triplet → CSC slot permutation, and
//!   runs the symbolic analysis once; later factorizations scatter new values
//!   through the cached slots and reuse the same [`rslab::MultifrontalSymbolic`];
//! * `multi_solve` with `new_matrix = false` back-substitutes against the
//!   stored factor.
//!
//! # Why the low-level RSLAB entry points
//!
//! RSLAB's ergonomic handle is `rslab::LdltSolver` (`LdltSymbolic::analyze`
//! → `.factor()` → `.solve()`), which also owns the supernodal panel solve.
//! The adapter does **not** use it, for one reason: `LdltSolver` exposes
//! `inertia()` and `n_perturbed()` but keeps its `LdltFactors` private, and
//! RSLAB has no `min_pivot_magnitude()` anywhere in the crate. POUNCE's
//! inertia-trust gate (pounce gh#540) is a comparison of the smallest accepted
//! pivot magnitude against a floor, so a backend that cannot report that
//! quantity cannot preserve the behaviour the gate exists for. `analyze_with`
//! + [`rslab::factor_numeric`] hand back `rslab::LdltNumeric`, whose
//! `d_diag` / `d_subdiag` / `two_by_two` are public — from which
//! [`inertia::scan_d`] recovers both the extent and a cancellation-free
//! inertia recount in one O(n) pass.
//!
//! The cost of that choice is paid in two places, both documented and both
//! measured by `examples/rslab_bench.rs`:
//!
//! * the solve runs RSLAB's CSC reference sweep ([`rslab::solve_ldlt_many`])
//!   rather than the supernodal panel sweep, because `SolvePlan` is
//!   `pub(crate)`;
//! * `LdltNumeric::into_factors` materializes `L` in CSC, one copy per factor.
//!
//! Neither is a numerical difference, and neither is worth fixing before the
//! numbers say RSLAB is worth keeping.
//!
//! # Equilibration
//!
//! `LdltSolver` symmetrically equilibrates (`A_hat = D A D`,
//! `s_i = 1/sqrt(max_j |A_ij|)`) before factoring, and the low-level entry
//! points do not — `equilibrate_with` is private. The adapter reproduces that
//! one-pass inf-norm step itself (see [`scaling`]) so its numerics match
//! RSLAB's own default rather than silently running unscaled, and so the pivot
//! magnitudes it reports live in the same scaled space feral's
//! `min_pivot_magnitude` does.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod compare;
pub mod inertia;
pub mod scaling;

pub use inertia::{InertiaInfo, inertia_trust_floor};

use std::sync::{Arc, Mutex};

use pounce_common::types::{Index, Number};
use pounce_linsol::summary::LinearSolverSummary;
use pounce_linsol::{EMatrixFormat, ESymSolverStatus, SparseSymLinearSolverInterface};
use rslab::{CscMatrix, LdltFactors, MultifrontalSymbolic, SolverSettings, ZeroPivotAction};

/// How the adapter asks RSLAB to treat a pivot it cannot eliminate.
///
/// This is an adapter-owned enum rather than a re-export of
/// [`ZeroPivotAction`], because RSLAB's three arms do not mean what their
/// names suggest to a POUNCE reader and because the useful static-pivot floor
/// is a function of the matrix, which the caller does not have at
/// configuration time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PivotPolicy {
    /// Exact factorization: a fully-summed block RSLAB cannot pivot is
    /// [`ESymSolverStatus::Singular`]. RSLAB's own `SolverSettings::default()`,
    /// and the adapter's.
    ///
    /// **This mode cannot factor a POUNCE KKT.** RSLAB restricts Bunch-Kaufman
    /// pivoting to each front's fully-summed block and has no delayed
    /// pivoting — its module docs say so outright: *"a fully-summed block that
    /// is singular in exact mode surfaces as `NumericallyRankDeficient`"*. A
    /// saddle-point row factors only when its 2×2 partner happens to land in
    /// the same front. Measured over every KKT system six real models handed
    /// their linear solver, that never happened: see
    /// `dev-notes/rslab-backend-assessment.md`.
    Exact,
    /// Static pivoting with an absolute floor: a pivot below `floor` is lifted
    /// to `sign(d)·floor`, and the factor is of `A + E` rather than `A`.
    StaticPivotAbsolute(f64),
    /// Static pivoting with the floor computed per factorization as
    /// `eps_rel · max|A_ij|` over the **equilibrated** matrix — RSLAB's own
    /// recommended recipe (`SolverSettings::preconditioner`'s doc names
    /// `eps_rel ∈ [1e-12, 1e-8]`).
    ///
    /// This is the only mode in which RSLAB completes a POUNCE KKT
    /// factorization, and what it produces is a *preconditioner*: the reported
    /// inertia is that of `A + E`, so the adapter marks it unreliable, and the
    /// solve needs iterative refinement against the unperturbed `A` to reach a
    /// direct solver's accuracy.
    StaticPivotRelative(f64),
}

impl PivotPolicy {
    /// The RSLAB action for this policy, given the largest magnitude in the
    /// equilibrated matrix about to be factored.
    fn to_rslab(self, a_max: f64) -> ZeroPivotAction {
        match self {
            Self::Exact => ZeroPivotAction::Fail,
            Self::StaticPivotAbsolute(f) => ZeroPivotAction::PerturbToEps {
                abs_floor: f.max(0.0),
            },
            Self::StaticPivotRelative(rel) => ZeroPivotAction::PerturbToEps {
                abs_floor: (rel * a_max.max(1.0)).max(0.0),
            },
        }
    }
}

/// Construction-time configuration for [`RslabSolverInterface`].
///
/// The defaults are chosen to make RSLAB behave the way POUNCE's FERAL path
/// behaves, not the way RSLAB's own examples do, so that a comparison between
/// the two backends is a comparison of their factorizations rather than of
/// their option defaults.
#[derive(Debug, Clone)]
pub struct RslabConfig {
    /// See [`PivotPolicy`]. Default [`PivotPolicy::Exact`].
    pub pivot: PivotPolicy,

    /// Symmetric equilibration applied before factoring. Default
    /// [`scaling::Equilibration::OnePassInfNorm`], reproducing
    /// `rslab::LdltSolver`'s built-in step.
    pub equilibration: scaling::Equilibration,

    /// Absolute floor under which a *mismatching* inertia count is treated as
    /// noise rather than as a measurement, mirroring
    /// `pounce_feral::FeralConfig::inertia_pivot_floor`: `Some(v)` pins an
    /// absolute floor (`Some(0.0)` disables the trigger), `None` selects the
    /// dimension-aware `n · ε` default computed by [`inertia_trust_floor`].
    ///
    /// Deliberately the same knob, computed the same way, so RSLAB and FERAL
    /// are judged against one threshold convention (pounce gh#540 / gh#592)
    /// instead of two.
    pub inertia_pivot_floor: Option<f64>,

    /// Absolute near-singularity floor (the MA57 `CNTL(2)` analog). When
    /// positive, a factorization whose smallest accepted pivot magnitude falls
    /// below it is reported [`ESymSolverStatus::Singular`] even though RSLAB
    /// completed it. Default `0.0` (off), matching
    /// `FeralConfig::singular_pivot_floor`'s default.
    pub singular_pivot_floor: f64,

    /// RSLAB factorization settings other than the pivot policy (ordering,
    /// threads, kernel knobs). Defaults to [`SolverSettings::default`], i.e.
    /// the left-looking path with the `Auto` ordering.
    pub settings: SolverSettings,
}

impl Default for RslabConfig {
    fn default() -> Self {
        Self {
            pivot: PivotPolicy::Exact,
            equilibration: scaling::Equilibration::OnePassInfNorm,
            inertia_pivot_floor: None,
            singular_pivot_floor: 0.0,
            settings: SolverSettings::default(),
        }
    }
}

impl RslabConfig {
    /// The configuration in which RSLAB can actually factor a POUNCE KKT:
    /// static pivoting at `eps_rel · max|A_ij|`. See
    /// [`PivotPolicy::StaticPivotRelative`] for what that costs.
    pub fn static_pivoting(eps_rel: f64) -> Self {
        Self {
            pivot: PivotPolicy::StaticPivotRelative(eps_rel),
            ..Self::default()
        }
    }

    /// The [`SolverSettings`] actually handed to RSLAB.
    fn effective_settings(&self, a_max: f64) -> SolverSettings {
        self.settings.clone().with_pivot(self.pivot.to_rslab(a_max))
    }
}

/// RSLAB solver implementing the IPM-side sparse symmetric backend contract.
pub struct RslabSolverInterface {
    cfg: RslabConfig,

    dim: Index,
    nonzeros: Index,

    /// 0-based row indices, fixed by `initialize_structure`.
    rows_0: Vec<usize>,
    /// 0-based column indices, fixed by `initialize_structure`.
    cols_0: Vec<usize>,
    /// Caller-filled numerical values, in the same order as `(rows_0, cols_0)`.
    values: Vec<Number>,

    /// Unscaled CSC view of the caller's triplets, rebuilt in place on each
    /// factorization through [`Self::slot`].
    matrix: Option<CscMatrix<f64>>,
    /// Triplet → CSC slot permutation: `slot[k]` is the index into
    /// `matrix.values` that triplet `k` lands in. RSLAB's `from_triplets`
    /// sorts each column and sums duplicates, so the structure is a function
    /// of the pattern alone and is reproduced identically on every refill.
    /// Recorded on the first factorization after `initialize_structure` and
    /// replayed as an allocation-free O(nnz) scatter thereafter — the same
    /// device `pounce-feral` uses (pounce gh#562).
    slot: Option<Vec<usize>>,

    /// Equilibrated values of `matrix` — what RSLAB actually factors.
    scaled: Vec<Number>,
    /// Symmetric equilibration diagonal `s` (`D = diag(s)`), length `dim`.
    scale: Vec<f64>,
    /// Row-major `n × nrhs` staging buffer for [`Self::backsolve`], grown once.
    rhs_scratch: Vec<Number>,

    /// Symbolic analysis, built once per pattern and reused across
    /// factorizations (RSLAB's PARDISO phase 1).
    symbolic: Option<MultifrontalSymbolic>,
    /// Current numeric factor in CSC form, ready for back-substitution.
    factors: Option<LdltFactors<f64>>,

    /// Inertia and reliability of the most recent successful factorization.
    inertia: InertiaInfo,
    negevals: Index,
    /// Slots of the most recent factor holding an exact zero. See the note in
    /// [`Self::factor`] on why `nnz(L)` is reported structurally.
    explicit_zeros: usize,

    summary: LinearSolverSummary,
    sink: Option<Arc<Mutex<LinearSolverSummary>>>,
}

impl std::fmt::Debug for RslabSolverInterface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RslabSolverInterface")
            .field("dim", &self.dim)
            .field("nonzeros", &self.nonzeros)
            .field("negevals", &self.negevals)
            .field("inertia", &self.inertia)
            .finish_non_exhaustive()
    }
}

impl Default for RslabSolverInterface {
    fn default() -> Self {
        Self::new()
    }
}

impl RslabSolverInterface {
    /// A backend with [`RslabConfig::default`].
    pub fn new() -> Self {
        Self::with_config(RslabConfig::default())
    }

    /// A backend with explicit configuration.
    pub fn with_config(cfg: RslabConfig) -> Self {
        Self {
            cfg,
            dim: 0,
            nonzeros: 0,
            rows_0: Vec::new(),
            cols_0: Vec::new(),
            values: Vec::new(),
            matrix: None,
            slot: None,
            scaled: Vec::new(),
            scale: Vec::new(),
            rhs_scratch: Vec::new(),
            symbolic: None,
            factors: None,
            inertia: InertiaInfo::empty(),
            negevals: 0,
            explicit_zeros: 0,
            summary: LinearSolverSummary {
                solver_name: "rslab".to_string(),
                ..Default::default()
            },
            sink: None,
        }
    }

    /// Install a shared summary sink, updated after every successful factor.
    /// Mirrors `pounce_feral::FeralSolverInterface::with_summary_sink`.
    pub fn with_summary_sink(mut self, sink: Arc<Mutex<LinearSolverSummary>>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Post-solve aggregate over this instance's lifetime.
    pub fn summary(&self) -> LinearSolverSummary {
        self.summary.clone()
    }

    /// Inertia and reliability of the most recent successful factorization.
    ///
    /// This is the adapter's answer to "what did RSLAB find, and may POUNCE
    /// act on it?". [`InertiaInfo::positive`] / `negative` / `zero` are
    /// RSLAB's own counts, reported unchanged; the rest is the adapter's.
    pub fn inertia_info(&self) -> &InertiaInfo {
        &self.inertia
    }

    /// The configuration this backend was built with.
    pub fn config(&self) -> &RslabConfig {
        &self.cfg
    }

    /// Slots of the most recent factor holding an exact zero
    /// (`rslab::LdltNumeric::n_zeros`).
    ///
    /// `LinearSolverSummary::last_nnz_l` reports the **structural** factor
    /// size, which is what FERAL reports and what memory costs. This is the
    /// part of it that carries no information — on a POUNCE KKT it is most of
    /// it, because the triplet pattern arrives full of explicit zeros. A
    /// caller comparing "useful" fill across backends wants
    /// `last_nnz_l - explicit_zeros`; a caller comparing memory wants
    /// `last_nnz_l`.
    pub fn explicit_zeros(&self) -> usize {
        self.explicit_zeros
    }

    /// Effective inertia-trust floor for the current dimension.
    fn trust_floor(&self) -> f64 {
        inertia_trust_floor(self.cfg.inertia_pivot_floor, self.dim as usize)
    }

    /// Build (first call) or refill (later calls) the unscaled CSC view.
    ///
    /// Returns `false` only when RSLAB rejects the pattern, which for a
    /// lower-triangle 1-based triplet set means the caller handed us an
    /// upper-triangle entry or an out-of-range index.
    fn refresh_matrix(&mut self) -> bool {
        if let (Some(m), Some(slot)) = (self.matrix.as_mut(), self.slot.as_ref()) {
            // Duplicates are summed by `from_triplets`, so the replay must
            // zero first and accumulate — not assign.
            m.values.iter_mut().for_each(|v| *v = 0.0);
            for (k, &s) in slot.iter().enumerate() {
                m.values[s] += self.values[k];
            }
            return true;
        }
        let m = match CscMatrix::<f64>::from_triplets(
            self.dim as usize,
            &self.rows_0,
            &self.cols_0,
            &self.values,
        ) {
            Ok(m) => m,
            Err(e) => {
                tracing::error!(
                    target: "pounce::linsol",
                    error = ?e,
                    "rslab: from_triplets rejected the KKT pattern"
                );
                return false;
            }
        };
        self.slot = Some(build_slot_map(&self.rows_0, &self.cols_0, &m));
        self.matrix = Some(m);
        true
    }

    /// Equilibrate the cached CSC into `self.scaled`, recording `self.scale`.
    fn equilibrate(&mut self) {
        let m = self.matrix.as_ref().expect("refresh_matrix stored one");
        self.scale = scaling::compute(self.cfg.equilibration, m);
        self.scaled.clear();
        self.scaled.reserve(m.values.len());
        for j in 0..m.n {
            for k in m.col_ptr[j]..m.col_ptr[j + 1] {
                let i = m.row_idx[k];
                self.scaled.push(m.values[k] * self.scale[i] * self.scale[j]);
            }
        }
    }

    /// Factor the current values and stash the inertia.
    ///
    /// The status ladder mirrors `pounce_feral::FeralSolverInterface::factor`
    /// step for step, so the two backends drive POUNCE's perturbation handler
    /// the same way:
    ///
    /// 1. a rank-deficient factorization is `Singular` (→ `δ_c`);
    /// 2. a mismatching negative count whose inertia is not trustworthy is
    ///    `Singular`, not `WrongInertia` (pounce gh#540);
    /// 3. a mismatching count on a trustworthy factor is `WrongInertia`
    ///    (→ `δ_w`);
    /// 4. a completed factor whose smallest pivot is below
    ///    `singular_pivot_floor` is `Singular`.
    fn factor(&mut self, check_neg_evals: bool, number_of_neg_evals: Index) -> ESymSolverStatus {
        if !self.refresh_matrix() {
            return ESymSolverStatus::FatalError;
        }
        self.equilibrate();

        // The static-pivot floor is relative to the matrix RSLAB actually
        // factors, i.e. after equilibration — a floor derived from the
        // unscaled entries would mean something different on every iterate.
        let a_max = self.scaled.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let opts = self.cfg.effective_settings(a_max);
        let m = self.matrix.as_ref().expect("refresh_matrix stored one");
        let scaled = CscMatrix::<f64> {
            n: m.n,
            col_ptr: m.col_ptr.clone(),
            row_idx: m.row_idx.clone(),
            values: self.scaled.clone(),
        };

        let pattern_reused = self.symbolic.is_some();
        if self.symbolic.is_none() {
            match rslab::analyze_with(scaled.n, &scaled.col_ptr, &scaled.row_idx, &opts) {
                Ok(s) => self.symbolic = Some(s),
                Err(e) => {
                    tracing::error!(
                        target: "pounce::linsol",
                        error = ?e, "rslab: symbolic analysis failed"
                    );
                    return ESymSolverStatus::FatalError;
                }
            }
        }
        let symb = self.symbolic.as_ref().expect("analysis stored above");

        let numeric = match rslab::factor_numeric(symb, &scaled, &opts) {
            Ok(nf) => nf,
            Err(rslab::RslabError::NumericallyRankDeficient) => {
                tracing::debug!(
                    target: "pounce::linsol",
                    dim = self.dim, "rslab: rank deficient; reporting singular"
                );
                self.factors = None;
                return ESymSolverStatus::Singular;
            }
            Err(e) => {
                tracing::error!(
                    target: "pounce::linsol",
                    error = ?e, "rslab: numeric factorization failed"
                );
                self.factors = None;
                return ESymSolverStatus::FatalError;
            }
        };

        // `nnz(L)` is reported **structurally** — every slot the factor
        // occupies — not as the count of numerically nonzero entries.
        //
        // The distinction is not pedantic on a POUNCE KKT. The triplet
        // pattern POUNCE hands over carries explicit zeros wherever the
        // Hessian or the (2,2) block has a structural slot but no value: on
        // `airport`'s first KKT, 1768 of 2016 stored entries are exactly
        // `0.0`. RSLAB propagates them and then drops them
        // (`LdltNumeric::n_zeros`, and `PanelFactor::to_csc` omits them
        // outright), so its "stored nonzeros" came to 330 against FERAL's
        // 5721 — a 17x difference that is entirely an accounting artefact.
        // `numeric.factor.nnz()` is the panel storage RSLAB actually holds,
        // which is the quantity FERAL's `nnz_l` also reports and the one a
        // memory comparison needs.
        let nnz_l = numeric.factor.nnz();
        let explicit_zeros = numeric.n_zeros;
        let (extent, stable, n_2x2) =
            inertia::scan_d(&numeric.d_diag, &numeric.d_subdiag, &numeric.two_by_two);
        let counts = (
            numeric.inertia.positive,
            numeric.inertia.negative,
            numeric.inertia.zero,
        );
        let min_abs = extent.map(|(lo, _)| lo);
        let max_abs = extent.map(|(_, hi)| hi);
        let reliable = inertia::classify_reliability(
            counts,
            stable,
            min_abs,
            numeric.n_perturbed,
            self.trust_floor(),
        );
        self.inertia = InertiaInfo {
            positive: counts.0,
            negative: counts.1,
            zero: counts.2,
            min_abs_pivot_or_block_eigenvalue: min_abs,
            max_abs_pivot_or_block_eigenvalue: max_abs,
            perturbed_pivots: numeric.n_perturbed,
            two_by_two_pivots: n_2x2,
            stable_recount: stable,
            reliable,
        };
        self.negevals = counts.1 as Index;
        self.explicit_zeros = explicit_zeros;
        self.factors = Some(numeric.into_factors());
        self.record_factor_stats(pattern_reused, nnz_l);

        if !self.inertia.recount_agrees() {
            tracing::warn!(
                target: "pounce::linsol",
                rslab = ?counts, stable = ?stable, dim = self.dim,
                "rslab: 2x2 determinant cancellation changed an inertia sign"
            );
        }

        // 1. Rank deficiency. Unreachable under `ZeroPivotAction::Fail` (RSLAB
        //    aborts first, handled above); reachable when a caller selects a
        //    perturbing policy, where RSLAB may book an accepted zero.
        if self.inertia.zero > 0 {
            tracing::debug!(
                target: "pounce::linsol",
                neg = counts.1, zero = counts.2, expected = number_of_neg_evals,
                dim = self.dim, "rslab: inertia singular"
            );
            return ESymSolverStatus::Singular;
        }

        if check_neg_evals && self.negevals != number_of_neg_evals {
            // 2. pounce gh#540: a count read off a factorization whose smallest
            //    pivot sits at the working-precision floor is noise, and
            //    `WrongInertia` would send the IPM up the `δ_w` ladder (×8 per
            //    retry) when the perturbation that repairs the underlying
            //    rank-deficient Jacobian is `δ_c`. `Singular` reaches for that
            //    one. `reliable` folds in two further disqualifiers the FERAL
            //    sibling has no analogue for — a perturbed pivot and a
            //    determinant whose sign cancellation moved — and both point the
            //    same way: do not spend `δ_w` on a number that is not a
            //    measurement.
            if !self.inertia.reliable {
                tracing::debug!(
                    target: "pounce::linsol",
                    got_neg = self.negevals, expected = number_of_neg_evals,
                    min_piv = min_abs, floor = self.trust_floor(),
                    perturbed = self.inertia.perturbed_pivots,
                    recount_agrees = self.inertia.recount_agrees(),
                    dim = self.dim,
                    "rslab: inertia untrustworthy; reporting singular"
                );
                return ESymSolverStatus::Singular;
            }
            // 3.
            tracing::debug!(
                target: "pounce::linsol",
                got_neg = self.negevals, expected = number_of_neg_evals,
                dim = self.dim, min_piv = min_abs, "rslab: inertia mismatch"
            );
            return ESymSolverStatus::WrongInertia;
        }

        // 4. Near-singularity (MA57 `CNTL(2)` analog). An absolute floor, not
        //    a `min/max` ratio: an interior-point KKT is *designed* to become
        //    ill-conditioned as `μ→0`, so the ratio collapses on healthy
        //    full-rank systems near the solution.
        if self.cfg.singular_pivot_floor > 0.0 {
            if let Some(mp) = min_abs {
                if mp < self.cfg.singular_pivot_floor {
                    return ESymSolverStatus::Singular;
                }
            }
        }

        ESymSolverStatus::Success
    }

    fn record_factor_stats(&mut self, pattern_reused: bool, nnz_l: usize) {
        let nnz_a = self.matrix.as_ref().map(|m| m.values.len()).unwrap_or(0);
        let min_abs = self.inertia.min_abs_pivot_or_block_eigenvalue;
        let max_abs = self.inertia.max_abs_pivot_or_block_eigenvalue;
        let triple = self.inertia.triple();
        let neg = self.inertia.negative;
        let fill = if nnz_a > 0 {
            nnz_l as f64 / nnz_a as f64
        } else {
            0.0
        };

        let s = &mut self.summary;
        s.n_factors += 1;
        if pattern_reused {
            s.n_pattern_reuse += 1;
        } else {
            s.n_pattern_changes += 1;
        }
        if nnz_a > 0 {
            s.max_fill_ratio = Some(s.max_fill_ratio.map_or(fill, |p: f64| p.max(fill)));
        }
        if let Some(mp) = min_abs {
            s.min_abs_pivot = Some(s.min_abs_pivot.map_or(mp, |p: f64| p.min(mp)));
        }
        if let Some(mp) = max_abs {
            s.max_abs_pivot = Some(s.max_abs_pivot.map_or(mp, |p: f64| p.max(mp)));
        }
        s.last_inertia = Some(triple);
        s.last_nnz_a = Some(nnz_a);
        s.last_nnz_l = Some(nnz_l);

        if let Some(sink) = self.sink.as_ref() {
            if let Ok(mut guard) = sink.lock() {
                *guard = s.clone();
            }
        }

        // The same tracing span fields `pounce-feral` records, so an existing
        // `linear_solve` subscriber reads an RSLAB solve without changes.
        let span = tracing::Span::current();
        span.record("n", self.dim);
        span.record("matrix_nnz", nnz_a);
        span.record("factor_nnz", nnz_l);
        span.record("inertia_neg", neg);
        span.record("fill_ratio", fill);
    }

    /// Back-substitute `nrhs` right-hand sides against the stored factor.
    ///
    /// Two conversions live here, and both are the adapter's job rather than
    /// the caller's:
    ///
    /// * **Layout.** POUNCE packs a multi-RHS block **column-major**
    ///   (`rhs_vals[c * n + i]` is column `c`, row `i`); RSLAB's
    ///   `solve_ldlt_many` reads and writes a **row-major** `n × nrhs` buffer
    ///   (`b[i * nrhs + c]`), so that one pass over `L` serves every column.
    ///   The two disagree for every `nrhs > 1`, silently and with no
    ///   dimension error to catch it — the transposed answer has the right
    ///   shape. `packed_multi_rhs_matches_one_at_a_time` is the guard.
    /// * **Equilibration.** RSLAB holds a factor of `A_hat = D A D`, so
    ///   `A x = b` is `x = D · (A_hat⁻¹ · (D b))`. Folded into the transpose
    ///   passes rather than run as two extra sweeps.
    ///
    /// `rhs_scratch` is grown once and reused: a back-solve is the IPM's hot
    /// path, and RSLAB's entry point already returns one owned `Vec` per call.
    fn backsolve(&mut self, nrhs: Index, rhs_vals: &mut [Number]) -> ESymSolverStatus {
        let n = self.dim as usize;
        let nrhs = nrhs as usize;
        debug_assert_eq!(rhs_vals.len(), n * nrhs);
        let Some(factors) = self.factors.as_ref() else {
            tracing::error!(
                target: "pounce::linsol",
                "rslab: solve requested with no stored factor"
            );
            return ESymSolverStatus::FatalError;
        };

        // Column-major → row-major, scaling in on the way.
        self.rhs_scratch.resize(n * nrhs, 0.0);
        for c in 0..nrhs {
            for i in 0..n {
                self.rhs_scratch[i * nrhs + c] = rhs_vals[c * n + i] * self.scale[i];
            }
        }
        let x = match rslab::solve_ldlt_many(factors, &self.rhs_scratch, nrhs) {
            Ok(x) => x,
            Err(e) => {
                tracing::error!(
                    target: "pounce::linsol",
                    error = ?e, "rslab: back-substitution failed"
                );
                return ESymSolverStatus::FatalError;
            }
        };
        // Row-major → column-major, scaling out on the way.
        for c in 0..nrhs {
            for i in 0..n {
                rhs_vals[c * n + i] = x[i * nrhs + c] * self.scale[i];
            }
        }
        ESymSolverStatus::Success
    }
}

/// Record, for each input triplet, which CSC slot it landed in.
///
/// `from_triplets` buckets by column, sorts each column by row and sums
/// duplicates; the resulting `(col_ptr, row_idx)` is therefore a function of
/// the pattern alone. A binary search of `row_idx` within each column
/// reproduces the mapping without re-running the sort, and it stays valid for
/// the life of the pattern.
fn build_slot_map(rows_0: &[usize], cols_0: &[usize], m: &CscMatrix<f64>) -> Vec<usize> {
    let mut slot = vec![0usize; rows_0.len()];
    for (k, s) in slot.iter_mut().enumerate() {
        let (i, j) = (rows_0[k], cols_0[k]);
        let lo = m.col_ptr[j];
        let hi = m.col_ptr[j + 1];
        // `from_triplets` sorted this range by row index.
        let pos = m.row_idx[lo..hi]
            .binary_search(&i)
            .expect("every input triplet has a CSC slot");
        *s = lo + pos;
    }
    slot
}

impl SparseSymLinearSolverInterface for RslabSolverInterface {
    fn initialize_structure(
        &mut self,
        dim: Index,
        nonzeros: Index,
        ia: &[Index],
        ja: &[Index],
    ) -> ESymSolverStatus {
        debug_assert_eq!(ia.len(), nonzeros as usize);
        debug_assert_eq!(ja.len(), nonzeros as usize);
        self.dim = dim;
        self.nonzeros = nonzeros;
        // Triplet format is 1-based; RSLAB's CSC is 0-based.
        self.rows_0 = ia.iter().map(|&v| (v - 1) as usize).collect();
        self.cols_0 = ja.iter().map(|&v| (v - 1) as usize).collect();
        self.values = vec![0.0; nonzeros as usize];
        // The pattern changed: every cache keyed on it is stale.
        self.matrix = None;
        self.slot = None;
        self.symbolic = None;
        self.factors = None;
        self.scale = vec![1.0; dim as usize];
        self.scaled.clear();
        self.inertia = InertiaInfo::empty();
        self.explicit_zeros = 0;
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
            let s = self.factor(check_neg_evals, number_of_neg_evals);
            if s != ESymSolverStatus::Success {
                return s;
            }
        }
        self.backsolve(nrhs, rhs_vals)
    }

    fn number_of_neg_evals(&self) -> Index {
        self.negevals
    }

    fn increase_quality(&mut self) -> bool {
        // RSLAB has no pivot-tolerance ladder to climb: Bunch-Kaufman's
        // `alpha = (1 + sqrt(17))/8` is a compile-time constant
        // (`dense::ldlt_generic::bk_alpha`) and no `SolverSettings` knob
        // reaches it. Reporting `false` says "already at maximum quality",
        // which is what MA57 reports once its own ladder is exhausted, so the
        // caller's retry loop terminates instead of spinning.
        false
    }

    fn provides_inertia(&self) -> bool {
        true
    }

    fn multi_solve_matches_single_solve(&self, _nrhs: usize) -> bool {
        // `rslab::solve_ldlt_many` splits the RHS columns across rayon workers
        // and runs a per-chunk block kernel over a gathered row-major `n × w`
        // sub-block, so a column's sums need not be associated the way the
        // `nrhs = 1` path associates them. Answering `false` — the trait's
        // conservative default — keeps opportunistic batching off, which is
        // the right answer until someone measures it (pounce gh#729 is what
        // the question exists for).
        false
    }

    fn matrix_format(&self) -> EMatrixFormat {
        EMatrixFormat::TripletFormat
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1-based lower-triangle triplets of a small SPD matrix.
    fn spd_2x2() -> (Vec<Index>, Vec<Index>, Vec<Number>) {
        (vec![1, 2, 2], vec![1, 1, 2], vec![2.0, 1.0, 3.0])
    }

    fn factor_and_solve(
        n: Index,
        irn: &[Index],
        jcn: &[Index],
        vals: &[Number],
        rhs: &mut [Number],
    ) -> (RslabSolverInterface, ESymSolverStatus) {
        let mut s = RslabSolverInterface::new();
        s.initialize_structure(n, irn.len() as Index, irn, jcn);
        s.values_array_mut().copy_from_slice(vals);
        let st = s.multi_solve(true, irn, jcn, 1, rhs, false, 0);
        (s, st)
    }

    #[test]
    fn factors_and_solves_an_spd_system() {
        let (irn, jcn, vals) = spd_2x2();
        let mut rhs = vec![3.0, 4.0];
        let (s, st) = factor_and_solve(2, &irn, &jcn, &vals, &mut rhs);
        assert_eq!(st, ESymSolverStatus::Success);
        assert!((rhs[0] - 1.0).abs() < 1e-12, "x0 = {}", rhs[0]);
        assert!((rhs[1] - 1.0).abs() < 1e-12, "x1 = {}", rhs[1]);
        assert_eq!(s.number_of_neg_evals(), 0);
        assert_eq!(s.inertia_info().triple(), (2, 0, 0));
        assert!(s.inertia_info().reliable);
    }

    /// The values refill replays through the cached slot map; a refactor must
    /// therefore answer the *new* system, not the old one.
    #[test]
    fn refactor_reuses_the_pattern_and_tracks_new_values() {
        let (irn, jcn, vals) = spd_2x2();
        let mut rhs = vec![3.0, 4.0];
        let (mut s, st) = factor_and_solve(2, &irn, &jcn, &vals, &mut rhs);
        assert_eq!(st, ESymSolverStatus::Success);

        // Perturb to [[4, 1], [1, 5]] and solve against (5, 6).
        s.values_array_mut().copy_from_slice(&[4.0, 1.0, 5.0]);
        let mut rhs2 = vec![5.0, 6.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs2, false, 0),
            ESymSolverStatus::Success
        );
        let r0 = 4.0 * rhs2[0] + rhs2[1] - 5.0;
        let r1 = rhs2[0] + 5.0 * rhs2[1] - 6.0;
        assert!(r0.abs() < 1e-10 && r1.abs() < 1e-10, "residual ({r0}, {r1})");
        // One analysis, two factorizations.
        let sum = s.summary();
        assert_eq!(sum.n_factors, 2);
        assert_eq!(sum.n_pattern_changes, 1);
        assert_eq!(sum.n_pattern_reuse, 1);
        assert_eq!(sum.solver_name, "rslab");
    }

    /// Duplicate triplets are summed, both on the build and on every refill —
    /// the replay must zero the CSC slot first rather than assign into it.
    #[test]
    fn duplicate_triplets_sum_on_the_refill_too() {
        // Two entries both at (1, 1), summing to the diagonal 2.0.
        let irn = vec![1, 1, 2, 2];
        let jcn = vec![1, 1, 1, 2];
        let vals = vec![1.5, 0.5, 1.0, 3.0];
        let mut rhs = vec![3.0, 4.0];
        let (mut s, st) = factor_and_solve(2, &irn, &jcn, &vals, &mut rhs);
        assert_eq!(st, ESymSolverStatus::Success);
        assert!((rhs[0] - 1.0).abs() < 1e-12 && (rhs[1] - 1.0).abs() < 1e-12);

        // Refill with the same effective matrix split differently.
        s.values_array_mut().copy_from_slice(&[0.25, 1.75, 1.0, 3.0]);
        let mut rhs2 = vec![3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs2, false, 0),
            ESymSolverStatus::Success
        );
        assert!(
            (rhs2[0] - 1.0).abs() < 1e-12 && (rhs2[1] - 1.0).abs() < 1e-12,
            "refill did not re-sum duplicates: {rhs2:?}"
        );
    }

    /// `new_matrix = false` is a pure back-substitution against the stored
    /// factor: it must not refactor, and it must answer the factored system.
    #[test]
    fn back_substitution_reuses_the_stored_factor() {
        let (irn, jcn, vals) = spd_2x2();
        let mut rhs = vec![3.0, 4.0];
        let (mut s, _) = factor_and_solve(2, &irn, &jcn, &vals, &mut rhs);
        let before = s.summary().n_factors;
        let mut rhs2 = vec![5.0, 5.0];
        assert_eq!(
            s.multi_solve(false, &irn, &jcn, 1, &mut rhs2, false, 0),
            ESymSolverStatus::Success
        );
        assert_eq!(s.summary().n_factors, before, "back-solve refactored");
        // A x = (5, 5) for A = [[2,1],[1,3]] → x = (2, 1).
        assert!((rhs2[0] - 2.0).abs() < 1e-12 && (rhs2[1] - 1.0).abs() < 1e-12);
    }

    /// A packed multi-RHS solve agrees with the columns solved one at a time.
    #[test]
    fn packed_multi_rhs_matches_one_at_a_time() {
        let (irn, jcn, vals) = spd_2x2();
        let mut warm = vec![0.0, 0.0];
        let (mut s, _) = factor_and_solve(2, &irn, &jcn, &vals, &mut warm);
        let mut packed = vec![3.0, 4.0, 5.0, 5.0, 2.0, 6.0];
        assert_eq!(
            s.multi_solve(false, &irn, &jcn, 3, &mut packed, false, 0),
            ESymSolverStatus::Success
        );
        for (c, col) in [[3.0, 4.0], [5.0, 5.0], [2.0, 6.0]].iter().enumerate() {
            let mut one = col.to_vec();
            assert_eq!(
                s.multi_solve(false, &irn, &jcn, 1, &mut one, false, 0),
                ESymSolverStatus::Success
            );
            for i in 0..2 {
                let got = packed[2 * c + i];
                assert!(
                    (got - one[i]).abs() <= 1e-12 * one[i].abs().max(1.0),
                    "column {c} entry {i}: packed {got}, single {}",
                    one[i]
                );
            }
        }
    }

    /// An exactly singular matrix is `Singular`, not a panic and not a
    /// silently-wrong answer.
    #[test]
    fn a_singular_matrix_reports_singular() {
        // [[1, 1], [1, 1]] — rank 1.
        let irn = vec![1, 2, 2];
        let jcn = vec![1, 1, 2];
        let mut rhs = vec![1.0, 1.0];
        let (_, st) = factor_and_solve(2, &irn, &jcn, &[1.0, 1.0, 1.0], &mut rhs);
        assert_eq!(st, ESymSolverStatus::Singular);
    }

    /// A mismatching negative count on a well-conditioned factor is
    /// `WrongInertia` — the `δ_w` branch.
    #[test]
    fn a_mismatching_count_on_a_healthy_factor_is_wrong_inertia() {
        let (irn, jcn, vals) = spd_2x2();
        let mut s = RslabSolverInterface::new();
        s.initialize_structure(2, 3, &irn, &jcn);
        s.values_array_mut().copy_from_slice(&vals);
        let mut rhs = vec![3.0, 4.0];
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 1),
            ESymSolverStatus::WrongInertia
        );
    }

    /// `as_any` downcasts back to the concrete backend, so a test can check
    /// how a factory configured it (the seam pounce gh#825 exists for).
    #[test]
    fn as_any_downcasts_to_the_backend() {
        let s = RslabSolverInterface::new();
        let obj: &dyn SparseSymLinearSolverInterface = &s;
        let back = obj
            .as_any()
            .and_then(|a| a.downcast_ref::<RslabSolverInterface>())
            .expect("downcast");
        assert_eq!(back.config().pivot, PivotPolicy::Exact);
    }
}
