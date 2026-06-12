//! Mutable algorithm state — port of `Algorithm/IpIpoptData.{hpp,cpp}`.
//!
//! Holds the iterate trio (`curr` / `trial` / `delta`), the affine
//! step `delta_aff`, current barrier parameter `mu`, fraction-to-the-
//! boundary `tau`, iteration count, tolerances, and the four PD
//! perturbations. No additional-data plug in v1.0 (CG penalty is
//! deferred to Phase 10).
//!
//! Phase 5 ships the full state holder; concrete strategies (line
//! search, mu update, etc.) read/write fields here as their inputs
//! and outputs.

use crate::iterates_vector::IteratesVector;
use pounce_common::timing::TimingStatistics;
use pounce_common::types::{Index, Number};
use pounce_linalg::SymMatrix;
use std::cell::RefCell;
use std::rc::Rc;

/// Primal-dual perturbation triple (`delta_x`, `delta_s`, `delta_c`,
/// `delta_d`) — port of the four `current_perturbation` fields in
/// `IpIpoptData.hpp`.
#[derive(Debug, Default, Clone, Copy)]
pub struct PdPerturbations {
    pub delta_x: Number,
    pub delta_s: Number,
    pub delta_c: Number,
    pub delta_d: Number,
}

/// KKT-factorization diagnostics captured for the interactive debugger
/// after a search-direction solve. Only populated when a debugger is
/// installed (see `IpoptAlgorithm`); inspected via `DebugCtx::kkt`.
#[derive(Clone, Debug, Default)]
pub struct KktDebug {
    /// The outer iteration this factorization was assembled at. Lets the
    /// debugger label `viz kkt` / `viz L` with the iteration the system
    /// actually came from — at an `iter_start` pause that's the *previous*
    /// iteration (the step that produced the current point), not the
    /// iterate you're standing on.
    pub iter: i32,
    /// Dimension of the augmented system (n + m).
    pub dim: i32,
    /// Negative eigenvalues reported by the factorization (-1 if the
    /// backend doesn't provide inertia).
    pub n_neg: i32,
    /// Whether the backend reports inertia at all.
    pub provides_inertia: bool,
    /// Debug string of the last factorization status.
    pub status: String,
    /// Assembled KKT triplets `(dim, irn, jcn, vals)`, 1-based lower
    /// triangle — for `viz kkt`. Captured when a debugger is attached.
    pub matrix: Option<(i32, Vec<i32>, Vec<i32>, Vec<f64>)>,
    /// `LDLᵀ` factor pattern (+ values) — for `viz L`. Captured only
    /// after the debugger opts in (it's the expensive piece).
    pub l_factor: Option<pounce_linsol::FactorPattern>,
}

/// Mutable state passed down through the algorithm. Owned by
/// `IpoptAlgorithm`; strategies access via `Rc<RefCell<IpoptData>>`.
pub struct IpoptData {
    pub curr: Option<IteratesVector>,
    pub trial: Option<IteratesVector>,
    pub delta: Option<IteratesVector>,
    pub delta_aff: Option<IteratesVector>,
    /// Pure centering step — solution of the primal-dual system with
    /// RHS `(0, 0, 0, 0, μ̄·1, μ̄·1, μ̄·1, μ̄·1)` (where μ̄ = avrg_compl)
    /// per upstream `IpQualityFunctionMuOracle.cpp:227-247`. Used by
    /// the quality-function oracle to assemble σ-step trial points
    /// without re-factorising for each candidate σ.
    pub delta_cen: Option<IteratesVector>,

    /// Hessian of the Lagrangian for the *current* iterate. Set by
    /// `HessianUpdater` (exact or quasi-Newton). Mirrors `IpIpoptData::W_`.
    pub w: Option<Rc<dyn SymMatrix>>,

    pub iter_count: Index,
    pub curr_mu: Number,
    pub curr_tau: Number,
    pub tol: Number,

    pub perturbations: PdPerturbations,

    /// KKT-factorization diagnostics for the debugger (set after a
    /// search-direction solve when a debugger is installed). The full
    /// matrix triplets and `LDLᵀ` factor are captured here whenever the
    /// debugger is stepping (see `DebugHook::wants_kkt_capture`) and
    /// dropped when it detaches, so `viz kkt` / `viz L` always have the
    /// previous iteration's system to look back at without paying the
    /// O(nnz) assembly during a free run.
    pub kkt_debug: Option<KktDebug>,

    /// Set after a successful trial-acceptance step in the line
    /// search. Cleared on accept.
    pub info_alpha_primal: Number,
    pub info_alpha_dual: Number,

    /// Mirrors `IpIpoptData::info_regu_x_`.
    pub info_regu_x: Number,

    /// Mirrors `IpIpoptData::info_skip_output_`.
    pub info_skip_output: bool,

    /// Mirrors `IpIpoptData::info_string_`. Free-form text the
    /// iteration output appends to its line.
    pub info_string: String,

    /// Mirrors `IpIpoptData::tiny_step_flag_`. Set by the line search
    /// when an alpha→0 trial is detected; the main loop reads it on
    /// the next pass to decide between "tiny step accept" and bail.
    pub tiny_step_flag: bool,

    /// Emergency restoration request from the μ-update layer. Set by
    /// [`AdaptiveMuUpdate`] when the probing oracle's input iterate
    /// is corrupted (`curr_avrg_compl` ≫ `curr_mu`) so the main loop
    /// invokes restoration instead of letting the oracle snap μ up
    /// many orders of magnitude. Pounce-specific guard; no upstream
    /// counterpart. See pounce#58.
    pub request_resto: bool,

    /// One-char marker the iteration output puts in front of
    /// `alpha_primal` (e.g. `'f'` for filter, `'r'` for restoration,
    /// `'h'` for the very first iterate). Mirrors
    /// `IpIpoptData::info_alpha_primal_char_`.
    pub info_alpha_primal_char: char,

    /// Number of trial points evaluated in the most recent line
    /// search. Mirrors `IpIpoptData::info_ls_count_`.
    pub info_ls_count: Index,

    /// The wall-clock at the last `OrigIterationOutput::WriteOutput`
    /// pass. Phase 7 uses this to decide whether to re-print the
    /// header. Mirrors `IpIpoptData::info_last_output_`.
    pub info_last_output: Number,

    /// Iterations since the iteration header was last printed. Phase
    /// 7 reprints every `print_frequency_iter` lines. Mirrors
    /// `IpIpoptData::info_iters_since_header_`.
    pub info_iters_since_header: Index,

    /// Shared per-subsystem timing accumulator. Mirrors upstream's
    /// `IpoptData::TimingStats_`. `IpoptApplication` constructs a single
    /// instance per solve and shares it (via `Rc`) with the algorithm,
    /// NLP, and KKT solver so each can record its own contribution.
    /// Defaults to a fresh empty instance for the structural unit tests
    /// that don't go through `IpoptApplication`.
    pub timing: Rc<TimingStatistics>,
}

impl Default for IpoptData {
    fn default() -> Self {
        Self::new()
    }
}

impl IpoptData {
    pub fn new() -> Self {
        Self {
            curr: None,
            trial: None,
            delta: None,
            delta_aff: None,
            delta_cen: None,
            w: None,
            iter_count: 0,
            curr_mu: 0.1,
            curr_tau: 0.99,
            tol: 1e-8,
            perturbations: PdPerturbations::default(),
            kkt_debug: None,
            info_alpha_primal: 0.0,
            info_alpha_dual: 0.0,
            info_regu_x: 0.0,
            info_skip_output: false,
            info_string: String::new(),
            tiny_step_flag: false,
            request_resto: false,
            info_alpha_primal_char: ' ',
            info_ls_count: 0,
            info_last_output: -1.0,
            info_iters_since_header: 0,
            timing: Rc::new(TimingStatistics::new()),
        }
    }

    /// Append text to `info_string`. Mirrors `IpIpoptData::Append_info_string`.
    pub fn append_info_string(&mut self, s: &str) {
        self.info_string.push_str(s);
    }

    /// Reset per-iteration info fields. Mirrors the top of
    /// `IpoptAlgorithm::Optimize`'s loop body.
    pub fn reset_info(&mut self) {
        self.info_string.clear();
        self.info_skip_output = false;
        self.info_alpha_primal_char = ' ';
        self.info_ls_count = 0;
        self.info_regu_x = 0.0;
    }

    /// Replace `curr` with the previously-set `trial`. Mirrors
    /// `IpIpoptData::AcceptTrialPoint`, which `DBG_ASSERT`s a trial is
    /// staged before promoting it (upstream always runs a line search that
    /// stages one). pounce additionally supports a bookkeeping-only
    /// `iterate()` path (no NLP + no search_dir, per the module docs) that
    /// runs the per-iteration bookkeeping without computing a step, so
    /// `trial` may be unset here. Promoting `None` would null out `curr`
    /// and make the next iteration's CQ accessor (`IpoptCq::curr_iv`) hit
    /// `unreachable!`; preserve `curr` when nothing is staged.
    pub fn accept_trial_point(&mut self) {
        if let Some(trial) = self.trial.take() {
            self.curr = Some(trial);
        }
    }

    /// Set the trial iterate from a primal step `delta_x`/`delta_s`
    /// scaled by `alpha_p` and a dual step scaled by `alpha_d`.
    /// Phase 5 ships only the structural plumbing; the actual
    /// arithmetic is implemented once the line search lands in
    /// Phase 7.
    pub fn set_trial(&mut self, trial: IteratesVector) {
        self.trial = Some(trial);
    }

    pub fn set_curr(&mut self, curr: IteratesVector) {
        self.curr = Some(curr);
    }

    pub fn set_delta(&mut self, d: IteratesVector) {
        self.delta = Some(d);
    }

    pub fn set_delta_aff(&mut self, d: IteratesVector) {
        self.delta_aff = Some(d);
    }

    pub fn set_delta_cen(&mut self, d: IteratesVector) {
        self.delta_cen = Some(d);
    }
}

/// Convenience handle. Mirrors how upstream passes the data object
/// around as `SmartPtr<IpoptData>`.
pub type IpoptDataHandle = Rc<RefCell<IpoptData>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iterates_vector::IteratesVector;
    use pounce_linalg::dense_vector::DenseVectorSpace;
    use pounce_linalg::Vector;
    use std::rc::Rc as StdRc;

    fn zero_iv() -> IteratesVector {
        let z = |n| StdRc::new(DenseVectorSpace::new(n).make_new_dense()) as StdRc<dyn Vector>;
        IteratesVector::new(z(2), z(1), z(1), z(1), z(2), z(2), z(1), z(1))
    }

    #[test]
    fn accept_trial_point_promotes_trial_to_curr() {
        let mut d = IpoptData::new();
        d.set_trial(zero_iv());
        assert!(d.curr.is_none());
        d.accept_trial_point();
        assert!(d.curr.is_some());
        assert!(d.trial.is_none());
    }

    // Regression for M2 (dev-notes/code-review-2026-06.md): in the
    // bookkeeping-only `iterate()` path (no NLP + no search_dir), step 5
    // is skipped so no trial is staged, yet `accept_trial_point()` is still
    // called. The old `curr = trial.take()` then nulled out `curr`, and the
    // next iteration's CQ accessor (`ipopt_cq.rs` `curr_iv`) hit
    // `unreachable!`. With no trial staged, `curr` must be preserved.
    #[test]
    fn accept_trial_point_preserves_curr_when_no_trial_staged() {
        let mut d = IpoptData::new();
        d.set_curr(zero_iv());
        assert!(d.curr.is_some());
        assert!(d.trial.is_none());
        d.accept_trial_point();
        // Must NOT destroy the current iterate when nothing is staged.
        assert!(
            d.curr.is_some(),
            "accept_trial_point() nulled curr with no trial staged"
        );
        assert!(d.trial.is_none());
    }
}
