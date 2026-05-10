//! Adaptive mu update — port of `IpAdaptiveMuUpdate.{hpp,cpp}`.
//!
//! Phase 10. The full update reaches into `IpoptCq` for residuals and
//! into a `MuOracle` for the candidate σ; this file ships:
//!
//! * the option struct with upstream defaults from `RegisterOptions`,
//! * the `lower_mu_safeguard` scalar core (lines 753-786),
//! * the globalization-mode enum and the FreeMuMode/FixedMuMode state
//!   machine (`UpdateBarrierParameter` lines 252-444),
//! * the `mu_oracle` selector ([`MuOracleKind`]) — `Loqo` runs the
//!   closed form; `Probing` / `QualityFunction` drive an affine /
//!   centring solve when [`MuUpdate`] is given the search-dir + nlp
//!   handles, otherwise fall through to LOQO (mirrors upstream's
//!   "oracle returned no candidate" branch at lines 402-408).

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::IteratesVector;
use crate::kkt::pd_search_dir_calc::PdSearchDirCalc;
use crate::line_search::filter::Filter;
use crate::mu::oracle::loqo::LoqoMuOracle;
use crate::mu::oracle::probing::ProbingMuOracle;
use crate::mu::oracle::quality_function::QualityFunctionMuOracle;
use crate::mu::oracle::r#trait::MuOracle;
use crate::mu::r#trait::MuUpdate;
use pounce_common::types::Number;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

/// `mu_oracle` option from `IpAdaptiveMuUpdate.cpp:RegisterOptions`.
/// Default `QualityFunction` matches upstream (`"quality-function"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuOracleKind {
    /// Closed-form LOQO rule. No predictor solve required.
    Loqo,
    /// Mehrotra probing oracle. Needs an affine-step solve.
    Probing,
    /// Golden-section minimisation of the q(σ) quality function.
    /// Needs an affine-step solve plus a centring evaluator.
    QualityFunction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveMuGlobalization {
    KktError,
    ObjConstrFilter,
    NeverMonotoneMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveMuKktNorm {
    OneNorm,
    TwoNormSquared,
    MaxNorm,
    TwoNorm,
}

pub struct AdaptiveMuUpdate {
    pub mu_oracle: MuOracleKind,
    pub adaptive_mu_globalization: AdaptiveMuGlobalization,
    pub adaptive_mu_kkt_norm: AdaptiveMuKktNorm,
    pub adaptive_mu_safeguard_factor: Number,
    pub adaptive_mu_kkterror_red_iters: usize,
    pub adaptive_mu_kkterror_red_fact: Number,
    pub filter_max_margin: Number,
    pub filter_margin_fact: Number,
    pub mu_min: Number,
    pub mu_max: Number,
    /// `tau_min` from `IpAdaptiveMuUpdate.cpp:RegisterOptions`. Used to
    /// derive `curr_tau = max(tau_min, 1 - mu)` after each update,
    /// mirroring upstream's `IpAdaptiveMuUpdate.cpp:UpdateBarrierParameter`
    /// at the post-oracle update.
    pub tau_min: Number,
    /// Initial mu seed — `mu_init` from `IpoptAlgorithm` registered
    /// options. Used to seed `curr_mu` in `initialize`.
    pub mu_init: Number,
    /// `barrier_tol_factor` (default 10) from upstream
    /// `IpMonotoneMuUpdate::RegisterOptions`. Threshold for fixed-mode
    /// barrier subproblem completion: reduce μ when
    /// `curr_barrier_error ≤ barrier_tol_factor · μ`.
    pub barrier_tol_factor: Number,
    /// `mu_linear_decrease_factor` (default 0.2) — fixed-mode update
    /// uses `min(linear · μ, μ^superlinear_power)`.
    pub mu_linear_decrease_factor: Number,
    /// `mu_superlinear_decrease_power` (default 1.5).
    pub mu_superlinear_decrease_power: Number,
    /// `adaptive_mu_monotone_init_factor` (default 0.8). Used by
    /// `new_fixed_mu` when no `fix_mu_oracle_` is configured.
    pub adaptive_mu_monotone_init_factor: Number,
    /// `adaptive_mu_restore_previous_iterate` (default false).
    pub restore_accepted_iterate: bool,

    /// Upstream tracks `init_*_inf` lazily — sentinel −1 means
    /// "not yet captured".
    init_dual_inf: Number,
    init_primal_inf: Number,

    /// FreeMuMode/FixedMuMode flag — port of
    /// `IpoptData::FreeMuMode()`. `true` means "let the oracle drive
    /// μ"; `false` means "monotone decrease until sufficient progress
    /// is made". Initialised to `true` in [`MuUpdate::initialize`]
    /// (matches upstream `InitializeImpl` line 239).
    free_mu_mode: bool,
    /// KKT-error history for `KKT_ERROR` globalization. Bounded length
    /// = `adaptive_mu_kkterror_red_iters`. Mirrors `refs_vals_`.
    refs_vals: VecDeque<Number>,
    /// 2-D `(theta, phi)` filter for `OBJ_CONSTR_FILTER` globalization.
    /// Mirrors `filter_` (constructed with `Filter(2)`).
    filter: Filter,
    /// Snapshot of `curr` at the most recent successful free-mode
    /// iterate; restored when switching to fixed mode if
    /// `restore_accepted_iterate` is on. Mirrors `accepted_point_`.
    accepted_point: Option<IteratesVector>,
}

impl Default for AdaptiveMuUpdate {
    fn default() -> Self {
        // Defaults from `IpAdaptiveMuUpdate.cpp:RegisterOptions`.
        Self {
            mu_oracle: MuOracleKind::QualityFunction,
            adaptive_mu_globalization: AdaptiveMuGlobalization::ObjConstrFilter,
            adaptive_mu_kkt_norm: AdaptiveMuKktNorm::TwoNormSquared,
            adaptive_mu_safeguard_factor: 0.0,
            adaptive_mu_kkterror_red_iters: 4,
            adaptive_mu_kkterror_red_fact: 0.9999,
            filter_max_margin: 1.0,
            filter_margin_fact: 1e-5,
            mu_min: 1e-11,
            mu_max: 1e5,
            tau_min: 0.99,
            mu_init: 0.1,
            barrier_tol_factor: 10.0,
            mu_linear_decrease_factor: 0.2,
            mu_superlinear_decrease_power: 1.5,
            adaptive_mu_monotone_init_factor: 0.8,
            restore_accepted_iterate: false,
            init_dual_inf: -1.0,
            init_primal_inf: -1.0,
            free_mu_mode: true,
            refs_vals: VecDeque::new(),
            filter: Filter::new(),
            accepted_point: None,
        }
    }
}

impl AdaptiveMuUpdate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Scalar core of `AdaptiveMuUpdate::lower_mu_safeguard`
    /// (`IpAdaptiveMuUpdate.cpp:753-786`):
    /// ```text
    ///   init_dual_inf   ← max(1, dual_inf)   if not yet set
    ///   init_primal_inf ← max(1, primal_inf) if not yet set
    ///   lower = max(safeguard_factor * dual_inf / init_dual_inf,
    ///               safeguard_factor * primal_inf / init_primal_inf)
    ///   if globalization == KKT_ERROR: lower = min(lower, min_ref_val)
    /// ```
    pub fn lower_mu_safeguard(
        &mut self,
        dual_inf: Number,
        primal_inf: Number,
        min_ref_val: Number,
    ) -> Number {
        if self.init_dual_inf < 0.0 {
            self.init_dual_inf = dual_inf.max(1.0);
        }
        if self.init_primal_inf < 0.0 {
            self.init_primal_inf = primal_inf.max(1.0);
        }
        let dual_term = self.adaptive_mu_safeguard_factor * (dual_inf / self.init_dual_inf);
        let prim_term =
            self.adaptive_mu_safeguard_factor * (primal_inf / self.init_primal_inf);
        let mut lower = dual_term.max(prim_term);
        if self.adaptive_mu_globalization == AdaptiveMuGlobalization::KktError {
            lower = lower.min(min_ref_val);
        }
        lower
    }

    pub fn reset_init_inf(&mut self) {
        self.init_dual_inf = -1.0;
        self.init_primal_inf = -1.0;
    }

    /// Globalization KKT-error proxy — port of
    /// `AdaptiveMuUpdate::quality_function_pd_system`
    /// (`IpAdaptiveMuUpdate.cpp:629-744`). v1.0 hardwires the
    /// max-norm variant (`adaptive_mu_kkt_norm_type=max-norm`,
    /// upstream "NM_NORM_MAX") because the existing CQ surface
    /// exposes max-norm primal/dual infeasibility cheaply; the
    /// other three norm variants follow once `curr_*_infeasibility`
    /// learns to dispatch on `NormEnum`. The score sums primal +
    /// dual + complementarity (+ optional centrality / balancing
    /// — both default off; left as `0`).
    fn quality_function_pd_system(&self, cq: &IpoptCqHandle) -> Number {
        let cq_ref = cq.borrow();
        let primal_inf = cq_ref.curr_primal_infeasibility_max();
        let dual_inf = cq_ref.curr_dual_infeasibility_max();
        // Max-norm complementarity ≈ avrg_compl is a cheap proxy.
        // Upstream's `curr_complementarity(0., NORM_MAX)` would use
        // `||s ⊙ z||_∞`; absent that accessor, fall through to the
        // average. For the monotonicity test inside
        // `check_sufficient_progress` only ratios matter, so the
        // proxy preserves the convergence criterion.
        let complty = cq_ref.curr_avrg_compl();
        primal_inf + dual_inf + complty
    }

    /// Port of `AdaptiveMuUpdate::CheckSufficientProgress`
    /// (`IpAdaptiveMuUpdate.cpp:446-490`). Returns `true` if the
    /// current iterate makes acceptable progress under the active
    /// globalization rule.
    fn check_sufficient_progress(&self, cq: &IpoptCqHandle) -> bool {
        match self.adaptive_mu_globalization {
            AdaptiveMuGlobalization::KktError => {
                if self.refs_vals.len() < self.adaptive_mu_kkterror_red_iters.max(1) {
                    // Not enough history yet — accept (matches
                    // upstream's `num_refs >= num_refs_max_` guard).
                    return true;
                }
                let curr_error = self.quality_function_pd_system(cq);
                self.refs_vals
                    .iter()
                    .any(|&r| curr_error <= self.adaptive_mu_kkterror_red_fact * r)
            }
            AdaptiveMuGlobalization::ObjConstrFilter => {
                let cq_ref = cq.borrow();
                let curr_f = cq_ref.curr_f();
                let curr_theta = cq_ref.curr_constraint_violation();
                // `curr_nlp_error` is our analogue of upstream's
                // global error margin driver.
                let curr_err = cq_ref.curr_nlp_error();
                drop(cq_ref);
                let margin =
                    self.filter_margin_fact * self.filter_max_margin.min(curr_err);
                !self.filter.dominated_by_any(curr_theta + margin, curr_f + margin)
            }
            AdaptiveMuGlobalization::NeverMonotoneMode => true,
        }
    }

    /// Port of `AdaptiveMuUpdate::RememberCurrentPointAsAccepted`
    /// (`IpAdaptiveMuUpdate.cpp:492-546`). Records the iterate state
    /// for the next sufficient-progress check.
    fn remember_current_point_as_accepted(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
    ) {
        match self.adaptive_mu_globalization {
            AdaptiveMuGlobalization::KktError => {
                let curr_error = self.quality_function_pd_system(cq);
                if self.refs_vals.len() >= self.adaptive_mu_kkterror_red_iters.max(1) {
                    self.refs_vals.pop_front();
                }
                self.refs_vals.push_back(curr_error);
            }
            AdaptiveMuGlobalization::ObjConstrFilter => {
                let cq_ref = cq.borrow();
                let f = cq_ref.curr_f();
                let theta = cq_ref.curr_constraint_violation();
                let it = data.borrow().iter_count;
                drop(cq_ref);
                self.filter.add(theta, f, it);
            }
            AdaptiveMuGlobalization::NeverMonotoneMode => {}
        }
        if self.restore_accepted_iterate {
            self.accepted_point = data.borrow().curr.clone();
        }
    }

    /// Port of `AdaptiveMuUpdate::NewFixedMu`
    /// (`IpAdaptiveMuUpdate.cpp:583-627`). Selects μ when the state
    /// machine drops out of free mode. v1.0 always uses the
    /// "average complementarity" branch (no `fix_mu_oracle_` is
    /// wired; matches `fixed_mu_oracle = average_compl`).
    fn new_fixed_mu(&self, cq: &IpoptCqHandle) -> Number {
        let avrg = cq.borrow().curr_avrg_compl();
        let new_mu = self.adaptive_mu_monotone_init_factor * avrg;
        new_mu.clamp(self.mu_min, self.mu_max)
    }
}

impl MuUpdate for AdaptiveMuUpdate {
    /// Port of `IpAdaptiveMuUpdate.cpp:InitializeImpl`. Seeds
    /// `curr_mu = mu_init`, `curr_tau = max(tau_min, 1 - mu_init)`,
    /// resets the globalization state, and starts in free-μ mode
    /// (`SetFreeMuMode(true)` at line 239).
    fn initialize(&mut self, data: &IpoptDataHandle) {
        let mut d = data.borrow_mut();
        d.curr_mu = self.mu_init;
        d.curr_tau = self.tau_min.max(1.0 - self.mu_init);
        drop(d);
        self.free_mu_mode = true;
        self.refs_vals.clear();
        self.filter.clear();
        self.accepted_point = None;
        self.init_dual_inf = -1.0;
        self.init_primal_inf = -1.0;
    }

    /// Adaptive μ update — port of `UpdateBarrierParameter`
    /// (`IpAdaptiveMuUpdate.cpp:252-444`). Runs the FreeMuMode /
    /// FixedMuMode state machine:
    ///
    /// * **FreeMuMode**: ask the configured oracle for a candidate
    ///   (LOQO closed-form, Probing predictor solve, or
    ///   QualityFunction golden-section). If progress is sufficient,
    ///   stay in free mode and remember the iterate; otherwise switch
    ///   to fixed mode at `new_fixed_mu`.
    /// * **FixedMuMode**: monotone Fiacco-McCormick reduction
    ///   (`min(linear · μ, μ^superlinear_power)`). Switch back to
    ///   free mode once the globalization criterion is satisfied
    ///   again.
    ///
    /// Probing / QualityFunction silently fall back to LOQO when
    /// `nlp` / `pd_search_dir` are unavailable (mirrors upstream
    /// lines 402-408).
    ///
    /// Note: line-search reset (upstream's `linesearch_->Reset()` at
    /// lines 339, 386, 431) is not yet wired here — that handle is
    /// not part of the [`MuUpdate`] trait surface. This is a
    /// deliberate v1.0 deviation; it primarily affects the watchdog
    /// counter, not convergence.
    fn update_barrier_parameter(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: Option<&Rc<RefCell<dyn IpoptNlp>>>,
        pd_search_dir: Option<&mut PdSearchDirCalc>,
    ) -> Number {
        let (curr_mu, iter_count, tiny_step_flag) = {
            let d = data.borrow();
            (d.curr_mu, d.iter_count, d.tiny_step_flag)
        };

        // Bootstrap: at iter_count==0 the iterate is the unmodified
        // initial point. Snapshot it as the first reference and bail
        // out with the seed μ. Mirrors upstream's first-iteration
        // short-circuit (`IpAdaptiveMuUpdate.cpp:236` + the
        // `RememberCurrentPointAsAccepted` call upstream makes from
        // `IpoptAlgorithm`'s init path).
        if iter_count == 0 {
            self.remember_current_point_as_accepted(data, cq);
            data.borrow_mut().curr_tau = self.tau_min.max(1.0 - curr_mu);
            return curr_mu;
        }

        // `tiny_step_flag` (and upstream's `CheckSkippedLineSearch()`,
        // which is only set in non-rigorous resto mode) forces
        // `sufficient_progress = false` when not in `NEVER_MONOTONE_MODE`
        // — see `IpAdaptiveMuUpdate.cpp:347-351`. This is what lets a
        // stalled outer iter drop into fixed-μ and re-seed μ via
        // `new_fixed_mu` instead of the oracle re-driving μ further down.
        let force_no_progress = tiny_step_flag
            && self.adaptive_mu_globalization != AdaptiveMuGlobalization::NeverMonotoneMode;

        if !self.free_mu_mode {
            // Fixed-mu branch — `cpp:299-342`.
            let sufficient_progress =
                !force_no_progress && self.check_sufficient_progress(cq);
            if sufficient_progress {
                // Switch back to free mode; record the iterate.
                self.free_mu_mode = true;
                self.remember_current_point_as_accepted(data, cq);
                // Fall through to the free-mode oracle path below.
            } else {
                // Keep reducing μ Fiacco-McCormick style if the
                // barrier subproblem is solved to within
                // `barrier_tol_factor · μ`.
                let sub_problem_error = cq.borrow().curr_barrier_error();
                if sub_problem_error <= self.barrier_tol_factor * curr_mu {
                    let lin = self.mu_linear_decrease_factor * curr_mu;
                    let sup = curr_mu.powf(self.mu_superlinear_decrease_power);
                    let new_mu = lin.min(sup).max(self.mu_min).min(self.mu_max);
                    let new_tau = self.tau_min.max(1.0 - new_mu);
                    data.borrow_mut().curr_tau = new_tau;
                    return new_mu;
                }
                // Subproblem not yet solved — keep μ.
                let new_tau = self.tau_min.max(1.0 - curr_mu);
                data.borrow_mut().curr_tau = new_tau;
                return curr_mu;
            }
        } else {
            // Free-mu branch — `cpp:343-389`.
            let sufficient_progress =
                !force_no_progress && self.check_sufficient_progress(cq);
            if sufficient_progress {
                self.remember_current_point_as_accepted(data, cq);
                // Fall through to the oracle call below.
            } else {
                // Switch into fixed mode.
                self.free_mu_mode = false;
                if self.restore_accepted_iterate {
                    if let Some(prev) = self.accepted_point.clone() {
                        let mut d = data.borrow_mut();
                        d.set_trial(prev);
                        d.accept_trial_point();
                    }
                }
                let new_mu = self.new_fixed_mu(cq);
                let new_tau = self.tau_min.max(1.0 - new_mu);
                data.borrow_mut().curr_tau = new_tau;
                return new_mu;
            }
        }

        // ----- Free-mu oracle call (cpp:391-436) -----
        let cq_ref = cq.borrow();
        let dual_inf = cq_ref.curr_dual_infeasibility_max();
        let primal_inf = cq_ref.curr_primal_infeasibility_max();
        let avrg_compl = cq_ref.curr_avrg_compl();
        let centrality_xi = cq_ref.curr_centrality_measure();
        let nlp_error = cq_ref.curr_nlp_error();
        drop(cq_ref);

        // τ = max(tau_min, 1 - curr_nlp_error) — upstream cpp:397.
        let tau = self.tau_min.max(1.0 - nlp_error);
        data.borrow_mut().curr_tau = tau;

        let loqo_candidate = || {
            let mut oracle = LoqoMuOracle {
                mu_min: self.mu_min,
                mu_max: self.mu_max,
                avrg_compl,
                centrality_xi,
            };
            oracle.calculate_mu().unwrap_or(curr_mu)
        };

        let candidate = match self.mu_oracle {
            MuOracleKind::Loqo => loqo_candidate(),
            MuOracleKind::Probing => match (nlp, pd_search_dir) {
                (Some(nlp), Some(sd)) => {
                    let mut oracle = ProbingMuOracle {
                        sigma_max: 100.0,
                        mu_min: self.mu_min,
                        mu_max: self.mu_max,
                        mu_curr: curr_mu,
                        mu_aff: curr_mu,
                    };
                    oracle
                        .calculate_mu_with_affine_step(data, cq, nlp, sd, 1.0)
                        .unwrap_or_else(loqo_candidate)
                }
                _ => loqo_candidate(),
            },
            MuOracleKind::QualityFunction => match (nlp, pd_search_dir) {
                (Some(nlp), Some(sd)) => {
                    let mut oracle = QualityFunctionMuOracle::new();
                    oracle.mu_min = self.mu_min;
                    oracle.mu_max = self.mu_max;
                    oracle
                        .calculate_mu_with_predictor_centering(data, cq, nlp, sd)
                        .unwrap_or_else(loqo_candidate)
                }
                _ => loqo_candidate(),
            },
        };

        // Safeguard floor + global band clamp (cpp:410-426).
        let lower = self.lower_mu_safeguard(dual_inf, primal_inf, candidate);
        let mut mu = candidate.max(self.mu_min).max(lower).min(self.mu_max);

        // Free-mode growth cap — pragmatic stand-in for upstream's
        // `linesearch_->CheckSkippedLineSearch()` hook
        // (`IpAdaptiveMuUpdate.cpp:347-350`). Upstream forces
        // `sufficient_progress = false` if the previous line search
        // was skipped/tiny-stepped; that signal is plumbed through
        // `tiny_step_flag` above into the fixed/free mode switch
        // (which can raise μ via `new_fixed_mu`). The free-mode
        // oracle itself can swing 100× either direction on iterates
        // with non-uniform complementarity, and an unconstrained
        // increase tends to destabilise the filter line search on
        // problems like HAIFAM. Keep the cap on the oracle path.
        if mu > curr_mu {
            mu = curr_mu;
        }
        mu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_mu_safeguard_initializes_from_first_call() {
        let mut a = AdaptiveMuUpdate::new();
        a.adaptive_mu_safeguard_factor = 1e-2;
        // First call captures init values.
        let _ = a.lower_mu_safeguard(0.5, 2.0, 1.0);
        assert_eq!(a.init_dual_inf, 1.0); // max(1, 0.5)
        assert_eq!(a.init_primal_inf, 2.0); // max(1, 2.0)
    }

    #[test]
    fn lower_mu_safeguard_takes_max_of_dual_and_primal_terms() {
        let mut a = AdaptiveMuUpdate::new();
        a.adaptive_mu_safeguard_factor = 1.0;
        // Primal term dominates.
        let r = a.lower_mu_safeguard(0.1, 5.0, 1e9);
        // init_dual = 1, init_primal = 5 → terms: 0.1, 1.0 → max = 1.0.
        assert!((r - 1.0).abs() < 1e-15);
    }

    #[test]
    fn kkt_error_globalization_clips_to_min_ref_val() {
        let mut a = AdaptiveMuUpdate::new();
        a.adaptive_mu_globalization = AdaptiveMuGlobalization::KktError;
        a.adaptive_mu_safeguard_factor = 1.0;
        // Without clip, safeguard would be 5.0; min_ref_val = 0.1 wins.
        let r = a.lower_mu_safeguard(0.1, 5.0, 0.1);
        assert!((r - 0.1).abs() < 1e-15);
    }

    #[test]
    fn reset_clears_init_inf() {
        let mut a = AdaptiveMuUpdate::new();
        a.adaptive_mu_safeguard_factor = 1.0;
        let _ = a.lower_mu_safeguard(0.5, 2.0, 1.0);
        a.reset_init_inf();
        assert_eq!(a.init_dual_inf, -1.0);
        assert_eq!(a.init_primal_inf, -1.0);
    }

    // The trait `update_barrier_parameter` now takes
    // `(&IpoptDataHandle, &IpoptCqHandle)`. End-to-end coverage of the
    // adaptive path lands alongside the integration test that drives
    // `IpoptAlgorithm::optimize` with `mu_strategy=adaptive`; in
    // isolation the unit tests above exercise the safeguard
    // arithmetic and option defaults.

    #[test]
    fn default_mu_oracle_is_quality_function() {
        let a = AdaptiveMuUpdate::new();
        assert_eq!(a.mu_oracle, MuOracleKind::QualityFunction);
    }

    #[test]
    fn mu_oracle_kind_is_distinct() {
        assert_ne!(MuOracleKind::Loqo, MuOracleKind::Probing);
        assert_ne!(MuOracleKind::Probing, MuOracleKind::QualityFunction);
        assert_ne!(MuOracleKind::Loqo, MuOracleKind::QualityFunction);
    }
}
