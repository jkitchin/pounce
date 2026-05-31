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
    /// Upper bound on μ. Sentinel `-1.0` means "not yet computed; init
    /// lazily on the first `update_barrier_parameter` call to
    /// `mu_max_fact * curr_avrg_compl()`". Mirrors
    /// `IpAdaptiveMuUpdate.cpp:160-165` (load step) and
    /// `IpAdaptiveMuUpdate.cpp:267-274` (lazy init).
    pub mu_max: Number,
    /// `mu_max_fact` (default 1e3) — factor for lazy init of `mu_max`.
    /// Upstream `IpAdaptiveMuUpdate.cpp:RegisterOptions` line 42.
    /// Ignored if the user explicitly sets `mu_max` to a non-sentinel
    /// value.
    pub mu_max_fact: Number,
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
    /// `sigma_max` / `sigma_min` forwarded to
    /// `QualityFunctionMuOracle` on every free-mode call. Defaults
    /// from `IpQualityFunctionMuOracle.cpp:RegisterOptions`.
    pub sigma_max: Number,
    pub sigma_min: Number,
    /// `quality_function_norm_type` (default `2-norm-squared`) —
    /// norm used to aggregate the three KKT components inside the
    /// quality function. Forwarded to `QualityFunctionMuOracle` on
    /// every free-mode call. Mirrors
    /// `IpQualityFunctionMuOracle.cpp:RegisterOptions`.
    pub qf_norm_type: crate::mu::oracle::quality_function::NormType,
    /// `quality_function_centrality` (default `none`) — penalty term
    /// added to the quality function for centrality deviation.
    pub qf_centrality_type: crate::mu::oracle::quality_function::CentralityType,
    /// `quality_function_balancing_term` (default `none`) — penalty
    /// term added to the quality function when the complementarity
    /// is far smaller than the infeasibilities.
    pub qf_balancing_term: crate::mu::oracle::quality_function::BalancingTermType,
    /// `quality_function_max_section_steps` (default 8) — cap on
    /// golden-section iterations when picking σ.
    pub qf_max_section_steps: i32,
    /// `quality_function_section_sigma_tol` (default 1e-2) — width
    /// tolerance in σ-space for the golden-section search.
    pub qf_section_sigma_tol: Number,
    /// `quality_function_section_qf_tol` (default 0.0) — relative
    /// flatness tolerance for the golden-section search.
    pub qf_section_qf_tol: Number,

    /// `probing_iterate_quality_factor` (default 1e4, pounce-specific;
    /// see pounce#58). When the probing (Mehrotra) μ-oracle is about
    /// to read `curr_avrg_compl()` for its `mu_curr` input, a single
    /// imbalanced `(s_i, z_i)` pair can inflate the average 5+ orders
    /// above the stored `data.curr_mu`. Probing then mathematically
    /// correctly returns `σ·mu_curr` ≫ previous μ, which throws the
    /// iterate out of the convergence neighborhood. This guard
    /// short-circuits that case: when `curr_avrg_compl / curr_mu >
    /// probing_iterate_quality_factor`, we signal restoration via
    /// [`IpoptData::request_resto`] and keep μ unchanged. Set to 0 or
    /// any non-positive value to disable.
    pub probing_iterate_quality_factor: Number,

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
    /// `no_bounds_` flag — port of `IpAdaptiveMuUpdate.cpp:282-287`.
    /// Set to `true` on the first `update_barrier_parameter` call when
    /// the iterate has zero bound multipliers (z_l, z_u, v_l, v_u all
    /// have dim 0 — e.g. BT3, GENHS28, HS50, equality-only TNLPs).
    /// Subsequent calls return `mu_min` immediately. Without this,
    /// `mu_max = mu_max_fact * curr_avrg_compl()` evaluates to 0 (no
    /// slacks → zero complementarity) and the later `clamp(mu_min,
    /// mu_max)` panics with `min > max`.
    no_bounds: bool,
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
            // Sentinel; lazy-initialised to `mu_max_fact * avrg_compl`
            // on the first `update_barrier_parameter` call. Upstream
            // `IpAdaptiveMuUpdate.cpp:164` sets `mu_max_ = -1.` when
            // the option is not user-specified.
            mu_max: -1.0,
            mu_max_fact: 1e3,
            tau_min: 0.99,
            mu_init: 0.1,
            barrier_tol_factor: 10.0,
            mu_linear_decrease_factor: 0.2,
            mu_superlinear_decrease_power: 1.5,
            adaptive_mu_monotone_init_factor: 0.8,
            restore_accepted_iterate: false,
            sigma_max: 1e2,
            sigma_min: 1e-6,
            qf_norm_type: crate::mu::oracle::quality_function::NormType::TwoNormSquared,
            qf_centrality_type: crate::mu::oracle::quality_function::CentralityType::None,
            qf_balancing_term: crate::mu::oracle::quality_function::BalancingTermType::None,
            qf_max_section_steps: 8,
            qf_section_sigma_tol: 1e-2,
            qf_section_qf_tol: 0.0,
            probing_iterate_quality_factor: 1e4,
            init_dual_inf: -1.0,
            init_primal_inf: -1.0,
            free_mu_mode: true,
            refs_vals: VecDeque::new(),
            filter: Filter::new(),
            accepted_point: None,
            no_bounds: false,
        }
    }
}

impl AdaptiveMuUpdate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pure-arithmetic predicate behind the probing-oracle iterate-
    /// quality guard (pounce#58). Returns `true` when the ratio
    /// `avrg_compl / curr_mu` exceeds `factor`. The two non-strict
    /// gates (`factor > 0`, `curr_mu > 0`) keep the predicate
    /// well-defined when the guard is disabled or when an unusual
    /// μ-strategy zeroes `curr_mu`.
    pub fn probing_iterate_guard_fires(
        factor: Number,
        curr_mu: Number,
        avrg_compl: Number,
    ) -> bool {
        factor > 0.0 && curr_mu > 0.0 && avrg_compl > factor * curr_mu
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
        let prim_term = self.adaptive_mu_safeguard_factor * (primal_inf / self.init_primal_inf);
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
                let margin = self.filter_margin_fact * self.filter_max_margin.min(curr_err);
                !self
                    .filter
                    .dominated_by_any(curr_theta + margin, curr_f + margin)
            }
            AdaptiveMuGlobalization::NeverMonotoneMode => true,
        }
    }

    /// Port of `AdaptiveMuUpdate::RememberCurrentPointAsAccepted`
    /// (`IpAdaptiveMuUpdate.cpp:492-546`). Records the iterate state
    /// for the next sufficient-progress check.
    fn remember_current_point_as_accepted(&mut self, data: &IpoptDataHandle, cq: &IpoptCqHandle) {
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
        // Mirror upstream `IpAdaptiveMuUpdate.cpp:246-247`:
        //   IpData().Set_mu(1.);
        //   IpData().Set_tau(0.);
        // These are placeholder values so `CalculateSafeSlack` and the
        // first output line have something to work with; the actual μ
        // is computed by the oracle at iter 0's `update_barrier_parameter`.
        // Setting curr_mu = mu_init here (as we used to) skipped the
        // oracle's iter-0 call and locked μ at mu_init for the first
        // Newton step — diverging from upstream's iter-0 behaviour
        // (PFIT3: upstream iter 0 oracle picked μ=1.6e-6, pounce was
        // stuck at μ=0.1, producing different iter-1 trial point).
        let mut d = data.borrow_mut();
        d.curr_mu = 1.0;
        d.curr_tau = 0.0;
        drop(d);
        self.free_mu_mode = true;
        self.refs_vals.clear();
        self.filter.clear();
        self.accepted_point = None;
        self.init_dual_inf = -1.0;
        self.init_primal_inf = -1.0;
        // Reset mu_max sentinel so a re-solve re-runs the lazy init
        // against the fresh starting iterate's curr_avrg_compl.
        // Upstream re-enters InitializeImpl on each solve which
        // (lines 160-165) resets `mu_max_ = -1.` when not user-set.
        self.mu_max = -1.0;
        // Reset no-bounds detection on re-solve.
        self.no_bounds = false;
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
        // Lazy `mu_max` init — port of `IpAdaptiveMuUpdate.cpp:267-274`.
        // Upstream computes `mu_max = mu_max_fact * curr_avrg_compl()`
        // on the first call when the user did not set `mu_max`
        // explicitly. Pounce previously hard-coded `mu_max = 1e5`,
        // which let `new_fixed_mu = 0.8 * curr_avrg_compl` cap at 1e5
        // — on DECONVBNE that allowed μ to jump from 2.5e-3 to ~2000
        // at iter 198, destabilising the rest of the run.
        if self.mu_max < 0.0 {
            let avrg = cq.borrow().curr_avrg_compl();
            self.mu_max = self.mu_max_fact * avrg;
        }

        // No-bounds short-circuit — port of `IpAdaptiveMuUpdate.cpp:282-296`.
        // Detect once on the first call whether the iterate has any
        // bound multipliers (z_l, z_u, v_l, v_u). When all four are
        // dim-zero (equality-only TNLPs: BT3, GENHS28, HS50, METHANL8,
        // ...), `curr_avrg_compl()` is 0, hence `mu_max = 0`, and the
        // later `clamp(mu_min, mu_max)` panics with `min > max`.
        // Upstream sets `mu = mu_min`, `tau = tau_min`, and short-
        // circuits all subsequent oracle work; we mirror that.
        if !self.no_bounds {
            let n_bounds = {
                let d = data.borrow();
                let c = d.curr.as_ref().expect("curr set");
                c.z_l.dim() + c.z_u.dim() + c.v_l.dim() + c.v_u.dim()
            };
            if n_bounds == 0 {
                self.no_bounds = true;
                let mut d = data.borrow_mut();
                d.curr_mu = self.mu_min;
                d.curr_tau = self.tau_min;
                return self.mu_min;
            }
        }
        if self.no_bounds {
            let mut d = data.borrow_mut();
            d.curr_mu = self.mu_min;
            d.curr_tau = self.tau_min;
            return self.mu_min;
        }

        // Read-and-clear `tiny_step_flag` — mirrors upstream
        // `IpAdaptiveMuUpdate.cpp:297-298`. The flag is consumed by
        // this call: without the clear, a single tiny-step detection
        // would persist forever and suppress `sufficient_progress` on
        // every later outer iter.
        let (curr_mu, iter_count, tiny_step_flag) = {
            let mut d = data.borrow_mut();
            let out = (d.curr_mu, d.iter_count, d.tiny_step_flag);
            d.tiny_step_flag = false;
            out
        };

        // NB: do NOT short-circuit at iter_count==0. Upstream's
        // `UpdateBarrierParameter` runs the oracle at iter 0 (the
        // initialize() above set μ=1.0 as a placeholder only). Skipping
        // the oracle here locked μ at the placeholder for the first
        // Newton step. Letting the iter-0 path flow through the
        // free-μ branch picks up the oracle's choice — the empty
        // `refs_vals_` makes `check_sufficient_progress` return true,
        // we remember the iterate, then call the oracle below.
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
            let sufficient_progress = !force_no_progress && self.check_sufficient_progress(cq);
            if sufficient_progress {
                // Switch back to free mode and record the iterate —
                // upstream `cpp:303-311`. Upstream does NOT return
                // here: after flipping `FreeMuMode` to true the first
                // if/else ends and control reaches the `if
                // FreeMuMode()` block at `cpp:391`, which runs the
                // oracle and picks a fresh μ in the SAME iteration.
                // Returning `curr_mu` here froze μ on the transition
                // iter — PALMER4's iter-15 fixed→free transition kept
                // μ at 2.4e-7 instead of letting the oracle drop it to
                // mu_min, stalling to Maximum_Iterations_Exceeded.
                // Fall through to the oracle call below.
                self.free_mu_mode = true;
                self.remember_current_point_as_accepted(data, cq);
            } else {
                // Keep reducing μ Fiacco-McCormick style if the
                // barrier subproblem is solved to within
                // `barrier_tol_factor · μ`, OR if a tiny step was
                // just detected (`cpp:320` `|| tiny_step_flag`).
                let sub_problem_error = cq.borrow().curr_barrier_error();
                if sub_problem_error <= self.barrier_tol_factor * curr_mu || tiny_step_flag {
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
            let sufficient_progress = !force_no_progress && self.check_sufficient_progress(cq);
            if sufficient_progress {
                self.remember_current_point_as_accepted(data, cq);
                // Fall through to the oracle call below.
            } else {
                if std::env::var("POUNCE_DBG_AMU").is_ok() {
                    let cqr = cq.borrow();
                    let theta = cqr.curr_constraint_violation();
                    let f = cqr.curr_f();
                    let nlp_err = cqr.curr_nlp_error();
                    let avrg = cqr.curr_avrg_compl();
                    drop(cqr);
                    let margin = self.filter_margin_fact * self.filter_max_margin.min(nlp_err);
                    let entries: Vec<(Number, Number, i32)> = self
                        .filter
                        .entries()
                        .iter()
                        .map(|e| (e.theta, e.phi, e.iter))
                        .collect();
                    tracing::debug!(target: "pounce::mu",
                        "[AMU] iter={} free->fixed: curr_mu={:.3e} theta={:.3e} f={:.3e} nlp_err={:.3e} margin={:.3e} avrg_compl={:.3e} new_mu={:.3e} | filter={:?} | force_no_progress={} tiny={}",
                        iter_count,
                        curr_mu,
                        theta,
                        f,
                        nlp_err,
                        margin,
                        avrg,
                        self.adaptive_mu_monotone_init_factor * avrg,
                        entries,
                        force_no_progress,
                        tiny_step_flag,
                    );
                }
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
            MuOracleKind::Probing => {
                // Iterate-quality guard (pounce#58). The probing
                // oracle uses `curr_avrg_compl()` for its `mu_curr`
                // input (see `mu/oracle/probing.rs:85`). When a single
                // imbalanced `(s_i, z_i)` pair inflates the average
                // many orders above the stored `data.curr_mu`,
                // probing's `σ·mu_curr` correctly returns the inflated
                // value and the resulting search direction throws the
                // iterate out of the convergence neighborhood. On
                // arki0012 this manifests as μ jumping 5 orders at
                // iter 155 followed by divergence to "Local
                // Infeasibility" at iter 284. We short-circuit by
                // signalling restoration and keeping μ unchanged; the
                // main loop in `ipopt_alg.rs` consumes the flag
                // before the search-direction step.
                if Self::probing_iterate_guard_fires(
                    self.probing_iterate_quality_factor,
                    curr_mu,
                    avrg_compl,
                ) {
                    if std::env::var("POUNCE_DBG_ORACLE").is_ok() {
                        tracing::debug!(target: "pounce::mu",
                            "[PN_PROBE_GUARD] iter={} curr_mu={:.3e} avrg_compl={:.3e} ratio={:.3e} > factor={:.3e} → request_resto",
                            iter_count,
                            curr_mu,
                            avrg_compl,
                            avrg_compl / curr_mu,
                            self.probing_iterate_quality_factor,
                        );
                    }
                    data.borrow_mut().request_resto = true;
                    return curr_mu;
                }
                match (nlp, pd_search_dir) {
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
                }
            }
            MuOracleKind::QualityFunction => match (nlp, pd_search_dir) {
                (Some(nlp), Some(sd)) => {
                    let mut oracle = QualityFunctionMuOracle::new();
                    oracle.mu_min = self.mu_min;
                    oracle.mu_max = self.mu_max;
                    oracle.sigma_min = self.sigma_min;
                    oracle.sigma_max = self.sigma_max;
                    oracle.norm_type = self.qf_norm_type;
                    oracle.centrality_type = self.qf_centrality_type;
                    oracle.balancing_term = self.qf_balancing_term;
                    oracle.max_section_steps = self.qf_max_section_steps;
                    oracle.section_sigma_tol = self.qf_section_sigma_tol;
                    oracle.section_qf_tol = self.qf_section_qf_tol;
                    // Mirrors upstream's `quality_function_search` timer
                    // around `CalculateMu` in `IpQualityFunctionMuOracle.cpp`.
                    let timing = data.borrow().timing.clone();
                    let _qf_guard = timing.quality_function_search.guard();
                    oracle
                        .calculate_mu_with_predictor_centering(data, cq, nlp, sd)
                        .unwrap_or_else(loqo_candidate)
                }
                _ => loqo_candidate(),
            },
        };

        // Safeguard floor + global band clamp (cpp:410-426).
        let lower = self.lower_mu_safeguard(dual_inf, primal_inf, candidate);
        let mu = candidate.max(self.mu_min).max(lower).min(self.mu_max);

        // NB: upstream `IpAdaptiveMuUpdate.cpp:410-426` does NOT require
        // `mu ≤ curr_mu` in free mode — the oracle is allowed to bump
        // μ back up. A prior attempt to cap growth here ("HAIFAM
        // stability hack") let DECONVBNE's μ plunge from 0.1 to 5e-10
        // in ~20 iters and never recover (upstream oscillates μ in
        // [-8,-1] for the same range), trapping `inf_du` at 1e13.
        // Tiny-step skips are already handled by the
        // `tiny_step_flag → force_no_progress → new_fixed_mu` path
        // above, which can raise μ via the fixed-mode branch.
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

    // pounce#58 guard predicate. Numbers below come from the issue
    // body's iter 154-155 trace on arki0012.
    #[test]
    fn probing_iterate_guard_fires_on_arki0012_iter155() {
        let curr_mu = 1.98e-11;
        let avrg_compl = 8.90e-6;
        assert!(AdaptiveMuUpdate::probing_iterate_guard_fires(
            1e4, curr_mu, avrg_compl
        ));
    }

    #[test]
    fn probing_iterate_guard_quiet_on_healthy_iter() {
        // iter 154 in the same trace — ratio ≈ 2.2; ought not fire.
        let curr_mu = 1.02e-11;
        let avrg_compl = 2.24e-11;
        assert!(!AdaptiveMuUpdate::probing_iterate_guard_fires(
            1e4, curr_mu, avrg_compl
        ));
    }

    #[test]
    fn probing_iterate_guard_disabled_at_zero_factor() {
        // factor=0 ⇒ guard off, even with extreme ratio.
        assert!(!AdaptiveMuUpdate::probing_iterate_guard_fires(
            0.0, 1e-11, 1.0
        ));
    }

    #[test]
    fn probing_iterate_guard_disabled_at_negative_factor() {
        assert!(!AdaptiveMuUpdate::probing_iterate_guard_fires(
            -1.0, 1e-11, 1.0
        ));
    }

    #[test]
    fn probing_iterate_guard_quiet_when_curr_mu_zero() {
        // Pathological `curr_mu = 0` (no-bounds branch zeroes it out).
        // Predicate must stay quiet rather than division-by-zero.
        assert!(!AdaptiveMuUpdate::probing_iterate_guard_fires(
            1e4, 0.0, 1e-6
        ));
    }

    #[test]
    fn probing_iterate_guard_threshold_at_factor_times_mu() {
        // Boundary: equality does NOT fire (strict >).
        let curr_mu = 1.0e-10;
        let factor = 1e4;
        assert!(!AdaptiveMuUpdate::probing_iterate_guard_fires(
            factor,
            curr_mu,
            factor * curr_mu
        ));
        // Just above the boundary fires.
        assert!(AdaptiveMuUpdate::probing_iterate_guard_fires(
            factor,
            curr_mu,
            factor * curr_mu * (1.0 + 1e-12)
        ));
    }
}
