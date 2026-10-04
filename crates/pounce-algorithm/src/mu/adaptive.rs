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

/// Primal infeasibility (scaled, max-norm) at or below which the probing
/// iterate-quality guard recentres instead of requesting restoration
/// (gh#986 item 5).
const PROBING_GUARD_FEASIBLE_TOL: Number = 1e-8;

/// What the probing iterate-quality guard does with an iterate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbingGuardAction {
    /// Guard quiet: run the probing oracle.
    Probe,
    /// Guard fired on an infeasible iterate: request restoration.
    Restoration,
    /// Guard fired on an already-feasible iterate: restoration has nothing
    /// to repair, take a centering (LOQO) step instead (gh#986).
    Recenter,
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
    /// Complementarity tolerance — option `compl_inf_tol`, default 1e-4 per
    /// `IpAlgorithmRegOp.cpp`. Not used directly by the adaptive update;
    /// enters only through [`Self::certificate_safe_mu_min`], which caps
    /// `mu_min` so a strongly scaled-down objective can still reach the
    /// termination certificate (pounce#266) — the μ floor lives in scaled
    /// space while `compl_inf_tol` is enforced on the *unscaled*
    /// complementarity.
    pub compl_inf_tol: Number,
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
    /// `sigma_max` / `sigma_min` forwarded to `QualityFunctionMuOracle`
    /// on every free-mode call. `sigma_max` is additionally forwarded to
    /// `ProbingMuOracle` (upstream `IpProbingMuOracle.cpp` reads the same
    /// `sigma_max` option to cap its centering parameter — L3). Defaults
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
    /// `adaptive_mu_max_free_returns` (pounce#749) — cap on how many
    /// times the strategy may switch back out of fixed-mu mode. `-1`
    /// is unlimited, reproducing upstream. POUNCE extension: it has no
    /// counterpart in `IpAdaptiveMuUpdate.cpp`.
    pub max_free_returns: i32,
    /// Number of fixed->free transitions taken so far, compared
    /// against [`Self::max_free_returns`].
    free_returns_taken: i32,
    /// `adaptive_mu_budget_pin_fraction` (pounce#753) — once this
    /// fraction of an explicitly-set CPU or wall-clock budget has been
    /// spent without converging, stop exploring in free-mu mode and
    /// finish in the cheap fixed-mu (monotone) endgame. `1.0` disables.
    /// POUNCE extension: no counterpart in `IpAdaptiveMuUpdate.cpp`.
    ///
    /// This is an *in-flight* switch, not a retry, and that is the whole
    /// point. `mu_strategy_fallback` (pounce#748) deliberately declines
    /// to retry a `Maximum_CpuTime_Exceeded` exit, because "the budget a
    /// retry needs is precisely the budget already spent" — a second
    /// solve starts from x0 and has nothing left to pay with. Switching
    /// in place keeps the iterate, so the time already spent is not
    /// wasted; it bought the point the monotone endgame starts from.
    pub budget_pin_fraction: Number,
    /// `max_cpu_time` / `max_wall_time` as the convergence check sees
    /// them, mirrored here so [`Self::budget_spent`] can compute the
    /// consumed fraction on the direct-driver path, where no shared
    /// [`pounce_common::timing::Deadline`] is installed.
    pub max_cpu_time: Number,
    pub max_wall_time: Number,
    /// Latched once [`Self::budget_spent`] first fires, so the endgame
    /// cannot flap back into free mode as the clock keeps running.
    budget_pinned: bool,
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
            compl_inf_tol: 1e-4,
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
            max_free_returns: -1,
            budget_pin_fraction: 0.75,
            max_cpu_time: 1e6,
            max_wall_time: 1e6,
            budget_pinned: false,
            free_returns_taken: 0,
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
    pub fn probing_guard_action(guard_fired: bool, primal_inf: Number) -> ProbingGuardAction {
        match (guard_fired, primal_inf <= PROBING_GUARD_FEASIBLE_TOL) {
            (false, _) => ProbingGuardAction::Probe,
            (true, true) => ProbingGuardAction::Recenter,
            (true, false) => ProbingGuardAction::Restoration,
        }
    }

    pub fn probing_iterate_guard_fires(
        factor: Number,
        curr_mu: Number,
        avrg_compl: Number,
    ) -> bool {
        factor > 0.0 && curr_mu > 0.0 && avrg_compl > factor * curr_mu
    }

    /// Scalar core of the lazy `mu_max` initialization
    /// (`IpAdaptiveMuUpdate.cpp:267-274`): on the first call, when the
    /// user did not set `mu_max` explicitly, upstream sets it to
    /// `mu_max_fact * curr_avrg_compl()`.
    ///
    /// A warm start (`warm_start_init_point=yes`) can hand us an iterate
    /// whose bound multipliers are all zero — pounce does not yet wire
    /// `warm_start_mult_bound_push`, so `seed_from_nlp` leaves
    /// `z_l`/`z_u`/`v_l`/`v_u` at 0. Then `curr_avrg_compl()` is 0 even
    /// though bounds exist, the `no_bounds` short-circuit does NOT fire,
    /// `mu_max` collapses to 0, and the later `new_mu.clamp(mu_min,
    /// mu_max)` panics with `min > max` (min = mu_min = 1e-11, max = 0).
    /// When `avrg` carries no positive complementarity signal (zero, or a
    /// NaN handed in by a pathological iterate) fall back to `mu_init` as
    /// the proxy — what a cold start's `avrg_compl` is ~scaled to — so the
    /// `[mu_min, mu_max]` band stays valid. The final `.max(mu_min)` is a
    /// belt-and-suspenders floor against pathological options.
    pub fn lazy_mu_max(
        mu_max_fact: Number,
        avrg: Number,
        mu_init: Number,
        mu_min: Number,
    ) -> Number {
        let avrg = if avrg > 0.0 { avrg } else { mu_init };
        (mu_max_fact * avrg).max(mu_min)
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

    /// `mu_min` capped so it can never block the termination certificate
    /// (pounce#266) — the adaptive twin of
    /// [`crate::mu::monotone::MonotoneMuUpdate::certificate_safe_mu_min`],
    /// which carries the full story. The raw absolute `mu_min` (default
    /// `1e-11`) lives in μ's scaled space while `compl_inf_tol` is enforced
    /// on the *unscaled* complementarity; below
    /// `|df| ≈ mu_min·(barrier_tol_factor+1)/compl_inf_tol` an uncapped
    /// floor pins the unscaled complementarity above `compl_inf_tol` and
    /// the strict certificate is unreachable — in adaptive mode the solve
    /// then degrades to `Solved_To_Acceptable_Level` (reduced accuracy, on an
    /// iterate sitting at the optimum).
    ///
    /// The restoration sub-builder's `mu_min = 100 · outer_mu_min`
    /// safeguard is unaffected for the same reason as in monotone mode:
    /// `RestoIpoptNlp` does not override `obj_scaling_factor`, so the resto
    /// inner IPM sees `df = 1` and the cap sits far above the safeguard.
    pub fn certificate_safe_mu_min(&self, obj_scaling_factor: Number) -> Number {
        crate::mu::certificate_safe_mu_min(
            self.mu_min,
            self.compl_inf_tol,
            self.barrier_tol_factor,
            obj_scaling_factor,
        )
    }

    /// Floor for the **fixed-mode** (monotone-mode) μ decrease — port of
    /// `IpAdaptiveMuUpdate.cpp:328-329`:
    ///
    /// ```cpp
    /// new_mu = Max(new_mu,
    ///     Min(compl_inf_tol_scaled, IpData().tol()) / (barrier_tol_factor_ + 1.));
    /// ```
    ///
    /// pounce#511: this branch used to floor at `mu_min` instead — `1e-11`
    /// against upstream's `9.09e-10` at default `tol = 1e-8`, ~91× lower,
    /// and further with a looser `tol` (at `tol = 1e-6` upstream's floor is
    /// `9.09e-8`, four orders up). `mu_min` is the *free*-mode clamp; once the
    /// strategy has switched to fixed mode upstream deliberately uses the
    /// looser, tolerance-derived floor — that is the point of the switch.
    /// Driving the Newton system down to `1e-11` past the accuracy the
    /// termination test asks for buys nothing and invites degenerate search
    /// directions on an ill-conditioned Jacobian.
    ///
    /// Two details mirror the monotone floor
    /// (`MonotoneMuUpdate::update_barrier_parameter`):
    ///
    /// * `compl_inf_tol` is converted into μ's scaled space first
    ///   (pounce#257 — upstream's `apply_obj_scaling`), since it is enforced
    ///   on the *unscaled* complementarity while μ and `tol` are scaled;
    /// * the result is additionally `max`ed with the certificate-safe
    ///   `mu_min` (pounce#266) so the restoration sub-builder's
    ///   `100 · outer_mu_min` safeguard still applies. Capped that way,
    ///   `mu_min` can only raise the floor, never push it under the
    ///   certificate.
    pub fn fixed_mode_mu_floor(&self, tol: Number, obj_scaling_factor: Number) -> Number {
        let dynamic_floor = tol.min(crate::mu::scaled_compl_inf_tol(
            self.compl_inf_tol,
            obj_scaling_factor,
        )) / (self.barrier_tol_factor + 1.0);
        self.certificate_safe_mu_min(obj_scaling_factor)
            .max(dynamic_floor)
    }

    /// Port of `AdaptiveMuUpdate::NewFixedMu`
    /// (`IpAdaptiveMuUpdate.cpp:583-627`). Selects μ when the state
    /// machine drops out of free mode. v1.0 always uses the
    /// "average complementarity" branch (no `fix_mu_oracle_` is
    /// wired; matches `fixed_mu_oracle = average_compl`).
    ///
    /// The lower clamp is the certificate-safe `mu_min` (pounce#266);
    /// capped ≤ raw `mu_min`, so the `[mu_min, mu_max]` band the lazy
    /// `mu_max` init guarantees stays valid.
    fn new_fixed_mu(&self, cq: &IpoptCqHandle, mu_min: Number) -> Number {
        let avrg = cq.borrow().curr_avrg_compl();
        let new_mu = self.adaptive_mu_monotone_init_factor * avrg;
        new_mu.clamp(mu_min, self.mu_max)
    }

    /// Upstream's tiny-step termination test (pounce#512), shared by the
    /// two sites that throw `TINY_STEP_DETECTED` in
    /// `IpAdaptiveMuUpdate.cpp` — `:330-333` in the fixed-mode
    /// Fiacco-McCormick decrease and `:377-380` on the free→fixed switch.
    /// Both read `tiny_step_flag && new_mu == mu`: a tiny step was
    /// detected *and* the update could not move μ, so no further
    /// progress is available and the honest exit is "problem solved to
    /// best possible numerical accuracy" (`STOP_AT_TINY_STEP`) rather
    /// than iterating to the limit.
    ///
    /// Exact equality, like upstream. Both callers reach "unchanged" by
    /// clamping to the same bound, which is bit-exact; an epsilon band
    /// would instead swallow a genuine — if minute — reduction and stop
    /// an iteration early.
    fn tiny_step_is_terminal(tiny_step_flag: bool, new_mu: Number, curr_mu: Number) -> bool {
        tiny_step_flag && new_mu == curr_mu
    }

    /// pounce#753 — has the caller's explicit time budget been consumed
    /// past [`Self::budget_pin_fraction`]?
    ///
    /// POUNCE extension; no counterpart upstream. Free-μ mode costs
    /// roughly 2.3x fixed-μ mode per iteration on nql180 (the oracle's
    /// affine + centering back-solves, plus the trajectory it steers
    /// into), and on that problem adaptive spends the whole tail
    /// oscillating free->fixed->free and never reaches an endgame:
    /// 444 iterations / 2234 s and a `Maximum_CpuTime_Exceeded` exit,
    /// against 105 iterations / 258 s to `Optimal` if it is made to stay
    /// in fixed mode. `mu_strategy_fallback` (pounce#748) already
    /// recovers the *unbudgeted* form of that failure by retrying the
    /// whole solve monotone after `Maximum_Iterations_Exceeded`, but it
    /// deliberately declines to retry a CPU/wall exit — there is no
    /// budget left to retry with. This is that recovery done in flight:
    /// keep the iterate, drop the oracle, finish monotone.
    ///
    /// Returns `false` unless a budget was actually set. Both the
    /// builder default (1e6 s) and the registered sentinel (1e20 s)
    /// leave the consumed fraction indistinguishable from zero, so a
    /// caller who never asked for a time limit never sees this fire —
    /// which is also why the fixture sweep, which sets no budget, is
    /// unaffected by construction.
    ///
    /// Latching matters: without [`Self::budget_pinned`] the pin would
    /// depend on where in the iteration the clock is read, and a
    /// borderline solve could flap back into free mode after paying for
    /// the switch.
    fn budget_spent(&mut self, data: &IpoptDataHandle) -> bool {
        if self.budget_pinned {
            return true;
        }
        if !(self.budget_pin_fraction < 1.0) {
            // NaN-safe: only a fraction strictly below 1 can pin.
            return false;
        }
        let d = data.borrow();
        let frac = if let Some(deadline) = d.deadline.as_ref() {
            // The shared deadline (pounce#242) measures from a fixed
            // start instant and is what the convergence check trusts,
            // so it is what we measure against too.
            let cpu = fraction_of(
                deadline.max_cpu(),
                deadline.max_cpu() - deadline.remaining_cpu(),
            );
            let wall = fraction_of(
                deadline.max_wall(),
                deadline.max_wall() - deadline.remaining_wall(),
            );
            cpu.max(wall)
        } else {
            // Direct-driver / unit-test path — no deadline installed;
            // mirror `conv_check::opt_error`'s fallback to `overall_alg`.
            let timing = &d.timing;
            let cpu = fraction_of(self.max_cpu_time, timing.overall_alg.live_cpu_time());
            let wall = fraction_of(self.max_wall_time, timing.overall_alg.live_wallclock_time());
            cpu.max(wall)
        };
        drop(d);
        if frac >= self.budget_pin_fraction {
            self.budget_pinned = true;
            tracing::debug!(target: "pounce::mu",
                "[AMU] pinning to fixed-mu mode: {:.0}% of the time budget spent (pounce#753)",
                frac * 100.0,
            );
            return true;
        }
        false
    }
}

/// `spent / budget`, or 0 when the budget is not a usable positive
/// number. A non-finite or non-positive budget means "no limit was
/// expressed", not "the limit is already blown".
fn fraction_of(budget: Number, spent: Number) -> Number {
    if budget.is_finite() && budget > 0.0 {
        (spent / budget).max(0.0)
    } else {
        0.0
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
        // Both mode-pinning mechanisms are per-solve state, and
        // `initialize` is what a re-solve calls. Carrying either across
        // would let the first solve's history pin the second one before
        // it has taken a step: `free_returns_taken` (pounce#749) is a
        // budget of transitions this solve is allowed, and
        // `budget_pinned` (pounce#753) is a decision about this solve's
        // clock.
        self.free_returns_taken = 0;
        self.budget_pinned = false;
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
    /// Line-search reset: upstream calls `linesearch_->Reset()` at
    /// three points — line 339 (fixed-mode decrease), line 386
    /// (free→fixed switch) and line 431 (**every** free-mode
    /// iteration, whether or not μ moved). The [`MuUpdate`] trait
    /// surface carries no line-search handle, so we raise
    /// [`IpoptData::request_ls_reset`] at exactly those three points
    /// and `IpoptAlgorithm::iterate` performs the reset right after
    /// this call returns — the same plumbing the pounce#58 probing
    /// guard uses for [`IpoptData::request_resto`]. See pounce#510:
    /// the previous "reset when μ changed" proxy in the caller is
    /// correct for the monotone update but not for this one, and left
    /// the filter holding pre-restoration entries whenever μ happened
    /// to stay put.
    ///
    /// [`IpoptData::request_ls_reset`]: crate::ipopt_data::IpoptData::request_ls_reset
    /// [`IpoptData::request_resto`]: crate::ipopt_data::IpoptData::request_resto
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
            self.mu_max = Self::lazy_mu_max(self.mu_max_fact, avrg, self.mu_init, self.mu_min);
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

        // Certificate-safe μ floor (pounce#266): every place below that
        // stops μ from descending — the fixed-mode reduction, the
        // fixed-mode re-seed, the oracles' internal clamps, and the final
        // band clamp — must use `mu_min` capped into the space the
        // certificate lives in, or a strongly scaled-down objective ends
        // `Solved_To_Acceptable_Level` on an iterate at the optimum. The
        // `no_bounds` short-circuit above keeps the raw `mu_min`: with no
        // bound multipliers there is no complementarity to certify.
        let obj_scaling_factor = cq.borrow().obj_scaling_factor();
        let mu_min = self.certificate_safe_mu_min(obj_scaling_factor);

        // pounce#753 — POUNCE extension. Read once per update so the
        // two mode-transition sites below agree within an iteration.
        let budget_spent = self.budget_spent(data);

        if !self.free_mu_mode {
            // Fixed-mu branch — `cpp:299-342`.
            //
            // The gate is `sufficient_progress && !tiny_step_flag`
            // (`cpp:304`) — plain `tiny_step_flag`, *not* the
            // globalization-conditional `force_no_progress`, which
            // upstream applies only in the free-mode branch below
            // (`cpp:347-351`). Reusing `force_no_progress` here let
            // `adaptive_mu_globalization=never-monotone-mode` switch back
            // to free mode on a flagged tiny step, which upstream never
            // does and which routed around the termination at `cpp:330`.
            // At the default `obj-constr-filter` the two are equal, so
            // this distinction only moves never-monotone-mode (pounce#512).
            let sufficient_progress = !tiny_step_flag && self.check_sufficient_progress(cq);
            // pounce#749 — POUNCE extension. Upstream returns to free
            // mode every time progress looks sufficient, which on some
            // problems (nql180) oscillates for the whole tail: the
            // strategy re-enters fixed mode a handful of iterations
            // later having paid the oracle's extra affine + centering
            // solves the entire time, and never runs the cheap
            // monotone endgame that closes the problem. Once the cap is
            // reached we stay in fixed mode, which is exactly the
            // Fiacco-McCormick reduction in the `else` arm below.
            // `-1` disables the cap and reproduces upstream.
            let returns_left =
                self.max_free_returns < 0 || self.free_returns_taken < self.max_free_returns;
            // pounce#753 — and once the time budget is nearly gone, do
            // not return to free mode at all, whatever the cap says.
            if sufficient_progress && returns_left && !budget_spent {
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
                self.free_returns_taken += 1;
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
                    // Fixed-mode floor is NOT `mu_min` — see
                    // [`Self::fixed_mode_mu_floor`] (pounce#511).
                    let tol = data.borrow().tol;
                    let floor = self.fixed_mode_mu_floor(tol, obj_scaling_factor);
                    let new_mu = lin.min(sup).max(floor).min(self.mu_max);
                    // `cpp:330-333` — a tiny step was flagged and the
                    // decrease left μ where it was (it is pinned at the
                    // floor), so there is nothing left to try. Upstream
                    // throws TINY_STEP_DETECTED *before* `Set_mu`/`Set_tau`;
                    // the flag is unchanged by construction, so returning
                    // it below is the same iterate either way. Pairing it
                    // with the #511 floor is upstream's own pairing: the
                    // termination triggers off the same floor the decrease
                    // stops at, so it now fires at the tolerance-derived
                    // floor instead of at `mu_min`.
                    if Self::tiny_step_is_terminal(tiny_step_flag, new_mu, curr_mu) {
                        data.borrow_mut().request_tiny_step_stop = true;
                    }
                    let new_tau = self.tau_min.max(1.0 - new_mu);
                    let mut d = data.borrow_mut();
                    d.curr_tau = new_tau;
                    // Upstream `cpp:339` — reset inside this branch,
                    // unconditionally, even when the clamps leave μ
                    // where it was (pounce#510).
                    d.request_ls_reset = true;
                    return new_mu;
                }
                // Subproblem not yet solved — keep μ. Upstream does NOT
                // reset the line search on this path (`cpp:335-341`).
                let new_tau = self.tau_min.max(1.0 - curr_mu);
                data.borrow_mut().curr_tau = new_tau;
                return curr_mu;
            }
        } else {
            // Free-mu branch — `cpp:343-389`.
            // pounce#753 — `!budget_spent` forces the free->fixed
            // switch below through the *existing* transition path
            // (accepted-iterate restore, `new_fixed_mu`, line-search
            // reset) rather than inventing a second one. Combined with
            // the gate above, the switch is then permanent.
            let sufficient_progress =
                !force_no_progress && !budget_spent && self.check_sufficient_progress(cq);
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
                let new_mu = self.new_fixed_mu(cq, mu_min);
                // `cpp:377-380` — the same termination on the other
                // throw site: the switch into fixed mode re-seeded μ to
                // the value it already had, so the tiny step cannot be
                // walked off by changing μ either. Ordered after the
                // free-mode flip and the accepted-iterate restore, as
                // upstream is.
                if Self::tiny_step_is_terminal(tiny_step_flag, new_mu, curr_mu) {
                    data.borrow_mut().request_tiny_step_stop = true;
                }
                let new_tau = self.tau_min.max(1.0 - new_mu);
                let mut d = data.borrow_mut();
                d.curr_tau = new_tau;
                // Upstream `cpp:386` — the free→fixed switch resets the
                // line search whether or not `new_fixed_mu` differs from
                // the μ we came in with (pounce#510).
                d.request_ls_reset = true;
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
                mu_min,
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
                let guard_fired = Self::probing_iterate_guard_fires(
                    self.probing_iterate_quality_factor,
                    curr_mu,
                    avrg_compl,
                );
                // gh#986 item 5. Restoration is the remedy for an iterate the
                // oracle's `sigma * mu_curr` would throw out of the convergence
                // neighborhood *because it is infeasible* (arki0012). At an
                // already-feasible iterate restoration has nothing to do —
                // it fails on entry — so the guard turned a merely off-centre
                // feasible iterate into `Restoration_Failed`. That is exactly
                // what `mehrotra_algorithm=yes` produces on a general NLP: the
                // first probing step picks sigma ~ 0 (mu ~ 1e-11) while the
                // iterate's complementarity is still ~1e-2 (clnlbeam: ratio
                // 2e9 at iteration 1, `inf_pr` 2e-14). There the right move is
                // a centering step: take mu from the iterate's own
                // complementarity (the LOQO rule) for this iteration and let
                // the probing oracle resume once the iterate is back on the
                // central path.
                if Self::probing_guard_action(guard_fired, primal_inf)
                    == ProbingGuardAction::Recenter
                {
                    if std::env::var("POUNCE_DBG_ORACLE").is_ok() {
                        tracing::debug!(target: "pounce::mu",
                            "[PN_PROBE_GUARD] iter={} ratio={:.3e} but inf_pr={:.3e} is feasible → centering (loqo) instead of restoration",
                            iter_count,
                            avrg_compl / curr_mu,
                            primal_inf,
                        );
                    }
                    loqo_candidate()
                } else if guard_fired {
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
                    // No `request_ls_reset` here: this early return is a
                    // pounce-specific guard with no upstream counterpart,
                    // it leaves μ untouched, and the caller hands the
                    // iterate straight to restoration.
                    data.borrow_mut().request_resto = true;
                    return curr_mu;
                } else {
                    match (nlp, pd_search_dir) {
                        (Some(nlp), Some(sd)) => {
                            let mut oracle = ProbingMuOracle {
                                // Forward the user-set `sigma_max` (default 1e2),
                                // matching upstream `IpProbingMuOracle.cpp`, which
                                // reads `options.GetNumericValue("sigma_max", ...)`
                                // and caps `sigma = Min(sigma, sigma_max_)`. This
                                // was hard-coded to 100.0, so a user-set `sigma_max`
                                // reached only the quality-function oracle (L3).
                                sigma_max: self.sigma_max,
                                mu_min,
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
            }
            MuOracleKind::QualityFunction => match (nlp, pd_search_dir) {
                (Some(nlp), Some(sd)) => {
                    let mut oracle = QualityFunctionMuOracle::new();
                    oracle.mu_min = mu_min;
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
        let mu = candidate.max(mu_min).max(lower).min(self.mu_max);

        // Upstream `cpp:431` — the free-mode block closes with an
        // unconditional `linesearch_->Reset()`. This is the point the
        // old caller-side "μ changed" proxy missed (pounce#510): it
        // fires on every free-mode iteration, including the ones where
        // the oracle re-picks the μ we already had, and including the
        // fixed→free transition that falls through to here. Filter
        // entries are keyed on a barrier parameter *and* an iterate;
        // "μ is unchanged" does not make yesterday's entries valid.
        data.borrow_mut().request_ls_reset = true;

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
    use crate::mu::test_fixture;

    /// pounce#749: `adaptive_mu_max_free_returns` caps how many times
    /// the strategy may climb back out of fixed-μ mode. The default
    /// (`-1`) must leave upstream's behavior exactly as it was, so the
    /// two arms are asserted against the same starting state.
    fn returns_to_free_mode(max_free_returns: i32) -> bool {
        let mut a = AdaptiveMuUpdate::new();
        a.max_free_returns = max_free_returns;
        let (data, cq) = test_fixture::fixture(0.1);
        // Drive the state machine into fixed mode the same way
        // `free_to_fixed_switch_requests_ls_reset` does: the first call
        // seeds the filter, the second finds the same (θ, f) dominated.
        let _ = a.update_barrier_parameter(&data, &cq, None, None);
        let _ = a.update_barrier_parameter(&data, &cq, None, None);
        assert!(!a.free_mu_mode, "fixture must reach fixed mode first");
        // Clearing the filter makes the next progress check succeed, so
        // the only thing that can hold the strategy in fixed mode is the
        // cap under test.
        a.filter = Filter::new();
        let _ = a.update_barrier_parameter(&data, &cq, None, None);
        a.free_mu_mode
    }

    #[test]
    fn unlimited_free_returns_is_upstream_behavior() {
        assert!(
            returns_to_free_mode(-1),
            "-1 must not cap the return to free mode"
        );
    }

    #[test]
    fn a_zero_cap_pins_the_strategy_in_the_monotone_endgame() {
        assert!(
            !returns_to_free_mode(0),
            "with no returns budgeted the strategy must stay in fixed mode"
        );
    }

    #[test]
    fn a_cap_of_one_spends_its_budget_and_then_pins() {
        assert!(returns_to_free_mode(1), "the first return is within budget");
    }

    /// pounce#753: same state machine as [`returns_to_free_mode`], but
    /// the thing under test is the time budget rather than the return
    /// cap. `budget` is `(max_wall, max_cpu)` for the shared
    /// [`pounce_common::timing::Deadline`] the application installs.
    fn returns_to_free_mode_under_budget(
        budget: (Number, Number),
        budget_pin_fraction: Number,
    ) -> bool {
        let mut a = AdaptiveMuUpdate::new();
        a.budget_pin_fraction = budget_pin_fraction;
        let (data, cq) = test_fixture::fixture(0.1);
        data.borrow_mut().deadline = Some(pounce_common::timing::Deadline::new(budget.0, budget.1));
        let _ = a.update_barrier_parameter(&data, &cq, None, None);
        let _ = a.update_barrier_parameter(&data, &cq, None, None);
        assert!(!a.free_mu_mode, "fixture must reach fixed mode first");
        a.filter = Filter::new();
        let _ = a.update_barrier_parameter(&data, &cq, None, None);
        a.free_mu_mode
    }

    /// The default 1e6 s budget — what a caller who never asked for a
    /// time limit gets — must leave the strategy behaving exactly as it
    /// did before pounce#753.
    #[test]
    fn an_unset_time_budget_does_not_pin() {
        assert!(
            returns_to_free_mode_under_budget((1e6, 1e6), 0.75),
            "the default budget is nowhere near spent, so nothing may change"
        );
    }

    /// A budget already consumed many times over pins the strategy in
    /// the monotone endgame instead of paying the oracle again.
    #[test]
    fn a_spent_time_budget_pins_the_monotone_endgame() {
        assert!(
            !returns_to_free_mode_under_budget((1e-9, 1e-9), 0.75),
            "with the budget spent the strategy must stay in fixed mode"
        );
    }

    /// `adaptive_mu_budget_pin_fraction = 1` is the documented off
    /// switch and must restore the pre-pounce#753 trajectory even on a
    /// budget that is comprehensively blown.
    #[test]
    fn a_pin_fraction_of_one_disables_the_mechanism() {
        assert!(
            returns_to_free_mode_under_budget((1e-9, 1e-9), 1.0),
            "a fraction of 1 must disable the pin"
        );
    }

    /// The other half of the mechanism: a solve *already* in free mode
    /// with an empty filter would sail on making "sufficient progress"
    /// forever. With the budget spent it must be pushed into fixed mode
    /// through the ordinary free->fixed path.
    #[test]
    fn a_spent_time_budget_forces_free_mode_out_of_the_oracle() {
        let mut a = AdaptiveMuUpdate::new();
        let (data, cq) = test_fixture::fixture(0.1);
        // Empty filter + free mode = sufficient progress on every call,
        // which is precisely the nql180 tail this issue is about.
        let _ = a.update_barrier_parameter(&data, &cq, None, None);
        a.filter = Filter::new();
        a.free_mu_mode = true;
        let _ = a.update_barrier_parameter(&data, &cq, None, None);
        assert!(a.free_mu_mode, "control: the oracle keeps free mode");

        a.filter = Filter::new();
        data.borrow_mut().deadline = Some(pounce_common::timing::Deadline::new(1e-9, 1e-9));
        let _ = a.update_barrier_parameter(&data, &cq, None, None);
        assert!(
            !a.free_mu_mode,
            "a spent budget must force the free->fixed switch"
        );
        assert!(
            data.borrow().request_ls_reset,
            "the switch must go through the ordinary free->fixed path, \
             which resets the line search"
        );
    }

    /// Once pinned, the strategy stays pinned: the latch means the
    /// decision does not depend on when in an iteration the clock is
    /// read, and a solve cannot flap back after paying for the switch.
    #[test]
    fn the_pin_latches() {
        let mut a = AdaptiveMuUpdate::new();
        let (data, _cq) = test_fixture::fixture(0.1);
        data.borrow_mut().deadline = Some(pounce_common::timing::Deadline::new(1e-9, 1e-9));
        assert!(a.budget_spent(&data));
        // Swap in a budget that is not spent at all; the latch holds.
        data.borrow_mut().deadline = Some(pounce_common::timing::Deadline::new(1e6, 1e6));
        assert!(a.budget_spent(&data), "the pin must not un-fire");
    }

    /// A nonsensical or absent budget is "no limit expressed", not "the
    /// limit is already blown" — otherwise a zero/NaN `max_cpu_time`
    /// would silently disable the mu oracle for every solve.
    #[test]
    fn a_degenerate_budget_is_not_a_spent_budget() {
        for (wall, cpu) in [
            (0.0, 0.0),
            (-1.0, -1.0),
            (Number::NAN, Number::NAN),
            (Number::INFINITY, Number::INFINITY),
        ] {
            let mut a = AdaptiveMuUpdate::new();
            let (data, _cq) = test_fixture::fixture(0.1);
            data.borrow_mut().deadline = Some(pounce_common::timing::Deadline::new(wall, cpu));
            assert!(
                !a.budget_spent(&data),
                "({wall}, {cpu}) expresses no budget and must not pin"
            );
        }
    }

    /// pounce#510: upstream resets the line search on **every** free-mode
    /// iteration (`IpAdaptiveMuUpdate.cpp:431`), not only when μ moves.
    /// The caller used to infer the reset from `next_mu != mu_before`,
    /// which silently skipped it whenever the oracle re-picked the μ we
    /// already had — leaving the filter holding entries computed against
    /// an iterate and a barrier parameter the algorithm had left behind.
    #[test]
    fn free_mode_requests_ls_reset_even_when_mu_is_unchanged() {
        let mut a = AdaptiveMuUpdate::new();
        // Never-monotone globalization keeps the state machine in free
        // mode across both calls, which is the endgame this issue is
        // about; the filter/KKT variants are covered below.
        a.adaptive_mu_globalization = AdaptiveMuGlobalization::NeverMonotoneMode;
        let (data, cq) = test_fixture::fixture(0.1);
        // First pass: free mode with an empty filter ⇒ sufficient
        // progress ⇒ the oracle picks μ.
        let mu1 = a.update_barrier_parameter(&data, &cq, None, None);
        assert!(a.free_mu_mode);
        assert!(data.borrow().request_ls_reset);

        // Re-enter at exactly the μ the oracle just chose, on the same
        // (unchanged) iterate: μ cannot move, and the pre-fix caller
        // would therefore never reset.
        data.borrow_mut().request_ls_reset = false;
        data.borrow_mut().curr_mu = mu1;
        let mu2 = a.update_barrier_parameter(&data, &cq, None, None);
        assert_eq!(mu2, mu1, "fixture must hold μ still for this test");
        assert!(
            data.borrow().request_ls_reset,
            "free-mode iteration must request a line-search reset with μ unchanged"
        );
    }

    /// pounce#510: the free→fixed switch is upstream's `cpp:386` reset,
    /// which likewise does not care whether `new_fixed_mu` differs from
    /// the incoming μ.
    #[test]
    fn free_to_fixed_switch_requests_ls_reset() {
        let mut a = AdaptiveMuUpdate::new();
        let (data, cq) = test_fixture::fixture(0.1);
        // Seed the filter with the current point, then re-run: the same
        // (θ, f) is now dominated, so progress is insufficient and the
        // update drops into fixed mode.
        let _ = a.update_barrier_parameter(&data, &cq, None, None);
        data.borrow_mut().request_ls_reset = false;
        let _ = a.update_barrier_parameter(&data, &cq, None, None);
        assert!(
            !a.free_mu_mode,
            "fixture must fall out of free mode for this test"
        );
        assert!(data.borrow().request_ls_reset);
    }

    /// pounce#510: the fixed-mode μ decrease is upstream's `cpp:339`
    /// reset. Note it fires inside the branch, so a decrease that the
    /// `mu_min`/`mu_max` clamps flatten still resets.
    #[test]
    fn fixed_mode_decrease_requests_ls_reset() {
        let mut a = AdaptiveMuUpdate::new();
        let (data, cq) = test_fixture::fixture(0.1);
        a.free_mu_mode = false;
        // Force "no sufficient progress" so the update stays in fixed
        // mode, and a barrier tolerance loose enough that the decrease
        // branch fires on this (far-from-optimal) iterate.
        a.adaptive_mu_globalization = AdaptiveMuGlobalization::KktError;
        a.adaptive_mu_kkterror_red_iters = 1;
        a.adaptive_mu_kkterror_red_fact = 0.0;
        a.refs_vals.push_back(1.0);
        a.barrier_tol_factor = 1e6;
        // Degenerate decrease factors: `min(1·μ, μ^1) = μ`. The branch is
        // taken but μ does not move, so the pre-fix `next_mu != mu_before`
        // proxy would have skipped the reset here as well.
        a.mu_linear_decrease_factor = 1.0;
        a.mu_superlinear_decrease_power = 1.0;
        let mu = a.update_barrier_parameter(&data, &cq, None, None);
        assert!(!a.free_mu_mode, "must stay in fixed mode for this test");
        assert_eq!(mu, 0.1, "flat decrease leaves μ where it was");
        assert!(data.borrow().request_ls_reset);
    }

    /// The one fixed-mode path upstream leaves alone (`cpp:335-341`):
    /// the barrier subproblem is not solved yet, μ stays, no reset.
    #[test]
    fn fixed_mode_without_decrease_does_not_request_ls_reset() {
        let mut a = AdaptiveMuUpdate::new();
        let (data, cq) = test_fixture::fixture(1e-8);
        a.free_mu_mode = false;
        // A far-from-optimal iterate at a tiny μ: the barrier error is
        // way above `barrier_tol_factor · μ`, and the filter is empty so
        // `check_sufficient_progress` must be forced to fail.
        a.adaptive_mu_globalization = AdaptiveMuGlobalization::KktError;
        a.adaptive_mu_kkterror_red_iters = 1;
        a.adaptive_mu_kkterror_red_fact = 0.0;
        a.refs_vals.push_back(1.0);
        let mu = a.update_barrier_parameter(&data, &cq, None, None);
        assert!(!a.free_mu_mode);
        assert_eq!(mu, 1e-8);
        assert!(!data.borrow().request_ls_reset);
    }

    /// pounce#266, adaptive twin of the monotone test: the raw `mu_min`
    /// clamp must yield to `compl_inf_tol·|df|/(barrier_tol_factor+1)` once
    /// |df| drops below `df* = mu_min·(barrier_tol_factor+1)/compl_inf_tol`,
    /// or the strict certificate is unreachable and the solve degrades to
    /// `Solved_To_Acceptable_Level` at the optimum.
    #[test]
    fn adaptive_mu_min_is_capped_so_certificate_stays_reachable() {
        let a = AdaptiveMuUpdate::new();
        let df_star = a.mu_min * (a.barrier_tol_factor + 1.0) / a.compl_inf_tol;
        assert!((df_star - 1.1e-6).abs() < 1e-21);
        for df in [1.0, -1.0, 1e-3, 1e-5, df_star] {
            assert_eq!(a.certificate_safe_mu_min(df), a.mu_min);
        }
        // HS71 × 1e8 computes df = 8.3e-8, under the cliff: the cap engages.
        let df = 8.3e-8;
        let capped = a.certificate_safe_mu_min(df);
        assert!(capped < a.mu_min);
        assert!((capped - 1e-4 * 8.3e-8 / 11.0).abs() < 1e-27);
        assert_eq!(a.certificate_safe_mu_min(-df), capped);
        // Degenerate factors fall back to the unconverted tolerance, whose
        // cap (9.09e-6) leaves mu_min alone.
        for df in [0.0, Number::NAN, Number::INFINITY] {
            assert_eq!(a.certificate_safe_mu_min(df), a.mu_min);
        }
        // The restoration sub-builder's `mu_min = 100 · outer_mu_min`
        // safeguard survives: the resto inner IPM sees df = 1.
        let mut resto = AdaptiveMuUpdate::new();
        resto.mu_min = 100.0 * a.mu_min;
        assert_eq!(resto.certificate_safe_mu_min(1.0), resto.mu_min);
    }

    /// pounce#511: the fixed-mode decrease must floor at upstream's
    /// `Min(compl_inf_tol_scaled, tol)/(barrier_tol_factor+1)`, not at
    /// `mu_min`. At default `tol=1e-8`, `compl_inf_tol=1e-4`,
    /// `barrier_tol_factor=10` that is `1e-8/11 ≈ 9.09e-10` — ~91× above
    /// `mu_min = 1e-11`, and further still at a looser `tol`.
    #[test]
    fn fixed_mode_floor_matches_upstream_not_mu_min() {
        let a = AdaptiveMuUpdate::new();
        let floor = a.fixed_mode_mu_floor(1e-8, 1.0);
        assert!((floor - 1e-8 / 11.0).abs() < 1e-20, "floor was {floor}");
        // ~91× above `mu_min` — the old floor — i.e. nearly two orders.
        assert!(floor / a.mu_min > 90.0, "floor was {floor}");
        // Looser `tol` raises the floor with it (upstream takes the min of
        // `tol` and `compl_inf_tol`, so `tol` binds until it exceeds 1e-4).
        assert!((a.fixed_mode_mu_floor(1e-6, 1.0) - 1e-6 / 11.0).abs() < 1e-18);
        // Beyond that `compl_inf_tol` binds.
        assert!((a.fixed_mode_mu_floor(1e-2, 1.0) - 1e-4 / 11.0).abs() < 1e-18);
    }

    /// The `compl_inf_tol` half of the floor is converted into μ's scaled
    /// space before the `Min` (upstream's `apply_obj_scaling`, pounce#257),
    /// so the two disagree whenever objective scaling is active.
    #[test]
    fn fixed_mode_floor_scales_compl_inf_tol() {
        let a = AdaptiveMuUpdate::new();
        // df = 1e-6 puts scaled compl_inf_tol at 1e-10, under `tol=1e-8`,
        // so it is the binding half: 1e-10/11 ≈ 9.09e-12.
        let df = 1e-6;
        let floor = a.fixed_mode_mu_floor(1e-8, df);
        assert!(
            (floor - 1e-4 * df / 11.0).abs() < 1e-24,
            "floor was {floor}"
        );
        // Sign of the scaling factor (maximization poses df < 0) is
        // irrelevant — the magnitude is what converts spaces.
        assert_eq!(a.fixed_mode_mu_floor(1e-8, -df), floor);
        // Degenerate factors fall back to the unconverted tolerance.
        for df in [0.0, Number::NAN, Number::INFINITY] {
            assert!((a.fixed_mode_mu_floor(1e-8, df) - 1e-8 / 11.0).abs() < 1e-20);
        }
    }

    /// The restoration sub-builder's `mu_min = 100 · outer_mu_min`
    /// safeguard still binds when it sits above the tolerance floor: the
    /// certificate-safe `mu_min` is `max`ed in, mirroring monotone mode.
    #[test]
    fn fixed_mode_floor_keeps_resto_mu_min_safeguard() {
        let mut resto = AdaptiveMuUpdate::new();
        resto.mu_min = 1e-6; // well above tol/(barrier_tol_factor+1) = 9.09e-10
        // `RestoIpoptNlp` does not override obj scaling — the resto inner
        // IPM sees df = 1, so the cap leaves `mu_min` alone and it wins.
        assert_eq!(resto.fixed_mode_mu_floor(1e-8, 1.0), 1e-6);
    }

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

    // gh#986 item 5: clnlbeam under `mehrotra_algorithm=yes` — ratio 2e9 at
    // iteration 1 with `inf_pr` 2.3e-14. Restoration at a feasible point
    // can only fail; the guard recentres instead. An infeasible iterate
    // (arki0012's case) still goes to restoration.
    #[test]
    fn probing_guard_recentres_a_feasible_iterate_and_restores_an_infeasible_one() {
        use ProbingGuardAction::*;
        let fired = AdaptiveMuUpdate::probing_iterate_guard_fires(1e4, 2.575e-11, 5.358e-2);
        assert!(fired);
        assert_eq!(
            AdaptiveMuUpdate::probing_guard_action(fired, 2.3e-14),
            Recenter
        );
        assert_eq!(
            AdaptiveMuUpdate::probing_guard_action(fired, 3e-3),
            Restoration
        );
        assert_eq!(AdaptiveMuUpdate::probing_guard_action(false, 3e-3), Probe);
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

    // Regression: `mu_strategy=adaptive` + `warm_start_init_point=yes`
    // used to panic in `new_mu.clamp(mu_min, mu_max)` with
    // "min > max ... min = 1e-11, max = 0.0" — the warm start zeroes the
    // bound multipliers, so `curr_avrg_compl()` reads 0 even though
    // bounds exist, collapsing `mu_max` to 0. `lazy_mu_max` must keep the
    // band valid (mu_max >= mu_min) regardless of the `avrg` it is fed.
    #[test]
    fn lazy_mu_max_keeps_band_valid_on_zero_avrg_compl() {
        let a = AdaptiveMuUpdate::new();
        // Warm-start pathology: avrg_compl == 0.
        let mu_max = AdaptiveMuUpdate::lazy_mu_max(a.mu_max_fact, 0.0, a.mu_init, a.mu_min);
        assert!(
            mu_max >= a.mu_min,
            "mu_max {mu_max} must not fall below mu_min {}",
            a.mu_min
        );
        // Falls back to the mu_init-scaled band: 1e3 * 0.1 = 100.
        assert!((mu_max - a.mu_max_fact * a.mu_init).abs() < 1e-12);
    }

    #[test]
    fn lazy_mu_max_unchanged_for_cold_start() {
        let a = AdaptiveMuUpdate::new();
        // A healthy cold start hands a positive avrg_compl; the band is
        // mu_max_fact * avrg, exactly as before the warm-start guard.
        let avrg = 2.5e-3;
        let mu_max = AdaptiveMuUpdate::lazy_mu_max(a.mu_max_fact, avrg, a.mu_init, a.mu_min);
        assert!((mu_max - a.mu_max_fact * avrg).abs() < 1e-15);
    }

    #[test]
    fn lazy_mu_max_survives_nan_avrg_compl() {
        let a = AdaptiveMuUpdate::new();
        // A NaN avrg (the other half of the original panic message) must
        // not propagate: `avrg > 0.0` is false for NaN, so we fall back.
        let mu_max = AdaptiveMuUpdate::lazy_mu_max(a.mu_max_fact, f64::NAN, a.mu_init, a.mu_min);
        assert!(mu_max.is_finite() && mu_max >= a.mu_min);
    }

    // pounce#512 — the shared condition behind both of upstream's
    // `TINY_STEP_DETECTED` throws (`IpAdaptiveMuUpdate.cpp:330-333`,
    // `:377-380`). Both conjuncts are load-bearing in opposite
    // directions: without the flag the update is just at its floor and
    // must keep iterating, and without the μ test a tiny step that the
    // update *can* still respond to would stop the solve early.
    #[test]
    fn tiny_step_is_terminal_needs_the_flag_and_an_unmoved_mu() {
        let mu = 1e-11;
        assert!(AdaptiveMuUpdate::tiny_step_is_terminal(true, mu, mu));
        // μ moved — the update has something left to try.
        assert!(!AdaptiveMuUpdate::tiny_step_is_terminal(true, 0.2 * mu, mu));
        // No tiny step: μ pinned at its floor is the ordinary end-game,
        // not a reason to stop.
        assert!(!AdaptiveMuUpdate::tiny_step_is_terminal(false, mu, mu));
        assert!(!AdaptiveMuUpdate::tiny_step_is_terminal(
            false,
            0.2 * mu,
            mu
        ));
    }

    /// Equality is exact, as upstream's `new_mu == mu` is. A reduction of
    /// one ulp is a reduction; an epsilon band would call it "unchanged"
    /// and terminate an iteration early.
    #[test]
    fn tiny_step_is_terminal_does_not_round_a_reduction_away() {
        let mu = 1e-11;
        let nudged = mu - f64::EPSILON * 1e-4;
        assert!(nudged < mu, "test setup: the nudge must actually reduce μ");
        assert!(!AdaptiveMuUpdate::tiny_step_is_terminal(true, nudged, mu));
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
