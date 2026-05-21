//! Filter line-search acceptor — port of
//! `Algorithm/IpFilterLSAcceptor.{hpp,cpp}`.
//!
//! Combines an [`super::filter::Filter`] with Fletcher-Leyffer's
//! filter logic: at each backtracking step it asks whether the
//! `(theta_trial, phi_trial)` pair is acceptable to the filter and
//! to a sufficient-decrease test. The decisions are:
//!
//! * **Switching condition**: `d_phi < 0 ∧
//!   alpha * (-d_phi)^s_phi > delta * theta^s_theta` — when true, we
//!   require an Armijo decrease in `phi`; otherwise we relax to the
//!   filter test.
//! * **Armijo decrease**:
//!   `phi_trial - phi <= eta_phi * alpha * d_phi`, compared via
//!   `Compare_le` (a round-off-tolerant `<=`) — see [`FilterLsAcceptor::armijo_holds`].
//! * **Filter acceptance**: `phi_trial < phi - gamma_phi * theta` OR
//!   `theta_trial < (1 - gamma_theta) * theta`.
//!
//! See `ref/Ipopt/AGENT_REFERENCE/LINE_SEARCH.md` §"Acceptance" for
//! the full statement; constants below default to upstream's
//! `RegisterOptions` values.

use crate::line_search::filter::Filter;
use crate::line_search::ls_acceptor::BacktrackingLsAcceptor;
use pounce_common::types::Number;

/// `POUNCE_DBG_LS=1` toggle, cached so per-trial overhead is one
/// atomic load instead of a syscall. Used by [`FilterLsAcceptor::check_acceptability`]
/// to emit per-trial `(α, θ, φ, d_phi, θ_trial, φ_trial, rapid_inc_ok,
/// suff_progress_ok)` for the pounce#21 W-B parity investigation.
fn dbg_ls_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("POUNCE_DBG_LS").as_deref() == Ok("1"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptDecision {
    /// `(theta_trial, phi_trial)` is acceptable to the filter and
    /// passes the decrease test for the current `alpha`.
    Accept,
    /// Trial point fails the filter or decrease check.
    Reject,
}

pub struct FilterLsAcceptor {
    pub filter: Filter,
    pub eta_phi: Number,
    pub delta_armijo: Number,
    pub theta_min_fact: Number,
    pub theta_max_fact: Number,
    pub gamma_phi: Number,
    pub gamma_theta: Number,
    pub s_phi: Number,
    pub s_theta: Number,
    pub max_soc: i32,
    /// `alpha_min_frac` from `IpFilterLSAcceptor.cpp:RegisterOptions` —
    /// safety factor applied to the dynamic alpha-min before declaring
    /// a tiny step / handing off to restoration. Default 0.05.
    pub alpha_min_frac: Number,
    /// Lazily initialised `theta_min_` (upstream
    /// `IpFilterLSAcceptor.cpp:333-339`). `None` until the first call to
    /// [`Self::calc_alpha_min`] sees a reference theta.
    theta_min: Option<Number>,
    /// Lazily initialised `theta_max_` (upstream
    /// `IpFilterLSAcceptor.cpp:325-331`). Locked on first encounter to
    /// `theta_max_fact * max(1, reference_theta)`. Any trial iterate
    /// with `theta_trial > theta_max` is rejected outright by the
    /// filter — this guards against the line search accepting a step
    /// that catastrophically inflates constraint violation (e.g. a
    /// Newton step from a poorly-scaled iterate landing far outside
    /// the feasible basin).
    theta_max: Option<Number>,
    /// Maximum number of filter resets allowed per solve (upstream
    /// option `max_filter_resets`, default 5). Set to `0` to disable
    /// the heuristic entirely.
    pub max_filter_resets: i32,
    /// Number of consecutive filter-rejected accepts that triggers a
    /// reset (upstream option `filter_reset_trigger`, default 5).
    pub filter_reset_trigger: i32,
    /// Resets used so far this solve. Bumped each time the heuristic
    /// fires; not cleared by [`Self::reset`] — only re-initialising
    /// the acceptor (a fresh `Default::default()`) zeroes it. Mirrors
    /// upstream `n_filter_resets_`, which is cleared only in
    /// `InitializeImpl` (per-solve), never in `Reset()`.
    n_filter_resets: i32,
    /// `true` when the most recent trial was rejected because of the
    /// filter (as opposed to the iterate-acceptability test). Mirrors
    /// upstream `last_rejection_due_to_filter_`. Read on the accept
    /// path to decide whether to bump `count_successive_filter_rejections`.
    last_rejection_due_to_filter: bool,
    /// Number of consecutive accepted trials whose immediately-preceding
    /// rejection was due to the filter. Reset whenever an accept
    /// follows a non-filter rejection (or the very first accept of the
    /// solve). Mirrors `count_successive_filter_rejections_`.
    count_successive_filter_rejections: i32,
}

impl Default for FilterLsAcceptor {
    fn default() -> Self {
        // Defaults from `IpFilterLSAcceptor.cpp:RegisterOptions`.
        Self {
            filter: Filter::new(),
            eta_phi: 1e-8,
            delta_armijo: 1.0,
            theta_min_fact: 1e-4,
            theta_max_fact: 1e4,
            gamma_phi: 1e-8,
            gamma_theta: 1e-5,
            s_phi: 2.3,
            s_theta: 1.1,
            max_soc: 4,
            alpha_min_frac: 0.05,
            theta_min: None,
            theta_max: None,
            max_filter_resets: 5,
            filter_reset_trigger: 5,
            n_filter_resets: 0,
            last_rejection_due_to_filter: false,
            count_successive_filter_rejections: 0,
        }
    }
}

impl FilterLsAcceptor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Switching condition — true ⇒ require Armijo decrease in phi
    /// rather than the filter check. Mirrors
    /// `IpFilterLSAcceptor.cpp:IsSwitchingCondition` (line ~590).
    pub fn is_switching_condition(
        &self,
        alpha_primal: Number,
        d_phi: Number,
        theta: Number,
    ) -> bool {
        if d_phi >= 0.0 {
            return false;
        }
        let lhs = alpha_primal * (-d_phi).powf(self.s_phi);
        let rhs = self.delta_armijo * theta.powf(self.s_theta);
        lhs > rhs
    }

    /// Armijo sufficient decrease in `phi`.
    /// Mirrors `IpFilterLSAcceptor.cpp:ArmijoHolds`, which compares with
    /// `Compare_le` — a `<=` carrying a `10·eps·|phi|` round-off slack —
    /// not a bare `<=`. The slack is essential near a solution: when the
    /// barrier objective is flat, `phi_trial - phi` is dominated by
    /// floating-point summation noise (a tiny *positive* value even on a
    /// genuine descent step), while `eta_phi·alpha·d_phi` is a tiny
    /// *negative* number. A bare `<=` can then never hold, so the line
    /// search backtracks to `alpha_min` and falls into restoration —
    /// mislabelling a converged iterate `Error_In_Step_Computation`
    /// (PALMER1/2, VESUVIALS, HIELOW, MGH10LS, ... — all reach the
    /// optimum, then stall here).
    pub fn armijo_holds(
        &self,
        alpha_primal: Number,
        d_phi: Number,
        phi: Number,
        phi_trial: Number,
    ) -> bool {
        pounce_common::utils::compare_le(
            phi_trial - phi,
            self.eta_phi * alpha_primal * d_phi,
            phi,
        )
    }

    /// Sufficient progress check (used when *not* in switching mode).
    /// Mirrors the OR-test in `IpFilterLSAcceptor.cpp:IsAcceptableToCurrentIterate`.
    pub fn is_sufficient_progress(
        &self,
        theta: Number,
        phi: Number,
        theta_trial: Number,
        phi_trial: Number,
    ) -> bool {
        phi_trial < phi - self.gamma_phi * theta || theta_trial < (1.0 - self.gamma_theta) * theta
    }

    /// Mirrors `FilterLSAcceptor::IsAcceptableToCurrentFilter`
    /// (`IpFilterLSAcceptor.cpp:501-504`). True iff `(theta, barr)` is
    /// not dominated by any entry in the filter.
    pub fn is_acceptable_to_current_filter(&self, trial_barr: Number, trial_theta: Number) -> bool {
        !self.filter.dominated_by_any(trial_theta, trial_barr)
    }

    /// Mirrors `FilterLSAcceptor::IsAcceptableToCurrentIterate`
    /// (`IpFilterLSAcceptor.cpp:471-499`).
    ///
    /// `reference_*` is the iterate-pair the line search is comparing
    /// against (typically the current iterate's `(barr, theta)` at the
    /// start of the line search). `obj_max_inc` is the upstream
    /// `obj_max_inc` option (default 5.0). `called_from_restoration`
    /// disables the rapid-barrier-increase guard — used by
    /// [`crate::line_search::filter_acceptor`] consumers in the
    /// restoration phase, since the resto sub-solver's barrier value
    /// has no direct comparability to the outer one.
    pub fn is_acceptable_to_current_iterate(
        &self,
        trial_barr: Number,
        trial_theta: Number,
        reference_barr: Number,
        reference_theta: Number,
        obj_max_inc: Number,
        called_from_restoration: bool,
    ) -> bool {
        if !called_from_restoration && trial_barr > reference_barr {
            // Rapid-increase guard: log-scale jump cap.
            let basval = if reference_barr.abs() > 10.0 {
                reference_barr.abs().log10()
            } else {
                1.0
            };
            if (trial_barr - reference_barr).log10() > obj_max_inc + basval {
                return false;
            }
        }
        // Filter-style sufficient-progress test (line 497-498).
        pounce_common::utils::compare_le(
            trial_theta,
            (1.0 - self.gamma_theta) * reference_theta,
            reference_theta,
        ) || pounce_common::utils::compare_le(
            trial_barr - reference_barr,
            -self.gamma_phi * reference_theta,
            reference_barr,
        )
    }

    /// Single-trial accept decision. Caller has already computed the
    /// trial `(theta, phi)` pair and the directional derivative
    /// `d_phi`. Mirrors the body of
    /// `IpFilterLSAcceptor::CheckAcceptabilityOfTrialPoint` (lines
    /// 311-437): the iterate test runs first (Armijo when F-type AND
    /// `theta <= theta_min`, otherwise `IsAcceptableToCurrentIterate`
    /// with the `obj_max_inc` rapid-increase guard); only on iterate
    /// acceptance do we then consult the filter.
    ///
    /// The `&mut self` receiver lets the method record the rejection
    /// reason in `last_rejection_due_to_filter` and run the filter-reset
    /// heuristic on the accept path
    /// (`IpFilterLSAcceptor.cpp:407-433`).
    pub fn check_acceptability(
        &mut self,
        alpha_primal: Number,
        theta: Number,
        phi: Number,
        d_phi: Number,
        theta_trial: Number,
        phi_trial: Number,
    ) -> AcceptDecision {
        // theta_min / theta_max may not yet have been initialised if
        // the caller skipped `calc_alpha_min` for some reason; fall
        // back to the same lazy formula upstream uses on first
        // encounter (`IpFilterLSAcceptor.cpp:325-339`).
        let theta_min = self
            .theta_min
            .unwrap_or_else(|| self.theta_min_fact * theta.max(1.0));
        let theta_max = self
            .theta_max
            .unwrap_or_else(|| self.theta_max_fact * theta.max(1.0));

        // `IpFilterLSAcceptor.cpp:341-348`: any trial iterate above
        // `theta_max` is rejected outright. Without this guard the
        // line search may accept a step that inflates constraint
        // violation by many orders of magnitude (POLAK6, ROSENMMX,
        // ACOPR14: theta jumps from 8 to 1e12 on iter 1).
        if theta_trial > theta_max {
            self.last_rejection_due_to_filter = false;
            return AcceptDecision::Reject;
        }

        let f_type = alpha_primal > 0.0 && self.is_switching_condition(alpha_primal, d_phi, theta);
        let take_armijo = f_type && theta <= theta_min;

        let iterate_ok = if take_armijo {
            self.armijo_holds(alpha_primal, d_phi, phi, phi_trial)
        } else {
            // `IsAcceptableToCurrentIterate` with `called_from_restoration=false`.
            // Rapid-barrier-increase guard (`obj_max_inc` default 5.0).
            let rapid_increase_ok = if phi_trial > phi {
                let basval = if phi.abs() > 10.0 {
                    phi.abs().log10()
                } else {
                    1.0
                };
                (phi_trial - phi).log10() <= 5.0 + basval
            } else {
                true
            };
            let suff_progress_ok = pounce_common::utils::compare_le(
                theta_trial,
                (1.0 - self.gamma_theta) * theta,
                theta,
            ) || pounce_common::utils::compare_le(
                phi_trial - phi,
                -self.gamma_phi * theta,
                phi,
            );
            // pounce#21 diagnostic — env-gated. Emits one line per
            // trial when POUNCE_DBG_LS=1 so the divergence-vs-Ipopt
            // investigation can correlate which branch (rapid-increase
            // guard vs sufficient-progress) was the rejection cause.
            // Env lookup cached in a `OnceLock` so the disabled case
            // costs one atomic load per trial.
            if dbg_ls_enabled() {
                eprintln!(
                    "DBG_LS alpha={:.3e} theta={:.3e} phi={:.3e} d_phi={:.3e} theta_trial={:.3e} phi_trial={:.3e} theta_max={:.3e} rapid_inc_ok={} suff_progress_ok={}",
                    alpha_primal,
                    theta,
                    phi,
                    d_phi,
                    theta_trial,
                    phi_trial,
                    theta_max,
                    rapid_increase_ok,
                    suff_progress_ok,
                );
            }
            rapid_increase_ok && suff_progress_ok
        };

        if !iterate_ok {
            // Iterate-acceptability rejection (the LS-test branch in
            // upstream's `CheckAcceptabilityOfTrialPoint`, lines 363-381).
            // Upstream sets `last_rejection_due_to_filter_ = false` on
            // both the unfortunate-Armijo-failure and the
            // sufficient-progress-failure paths.
            self.last_rejection_due_to_filter = false;
            return AcceptDecision::Reject;
        }

        if self.filter.dominated_by_any(theta_trial, phi_trial) {
            // Iterate test passed but filter dominates → mark this
            // rejection as filter-due (line 397).
            self.last_rejection_due_to_filter = true;
            return AcceptDecision::Reject;
        }

        // Trial accepted. Run the filter-reset heuristic
        // (`IpFilterLSAcceptor.cpp:407-433`).
        if self.max_filter_resets > 0 && self.n_filter_resets < self.max_filter_resets {
            if self.last_rejection_due_to_filter {
                self.count_successive_filter_rejections += 1;
                if self.count_successive_filter_rejections >= self.filter_reset_trigger {
                    self.filter.clear();
                    self.n_filter_resets += 1;
                    self.count_successive_filter_rejections = 0;
                }
            } else {
                self.count_successive_filter_rejections = 0;
            }
        }
        // Clear for the next outer iteration's α-loop (line 434).
        self.last_rejection_due_to_filter = false;

        AcceptDecision::Accept
    }
}

impl FilterLsAcceptor {
    /// Lazy initialiser for `theta_min` matching upstream
    /// `IpFilterLSAcceptor.cpp:333-339`: on first invocation (and never
    /// again — upstream resets only `theta_min_ = -1` in
    /// `InitializeImpl`, not in `Reset()`), set
    /// `theta_min = theta_min_fact * max(1, reference_theta)`.
    fn ensure_theta_min(&mut self, reference_theta: Number) -> Number {
        *self
            .theta_min
            .get_or_insert_with(|| self.theta_min_fact * reference_theta.max(1.0))
    }

    /// Lazy initialiser for `theta_max_` matching upstream
    /// `IpFilterLSAcceptor.cpp:325-331`: locked once on first
    /// encounter to `theta_max_fact * max(1, reference_theta)`.
    fn ensure_theta_max(&mut self, reference_theta: Number) -> Number {
        *self
            .theta_max
            .get_or_insert_with(|| self.theta_max_fact * reference_theta.max(1.0))
    }
}

impl BacktrackingLsAcceptor for FilterLsAcceptor {
    fn reset(&mut self) {
        // Mirrors upstream `FilterLSAcceptor::Reset`
        // (`IpFilterLSAcceptor.cpp:524-532`): clears the filter and the
        // per-LS rejection-tracking state, but **does not** clear
        // `n_filter_resets` (that's only cleared in `InitializeImpl`,
        // i.e. via constructing a fresh acceptor).
        self.filter.clear();
        self.last_rejection_due_to_filter = false;
        self.count_successive_filter_rejections = 0;
    }

    /// Port of `IpFilterLSAcceptor.cpp:CalculateAlphaMin` (lines
    /// 450-469). Returns `alpha_min_frac * alpha_min` where
    /// `alpha_min` is `gamma_theta` by default, tightened to
    /// `gamma_phi * theta / (-d_phi)` when `d_phi < 0`, and further
    /// tightened to `delta * theta^s_theta / (-d_phi)^s_phi` when
    /// `theta <= theta_min`.
    fn calc_alpha_min(&mut self, d_phi: Number, theta: Number) -> Number {
        let theta_min = self.ensure_theta_min(theta);
        let _ = self.ensure_theta_max(theta);
        let mut alpha_min = self.gamma_theta;
        if d_phi < 0.0 {
            alpha_min = alpha_min.min(self.gamma_phi * theta / (-d_phi));
            if theta <= theta_min {
                alpha_min = alpha_min
                    .min(self.delta_armijo * theta.powf(self.s_theta) / (-d_phi).powf(self.s_phi));
            }
        }
        self.alpha_min_frac * alpha_min
    }

    fn check_trial_point(
        &mut self,
        alpha_primal: Number,
        theta: Number,
        phi: Number,
        d_phi: Number,
        theta_trial: Number,
        phi_trial: Number,
    ) -> AcceptDecision {
        self.check_acceptability(alpha_primal, theta, phi, d_phi, theta_trial, phi_trial)
    }

    /// Port of `IpFilterLSAcceptor::UpdateForNextIteration` (lines
    /// 881-895). Returns `'f'` when both the switching condition fires
    /// AND Armijo holds (in which case the filter is **not** augmented);
    /// otherwise returns `'h'` and augments the filter with the
    /// pre-shrunken envelope `(theta_add, phi_add) = ((1 - γ_θ)·θ_ref,
    /// φ_ref - γ_φ·θ_ref)`.
    fn update_for_next_iteration(
        &mut self,
        alpha_primal: Number,
        theta: Number,
        phi: Number,
        d_phi: Number,
        phi_trial: Number,
    ) -> char {
        let is_ftype = self.is_switching_condition(alpha_primal, d_phi, theta);
        let armijo = self.armijo_holds(alpha_primal, d_phi, phi, phi_trial);
        if !is_ftype || !armijo {
            let phi_add = phi - self.gamma_phi * theta;
            let theta_add = (1.0 - self.gamma_theta) * theta;
            self.filter.add(theta_add, phi_add, 0);
            'h'
        } else {
            'f'
        }
    }

    /// Build the orig-progress callback for the inner restoration IPM.
    /// Clones the current filter and the iterate-acceptance constants
    /// into the closure so it can be evaluated repeatedly without
    /// holding a borrow on the acceptor.
    fn make_orig_progress_check(
        &self,
        reference_theta: Number,
        reference_barr: Number,
        // `called_from_restoration=true` in the closure below disables
        // the rapid-barrier-increase guard, which is the only place
        // upstream consumes `obj_max_inc`. Param kept on the trait
        // surface for parity with upstream's signature.
        _obj_max_inc: Number,
    ) -> Option<crate::restoration::OrigProgressCallback> {
        let filter_snapshot = self.filter.clone();
        let gamma_theta = self.gamma_theta;
        let gamma_phi = self.gamma_phi;
        Some(Box::new(move |trial_barr: Number, trial_theta: Number| {
            // 1. Filter acceptance — `IsAcceptableToCurrentFilter`.
            if filter_snapshot.dominated_by_any(trial_theta, trial_barr) {
                return false;
            }
            // 2. Iterate acceptance with `called_from_restoration=true`
            //    — disables the rapid-barrier-increase guard and runs
            //    only the sufficient-progress branch
            //    (`IpFilterLSAcceptor.cpp:495-498`).
            pounce_common::utils::compare_le(
                trial_theta,
                (1.0 - gamma_theta) * reference_theta,
                reference_theta,
            ) || pounce_common::utils::compare_le(
                trial_barr - reference_barr,
                -gamma_phi * reference_theta,
                reference_barr,
            )
        }))
    }

    /// Port of `IpFilterLSAcceptor::PrepareRestoPhaseStart` →
    /// `AugmentFilter` (`IpFilterLSAcceptor.cpp:297-308, 898-901`).
    /// Called by the algorithm immediately before restoration; adds
    /// the resto-entry envelope to the filter so that after recovery,
    /// the outer's Newton step is forced to make real progress vs
    /// the entry point. Without this, pounce on DECONVBNE enters
    /// restoration 323× (vs ipopt 21×): the outer accepts null-progress
    /// 'h' steps and immediately re-enters restoration.
    fn prepare_resto_phase_start(&mut self, reference_theta: Number, reference_barr: Number) {
        let phi_add = reference_barr - self.gamma_phi * reference_theta;
        let theta_add = (1.0 - self.gamma_theta) * reference_theta;
        self.filter.add(theta_add, phi_add, 0);
    }

    fn set_theta_max_fact(&mut self, theta_max_fact: Number) {
        self.theta_max_fact = theta_max_fact;
        self.theta_max = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switching_condition_requires_descent() {
        let a = FilterLsAcceptor::new();
        // d_phi >= 0 always returns false.
        assert!(!a.is_switching_condition(1.0, 0.0, 1.0));
        assert!(!a.is_switching_condition(1.0, 0.5, 1.0));
    }

    #[test]
    fn switching_condition_holds_when_descent_dominates_theta() {
        let a = FilterLsAcceptor::new();
        // alpha * (-d_phi)^s_phi vs delta * theta^s_theta.
        // alpha=1, d_phi=-1 → lhs=1; theta=1e-3, s_theta=1.1
        //   → rhs ≈ 1.0 * (1e-3)^1.1 ≈ 5e-4. lhs > rhs.
        assert!(a.is_switching_condition(1.0, -1.0, 1e-3));
    }

    #[test]
    fn armijo_strict_decrease() {
        let a = FilterLsAcceptor::new();
        // phi - phi_trial >= -eta_phi * alpha * d_phi  (with d_phi<0).
        // alpha=1, d_phi=-1, eta_phi=1e-8 → req: phi_trial - phi <= -1e-8.
        assert!(a.armijo_holds(1.0, -1.0, 0.0, -1e-7));
        assert!(!a.armijo_holds(1.0, -1.0, 0.0, 1e-7));
    }

    #[test]
    fn accept_when_filter_clear_and_progress_in_phi() {
        let mut a = FilterLsAcceptor::new();
        // Not in switching mode (small descent, small alpha).
        // Sufficient progress: phi_trial = -1 < phi=0 - gamma_phi*theta=0.
        let d = a.check_acceptability(1e-12, 1.0, 0.0, -1e-12, 1.0, -1.0);
        assert_eq!(d, AcceptDecision::Accept);
    }

    #[test]
    fn reject_when_filter_dominates_trial() {
        let mut a = FilterLsAcceptor::new();
        a.filter.add(0.5, -0.5, 0);
        // Trial (1.0, 1.0) is dominated by (0.5, -0.5).
        let d = a.check_acceptability(1.0, 1.0, 0.0, -1.0, 1.0, 1.0);
        assert_eq!(d, AcceptDecision::Reject);
    }

    #[test]
    fn reject_when_switching_but_armijo_fails() {
        let mut a = FilterLsAcceptor::new();
        // Switching mode active, but phi_trial is *worse* than phi.
        let d = a.check_acceptability(1.0, 1e-3, 0.0, -1.0, 1e-3, 1.0);
        assert_eq!(d, AcceptDecision::Reject);
    }

    #[test]
    fn accept_when_switching_and_armijo_holds() {
        let mut a = FilterLsAcceptor::new();
        // Switching mode and phi_trial is much smaller than phi.
        let d = a.check_acceptability(1.0, 1e-3, 0.0, -1.0, 1e-3, -1.0);
        assert_eq!(d, AcceptDecision::Accept);
    }

    #[test]
    fn calc_alpha_min_floor_at_alpha_min_frac_times_gamma_theta_when_no_descent() {
        let mut a = FilterLsAcceptor::new();
        // d_phi >= 0 → upstream skips both descent-based tightenings;
        // alpha_min = alpha_min_frac * gamma_theta.
        let v = a.calc_alpha_min(0.0, 1.0);
        assert!((v - 0.05 * 1e-5).abs() < 1e-20);
    }

    #[test]
    fn calc_alpha_min_uses_descent_term_when_d_phi_negative() {
        let mut a = FilterLsAcceptor::new();
        // theta=1 (large), so the second tightening (theta <= theta_min)
        // does NOT fire on first call; theta_min lazy-inits to
        // theta_min_fact*max(1,theta) = 1e-4. theta=1 > 1e-4.
        // alpha_min = min(gamma_theta, gamma_phi*theta/(-d_phi))
        //           = min(1e-5, 1e-8*1/1)  = 1e-8
        // returned = alpha_min_frac * 1e-8 = 5e-10.
        let v = a.calc_alpha_min(-1.0, 1.0);
        assert!((v - 0.05 * 1e-8).abs() < 1e-25);
    }

    #[test]
    fn calc_alpha_min_lazy_inits_theta_min_from_first_reference() {
        let mut a = FilterLsAcceptor::new();
        let _ = a.calc_alpha_min(0.0, 0.5);
        // theta_min should now be 1e-4 * max(1, 0.5) = 1e-4.
        assert!((a.theta_min.unwrap() - 1e-4).abs() < 1e-15);
        // Subsequent calls keep the original theta_min (matches
        // upstream where theta_min_ stays set across iterations).
        let _ = a.calc_alpha_min(0.0, 100.0);
        assert!((a.theta_min.unwrap() - 1e-4).abs() < 1e-15);
    }

    #[test]
    fn make_orig_progress_check_accepts_when_filter_clear_and_theta_drops() {
        use crate::line_search::ls_acceptor::BacktrackingLsAcceptor;
        let a = FilterLsAcceptor::new();
        // Empty filter, reference (theta=1.0, barr=0.0), trial below
        // (1-gamma_theta)*reference_theta ⇒ accepted via theta branch.
        let cb = a
            .make_orig_progress_check(1.0, 0.0, 5.0)
            .expect("FilterLsAcceptor returns Some");
        assert!(cb(2.0, 0.5)); // trial_barr=2.0 worse, but theta=0.5<<1.0
    }

    #[test]
    fn make_orig_progress_check_rejects_when_filter_dominates() {
        use crate::line_search::ls_acceptor::BacktrackingLsAcceptor;
        let mut a = FilterLsAcceptor::new();
        // Plant a filter entry that dominates the trial.
        a.filter.add(0.05, 0.0, 0);
        let cb = a
            .make_orig_progress_check(1.0, 0.0, 5.0)
            .expect("FilterLsAcceptor returns Some");
        // (theta_trial=0.1, barr_trial=0.5) is dominated by (0.05, 0.0).
        assert!(!cb(0.5, 0.1));
    }

    #[test]
    fn make_orig_progress_check_rejects_when_no_progress() {
        use crate::line_search::ls_acceptor::BacktrackingLsAcceptor;
        let a = FilterLsAcceptor::new();
        // Reference (theta=1.0, barr=0.0). Trial (theta=1.0, barr=2.0)
        // — no theta progress and barr increases. force_armijo skips
        // the rapid-increase guard, but the sufficient-progress
        // disjunction still fails on both branches.
        let cb = a
            .make_orig_progress_check(1.0, 0.0, 5.0)
            .expect("FilterLsAcceptor returns Some");
        assert!(!cb(2.0, 1.0));
    }

    #[test]
    fn filter_reset_heuristic_clears_after_trigger_consecutive_filter_rejected_accepts() {
        let mut a = FilterLsAcceptor::new();
        a.filter_reset_trigger = 2;
        a.max_filter_resets = 5;
        // Plant a filter entry that dominates the planned filter-reject
        // trial. Accept iterates between two trial sets:
        //   - theta_trial=1.0, phi_trial=10.0 (dominated by (0.5, 9.0))
        //   - theta_trial=0.4, phi_trial=8.0 (passes filter, accepted)
        // For the heuristic to fire we need the LS to record a
        // filter-due rejection just before each accept.
        a.filter.add(0.5, 9.0, 0);

        // Backtrack 1: filter-reject (sets last_rejection_due_to_filter).
        let r1 = a.check_acceptability(1.0, 1.0, 10.0, -1.0, 1.0, 9.5);
        assert_eq!(r1, AcceptDecision::Reject);
        assert!(a.last_rejection_due_to_filter);
        // Backtrack 2: accept on a smaller alpha → bumps count to 1.
        let r2 = a.check_acceptability(0.5, 1.0, 10.0, -1.0, 0.4, 8.0);
        assert_eq!(r2, AcceptDecision::Accept);
        assert_eq!(a.count_successive_filter_rejections, 1);
        assert_eq!(a.n_filter_resets, 0);
        assert!(!a.filter.entries().is_empty());

        // Re-plant filter and repeat for the second accept → count
        // reaches the trigger=2, filter is reset.
        a.filter.add(0.5, 9.0, 0);
        let r3 = a.check_acceptability(1.0, 1.0, 10.0, -1.0, 1.0, 9.5);
        assert_eq!(r3, AcceptDecision::Reject);
        let r4 = a.check_acceptability(0.5, 1.0, 10.0, -1.0, 0.4, 8.0);
        assert_eq!(r4, AcceptDecision::Accept);
        assert_eq!(a.n_filter_resets, 1);
        assert_eq!(a.count_successive_filter_rejections, 0);
        assert!(a.filter.entries().is_empty());
    }

    #[test]
    fn filter_reset_count_clears_when_last_rejection_was_iterate_test() {
        let mut a = FilterLsAcceptor::new();
        a.filter_reset_trigger = 2;
        a.filter.add(0.5, 9.0, 0);

        // First sequence: filter-rejection + accept → count = 1.
        let _ = a.check_acceptability(1.0, 1.0, 10.0, -1.0, 1.0, 9.5);
        let _ = a.check_acceptability(0.5, 1.0, 10.0, -1.0, 0.4, 8.0);
        assert_eq!(a.count_successive_filter_rejections, 1);

        // Second sequence: iterate-reject (theta and phi both fail
        // sufficient progress AND switching/Armijo) + accept → count
        // resets to 0, no filter reset.
        a.filter.clear();
        // Reject: not switching (small alpha), no theta progress
        // (theta_trial == theta), and phi is essentially unchanged
        // (no sufficient barrier decrease).
        let r1 = a.check_acceptability(1e-12, 1.0, 0.0, -1e-12, 1.0, 0.0);
        assert_eq!(r1, AcceptDecision::Reject);
        assert!(!a.last_rejection_due_to_filter);
        // Accept on a trial with theta progress.
        let r2 = a.check_acceptability(1e-12, 1.0, 0.0, -1e-12, 0.5, 0.0);
        assert_eq!(r2, AcceptDecision::Accept);
        assert_eq!(a.count_successive_filter_rejections, 0);
        assert_eq!(a.n_filter_resets, 0);
    }

    #[test]
    fn filter_reset_disabled_when_max_filter_resets_zero() {
        let mut a = FilterLsAcceptor::new();
        a.max_filter_resets = 0;
        a.filter_reset_trigger = 1;
        a.filter.add(0.5, 9.0, 0);

        // Any number of filter-reject + accept cycles must not reset.
        for _ in 0..5 {
            a.filter.add(0.5, 9.0, 0);
            let _ = a.check_acceptability(1.0, 1.0, 10.0, -1.0, 1.0, 9.5);
            let _ = a.check_acceptability(0.5, 1.0, 10.0, -1.0, 0.4, 8.0);
        }
        assert_eq!(a.n_filter_resets, 0);
        assert!(!a.filter.entries().is_empty());
    }

    #[test]
    fn accept_via_theta_progress_when_phi_unchanged() {
        let mut a = FilterLsAcceptor::new();
        // Not in switching mode; phi stays put but theta drops by
        // more than gamma_theta. Sufficient progress in theta.
        // theta=1, theta_trial=0.5 < (1-1e-5)*1 = 0.99999.
        let d = a.check_acceptability(1e-12, 1.0, 0.0, -1e-12, 0.5, 0.0);
        assert_eq!(d, AcceptDecision::Accept);
    }
}
